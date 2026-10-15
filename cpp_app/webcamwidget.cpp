#include "webcamwidget.h"
#include "normalizer.h"

#include <QCoreApplication>
#include <QDebug>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QImage>
#include <QJsonObject>
#include <QKeyEvent>
#include <QMessageBox>
#include <QPixmap>
#include <QScreen>
#include <algorithm>
#include <chrono>
#include <cstdio>

namespace {

// Walk up from the .exe folder and the working directory to the folder containing
// python_research/ - works from build/Debug, the repo root or a deployed copy.
QDir findProjectRoot()
{
    for (const QString& start : {QCoreApplication::applicationDirPath(), QDir::currentPath()}) {
        QDir dir(start);
        do {
            if (dir.exists("python_research/pose_server.py")) return dir;
        } while (dir.cdUp());
    }
    return QDir::current();
}

// Interpreter for pose_server.py: $ACTION_APP_PYTHON, else the project venv, else "python" on PATH.
QString pythonExecutable(const QDir& root)
{
    const QString fromEnv = qEnvironmentVariable("ACTION_APP_PYTHON");
    if (!fromEnv.isEmpty()) return fromEnv;
    const QString venv = root.filePath("python_research/venv/Scripts/python.exe");
    return QFileInfo::exists(venv) ? venv : QStringLiteral("python");
}

cv::Point toPoint(const QJsonValue& xy)  // truncated, as numpy's astype(int)
{
    return {static_cast<int>(xy.toArray()[0].toDouble()), static_cast<int>(xy.toArray()[1].toDouble())};
}

// Debug view: the hand skeletons - green when held open, orange when gripping, grey otherwise -
// and a line between two hands that share an object (mirrors draw_hands in common/hands.py and
// draw_pairs in ar_objects.py).
void drawHands(cv::Mat& frame, const QJsonArray& hands, const std::vector<HeldObject>& objects)
{
    const int thick = std::max(2, cvRound(2 * frame.rows / 480.0)), radius = std::max(4, cvRound(4 * frame.rows / 480.0));
    static const std::vector<std::vector<int>> LINES = {
        {0, 1, 2, 3, 4}, {0, 5, 6, 7, 8}, {0, 9, 10, 11, 12}, {0, 13, 14, 15, 16}, {0, 17, 18, 19, 20}, {5, 9, 13, 17}};
    for (const QJsonValue& value : hands) {
        const QJsonObject hand = value.toObject();
        const QString hold = hand["hold"].toString();
        const cv::Scalar colour = hold == "open" ? cv::Scalar(0, 220, 0)
                                : hold == "grip" ? cv::Scalar(0, 160, 255) : cv::Scalar(160, 160, 160);
        std::vector<cv::Point> points;
        for (const QJsonValue& p : hand["points"].toArray()) points.push_back(toPoint(p));
        if (points.size() < 21) continue;
        for (const auto& line : LINES) {
            std::vector<cv::Point> polyline;
            for (int i : line) polyline.push_back(points[i]);
            cv::polylines(frame, std::vector<std::vector<cv::Point>>{polyline}, false, colour, thick);
        }
        cv::circle(frame, toPoint(hand["centre"]), radius, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    for (const HeldObject& o : objects)
        if (o.centres.size() == 2) cv::line(frame, o.centres[0], o.centres[1], cv::Scalar(0, 255, 255), thick);
}

}  // namespace

WebcamWidget::WebcamWidget(QWidget* parent)
    : QWidget(parent)
    , root_(findProjectRoot())
{
    const QStringList args = QCoreApplication::arguments();
    const bool withObjects = !args.contains("--no-objects");
    setWindowTitle(withObjects ? "Action Recognition + Museum Objects in Your Hands (Qt/C++)"
                               : "Module 1 - Action Recognition (Qt/C++) | S = Save, Q = Quit");
    setFocusPolicy(Qt::StrongFocus);
    QPalette black = palette();
    black.setColor(QPalette::Window, Qt::black);  // around the image when the window has another shape
    setPalette(black);
    setAutoFillBackground(true);
    imageLabel_ = new QLabel(this);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);  // the image follows the window
    imageLabel_->setMinimumSize(320, 240);
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(imageLabel_);

    const QString models = root_.filePath("python_research/models/");
    actionInference_ = std::make_unique<ActionInference>((models + "action_classifier.onnx").toStdWString(),
                                                         (models + "action_classifier.labels.txt").toStdWString());
    if (!actionInference_->isLoaded())
        QMessageBox::warning(this, "ONNX model", QString("Could not load the classifier from %1\n%2\nRun export_onnx.py.")
                                                     .arg(models, QString::fromStdString(actionInference_->error())));
    if (withObjects) objectRenderer_ = std::make_unique<ObjectRenderer>(root_.filePath("python_research/data/objects"));

    const QString python = pythonExecutable(root_);
    qInfo().noquote() << "Project root:" << root_.absolutePath() << "| Python:" << python;
    poseExtractor_ = new PoseExtractor(python, root_.filePath("python_research/pose_server.py"), root_.absolutePath(),
                                       withObjects ? QStringList{} : QStringList{"--no-objects"}, this);
    connect(poseExtractor_, &PoseExtractor::replyReady, this, &WebcamWidget::onReply);
    connect(poseExtractor_, &PoseExtractor::stopped, this, [this](const QString& reason) {
        status_ = "Pose server stopped";
        answered_ = false;  // show the camera as it is
        pendingFrames_.clear();
        qWarning().noquote() << reason;
    });

    const int video = args.indexOf("--video");
    const bool fromFile = video >= 0 && video + 1 < args.size();
    if (fromFile) {
        fromVideo_ = capture_.open(args[video + 1].toStdString());
        if (!fromVideo_) {
            status_ = "No video";
            QMessageBox::warning(this, "Video error", "Could not open " + args[video + 1]);
        }
        const double w = capture_.get(cv::CAP_PROP_FRAME_WIDTH), h = capture_.get(cv::CAP_PROP_FRAME_HEIGHT);
        if (w > 0 && h > 0) frameSize_ = {cvRound(w), cvRound(h)};
    }
    const QRect screen = QGuiApplication::primaryScreen()->availableGeometry();  // a Full HD frame must fit it
    const double scale = std::min({WINDOW_SCALE, 0.62 * screen.height() / frameSize_.height, 0.9 * screen.width() / frameSize_.width});
    resize(qRound(frameSize_.width * scale), qRound(frameSize_.height * scale));
    clock_.start();
    showStatus();
    if (fromVideo_ || !fromFile) grabber_ = std::thread(&WebcamWidget::grabLoop, this);  // it opens the webcam
}

WebcamWidget::~WebcamWidget()
{
    stopGrabbing_ = true;
    if (grabber_.joinable()) grabber_.join();
}

// The capture thread: the newest frame and its capture time, for onFrame() in the GUI thread.
// The webcam is opened here (that takes seconds) and sets the pace; a video file is played at
// its own frame rate, in a loop.
void WebcamWidget::grabLoop()
{
    if (!fromVideo_ && !capture_.open(0)) {
        QMetaObject::invokeMethod(this, [this] {
            status_ = "No webcam";
            showStatus();
            QMessageBox::warning(this, "Webcam error", "Could not open the webcam (index 0).");
        }, Qt::QueuedConnection);
        return;
    }
    if (!fromVideo_) {  // the webcam's Full HD, or the closest resolution it has
        capture_.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    }
    using Clock = std::chrono::steady_clock;
    const double fps = capture_.get(cv::CAP_PROP_FPS);
    const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / (fps > 0 ? fps : 30.0)));
    auto next = Clock::now();
    while (!stopGrabbing_) {
        cv::Mat frame;
        if (!capture_.read(frame) || frame.empty()) {
            if (fromVideo_ && capture_.get(cv::CAP_PROP_POS_FRAMES) > 0)
                capture_.set(cv::CAP_PROP_POS_FRAMES, 0);  // the end of the video: from the start again
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(20));  // no frame (camera busy or gone)
            continue;
        }
        if (fromVideo_) {
            next = std::max(next + period, Clock::now() - period);
            std::this_thread::sleep_until(next);
        } else {
            cv::flip(frame, frame, 1);  // mirror the webcam
        }
        {
            const std::lock_guard<std::mutex> lock(frameMutex_);
            latestFrame_ = frame;
            latestTime_ = clock_.nsecsElapsed() / 1e9;
        }
        if (!frameNotified_.exchange(true)) QMetaObject::invokeMethod(this, [this] { onFrame(); }, Qt::QueuedConnection);
    }
}

void WebcamWidget::onFrame()
{
    frameNotified_ = false;
    if (!answered_) {  // until pose_server.py answers (or if it stopped): the camera as it is
        cv::Mat frame;
        {
            const std::lock_guard<std::mutex> lock(frameMutex_);
            frame = latestFrame_.clone();  // a copy: the frame itself is still to be sent
        }
        if (!frame.empty()) {
            drawOverlay(frame);
            display(frame);
        }
    }
    if (pendingFrames_.size() < IN_FLIGHT) sendNewest();
}

// Send the newest frame to pose_server.py, unless it has already had it.
void WebcamWidget::sendNewest()
{
    if (!poseExtractor_->isRunning()) return;
    cv::Mat frame;
    double time = 0.0;
    {
        const std::lock_guard<std::mutex> lock(frameMutex_);
        if (latestFrame_.empty()) return;
        frame = latestFrame_;
        time = latestTime_;
        latestFrame_.release();  // taken: only this frame holds the pixels now, they can be drawn on
    }
    cv::Mat small = frame;  // the hand and pose models work on smaller images anyway
    double scale = 1.0;
    if (frame.rows > DETECT_ROWS) {
        scale = frame.rows / static_cast<double>(DETECT_ROWS);
        cv::resize(frame, small, {cvRound(frame.cols / scale), DETECT_ROWS}, 0, 0, cv::INTER_AREA);
    }
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", small, jpeg, {cv::IMWRITE_JPEG_QUALITY, 90});
    if (poseExtractor_->submitFrame(QByteArray(reinterpret_cast<const char*>(jpeg.data()), static_cast<qsizetype>(jpeg.size())),
                                    time, scale))
        pendingFrames_.push_back(frame);
}

void WebcamWidget::onReply(PoseReply reply)
{
    answered_ = true;
    cv::Mat frame;
    if (!pendingFrames_.empty()) {  // the frame this answer is about: the oldest one sent
        frame = std::move(pendingFrames_.front());
        pendingFrames_.pop_front();
    }
    sendNewest();  // pose_server.py already has the next one; this one is for after it, and is made while this is drawn

    videoRate_.tick(clock_.nsecsElapsed() / 1e9);
    if (reply.hasPose) classify(std::move(reply.keypoints), reply.poseTime);
    if (objectRenderer_) {
        objects_ = ObjectRenderer::parse(reply.objects);
        hands_ = reply.hands;
        harmonyReady_ = reply.harmony;
    }
    if (frame.empty()) return;
    // drawn on the frame the hands were found in: the objects never lag behind the hands
    if (objectRenderer_ && showObjects_) {
        if (--noiseCountdown_ <= 0) {  // their grain: the camera's own, measured on the image they go into
            cameraNoise_ = ObjectRenderer::cameraNoise(frame);
            noiseCountdown_ = NOISE_EVERY;
        }
        objectRenderer_->draw(frame, objects_, harmonize_, cameraNoise_);
    }
    if (showHands_) drawHands(frame, hands_, objects_);
    drawOverlay(frame);
    display(frame);
}

void WebcamWidget::classify(std::vector<float> keypoints, double t)
{
    if (!times_.empty() && t <= times_.back()) return;  // not newer than what is buffered
    poseRate_.tick(t);
    personVisible_ = keypoints.size() == static_cast<size_t>(Normalizer::NUM_FEATURES);
    if (!personVisible_) {  // keep the buffer, just skip frames without a person
        status_ = "No person detected";
        return;
    }

    times_.push_back(t);
    frames_.push_back(std::move(keypoints));
    while (times_.front() < t - WINDOW_SECONDS - 0.5) {  // keep a little more than one window
        times_.pop_front();
        frames_.pop_front();
    }
    const double covered = (times_.back() - times_.front()) / WINDOW_SECONDS;
    if (covered < 1.0) {
        status_ = "Collecting frames " + std::to_string(static_cast<int>(100 * covered)) + "%";
        return;
    }
    const std::vector<float> probs = actionInference_->predict(Normalizer::normalizeSequence(sampleWindow(t)));
    if (probs.empty()) {
        status_ = "Model error";
        return;
    }
    if (smoothed_.size() != probs.size()) smoothed_ = probs;
    for (size_t i = 0; i < probs.size(); ++i)  // exponential moving average -> no flicker
        smoothed_[i] = SMOOTHING * smoothed_[i] + (1.0f - SMOOTHING) * probs[i];

    const size_t best = std::max_element(smoothed_.begin(), smoothed_.end()) - smoothed_.begin();
    const std::string label = smoothed_[best] < MIN_CONFIDENCE ? "uncertain" : actionInference_->labels()[best];
    char text[96];
    std::snprintf(text, sizeof(text), "%s  %.0f%%", label.c_str(), 100.0f * smoothed_[best]);
    status_ = text;
    if (label != lastLabel_) {
        lastLabel_ = label;
        qInfo().noquote() << "Action:" << text;
    }
}

void WebcamWidget::drawOverlay(cv::Mat& frame) const
{
    const double u = frame.rows / 480.0;  // sizes are for a 480-row frame
    const auto px = [u](double v) { return cvRound(v * u); };
    const int thin = std::max(1, px(1)), bold = std::max(2, px(2));
    const std::string size = std::to_string(frame.cols) + "x" + std::to_string(frame.rows);
    // Module 1: the action, at the top - its bar drawn again only when its text changes
    const std::string pose = "pose " + std::to_string(static_cast<int>(poseRate_.perSecond() + 0.5)) + " fps";
    const std::string topKey = status_ + '\n' + pose + '\n' + (personVisible_ ? "person" : "none") + '\n' + size;
    if (topBar_.key != topKey) {
        topBar_.image.create(px(56), frame.cols, CV_8UC3);
        topBar_.image.setTo(cv::Scalar(20, 20, 20));
        cv::putText(topBar_.image, status_, {px(10), px(38)}, cv::FONT_HERSHEY_SIMPLEX, u,
                    personVisible_ ? cv::Scalar(0, 230, 0) : cv::Scalar(0, 180, 255), bold, cv::LINE_AA);
        cv::putText(topBar_.image, pose, {frame.cols - px(130), px(36)}, cv::FONT_HERSHEY_SIMPLEX, 0.55 * u,
                    cv::Scalar(170, 170, 170), thin, cv::LINE_AA);
        topBar_.key = topKey;
    }
    topBar_.image.copyTo(frame(cv::Rect(0, 0, frame.cols, topBar_.image.rows)));
    if (!objectRenderer_) return;

    // Modules 1 + 2: the objects, at the bottom (the lines of draw_status in ar_objects.py)
    const cv::Scalar grey(210, 210, 210), green(90, 220, 90), red(80, 80, 255);
    std::vector<std::pair<std::string, cv::Scalar>> lines;
    char text[200];
    std::snprintf(text, sizeof(text), "FPS %d | H Harmonization | O Objects | N Next Object | S Save | Q Quit",
                  static_cast<int>(videoRate_.perSecond() + 0.5));
    lines.emplace_back(text, grey);
    const bool allHarmonized = std::all_of(objects_.begin(), objects_.end(), [](const HeldObject& o) { return !o.raw; });
    if (!showObjects_)
        lines.emplace_back("Objects Hidden (O Shows Them)", red);
    else if (!harmonyReady_)
        lines.emplace_back("Loading the Harmonization Networks...", grey);
    else if (harmonize_ && allHarmonized)  // both hands up switch it on and off by themselves
        lines.emplace_back("Harmonization ON: Light, Sharpness and Noise Matched to the Camera", green);
    else
        lines.emplace_back("Harmonization OFF: Objects Shown Raw, as Photographed in the Museum", red);
    if (showObjects_) {
        for (const HeldObject& o : objects_) {
            std::snprintf(text, sizeof(text), "%d Hand%s, %s: %s (%.0f cm)%s", o.numHands, o.numHands > 1 ? "s" : "",
                          o.grip ? "Grip" : "Open", ObjectRenderer::displayName(o.name).c_str(), 100.0 * o.sizeM,
                          !o.busy ? "" : harmonize_ && !o.raw ? " - Harmonized" : " - Raw");
            lines.emplace_back(text, grey);
        }
    }
    std::string bottomKey = size;  // the same text in the same colours: the same bar
    for (const auto& [line, colour] : lines)
        bottomKey += '\n' + line + '/' + std::to_string(static_cast<int>(colour[1]));
    const int step = px(20), height = px(12) + step * static_cast<int>(lines.size());
    if (bottomBar_.key != bottomKey) {
        bottomBar_.image.create(height, frame.cols, CV_8UC3);
        bottomBar_.image.setTo(cv::Scalar(20, 20, 20));
        for (size_t i = 0; i < lines.size(); ++i)
            cv::putText(bottomBar_.image, lines[i].first, {px(8), px(18) + step * static_cast<int>(i)}, cv::FONT_HERSHEY_SIMPLEX,
                        0.5 * u, lines[i].second, thin, cv::LINE_AA);
        bottomBar_.key = bottomKey;
    }
    bottomBar_.image.copyTo(frame(cv::Rect(0, frame.rows - height, frame.cols, height)));
}

// A black image with the status: until the first camera frame, or without a camera.
void WebcamWidget::showStatus()
{
    cv::Mat blank(frameSize_, CV_8UC3, cv::Scalar::all(0));
    drawOverlay(blank);
    display(blank);
}

// Show the frame scaled to the window (keeping its shape), sharp on high-DPI screens too.
void WebcamWidget::display(const cv::Mat& frame)
{
    shown_ = frame;
    cv::Mat screen = frame;
    if (clock_.nsecsElapsed() / 1e9 < noteUntil_) {  // under the top bar, on a copy: S saves the frame without it
        screen = frame.clone();
        const double u = frame.rows / 480.0;  // sizes are for a 480-row frame
        const int thin = std::max(1, cvRound(u)), pad = cvRound(8 * u);
        int base = 0;
        const cv::Size size = cv::getTextSize(note_, cv::FONT_HERSHEY_SIMPLEX, 0.55 * u, thin, &base);
        const cv::Point at(screen.cols - size.width - cvRound(14 * u), cvRound(64 * u) + size.height);
        cv::rectangle(screen, {at.x - pad, at.y - size.height - pad, size.width + 2 * pad, size.height + 2 * pad},
                      cv::Scalar(20, 20, 20), cv::FILLED);
        cv::putText(screen, note_, at, cv::FONT_HERSHEY_SIMPLEX, 0.55 * u, cv::Scalar(90, 220, 90), thin, cv::LINE_AA);
    }
    const qreal dpr = devicePixelRatioF();
    const double scale = std::min(imageLabel_->width() * dpr / screen.cols, imageLabel_->height() * dpr / screen.rows);
    cv::Mat resized;  // INTER_AREA only where linear interpolation would skip pixels
    cv::resize(screen, resized, {std::max(1, qRound(screen.cols * scale)), std::max(1, qRound(screen.rows * scale))}, 0, 0,
               scale < 0.5 ? cv::INTER_AREA : cv::INTER_LINEAR);
    QImage image(resized.cols, resized.rows, QImage::Format_RGB32);  // the screen's own format: no conversion later
    cv::Mat target(image.height(), image.width(), CV_8UC4, image.bits(), image.bytesPerLine());
    cv::cvtColor(resized, target, cv::COLOR_BGR2BGRA);
    image.setDevicePixelRatio(dpr);
    imageLabel_->setPixmap(QPixmap::fromImage(std::move(image)));
}

// The S key: the frame shown, as images/qt_cpp_app/<name>_<n>.png - name = "objects", or with
// --no-objects the action shown - with n the first free number.
void WebcamWidget::saveShot()
{
    if (shown_.empty()) return;
    const QString name = objectRenderer_ ? QStringLiteral("objects")
                       : personVisible_ && !lastLabel_.empty() ? QString::fromStdString(lastLabel_) : QStringLiteral("frame");
    const QDir dir(root_.filePath("images/qt_cpp_app"));
    dir.mkpath(".");
    int n = 1;
    while (dir.exists(QString("%1_%2.png").arg(name).arg(n))) ++n;
    const QString path = dir.filePath(QString("%1_%2.png").arg(name).arg(n));
    if (!cv::imwrite(path.toStdString(), shown_)) {
        qWarning().noquote() << "Could not save" << QDir::toNativeSeparators(path);
        return;
    }
    qInfo().noquote() << "Saved:" << QDir::toNativeSeparators(path);
    note_ = "Saved " + QFileInfo(path).fileName().toStdString();
    noteUntil_ = clock_.nsecsElapsed() / 1e9 + NOTE_SECONDS;
    display(shown_);  // the note at once, even if no new frame comes
}

void WebcamWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    if (!shown_.empty()) display(shown_);
}

void WebcamWidget::keyPressEvent(QKeyEvent* event)
{
    if (event->isAutoRepeat()) return;  // a key held down toggles once
    switch (event->key()) {
    case Qt::Key_Q:
    case Qt::Key_Escape:
        close();
        break;
    case Qt::Key_H:
        harmonize_ = !harmonize_;
        break;
    case Qt::Key_O:
        showObjects_ = !showObjects_;
        poseExtractor_->sendCommand(showObjects_ ? "objects 1" : "objects 0");  // no hand tracking while hidden
        if (!showObjects_) objects_.clear();
        break;
    case Qt::Key_N:
        poseExtractor_->sendCommand("next");
        break;
    case Qt::Key_D:
        showHands_ = !showHands_;
        break;
    case Qt::Key_S:
        saveShot();
        break;
    default:
        QWidget::keyPressEvent(event);
    }
}

// SEQUENCE_LENGTH skeletons at 1/SAMPLE_RATE spacing ending at `end`, linearly
// interpolated between the buffered frames (mirrors sample_window() in common/pose.py).
std::vector<float> WebcamWidget::sampleWindow(double end) const
{
    std::vector<float> flat;
    flat.reserve(ActionInference::INPUT_SIZE);
    for (int k = ActionInference::SEQUENCE_LENGTH - 1; k >= 0; --k) {
        const double q = end - k / SAMPLE_RATE;
        const auto upper = static_cast<size_t>(std::lower_bound(times_.begin(), times_.end(), q) - times_.begin());
        const size_t i = std::clamp<size_t>(upper, 1, times_.size() - 1);
        const double span = std::max(times_[i] - times_[i - 1], 1e-9);
        const float w = static_cast<float>(std::clamp((q - times_[i - 1]) / span, 0.0, 1.0));
        for (int j = 0; j < Normalizer::NUM_FEATURES; ++j)
            flat.push_back(frames_[i - 1][j] * (1 - w) + frames_[i][j] * w);
    }
    return flat;
}
