#pragma once

#include <QDir>
#include <QElapsedTimer>
#include <QJsonArray>
#include <QLabel>
#include <QWidget>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "action_inference.h"
#include "object_renderer.h"
#include "pose_extractor.h"

// Modules 1 + 2 live in Qt/C++.
//
// Per webcam frame:
//   1. A capture thread opens the webcam in Full HD (that takes seconds; the window does not wait)
//      and grabs and mirrors the frames as the camera delivers them.
//   2. Whenever pose_server.py is free, a 540-row copy of the newest frame (the hand and pose
//      models work on smaller images anyway) is JPEG-encoded and sent with its capture time and
//      scale (one frame in flight -> latency cannot build up); the answer comes back at the full
//      frame's scale. The next one is sent as soon as the answer arrives, before that answer is
//      drawn, so the drawing overlaps the server's work.
//   3. Module 1: each new pose (33 keypoints) is buffered with the capture time of its frame;
//      SEQUENCE_LENGTH samples, SAMPLE_RATE per second, are interpolated from the buffer by time -
//      so the model sees the same motion speed at any frame rate - normalised (normalizer.h) and
//      classified by ActionInference (ONNX Runtime).
//   4. Modules 1 + 2: the museum objects pose_server.py places in the hands held out are drawn
//      by ObjectRenderer - harmonized, at their real size, at the Full HD detail of the museum
//      photos - onto the very frame the hands were found in, so they never lag behind them.
//   5. The action, the objects and the frame rates are drawn on the video.
// Keys: H = harmonization on/off, O = objects on/off, N = next object, D = hand landmarks,
//       S = save the frame shown (images/qt_cpp_app/objects_<n>.png, or <action>_<n>.png with
//       --no-objects), Q / Esc = quit (closing the window quits too).
// Arguments: --video FILE (a recorded video, played in a loop, instead of the webcam),
//            --no-objects (Module 1 only).
// SAMPLE_RATE / SEQUENCE_LENGTH must match python_research/common/pose.py.

// Events per second, from the moving average of the intervals between them (averaging the
// intervals, not their inverses, which would overstate an irregular rate).
struct RateMeter {
    double last = -1.0, interval = 0.0;
    void tick(double t)
    {
        if (last >= 0.0 && t > last) interval = interval > 0.0 ? 0.9 * interval + 0.1 * (t - last) : t - last;
        if (t > last) last = t;
    }
    double perSecond() const { return interval > 0.0 ? 1.0 / interval : 0.0; }
};

class WebcamWidget : public QWidget
{
    Q_OBJECT

public:
    explicit WebcamWidget(QWidget* parent = nullptr);
    ~WebcamWidget() override;

protected:
    void keyPressEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void onFrame();
    void onReply(PoseReply reply);

private:
    static constexpr double SAMPLE_RATE    = 15.0;
    static constexpr double WINDOW_SECONDS = (ActionInference::SEQUENCE_LENGTH - 1) / SAMPLE_RATE;
    static constexpr float  SMOOTHING      = 0.8f;  // weight of the previous probabilities (~0.2 s memory)
    static constexpr float  MIN_CONFIDENCE = 0.6f;  // below this the action is shown as "uncertain"
    static constexpr double WINDOW_SCALE   = 1.3;   // the window starts a little larger than the camera image,
                                                    // or smaller so that a Full HD one fits the screen
    static constexpr int    CAMERA_WIDTH   = 1920;  // the webcam's resolution: Full HD (it gives the closest it has)
    static constexpr int    CAMERA_HEIGHT  = 1080;
    static constexpr int    DETECT_ROWS    = 540;   // pose_server.py gets a copy of the frame this tall (Full HD / 2)
    static constexpr size_t IN_FLIGHT      = 2;     // frames at pose_server.py at once: the next one is there when it
                                                    // finishes one, while this side resizes and encodes the one after
    static constexpr int    NOISE_EVERY    = 15;    // frames between two measurements of the camera's noise
    static constexpr double NOTE_SECONDS   = 2.0;   // how long "Saved ..." stays on screen

    QDir                             root_;  // project root (contains python_research/)
    cv::Size                         frameSize_{CAMERA_WIDTH, CAMERA_HEIGHT};  // the camera's (until known: asked)
    QLabel*                          imageLabel_;
    QElapsedTimer                    clock_;
    PoseExtractor*                   poseExtractor_;
    std::unique_ptr<ActionInference> actionInference_;
    std::unique_ptr<ObjectRenderer>  objectRenderer_;  // null with --no-objects

    // The capture thread and the newest frame it has grabbed (empty once taken)
    cv::VideoCapture  capture_;
    bool              fromVideo_ = false;
    std::thread       grabber_;
    std::atomic<bool> stopGrabbing_{false};
    std::atomic<bool> frameNotified_{false};  // an onFrame() call is already queued
    std::mutex        frameMutex_;
    cv::Mat           latestFrame_;
    double            latestTime_ = 0.0;

    std::deque<double>             times_;   // capture time (s) of each buffered skeleton
    std::deque<std::vector<float>> frames_;  // 66 floats each: x, y of the 33 landmarks
    std::deque<cv::Mat> pendingFrames_;      // the frames sent to pose_server.py, oldest first (its replies come in order)
    cv::Mat     shown_;                      // the last frame shown: redrawn when the window is resized, saved by S
    std::string note_;                       // a short message on screen ("Saved ...") until noteUntil_
    double      noteUntil_ = -1.0;
    bool        answered_ = false;           // pose_server.py has answered (false again if it stops)
    RateMeter   videoRate_, poseRate_;       // frames shown, new poses (capture times)
    bool        personVisible_ = false;
    std::string status_{"Starting pose server..."};
    std::vector<float> smoothed_;            // time-smoothed class probabilities
    std::string lastLabel_;                  // last shown action, logged to the console on change

    std::vector<HeldObject> objects_;
    float       cameraNoise_ = -1.0f;        // the camera's sensor noise, for the objects' grain (< 0: unknown)
    int         noiseCountdown_ = 0;
    QJsonArray  hands_;
    bool        harmonyReady_ = false, harmonize_ = true, showObjects_ = true, showHands_ = false;

    struct TextBar {  // a bar of the overlay, kept: anti-aliased text in Full HD takes ~4 ms, it changes rarely
        std::string key;
        cv::Mat     image;
    };
    mutable TextBar topBar_, bottomBar_;

    void grabLoop();
    void sendNewest();
    void classify(std::vector<float> keypoints, double t);
    void drawOverlay(cv::Mat& frame) const;
    void showStatus();
    void display(const cv::Mat& frame);
    void saveShot();
    std::vector<float> sampleWindow(double end) const;
};
