#include "pose_extractor.h"

#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>

PoseExtractor::PoseExtractor(const QString& pythonExecutable, const QString& scriptPath, const QString& workingDir,
                             const QStringList& extraArgs, QObject* parent)
    : QObject(parent)
    , process_(new QProcess(this))
{
    connect(process_, &QProcess::readyReadStandardOutput, this, &PoseExtractor::onReadyRead);
    connect(process_, &QProcess::readyReadStandardError, this, [this] {
        for (const QByteArray& line : process_->readAllStandardError().split('\n'))
            if (!line.trimmed().isEmpty()) qWarning().noquote() << "[pose_server]" << line.trimmed();
    });
    connect(process_, &QProcess::errorOccurred, this, [this, pythonExecutable](QProcess::ProcessError) {
        emit stopped(QString("cannot run %1: %2").arg(pythonExecutable, process_->errorString()));
    });
    connect(process_, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus) {
        emit stopped(QString("pose_server.py exited with code %1 (see console)").arg(exitCode));
    });

    process_->setWorkingDirectory(workingDir);
    process_->start(pythonExecutable, QStringList{"-u", scriptPath} + extraArgs);  // -u: unbuffered stdout
}

PoseExtractor::~PoseExtractor()
{
    if (process_->state() == QProcess::NotRunning) return;
    process_->blockSignals(true);
    process_->closeWriteChannel();  // EOF on stdin ends the server loop (it also stops its helper process)
    if (!process_->waitForFinished(3000)) process_->kill();
}

bool PoseExtractor::submitFrame(const QByteArray& jpegData, double time, double scale)
{
    if (!isRunning()) return false;
    process_->write(QByteArray::number(time, 'f', 4) + ' ' + QByteArray::number(scale, 'f', 6) + ' ' + jpegData.toBase64() + '\n');
    return true;
}

void PoseExtractor::sendCommand(const QByteArray& command)
{
    if (isRunning()) process_->write('#' + command + '\n');
}

bool PoseExtractor::isRunning() const
{
    return process_->state() == QProcess::Running;
}

void PoseExtractor::onReadyRead()
{
    readBuffer_ += process_->readAllStandardOutput();
    qsizetype newline;
    while ((newline = readBuffer_.indexOf('\n')) != -1) {
        const QByteArray line = readBuffer_.left(newline).trimmed();
        readBuffer_.remove(0, newline + 1);

        // {"pose": [[x0, y0], ..., [x32, y32]] | [] | null, "pose_time": t, "objects": [...], "hands": [...], "harmony": b}
        const QJsonObject json = QJsonDocument::fromJson(line).object();
        PoseReply reply;
        reply.hasPose = json["pose"].isArray();
        for (const QJsonValue& v : json["pose"].toArray()) {
            const QJsonArray pair = v.toArray();
            reply.keypoints.push_back(static_cast<float>(pair[0].toDouble()));
            reply.keypoints.push_back(static_cast<float>(pair[1].toDouble()));
        }
        reply.poseTime = json["pose_time"].toDouble();
        reply.objects  = json["objects"].toArray();
        reply.hands    = json["hands"].toArray();
        reply.harmony  = json["harmony"].toBool();
        emit replyReady(std::move(reply));
    }
}
