#pragma once

#include <QByteArray>
#include <QJsonArray>
#include <QObject>
#include <QProcess>
#include <QStringList>
#include <vector>

// Runs python_research/pose_server.py as a persistent child process and exchanges JPEG frames /
// JSON replies over stdin/stdout (the protocol is described in pose_server.py).
// Python's stderr (startup messages, tracebacks) is forwarded to qWarning().

struct PoseReply {
    bool               hasPose = false;  // false: no new pose since the previous reply
    std::vector<float> keypoints;        // 33*2 = 66 floats (x, y per landmark), empty if nobody is in view
    double             poseTime = 0.0;   // capture time (s) of the frame the pose comes from
    QJsonArray         objects;          // the museum objects to draw (ObjectRenderer::parse)
    QJsonArray         hands;            // every hand found: {"points": 21 [x, y], "centre": [x, y], "hold": "open"/"grip"/null}
    bool               harmony = false;  // the harmonization networks are loaded
};

class PoseExtractor : public QObject
{
    Q_OBJECT

public:
    PoseExtractor(const QString& pythonExecutable, const QString& scriptPath, const QString& workingDir,
                  const QStringList& extraArgs = {}, QObject* parent = nullptr);
    ~PoseExtractor() override;

    // Send one frame, captured at `time` seconds, `scale` times smaller than the image the answer is
    // for (places and sizes come back at that image's scale); replyReady() follows asynchronously.
    bool submitFrame(const QByteArray& jpegData, double time, double scale = 1.0);
    // A command without reply: "next", "objects 0", "objects 1".
    void sendCommand(const QByteArray& command);
    bool isRunning() const;

signals:
    void replyReady(PoseReply reply);
    void stopped(QString reason);

private slots:
    void onReadyRead();

private:
    QProcess*  process_;
    QByteArray readBuffer_;
};
