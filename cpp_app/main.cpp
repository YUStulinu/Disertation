#include <QApplication>
#include <QScreen>
#include "webcamwidget.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    WebcamWidget window;

    // Open centred on the screen, in front of the console that started it
    window.winId();  // create the native window first, so frameGeometry() includes the title bar
    QRect frame = window.frameGeometry();
    frame.moveCenter(window.screen()->availableGeometry().center());
    window.move(frame.topLeft());
    window.show();
    window.raise();
    window.activateWindow();
    return app.exec();
}
