#pragma once

#include <QJsonArray>
#include <QString>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

// Modules 1 + 2 in the C++ app: draws the museum objects that pose_server.py places in the hands
// held out to the camera. pose_server.py decides which object, where and how it is held and
// supplies the harmonization fit (HandheldScene.describe in python_research/common/handheld.py);
// this class composites it exactly as HandheldScene.render() does - keep the two in step:
//   colour     the 3x4 colour matrix fitted to the harmonization networks' output
//   camera     the blur (or sharpening) of the hands holding the object and the camera's sensor noise
//   occlusion  the hands stay in front of an object held at its belly or handles
//   shadow     a soft contact shadow under an object standing on the palm
// Everything is drawn straight into the 8-bit frame (cv::blendLinear), from cut-outs whose colours
// are bled into their transparent pixels; only what changed is made again: an object smaller than
// its cut-out is resized whole (INTER_AREA) when its size or the harmonization update changes, one
// near or above its size (close to the camera) is sampled every frame, only where it is in the frame - into
// kept buffers: on Windows fresh memory costs ~0.2-0.5 ms per MB in page faults.

struct HeldObject {
    int         id = 0;
    std::string name;                   // sprite: python_research/data/objects/<name>.png
    cv::Size    size;                   // in the frame, pixels
    cv::Point   topLeft;
    std::string mode;                   // "palm" (stands on it), "sides" (held at its belly) or "handles"
    float       opacity = 0.0f;
    bool        raw = false;            // in its raw phase (both hands up: harmonized and raw in turn)
    bool        hasFit = false;         // false until the harmonization networks have seen it
    cv::Matx34f colour;                 // new BGR = colour * [b, g, r, 1], in 0-255 units
    float       sharpness = 0.0f;       // > 0: Gaussian blur sigma, < 0: unsharp-mask amount
    float       noise = 0.0f;           // sensor noise sigma, 0-255 units
    int         version = 0;            // changes with every harmonization update
    std::vector<std::vector<cv::Point2f>> hands;  // 21 points per hand that goes in front of it
    std::vector<cv::Point> centres;     // the centres of the hands holding it (debug view)
    int         numHands = 1;
    bool        grip = false;           // held by its handles
    bool        busy = false;           // both of the person's hands are up (automatic objects)
    bool        shadow = false;         // a contact shadow under its base (it stands on the palm)
    double      heightM = 0.0;          // real height
    double      sizeM = 0.0;            // the size shown: the catalog's (a sword lying lengthwise: its length)
};

class ObjectRenderer {
public:
    // Loads (and prepares) every cut-out in objectsDir now, so no object's first appearance holds up a frame.
    explicit ObjectRenderer(const QString& objectsDir);

    // The "objects" array of a pose_server.py reply.
    static std::vector<HeldObject> parse(const QJsonArray& objects);

    // Draw the objects onto the BGR frame, in place. harmonize = the H key: off, every object is
    // drawn raw; on, those with a fit and not in their raw phase are harmonized. noise >= 0: the
    // grain to give them (the frame's own, cameraNoise()); < 0: each object's fit's.
    void draw(cv::Mat& frame, const std::vector<HeldObject>& objects, bool harmonize, float noise = -1.0f);

    // Sensor noise sigma of a camera frame at its own resolution, on a central crop (mirrors
    // camera_noise in python_research/common/harmonization.py).
    static float cameraNoise(const cv::Mat& frame);

    // "Amphora Large Apulian" for "amphora_large_apulian".
    static std::string displayName(const std::string& name);

private:
    struct Part {          // (part of) an object at its size in the frame
        cv::Mat colours;   // BGR, 8-bit
        cv::Mat weight;    // its alpha, 0-1 (float)
        cv::Mat rest;      // 1 - weight
    };
    using SizeKey  = std::tuple<std::string, int, int, int>;  // name, width, height, update (-1: raw)
    struct Styled {        // an object's cut-out re-coloured for one update, and sharpened for one pixel step
        std::pair<std::string, int> key;  // name, update
        cv::Mat bgra;
        double  px = 0.0;                 // the step the sharpened one is for (0: none yet)
        cv::Mat sharpened;
    };
    static constexpr double SAMPLED_RATIO = 1.4;  // drawn taller than its cut-out / this: sampled where seen (as handheld.py)
    struct Kept {          // an object at its size: the top-left corners of buffers as large as its cut-out
        SizeKey key;
        cv::Mat sized, colours, alpha, weight, rest;
    };
    enum Work { DRAWN, ALPHA, COLOURS, WEIGHT, REST, OPAQUE_WEIGHT, OPAQUE_REST, GRAINED, BLURRED, WORK_COUNT };

    const cv::Mat* texture(const std::string& name);  // the bled cut-out (8-bit BGRA), nullptr if missing
    const cv::Mat& styled(const HeldObject& object, const cv::Mat& texture, double px);  // px 0: not sharpened
    Part part(const HeldObject& object, const cv::Mat& texture, const cv::Rect& visible, bool harmonized);
    const cv::Mat& shadow(const HeldObject& object, const cv::Mat& texture, cv::Point& at);
    cv::Mat grainWindow(cv::Size size);
    cv::Mat work(Work slot, cv::Size size, int type);  // a kept buffer's top-left corner: no new memory every frame

    std::string                                   dir_;
    std::map<std::string, cv::Mat>               textures_;  // per name (empty if missing)
    std::map<std::string, double>                spans_;     // per name: the width of the base, cut-out pixels
    std::map<int, Styled>                         styled_;    // per object id: its re-coloured cut-out
    std::map<int, Kept>                           sprites_;   // per object id: the whole object at its size
    cv::Mat                                       work_[WORK_COUNT];
    std::map<int, cv::Mat>                        shadows_;   // per radius: alpha canvas of the contact shadow
    cv::Mat                                       grain_;     // sensor noise as 128 + 32 n, the same in the 3 channels
    cv::Mat                                       weightLut_, restLut_;  // alpha (8-bit) -> weight, 1 - weight
    cv::RNG                                       rng_;
};
