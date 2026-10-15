#include "object_renderer.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonObject>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>

namespace {

// dst = weight * src + (1 - weight) * dst where weight (one-channel float, 0-1) placed at `at`
// overlaps dst, in place: 8-bit images (src empty = black, src as large as weight; rest = 1 - weight
// if made already), through cv::blendLinear - mirrors _blend() in common/handheld.py.
void blend(cv::Mat& dst, const cv::Mat& src, const cv::Mat& weight, cv::Point at, const cv::Mat& rest = cv::Mat())
{
    const cv::Rect area = cv::Rect(at, weight.size()) & cv::Rect(0, 0, dst.cols, dst.rows);
    if (area.empty()) return;
    const cv::Rect inside = area - at;
    cv::Mat part = dst(area);
    const cv::Mat w = weight(inside);
    cv::blendLinear(part, src.empty() ? cv::Mat(cv::Mat::zeros(part.size(), part.type())) : src(inside),
                    rest.empty() ? cv::Mat(1.0 - w) : rest(inside), w, part);
}

// The cut-out (8-bit BGRA) with every transparent pixel given the colour of the object around it -
// pull-push through a premultiplied pyramid - so that resizing or blurring its straight colours
// never pulls the photo's background or a dark rim into its edge (mirrors _bleed).
cv::Mat bleed(const cv::Mat& sprite)
{
    cv::Mat premultiplied;
    sprite.convertTo(premultiplied, CV_32F);
    premultiplied.forEach<cv::Vec4f>([](cv::Vec4f& p, const int*) {  // alpha 0-1, then colours * alpha, as numpy does
        p[3] /= 255.0f;
        for (int c = 0; c < 3; ++c) p[c] *= p[3];
    });
    std::vector<cv::Mat> levels{premultiplied};
    while (std::min(levels.back().rows, levels.back().cols) > 1) {
        cv::Mat down;
        cv::pyrDown(levels.back(), down);
        levels.push_back(down);
    }
    cv::Mat colours = cv::Mat::zeros(levels.back().size(), CV_32FC3);
    for (auto level = levels.rbegin(); level != levels.rend(); ++level) {
        if (colours.size() != level->size()) {
            cv::Mat up;
            cv::pyrUp(colours, up, level->size());
            colours = up;
        }
        for (int y = 0; y < level->rows; ++y) {  // each level's own colours wherever it has any alpha
            const cv::Vec4f* in = level->ptr<cv::Vec4f>(y);
            cv::Vec3f* out = colours.ptr<cv::Vec3f>(y);
            for (int x = 0; x < level->cols; ++x)
                if (in[x][3] > 0.0f) out[x] = {in[x][0] / in[x][3], in[x][1] / in[x][3], in[x][2] / in[x][3]};
        }
    }
    cv::Mat bgr, alpha, bled;
    colours.convertTo(bgr, CV_8U);  // rounded to the nearest (ties to even) and saturated
    cv::extractChannel(sprite, alpha, 3);
    std::vector<cv::Mat> channels;
    cv::split(bgr, channels);
    channels.push_back(alpha);
    cv::merge(channels, bled);
    return bled;
}

// In place: t > 0: Gaussian blur with sigma t; t < 0: unsharp mask with amount -t (sigma 1); px =
// the image's pixels per pixel of the image t is meant for; blurred = a buffer of image's size and
// type (mirrors _sharpness in common/harmonization.py). The image may be the corner of a larger
// buffer: its border is reflected, not read from beyond it (BORDER_ISOLATED), as numpy's arrays.
void sharpen(cv::Mat& image, float t, double px, cv::Mat blurred)
{
    const int border = cv::BORDER_REFLECT_101 | cv::BORDER_ISOLATED;
    if (t >= 0) {
        if (t > 0.05f) cv::GaussianBlur(image, image, {0, 0}, t * px, 0.0, border);
        return;
    }
    cv::GaussianBlur(image, blurred, {0, 0}, px, 0.0, border);
    cv::addWeighted(image, 1.0 - t, blurred, t, 0.0, image);
}

// Unit Gaussian sensor noise as 8-bit 128 + 32 n, the same in the three channels.
cv::Mat noiseImage(cv::Size size, cv::RNG& rng)
{
    cv::Mat unit(size, CV_32F), level, out;
    rng.fill(unit, cv::RNG::NORMAL, 0.0, 1.0);
    unit.convertTo(level, CV_8U, 32.0, 128.0);
    cv::merge(std::vector<cv::Mat>{level, level, level}, out);
    return out;
}

// Soft mask (0-1) of a hand's skin: the palm polygon and the fingers drawn as lines as thick as
// real fingers - no background between the fingers (mirrors skin_mask / _hand_mask); soft = the
// edge's blur.
cv::Mat handMask(const std::vector<cv::Point2f>& points, cv::Point offset, cv::Size size, double soft)
{
    static const std::vector<std::vector<int>> FINGERS = {
        {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}, {17, 18, 19, 20}};
    cv::Mat mask = cv::Mat::zeros(size, CV_8U);
    const auto at = [&](int i) {  // truncation towards zero, as numpy's astype(int32)
        return cv::Point(static_cast<int>(points[i].x - offset.x), static_cast<int>(points[i].y - offset.y));
    };
    const std::vector<cv::Point> palm{at(0), at(1), at(5), at(9), at(13), at(17)};
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{palm}, cv::Scalar(1));
    std::vector<std::vector<cv::Point>> fingers;
    for (const auto& chain : FINGERS) {
        fingers.emplace_back();
        for (int i : chain) fingers.back().push_back(at(i));
    }
    cv::polylines(mask, fingers, false, cv::Scalar(1), std::max(2, static_cast<int>(std::lround(0.3 * cv::norm(points[5] - points[17])))));
    cv::Mat smooth;
    mask.convertTo(smooth, CV_32F);
    cv::GaussianBlur(smooth, smooth, {0, 0}, soft);
    return smooth;
}

// Where a hand's occlusion mask can be above zero, inside the object's area: its box, the finger
// width and the blur's reach (as _occlude).
cv::Rect handBox(const std::vector<cv::Point2f>& hand, const cv::Rect& area, double soft)
{
    const int reach = static_cast<int>(std::lround(0.15 * cv::norm(hand[5] - hand[17]))) + static_cast<int>(std::ceil(4 * soft)) + 3;
    const cv::Rect box = cv::boundingRect(hand);
    return cv::Rect(box.x - reach, box.y - reach, box.width + 2 * reach, box.height + 2 * reach) & area;
}

}  // namespace

ObjectRenderer::ObjectRenderer(const QString& objectsDir)
    : dir_(objectsDir.toStdString())
{
    weightLut_.create(1, 256, CV_32F);
    restLut_.create(1, 256, CV_32F);
    for (int i = 0; i < 256; ++i) {  // as WEIGHT and REST in common/handheld.py
        weightLut_.at<float>(i) = static_cast<float>(i) * static_cast<float>(1.0 / 255);
        restLut_.at<float>(i) = 1.0f - weightLut_.at<float>(i);
    }
    grain_ = noiseImage({2048, 1152}, rng_);  // any object in a Full HD frame gets a random window of it
    std::vector<std::string> names;
    for (const QString& file : QDir(objectsDir).entryList({"*.png"}, QDir::Files))
        names.push_back(QFileInfo(file).completeBaseName().toStdString());
    std::vector<cv::Mat> loaded(names.size());
    const auto load = [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            const cv::Mat sprite = cv::imread(dir_ + "/" + names[i] + ".png", cv::IMREAD_UNCHANGED);
            loaded[i] = sprite.channels() == 4 ? bleed(sprite) : cv::Mat();
        }
    };
#ifdef NDEBUG  // in parallel, as HandheldScene does: 30 cut-outs in 0.6 s instead of 1.9 s
    cv::parallel_for_(cv::Range(0, static_cast<int>(names.size())), load);
#else  // the Debug heap serializes allocations: there threads make it slower (26 s instead of 13 s)
    load(cv::Range(0, static_cast<int>(names.size())));
#endif
    for (size_t i = 0; i < names.size(); ++i) textures_.emplace(names[i], std::move(loaded[i]));
}

std::vector<HeldObject> ObjectRenderer::parse(const QJsonArray& objects)
{
    std::vector<HeldObject> out;
    for (const QJsonValue& value : objects) {
        const QJsonObject json = value.toObject();
        HeldObject o;
        o.id      = json["id"].toInt();
        o.name    = json["name"].toString().toStdString();
        o.size    = {json["w"].toInt(), json["h"].toInt()};
        o.topLeft = {json["x"].toInt(), json["y"].toInt()};
        o.mode    = json["mode"].toString().toStdString();
        o.opacity = static_cast<float>(json["opacity"].toDouble());
        o.raw     = json["raw"].toBool();
        const QJsonArray fit = json["fit"].toArray();
        if (fit.size() == 14) {  // 4x3 matrix M row by row (rgb' = [rgb, 1] M in 0-1 units), sharpness, noise
            o.hasFit = true;
            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < 3; ++k) o.colour(c, k) = static_cast<float>(fit[k * 3 + c].toDouble());
                o.colour(c, 3) = 255.0f * static_cast<float>(fit[9 + c].toDouble());
            }
            o.sharpness = static_cast<float>(fit[12].toDouble());
            o.noise     = static_cast<float>(fit[13].toDouble());
        }
        o.version = json["version"].toInt();
        for (const QJsonValue& hand : json["hands"].toArray()) {
            o.hands.emplace_back();
            for (const QJsonValue& point : hand.toArray()) {
                const QJsonArray xy = point.toArray();
                o.hands.back().emplace_back(static_cast<float>(xy[0].toDouble()), static_cast<float>(xy[1].toDouble()));
            }
        }
        for (const QJsonValue& centre : json["centres"].toArray())  // truncated, as numpy's astype(int)
            o.centres.emplace_back(static_cast<int>(centre.toArray()[0].toDouble()), static_cast<int>(centre.toArray()[1].toDouble()));
        o.numHands = json["n_hands"].toInt(1);
        o.grip     = json["grip"].toBool();
        o.busy     = json["busy"].toBool();
        o.shadow   = json["shadow"].toBool(o.mode == "palm");
        o.heightM  = json["height_m"].toDouble();
        o.sizeM    = json["size_m"].toDouble(o.heightM);
        out.push_back(std::move(o));
    }
    return out;
}

std::string ObjectRenderer::displayName(const std::string& name)
{
    std::string text = name;
    bool start = true;
    for (char& c : text) {
        if (c == '_') c = ' ';
        c = static_cast<char>(start ? std::toupper(static_cast<unsigned char>(c)) : std::tolower(static_cast<unsigned char>(c)));
        start = c == ' ';
    }
    return text;
}

float ObjectRenderer::cameraNoise(const cv::Mat& frame)
{
    // Immerkaer's estimator (CVIU 1996) on the flatter half of the crop, as _noise_level
    const int w = std::min(frame.cols, 640), h = std::min(frame.rows, 360);
    cv::Mat gray, response, gx, gy, grad;
    cv::cvtColor(frame(cv::Rect((frame.cols - w) / 2, (frame.rows - h) / 2, w, h)), gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F);
    cv::filter2D(gray, response, -1, cv::Matx33f(1, -2, 1, -2, 4, -2, 1, -2, 1));
    response = cv::abs(response);
    cv::Sobel(gray, gx, -1, 1, 0);
    cv::Sobel(gray, gy, -1, 0, 1);
    cv::magnitude(gx, gy, grad);
    std::vector<float> values(grad.begin<float>(), grad.end<float>());
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    const cv::Mat flat = grad <= values[values.size() / 2];
    return static_cast<float>(std::sqrt(CV_PI / 2) / 6 * cv::mean(response, flat)[0]);
}

const cv::Mat* ObjectRenderer::texture(const std::string& name)
{
    auto it = textures_.find(name);
    if (it == textures_.end()) {
        const cv::Mat sprite = cv::imread(dir_ + "/" + name + ".png", cv::IMREAD_UNCHANGED);
        it = textures_.emplace(name, sprite.channels() == 4 ? bleed(sprite) : cv::Mat()).first;
    }
    return it->second.empty() ? nullptr : &it->second;
}

const cv::Mat& ObjectRenderer::styled(const HeldObject& object, const cv::Mat& texture, double px)
{
    // the cut-out re-coloured as the update says, once per update whichever way the object is drawn -
    // and, if px > 0 and the fit changes the sharpness, sharpened at its own resolution, px of its
    // pixels per frame pixel (kept beside it, for one px) - mirrors HandheldScene._styled
    const auto withAlpha = [&](const cv::Mat& bgr) {
        cv::Mat alpha, out;
        cv::extractChannel(texture, alpha, 3);
        std::vector<cv::Mat> channels;
        cv::split(bgr, channels);
        channels.push_back(alpha);
        cv::merge(channels, out);
        return out;
    };
    Styled& s = styled_[object.id];
    const std::pair<std::string, int> key{object.name, object.version};
    if (s.bgra.empty() || s.key != key) {
        cv::Mat bgr, coloured;
        cv::cvtColor(texture, bgr, cv::COLOR_BGRA2BGR);
        cv::transform(bgr, coloured, cv::Mat(object.colour));
        s = {key, withAlpha(coloured), 0.0, cv::Mat()};
    }
    if (px <= 0.0 || (object.sharpness >= 0.0f && object.sharpness <= 0.05f)) return s.bgra;  // the sharpness stays
    if (s.px != px) {
        cv::Mat bgr;
        cv::cvtColor(s.bgra, bgr, cv::COLOR_BGRA2BGR);
        sharpen(bgr, object.sharpness, px, cv::Mat(bgr.size(), bgr.type()));
        s.px = px;
        s.sharpened = withAlpha(bgr);
    }
    return s.sharpened;
}

ObjectRenderer::Part ObjectRenderer::part(const HeldObject& o, const cv::Mat& texture, const cv::Rect& visible, bool harmonized)
{
    // mirrors HandheldScene._sprite: near or above its cut-out's size, only the visible part, every
    // frame; well below it, the whole object, made again when its size or the update changes
    if (o.size.height * SAMPLED_RATIO > texture.rows) {
        const int step = static_cast<int>(std::floor(4 * std::log2(static_cast<double>(texture.rows) / o.size.height) + 0.5));
        const cv::Mat& source = harmonized ? styled(o, texture, std::pow(2.0, step / 4.0)) : texture;
        const double sx = static_cast<double>(o.size.width) / source.cols, sy = static_cast<double>(o.size.height) / source.rows;
        const cv::Matx23d m(1 / sx, 0, (visible.x - o.topLeft.x + 0.5) / sx - 0.5,  // cv::resize's pixel alignment
                            0, 1 / sy, (visible.y - o.topLeft.y + 0.5) / sy - 0.5);
        cv::Mat drawn = work(DRAWN, visible.size(), CV_8UC4), alpha = work(ALPHA, visible.size(), CV_8U);
        Part p{work(COLOURS, visible.size(), CV_8UC3), work(WEIGHT, visible.size(), CV_32F), work(REST, visible.size(), CV_32F)};
        cv::warpAffine(source, drawn, m, visible.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
        cv::extractChannel(drawn, alpha, 3);
        cv::cvtColor(drawn, p.colours, cv::COLOR_BGRA2BGR);
        cv::LUT(alpha, weightLut_, p.weight);
        cv::LUT(alpha, restLut_, p.rest);
        return p;
    }
    const SizeKey key{o.name, o.size.width, o.size.height, harmonized ? o.version : -1};
    Kept& kept = sprites_[o.id];
    if (kept.colours.empty() || kept.key != key) {
        const cv::Size full(std::max(texture.cols, o.size.width), texture.rows);  // as large as the object can be here
        const auto corner = [&](cv::Mat& buffer, int type) {
            if (buffer.size() != full || buffer.type() != type) buffer.create(full, type);
            return buffer(cv::Rect({0, 0}, o.size));
        };
        cv::Mat sized = corner(kept.sized, CV_8UC4), colours = corner(kept.colours, CV_8UC3), alpha = corner(kept.alpha, CV_8U);
        cv::Mat weight = corner(kept.weight, CV_32F), rest = corner(kept.rest, CV_32F);
        cv::resize(harmonized ? styled(o, texture, 0.0) : texture, sized, o.size, 0, 0,
                   o.size.height < texture.rows ? cv::INTER_AREA : cv::INTER_LINEAR);
        cv::cvtColor(sized, colours, cv::COLOR_BGRA2BGR);
        if (harmonized) sharpen(colours, o.sharpness, 1.0, work(BLURRED, o.size, CV_8UC3));
        cv::extractChannel(sized, alpha, 3);
        cv::LUT(alpha, weightLut_, weight);
        cv::LUT(alpha, restLut_, rest);
        kept.key = key;
    }
    const cv::Rect seen = visible - o.topLeft;
    return {kept.colours(seen), kept.weight(seen), kept.rest(seen)};
}

const cv::Mat& ObjectRenderer::shadow(const HeldObject& o, const cv::Mat& texture, cv::Point& at)
{
    // a blurred ellipse under the base, as wide as it (mirrors _contact_shadow)
    auto span = spans_.find(o.name);
    if (span == spans_.end()) {  // the base's columns in the cut-out
        const int rows = std::max(2, texture.rows / 20);
        int first = -1, last = -1;
        for (int x = 0; x < texture.cols; ++x)
            for (int y = texture.rows - rows; y < texture.rows; ++y)
                if (texture.at<cv::Vec4b>(y, x)[3] > 127) {
                    if (first < 0) first = x;
                    last = x;
                    break;
                }
        span = spans_.emplace(o.name, first >= 0 ? last - first : texture.cols / 2.0).first;
    }
    const int rx = std::max(3, static_cast<int>(std::floor(0.6 * span->second * o.size.width / texture.cols + 0.5)));
    auto canvas = shadows_.find(rx);
    if (canvas == shadows_.end()) {
        if (shadows_.size() > 64) shadows_.clear();
        cv::Mat ellipse = cv::Mat::zeros(4 * std::max(2, rx / 3), 3 * rx, CV_32F);
        cv::ellipse(ellipse, {ellipse.cols / 2, ellipse.rows / 2}, {rx, std::max(2, rx / 4)}, 0, 0, 360, cv::Scalar(1.0), cv::FILLED);
        cv::GaussianBlur(ellipse, ellipse, {0, 0}, std::max(1.0, rx / 4.0));
        canvas = shadows_.emplace(rx, cv::Mat(ellipse * 0.45)).first;
    }
    at = {o.topLeft.x + o.size.width / 2 - canvas->second.cols / 2, o.topLeft.y + o.size.height - canvas->second.rows / 2};
    return canvas->second;
}

cv::Mat ObjectRenderer::grainWindow(cv::Size size)
{
    if (size.width > grain_.cols || size.height > grain_.rows) return noiseImage(size, rng_);  // beyond Full HD: fresh
    const int x = rng_.uniform(0, grain_.cols - size.width + 1), y = rng_.uniform(0, grain_.rows - size.height + 1);
    return grain_(cv::Rect({x, y}, size));  // a random window: new noise every frame
}

cv::Mat ObjectRenderer::work(Work slot, cv::Size size, int type)
{
    cv::Mat& buffer = work_[slot];
    if (buffer.type() != type || buffer.cols < size.width || buffer.rows < size.height)
        buffer.create(std::max(buffer.rows, size.height), std::max(buffer.cols, size.width), type);
    return buffer(cv::Rect({0, 0}, size));
}

void ObjectRenderer::draw(cv::Mat& frame, const std::vector<HeldObject>& objects, bool harmonize, float noise)
{
    const auto gone = [&](int id) {
        return std::none_of(objects.begin(), objects.end(), [&](const HeldObject& o) { return o.id == id; });
    };
    for (auto it = sprites_.begin(); it != sprites_.end();) it = gone(it->first) ? sprites_.erase(it) : std::next(it);
    for (auto it = styled_.begin(); it != styled_.end();) it = gone(it->first) ? styled_.erase(it) : std::next(it);
    if (objects.empty()) return;
    const double soft = std::max(1.0, frame.rows / 480.0);  // the hands' soft edge, as at 480 rows
    const cv::Rect inFrame(0, 0, frame.cols, frame.rows);
    struct Patch {  // the camera's pixels where a hand in front of an object is, before anything is drawn
        const std::vector<cv::Point2f>* hand;
        cv::Rect box;
        cv::Mat camera;
    };
    std::vector<std::vector<Patch>> patches(objects.size());
    for (size_t i = 0; i < objects.size(); ++i)
        if (objects[i].mode != "palm")  // held at its belly or handles: the hands are in front of it
            for (const auto& hand : objects[i].hands)
                if (hand.size() >= 21)
                    if (const cv::Rect box = handBox(hand, cv::Rect(objects[i].topLeft, objects[i].size) & inFrame, soft); !box.empty())
                        patches[i].push_back({&hand, box, frame(box).clone()});
    for (size_t i = 0; i < objects.size(); ++i) {
        const HeldObject& o = objects[i];
        if (o.opacity <= 0.0f || o.size.area() <= 0) continue;
        const cv::Mat* tex = texture(o.name);
        if (!tex) continue;
        const bool harmonized = harmonize && !o.raw && o.hasFit;
        const bool shadowed = o.shadow;  // standing on the palm (not a sword lying across two)
        cv::Rect area(o.topLeft, o.size);
        cv::Mat shadowAlpha;
        cv::Point shadowAt;
        if (shadowed) {  // a contact shadow on the palm, under the base
            shadowAlpha = shadow(o, *tex, shadowAt) * o.opacity;
            area |= cv::Rect(shadowAt, shadowAlpha.size());
        }
        area &= inFrame;
        if (area.empty()) continue;
        cv::Mat roi = frame(area);  // drawn into in place
        if (shadowed) blend(roi, cv::Mat(), shadowAlpha, shadowAt - area.tl());
        const cv::Rect visible = cv::Rect(o.topLeft, o.size) & inFrame;
        if (!visible.empty()) {  // the part of it in the frame
            Part p = part(o, *tex, visible, harmonized);
            if (o.opacity < 1.0f) {  // into other buffers: p may show the kept sprite's
                cv::Mat weight = work(OPAQUE_WEIGHT, visible.size(), CV_32F), rest = work(OPAQUE_REST, visible.size(), CV_32F);
                p.weight.convertTo(weight, -1, o.opacity);
                weight.convertTo(rest, -1, -1.0, 1.0);
                p.weight = weight;
                p.rest = rest;
            }
            const float sigma = noise >= 0.0f ? noise : o.noise;
            if (harmonized && sigma > 0.0f) {  // fresh sensor noise every frame, like the camera's
                cv::Mat grained = work(GRAINED, visible.size(), CV_8UC3);
                cv::addWeighted(p.colours, 1.0, grainWindow(visible.size()), sigma / 32.0, -4.0 * sigma, grained);
                p.colours = grained;
            }
            blend(roi, p.colours, p.weight, visible.tl() - area.tl(), p.rest);
        }
        for (const Patch& patch : patches[i]) {  // the camera's own pixels back where the hand is
            cv::Mat part = frame(patch.box);
            blend(part, patch.camera, handMask(*patch.hand, patch.box.tl(), patch.box.size(), soft), {0, 0});
        }
    }
}
