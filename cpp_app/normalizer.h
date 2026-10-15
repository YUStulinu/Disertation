#pragma once

#include <cmath>
#include <vector>

// Mirrors normalize_sequence() in python_research/common/pose.py.
// Input:  numFrames * 33 * 2 floats (x, y interleaved per landmark).
// Output: same layout, centred on the hip midpoint and scaled by the
//         shoulder-to-hip distance (position/scale invariant).

namespace Normalizer {

constexpr int NUM_LANDMARKS  = 33;
constexpr int NUM_FEATURES   = NUM_LANDMARKS * 2;
constexpr int LEFT_SHOULDER  = 11;
constexpr int RIGHT_SHOULDER = 12;
constexpr int LEFT_HIP       = 23;
constexpr int RIGHT_HIP      = 24;

inline std::vector<float> normalizeSequence(const std::vector<float>& flat)
{
    std::vector<float> result(flat.size());
    for (size_t base = 0; base + NUM_FEATURES <= flat.size(); base += NUM_FEATURES) {
        auto x = [&](int lm) { return flat[base + lm * 2]; };
        auto y = [&](int lm) { return flat[base + lm * 2 + 1]; };

        const float hipX = (x(LEFT_HIP) + x(RIGHT_HIP)) * 0.5f;
        const float hipY = (y(LEFT_HIP) + y(RIGHT_HIP)) * 0.5f;
        const float dx = (x(LEFT_SHOULDER) + x(RIGHT_SHOULDER)) * 0.5f - hipX;
        const float dy = (y(LEFT_SHOULDER) + y(RIGHT_SHOULDER)) * 0.5f - hipY;
        float scale = std::sqrt(dx * dx + dy * dy);
        if (scale < 1e-6f) scale = 1.0f;  // avoid divide-by-zero on occluded frames

        for (int lm = 0; lm < NUM_LANDMARKS; ++lm) {
            result[base + lm * 2]     = (x(lm) - hipX) / scale;
            result[base + lm * 2 + 1] = (y(lm) - hipY) / scale;
        }
    }
    return result;
}

}  // namespace Normalizer
