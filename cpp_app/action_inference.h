#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

// Module 1: load action_classifier.onnx and classify one skeleton window.
//
// Input:  SEQUENCE_LENGTH * NUM_FEATURES floats, time-sampled and normalised
//         (see WebcamWidget::sampleWindow and normalizer.h).
// Output: the softmax probability of every action in labels() (empty on error).

class ActionInference {
public:
    static constexpr int SEQUENCE_LENGTH = 30;
    static constexpr int NUM_FEATURES    = 33 * 2;  // x, y per landmark
    static constexpr int INPUT_SIZE      = SEQUENCE_LENGTH * NUM_FEATURES;

    ActionInference(const std::filesystem::path& modelPath, const std::filesystem::path& labelsPath);

    std::vector<float> predict(const std::vector<float>& window);

    bool isLoaded() const { return loaded_; }
    const std::string& error() const { return error_; }
    const std::vector<std::string>& labels() const { return labels_; }

private:
    Ort::Env                 env_;
    Ort::Session             session_{nullptr};
    std::vector<std::string> labels_;
    bool                     loaded_ = false;
    std::string              error_;
};
