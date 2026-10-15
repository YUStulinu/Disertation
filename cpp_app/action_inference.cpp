#include "action_inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>

ActionInference::ActionInference(const std::filesystem::path& modelPath, const std::filesystem::path& labelsPath)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ActionClassifier")
{
    std::ifstream file(labelsPath);
    for (std::string line; std::getline(file, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) labels_.push_back(line);
    }
    if (labels_.empty()) {
        error_ = "cannot read labels file " + labelsPath.string();
        return;
    }
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);  // a tiny LSTM does not need more
    try {
        session_ = Ort::Session(env_, modelPath.c_str(), opts);  // wchar_t path on Windows
        loaded_  = true;
    } catch (const Ort::Exception& e) {
        error_ = e.what();
    }
}

std::vector<float> ActionInference::predict(const std::vector<float>& window)
{
    if (!loaded_ || static_cast<int>(window.size()) != INPUT_SIZE) return {};

    const std::array<int64_t, 3> shape{1, SEQUENCE_LENGTH, NUM_FEATURES};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // ONNX Runtime only reads the input buffer, so the const_cast is safe
    auto input = Ort::Value::CreateTensor<float>(memoryInfo, const_cast<float*>(window.data()),
                                                 window.size(), shape.data(), shape.size());
    const char* inputNames[]  = {"keypoint_sequence"};
    const char* outputNames[] = {"action_logits"};
    try {
        auto outputs = session_.Run(Ort::RunOptions{nullptr}, inputNames, &input, 1, outputNames, 1);
        if (outputs[0].GetTensorTypeAndShapeInfo().GetShape().back() != static_cast<int64_t>(labels_.size()))
            return {};  // labels file and model disagree (re-run export_onnx.py)
        const float* logits = outputs[0].GetTensorData<float>();
        std::vector<float> probs(logits, logits + labels_.size());
        const float top = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (float& p : probs) sum += (p = std::exp(p - top));  // softmax
        for (float& p : probs) p /= sum;
        return probs;
    } catch (const Ort::Exception&) {
        return {};
    }
}
