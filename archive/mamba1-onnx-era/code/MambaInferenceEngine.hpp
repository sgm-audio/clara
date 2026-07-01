#pragma once

#include "ConditioningFrame.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4324)
#endif

class MambaInferenceEngine {
public:
    struct Config {
        std::string modelPath;
        size_t dModel = 64;
        size_t dState = 16;
        size_t nLayers = 4;
        size_t convWidth = 4;
        size_t maxBlockSize = 256;
        // Conditioning projection weights (1x1 matrices for delta and B).
        // Each vector must be kCondDim elements. Empty vectors → zeros (no effect).
        std::vector<float> deltaWeights;
        std::vector<float> bWeights;
    };

    explicit MambaInferenceEngine(const Config &config);
    ~MambaInferenceEngine();

    bool init();
    void loadCondWeights();
    void resetState();
    void processBlock(const float *input, float *output, int numSamples, bool *ok = nullptr);

    // Set conditioning vector from any thread (double-buffered atomic swap).
    // Writes to the non-active buffer, then atomically swaps the pointer.
    void setConditioning(const float *cond, size_t dim);

    const ConditioningFrame *getActiveFrameForTest() const noexcept {
        return activeFrame_.load(std::memory_order_acquire);
    }

    const float *getActiveConditioningForTest() const noexcept {
        auto *frame = activeFrame_.load(std::memory_order_acquire);
        return frame ? frame->data.data() : nullptr;
    }

    bool isInitialized() const noexcept {
        return initialized_;
    }
    bool condWeightsLoaded() const noexcept {
        return condWeightsLoaded_;
    }
    const std::vector<float> &getCombinedWeights() const noexcept {
        return combinedWeights_;
    }

    // Pre-allocate thread_local vectors on the calling thread.
    // Call once per audio thread before real-time processing begins.
    void warmUp();

private:
    Config config_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MambaInference"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    static constexpr size_t kMaxLayers = 4;
    static constexpr int kSubBlockSize = 64;
    static constexpr size_t COND_DIM = ConditioningFrame::kCondDim;

    std::atomic<bool> resetRequested_{false};

    std::vector<int64_t> convLayerShape_;
    std::vector<int64_t> ssmLayerShape_;
    std::vector<int64_t> convStackShape_;
    std::vector<int64_t> ssmStackShape_;

    std::vector<float> inputBuffer_;
    std::vector<float> convStateIn_[kMaxLayers];
    std::vector<float> ssmStateIn_[kMaxLayers];
    std::vector<float> processedAudioOut_;
    std::vector<float> convStateOutStack_;
    std::vector<float> ssmStateOutStack_;

    static constexpr size_t kNumInputs = 10;
    static constexpr size_t kNumOutputs = 3;
    std::array<const char *, kNumInputs> inputNames_;
    std::array<const char *, kNumOutputs> outputNames_;

    // Conditioning: double-buffered ConditioningFrame with atomic pointer swap.
    ConditioningFrame frameA_;
    ConditioningFrame frameB_;
    std::atomic<ConditioningFrame *> activeFrame_{nullptr};

    // Projection weights pre-combined at activate() time.
    // combinedWeights_[d] = deltaWeights[d] + bWeights[d]
    // Applied as: output *= (1.0f + sum(cond[d] * combinedWeights_[d]))
    std::vector<float> combinedWeights_;
    bool condWeightsLoaded_{false};

    bool initialized_{false};

    void validateManifest();
};

#ifdef _MSC_VER
#    pragma warning(pop)
#endif
