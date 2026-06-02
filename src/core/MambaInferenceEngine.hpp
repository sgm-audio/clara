#pragma once

#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <array>
#include <algorithm>

class MambaInferenceEngine {
public:
    struct Config {
        std::string modelPath;
        size_t dModel = 64;
        size_t dState = 16;
        size_t nLayers = 4;
        size_t convWidth = 4;
        size_t maxBlockSize = 256;
    };

    explicit MambaInferenceEngine(const Config& config);
    ~MambaInferenceEngine();

    bool init();
    void resetState();
    void processBlock(const float* input, float* output, int numSamples);

    // Set conditioning vector from any thread (double-buffered atomic swap)
    void setConditioning(const float* cond, size_t dim);
    // Load projection weights (must be called from non-RT thread, typically activate())
    void loadCondWeights(const float* weights, size_t dim);

    // Test-only accessors for verifying the double-buffer swap and weights
    // loading state. Not part of the production API. Returns the address of
    // the currently active conditioning buffer (nullptr before any
    // setConditioning() call).
    const float* getActiveConditioningForTest() const noexcept {
        return activeCond_.load(std::memory_order_acquire);
    }
    // Test-only: returns true if loadCondWeights() populated condWeights_.
    bool getCondWeightsLoadedForTest() const noexcept {
        return condWeightsLoaded_;
    }

private:
    Config config_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MambaInference"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    static constexpr size_t kMaxLayers = 4;

    std::atomic<bool> resetRequested_{false};

    std::vector<int64_t> audioChunkShape_;
    std::vector<int64_t> processedAudioShape_;
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

    static constexpr size_t kNumInputs = 9;
    static constexpr size_t kNumOutputs = 3;
    std::array<const char*, kNumInputs> inputNames_;
    std::array<const char*, kNumOutputs> outputNames_;

    static constexpr size_t COND_DIM = 8;

    // Conditioning vector: double-buffered for RT-safe updates
    std::array<float, COND_DIM> condBufA_{};
    std::array<float, COND_DIM> condBufB_{};
    std::atomic<float*> activeCond_{nullptr};
    std::vector<float> condWeights_;  // COND_DIM scalars, loaded at activate()
    bool condWeightsLoaded_{false};

    bool initialized_{false};
};
