#include "MambaInferenceEngine.hpp"
#include <algorithm>
#include <cstring>
MambaInferenceEngine::MambaInferenceEngine(const Config& config) : config_(config) {
    convLayerShape_ = {1, static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.convWidth)};
    ssmLayerShape_ = {1, static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.dState)};
    convStackShape_ = {1, static_cast<int64_t>(config.nLayers), static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.convWidth)};
    ssmStackShape_ = {1, static_cast<int64_t>(config.nLayers), static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.dState)};

    inputNames_ = {"audio_chunk", "conv0", "conv1", "conv2", "conv3", "ssm0", "ssm1", "ssm2", "ssm3"};
    outputNames_ = {"processed_audio", "new_conv_state", "new_ssm_state"};
}

MambaInferenceEngine::~MambaInferenceEngine() = default;

bool MambaInferenceEngine::init() {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, config_.modelPath.c_str(), sessionOptions);
    } catch (const Ort::Exception&) {
        return false;
    }

    const size_t convLayerSize = config_.dModel * config_.convWidth;
    const size_t ssmLayerSize = config_.dModel * config_.dState;

    inputBuffer_.resize(config_.maxBlockSize, 0.0f);
    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        convStateIn_[i].resize(convLayerSize, 0.0f);
        ssmStateIn_[i].resize(ssmLayerSize, 0.0f);
    }
    processedAudioOut_.resize(config_.maxBlockSize, 0.0f);
    convStateOutStack_.resize(config_.nLayers * convLayerSize, 0.0f);
    ssmStateOutStack_.resize(config_.nLayers * ssmLayerSize, 0.0f);

    audioChunkShape_ = {1, static_cast<int64_t>(config_.maxBlockSize), 1};
    processedAudioShape_ = {1, static_cast<int64_t>(config_.maxBlockSize), 1};

    initialized_ = true;
    return true;
}

void MambaInferenceEngine::resetState() {
    // Deferred clear — the actual zeroing happens at the start of the next
    // processBlock() on the audio thread. This makes resetState() safe to call
    // from any thread without synchronizing with the audio thread.
    resetRequested_.store(true, std::memory_order_release);
}

void MambaInferenceEngine::processBlock(const float* input, float* output, int numSamples) {
    if (!initialized_ || !session_) return;

    // Deferred reset: if resetState() was called from another thread, clear
    // state buffers now on the audio thread before processing this block.
    if (resetRequested_.exchange(false, std::memory_order_acq_rel)) {
        for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
            std::fill(convStateIn_[i].begin(), convStateIn_[i].end(), 0.0f);
            std::fill(ssmStateIn_[i].begin(), ssmStateIn_[i].end(), 0.0f);
        }
    }

    const int blockSamples = std::min(numSamples, static_cast<int>(config_.maxBlockSize));

    std::memcpy(inputBuffer_.data(), input, blockSamples * sizeof(float));

    // Use dynamic shape based on actual block size — model was exported with
    // dynamic_axes={'audio_chunk': {1: 'seq_len'}, 'processed_audio': {1: 'seq_len'}}.
    // This avoids feeding zeros into the SSM recurrent state for partial blocks.
    const int64_t audioShape[] = {1, blockSamples, 1};
    const int64_t procAudioShape[] = {1, blockSamples, 1};

    thread_local std::vector<Ort::Value> inputTensors_;
    inputTensors_.clear();
    inputTensors_.reserve(kNumInputs);

    inputTensors_.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, inputBuffer_.data(), blockSamples,
        audioShape, 3));

    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        inputTensors_.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, convStateIn_[i].data(), convStateIn_[i].size(),
            convLayerShape_.data(), 3));
    }
    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        inputTensors_.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, ssmStateIn_[i].data(), ssmStateIn_[i].size(),
            ssmLayerShape_.data(), 3));
    }

    thread_local std::vector<Ort::Value> outputTensors_;
    outputTensors_.clear();
    outputTensors_.reserve(kNumOutputs);

    outputTensors_.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, processedAudioOut_.data(), blockSamples,
        procAudioShape, 3));
    outputTensors_.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, convStateOutStack_.data(), convStateOutStack_.size(),
        convStackShape_.data(), 4));
    outputTensors_.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, ssmStateOutStack_.data(), ssmStateOutStack_.size(),
        ssmStackShape_.data(), 4));

    try {
        session_->Run(Ort::RunOptions(),
            inputNames_.data(), inputTensors_.data(), kNumInputs,
            outputNames_.data(), outputTensors_.data(), kNumOutputs);
    } catch (const Ort::Exception&) {
        std::memset(output, 0, numSamples * sizeof(float));
        return;
    }

    std::memcpy(output, processedAudioOut_.data(), blockSamples * sizeof(float));

    const size_t convLayerSize = config_.dModel * config_.convWidth;
    const size_t ssmLayerSize = config_.dModel * config_.dState;

    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        std::memcpy(convStateIn_[i].data(),
            convStateOutStack_.data() + i * convLayerSize,
            convLayerSize * sizeof(float));
        std::memcpy(ssmStateIn_[i].data(),
            ssmStateOutStack_.data() + i * ssmLayerSize,
            ssmLayerSize * sizeof(float));
    }
}
