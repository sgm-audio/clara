#include "MambaInferenceEngine.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

MambaInferenceEngine::MambaInferenceEngine(const Config& config) : config_(config) {
    convLayerShape_ = {1, static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.convWidth)};
    ssmLayerShape_ = {1, static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.dState)};
    convStackShape_ = {1, static_cast<int64_t>(config.nLayers), static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.convWidth)};
    ssmStackShape_ = {1, static_cast<int64_t>(config.nLayers), static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.dState)};

    inputNames_ = {"audio_chunk", "conv0", "conv1", "conv2", "conv3", "ssm0", "ssm1", "ssm2", "ssm3"};
    outputNames_ = {"processed_audio", "new_conv_state", "new_ssm_state"};
}

MambaInferenceEngine::~MambaInferenceEngine() = default;

void MambaInferenceEngine::init() {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, config_.modelPath.c_str(), sessionOptions);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX session creation failed: " + std::string(e.what()));
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
}

void MambaInferenceEngine::resetState() {
    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        std::fill(convStateIn_[i].begin(), convStateIn_[i].end(), 0.0f);
        std::fill(ssmStateIn_[i].begin(), ssmStateIn_[i].end(), 0.0f);
    }
}

void MambaInferenceEngine::processBlock(const float* input, float* output, int numSamples) {
    if (!initialized_ || !session_) return;

    const int blockSamples = std::min(numSamples, static_cast<int>(config_.maxBlockSize));

    std::memcpy(inputBuffer_.data(), input, blockSamples * sizeof(float));
    if (blockSamples < static_cast<int>(config_.maxBlockSize)) {
        std::memset(inputBuffer_.data() + blockSamples, 0,
            (config_.maxBlockSize - blockSamples) * sizeof(float));
    }

    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(kNumInputs);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, inputBuffer_.data(), inputBuffer_.size(),
        audioChunkShape_.data(), 3));

    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, convStateIn_[i].data(), convStateIn_[i].size(),
            convLayerShape_.data(), 3));
    }
    for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, ssmStateIn_[i].data(), ssmStateIn_[i].size(),
            ssmLayerShape_.data(), 3));
    }

    std::vector<Ort::Value> outputTensors;
    outputTensors.reserve(kNumOutputs);

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, processedAudioOut_.data(), processedAudioOut_.size(),
        processedAudioShape_.data(), 3));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, convStateOutStack_.data(), convStateOutStack_.size(),
        convStackShape_.data(), 4));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, ssmStateOutStack_.data(), ssmStateOutStack_.size(),
        ssmStackShape_.data(), 4));

    try {
        session_->Run(Ort::RunOptions(),
            inputNames_.data(), inputTensors.data(), kNumInputs,
            outputNames_.data(), outputTensors.data(), kNumOutputs);
    } catch (const Ort::Exception& e) {
        std::memset(output, 0, numSamples * sizeof(float));
        return;
    }

    std::memcpy(output, processedAudioOut_.data(), blockSamples * sizeof(float));

    // Unstack conv/ssm state outputs back into per-layer buffers
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
