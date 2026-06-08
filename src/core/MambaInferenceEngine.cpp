#include "MambaInferenceEngine.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <atomic>

static constexpr size_t COND_DIM = ConditioningFrame::kCondDim;

MambaInferenceEngine::MambaInferenceEngine(const Config& config) : config_(config) {
    convLayerShape_ = {1, static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.convWidth)};
    ssmLayerShape_ = {1, static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.dState)};
    convStackShape_ = {1, static_cast<int64_t>(config.nLayers), static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.convWidth)};
    ssmStackShape_ = {1, static_cast<int64_t>(config.nLayers), static_cast<int64_t>(config.dModel), static_cast<int64_t>(config.dState)};

    inputNames_ = {"audio_chunk", "conv0", "conv1", "conv2", "conv3", "ssm0", "ssm1", "ssm2", "ssm3", "cond"};
    outputNames_ = {"processed_audio", "new_conv_state", "new_ssm_state"};
}

MambaInferenceEngine::~MambaInferenceEngine() = default;

bool MambaInferenceEngine::init() {
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
#ifdef _WIN32
        // Convert UTF-8 path to UTF-16 for Windows ONNX Runtime.
        int wlen = MultiByteToWideChar(CP_UTF8, 0, config_.modelPath.c_str(), -1, nullptr, 0);
        if (wlen <= 0) return false;
        std::wstring wPath(static_cast<size_t>(wlen) - 1, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, config_.modelPath.c_str(), -1, &wPath[0], wlen);
        session_ = std::make_unique<Ort::Session>(env_, wPath.c_str(), sessionOptions);
#else
        session_ = std::make_unique<Ort::Session>(env_, config_.modelPath.c_str(), sessionOptions);
#endif
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

    validateManifest();
    loadCondWeights();

    initialized_ = true;
    activeFrame_.store(nullptr, std::memory_order_release);
    return true;
}

void MambaInferenceEngine::loadCondWeights() {
    if (config_.deltaWeights.size() == COND_DIM &&
        config_.bWeights.size() == COND_DIM) {
        combinedWeights_.resize(COND_DIM);
        for (size_t d = 0; d < COND_DIM; ++d) {
            combinedWeights_[d] = config_.deltaWeights[d] + config_.bWeights[d];
        }
        condWeightsLoaded_ = true;
    } else {
        combinedWeights_.clear();
        condWeightsLoaded_ = false;
    }
}

static std::string jsonStringValue(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return {};

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return {};

    while (++pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) {}

    if (pos >= json.size()) return {};

    if (json[pos] == '"') {
        auto end = json.find('"', pos + 1);
        if (end == std::string::npos) return {};
        return json.substr(pos + 1, end - pos - 1);
    }

    if (json[pos] == '-' || (json[pos] >= '0' && json[pos] <= '9')) {
        auto end = pos;
        while (end < json.size() && (json[end] == '-' || (json[end] >= '0' && json[end] <= '9') || json[end] == '.')) {
            ++end;
        }
        return json.substr(pos, end - pos);
    }

    return {};
}

static int jsonIntValue(const std::string& json, const std::string& key) {
    auto s = jsonStringValue(json, key);
    if (s.empty()) return -1;
    try {
        return std::stoi(s);
    } catch (const std::exception&) {
        return -1;
    }
}

void MambaInferenceEngine::validateManifest() {
    std::string manifestPath = config_.modelPath;
    auto sep = manifestPath.rfind('/');
    if (sep == std::string::npos) sep = manifestPath.rfind('\\');
    if (sep != std::string::npos) {
        manifestPath = manifestPath.substr(0, sep + 1) + "manifest.json";
    } else {
        manifestPath = "models/manifest.json";
    }

    std::ifstream mf(manifestPath);
    if (!mf) {
        return;
    }

    std::string json((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());

    int schemaVer = jsonIntValue(json, "schema_version");
    if (schemaVer < 1) {
        return;
    }

    int manifestDModel = jsonIntValue(json, "d_model");
    int manifestDState = jsonIntValue(json, "d_state");
    int manifestNLayers = jsonIntValue(json, "n_layers");
    int manifestConvWidth = jsonIntValue(json, "conv_width");

    if (manifestDModel > 0 && static_cast<size_t>(manifestDModel) != config_.dModel) {
        fprintf(stderr, "[CLARA] WARNING: manifest d_model=%d but engine expects %zu\n",
                manifestDModel, config_.dModel);
    }
    if (manifestDState > 0 && static_cast<size_t>(manifestDState) != config_.dState) {
        fprintf(stderr, "[CLARA] WARNING: manifest d_state=%d but engine expects %zu\n",
                manifestDState, config_.dState);
    }
    if (manifestNLayers > 0 && static_cast<size_t>(manifestNLayers) != config_.nLayers) {
        fprintf(stderr, "[CLARA] WARNING: manifest n_layers=%d but engine expects %zu\n",
                manifestNLayers, config_.nLayers);
    }
    if (manifestConvWidth > 0 && static_cast<size_t>(manifestConvWidth) != config_.convWidth) {
        fprintf(stderr, "[CLARA] WARNING: manifest conv_width=%d but engine expects %zu\n",
                manifestConvWidth, config_.convWidth);
    }
}

void MambaInferenceEngine::resetState() {
    resetRequested_.store(true, std::memory_order_release);
}

void MambaInferenceEngine::setConditioning(const float* cond, size_t dim) {
    if (!cond || dim == 0) return;

    ConditioningFrame* active = activeFrame_.load(std::memory_order_acquire);
    ConditioningFrame* target = (active == &frameA_) ? &frameB_ : &frameA_;

    target->fill(cond, dim);
    target->seq.fetch_add(1, std::memory_order_relaxed);

    activeFrame_.store(target, std::memory_order_release);
}

void MambaInferenceEngine::warmUp() {
    if (!initialized_) return;
    // Process a single block of silence to trigger thread_local
    // std::vector<Ort::Value> allocation on the calling thread.
    static constexpr float dummy = 0.0f;
    float dummyOutput[64];
    processBlock(&dummy, dummyOutput, 64, nullptr);
    resetState();
}

void MambaInferenceEngine::processBlock(const float* input, float* output, int numSamples, bool* ok) {
    if (!input || !output) {
        if (ok) { *ok = false; }
        return;
    }

    if (numSamples <= 0) {
        if (ok) { *ok = true; }
        return;
    }

    if (!initialized_ || !session_) {
        static std::atomic<bool> warnedOnce{false};
        if (!warnedOnce.exchange(true, std::memory_order_relaxed)) {
            fprintf(stderr, "[CLARA] WARNING: processBlock() called on uninitialized engine -- outputting silence\n");
        }
        if (ok) { *ok = false; }
        std::memset(output, 0, numSamples * sizeof(float));
        return;
    }

    if (resetRequested_.exchange(false, std::memory_order_acq_rel)) {
        for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
            std::fill(convStateIn_[i].begin(), convStateIn_[i].end(), 0.0f);
            std::fill(ssmStateIn_[i].begin(), ssmStateIn_[i].end(), 0.0f);
        }
    }

    const int totalSamples = (numSamples > 0)
        ? std::min(numSamples, static_cast<int>(config_.maxBlockSize))
        : 0;

    static constexpr int SB = kSubBlockSize;
    const int64_t audioShape[] = {1, SB, 1};
    const int64_t procAudioShape[] = {1, SB, 1};
    const int64_t condShape[] = {1, static_cast<int64_t>(COND_DIM)};

    const size_t convLayerSize = config_.dModel * config_.convWidth;
    const size_t ssmLayerSize = config_.dModel * config_.dState;

    int offset = 0;
    while (offset < totalSamples) {
        const int chunkSamples = std::min(SB, totalSamples - offset);

        std::memcpy(inputBuffer_.data(), input + offset, chunkSamples * sizeof(float));
        if (chunkSamples < SB) {
            std::memset(inputBuffer_.data() + chunkSamples, 0,
                (SB - chunkSamples) * sizeof(float));
        }

        thread_local std::vector<Ort::Value> inputTensors_;
        inputTensors_.clear();
        inputTensors_.reserve(kNumInputs);

        inputTensors_.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, inputBuffer_.data(), SB, audioShape, 3));

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

        // Snapshot conditioning once per block to avoid TOCTOU race
        // with concurrent setConditioning() calls.
        float* activeCondData = nullptr;
        const auto* condFrame = activeFrame_.load(std::memory_order_acquire);
        if (condFrame) {
            activeCondData = const_cast<float*>(condFrame->data.data());
        } else {
            static const std::array<float, COND_DIM> zeroCond{};
            activeCondData = const_cast<float*>(zeroCond.data());
        }
        inputTensors_.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, activeCondData, COND_DIM, condShape, 2));

        thread_local std::vector<Ort::Value> outputTensors_;
        outputTensors_.clear();
        outputTensors_.reserve(kNumOutputs);

        outputTensors_.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo_, processedAudioOut_.data(), SB, procAudioShape, 3));
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
            if (ok) { *ok = false; }
            std::memset(output, 0, numSamples * sizeof(float));
            return;
        }

        if (condWeightsLoaded_ && condFrame) {
            float condScale = 1.0f;
            for (size_t d = 0; d < COND_DIM; ++d) {
                condScale += condFrame->data[d] * combinedWeights_[d];
            }
            if (condScale != 1.0f) {
                for (int s = 0; s < chunkSamples; ++s) {
                    processedAudioOut_[s] *= condScale;
                }
            }
        }

        std::memcpy(output + offset, processedAudioOut_.data(),
            chunkSamples * sizeof(float));

        for (size_t i = 0; i < config_.nLayers && i < kMaxLayers; ++i) {
            std::memcpy(convStateIn_[i].data(),
                convStateOutStack_.data() + i * convLayerSize,
                convLayerSize * sizeof(float));
            std::memcpy(ssmStateIn_[i].data(),
                ssmStateOutStack_.data() + i * ssmLayerSize,
                ssmLayerSize * sizeof(float));
        }

        offset += chunkSamples;
    }

    if (ok) { *ok = true; }
}
