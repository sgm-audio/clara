#pragma once

#include <clap/clap.h>
#include <clap/ext/state.h>
#include "../core/MambaInferenceEngine.hpp"
#include <atomic>
#include <vector>
#include <memory>

enum ParamId {
    PARAM_BYPASS = 0,
    PARAM_WET_DRY = 1,
    PARAM_INTENSITY = 2,
    PARAM_COUNT
};

struct MambaPluginState {
    std::unique_ptr<MambaInferenceEngine> engine;
    const clap_host_t* host{nullptr};
    std::string modelPath;

    std::atomic<float> bypass{0.0f};
    std::atomic<float> wetDry{1.0f};
    std::atomic<float> intensity{0.5f};

    float sampleRate{44100.0f};
    float currentBypassValue{0.0f};
    float crossfadeStep{0.001f};

    std::vector<float> chIn[2];
    std::vector<float> chOut[2];
};

extern "C" const clap_plugin_descriptor_t mamba_plugin_info;

extern "C" const clap_plugin_state_t mamba_state_extension;

extern const clap_plugin_t mamba_plugin;
