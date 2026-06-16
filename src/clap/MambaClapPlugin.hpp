#pragma once

#include <clap/clap.h>
#include <clap/ext/state.h>
#include <clap/ext/state-context.h>
#include <clap/ext/render.h>
#include <clap/ext/tail.h>
#include "../core/SsmInferenceCore.h"
#include "../core/SPSCQueue.hpp"
#include <atomic>
#include <vector>
#include <memory>
#include <string>

enum ParamId {
    PARAM_BYPASS = 0,
    PARAM_WET_DRY = 1,
    PARAM_INTENSITY = 2,
    PARAM_COUNT
};

using namespace clara::mamba3;

struct MambaPluginState {
    // One engine per channel (stereo). SsmInferenceCore holds SSM state internally.
    std::unique_ptr<SsmInferenceCore> engineL;
    std::unique_ptr<SsmInferenceCore> engineR;
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

    // Deferred task queue (SPSC, lock-free)
    enum class DeferredTask : uint8_t {
        ModelReload = 0,
        AnalysisQueueDrain,
    };
    SPSCQueue<DeferredTask, 64> deferredTaskQueue;

    // Flag to signal audio thread to go passthrough during model reload
    std::atomic<bool> modelReloadPending{false};

    // Atomic engine pointers for RT-safe access from audio thread.
    // The unique_ptrs own the engines; the atomics provide lock-free access.
    std::atomic<SsmInferenceCore*> enginePtrL{nullptr};
    std::atomic<SsmInferenceCore*> enginePtrR{nullptr};

    // Conditioning (shared across both channels)
    float condBuffer[COND_DIM] = {};
};

extern "C" const clap_plugin_descriptor_t mamba_plugin_info;

extern const clap_plugin_t mamba_plugin;

extern "C" const clap_plugin_state_t mamba_state_extension;
extern const clap_plugin_state_context_t mamba_state_context_extension;
extern const clap_plugin_render_t mamba_render_extension;
extern const clap_plugin_tail_t mamba_tail_extension;

const void* mamba_get_extension(const clap_plugin_t* plugin, const char* extensionId);
