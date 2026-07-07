#pragma once

#include "../core/SPSCQueue.hpp"
#include "../core/SsmInferenceCore.h"

// Forward declare GUI states (defined in MambaWebviewGUI.hpp / MambaNativeGUI.hpp, included in .cpp)
namespace clara::mamba3 {
struct MambaWebviewState;
struct MambaNativeGUIState;
} // namespace clara::mamba3

#include <clap/clap.h>
#include <clap/ext/render.h>
#include <clap/ext/state-context.h>
#include <clap/ext/state.h>
#include <clap/ext/tail.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum ParamId { PARAM_BYPASS = 0, PARAM_WET_DRY = 1, PARAM_INTENSITY = 2, PARAM_COUNT };

struct MambaPluginState {
    // One engine per channel (stereo). SsmInferenceCore holds SSM state internally.
    std::unique_ptr<clara::mamba3::SsmInferenceCore> engineL;
    std::unique_ptr<clara::mamba3::SsmInferenceCore> engineR;
    const clap_host_t *host{nullptr};
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
    std::atomic<clara::mamba3::SsmInferenceCore *> enginePtrL{nullptr};
    std::atomic<clara::mamba3::SsmInferenceCore *> enginePtrR{nullptr};

    // Conditioning (shared across both channels)
    float condBuffer[clara::mamba3::COND_DIM] = {};

    // Preset system: precomputed CLAP embeddings for instant recall.
    // Key = preset name hash, Value = 32-float embedding in cond slots 8-39.
    // Populated by CLAP text encoder on main thread (Phase 7 infra).
    // ponytail: map-based store, ~100 presets max = negligible memory.
    struct PresetEntry {
        float embedding[32] = {}; // cond slots 8-39
    };
    std::vector<PresetEntry> presets; // indexed by preset ID
    int activePreset = -1;            // -1 = none active

    // GUI state (heap-allocated to avoid header dependency)
    clara::mamba3::MambaWebviewState *webviewState{nullptr};
    clara::mamba3::MambaNativeGUIState *nativeGUI{nullptr};

    // Level meter (written on audio thread, read on main thread)
    std::atomic<float> peakIn{0.0f};
    std::atomic<float> peakOut{0.0f};
};

extern "C" const clap_plugin_descriptor_t mamba_plugin_info;

extern const clap_plugin_t mamba_plugin;

extern "C" const clap_plugin_state_t mamba_state_extension;
extern const clap_plugin_state_context_t mamba_state_context_extension;
extern const clap_plugin_render_t mamba_render_extension;
extern const clap_plugin_tail_t mamba_tail_extension;

const void *mamba_get_extension(const clap_plugin_t *plugin, const char *extensionId);
