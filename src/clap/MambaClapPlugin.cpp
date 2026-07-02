// CLARA CLAP Plugin — MambaSuiteClap.clap audio effect
//
// Per-channel engine instances, 128-dim conditioning.

#include "MambaClapPlugin.hpp"

#include "core/ConditioningGenerator.hpp"
#include "core/Platform.hpp"

#include <clap/ext/audio-ports.h>
#include <clap/ext/latency.h>
#include <clap/ext/log.h>
#include <clap/ext/params.h>
#include <clap/ext/render.h>
#include <clap/ext/state-context.h>
#include <clap/ext/state.h>
#include <clap/ext/tail.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

using namespace clara::mamba3;
#include <cstdio>
#include <cstring>
#include <string>

// -----------------------------------------------------------------------------
// Plugin Info
// -----------------------------------------------------------------------------

static const char *const mamba_features[] = {CLAP_PLUGIN_FEATURE_AUDIO_EFFECT, CLAP_PLUGIN_FEATURE_UTILITY, nullptr};

const clap_plugin_descriptor_t mamba_plugin_info = {
    CLAP_VERSION_INIT,
    "com.mamba.audio.suite",
    "Mamba-3 SSM Processor",
    "Elite DSP",
    MAMBA_PROJECT_URL,
    nullptr,
    nullptr,
    "1.2.0",
    "Real-time Mamba-3 SSM audio processor with ARA2 integration",
    mamba_features,
};

// -----------------------------------------------------------------------------
// Parameters Extension (unchanged except COND_DIM)
// -----------------------------------------------------------------------------

static uint32_t clap_params_count(const clap_plugin_t *) {
    return PARAM_COUNT;
}

static bool clap_params_get_info(const clap_plugin_t *, uint32_t param_index, clap_param_info_t *param_info) {
    std::memset(param_info, 0, sizeof(*param_info));
    param_info->cookie = nullptr;
    switch (param_index) {
        case PARAM_BYPASS:
            param_info->id = PARAM_BYPASS;
            param_info->flags = CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS;
            std::strncpy(param_info->name, "Bypass", sizeof(param_info->name));
            std::strncpy(param_info->module, "Main", sizeof(param_info->module));
            param_info->min_value = 0.0;
            param_info->max_value = 1.0;
            param_info->default_value = 0.0;
            return true;
        case PARAM_WET_DRY:
            param_info->id = PARAM_WET_DRY;
            param_info->flags = CLAP_PARAM_IS_AUTOMATABLE;
            std::strncpy(param_info->name, "Wet/Dry", sizeof(param_info->name));
            std::strncpy(param_info->module, "Mix", sizeof(param_info->module));
            param_info->min_value = 0.0;
            param_info->max_value = 1.0;
            param_info->default_value = 1.0;
            return true;
        case PARAM_INTENSITY:
            param_info->id = PARAM_INTENSITY;
            param_info->flags = CLAP_PARAM_IS_AUTOMATABLE;
            std::strncpy(param_info->name, "Intensity", sizeof(param_info->name));
            std::strncpy(param_info->module, "Processing", sizeof(param_info->module));
            param_info->min_value = 0.0;
            param_info->max_value = 1.0;
            param_info->default_value = 0.5;
            return true;
        default:
            return false;
    }
}

static bool clap_params_get_value(const clap_plugin_t *plugin, clap_id param_id, double *value) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    switch (param_id) {
        case PARAM_BYPASS:
            *value = state->bypass.load(std::memory_order_acquire);
            return true;
        case PARAM_WET_DRY:
            *value = state->wetDry.load(std::memory_order_acquire);
            return true;
        case PARAM_INTENSITY:
            *value = state->intensity.load(std::memory_order_acquire);
            return true;
        default:
            return false;
    }
}

static bool clap_params_value_to_text(const clap_plugin_t *, clap_id param_id, double value, char *display,
                                      uint32_t size) {
    switch (param_id) {
        case PARAM_BYPASS:
            std::snprintf(display, size, "%s", value > 0.5 ? "On" : "Off");
            return true;
        case PARAM_WET_DRY:
            std::snprintf(display, size, "%.1f%%", value * 100.0);
            return true;
        case PARAM_INTENSITY:
            std::snprintf(display, size, "%.1f%%", value * 100.0);
            return true;
        default:
            return false;
    }
}

static bool clap_params_text_to_value(const clap_plugin_t *, clap_id param_id, const char *display, double *value) {
    switch (param_id) {
        case PARAM_BYPASS:
            *value = (std::strcmp(display, "On") == 0 || std::strcmp(display, "1") == 0) ? 1.0 : 0.0;
            return true;
        case PARAM_WET_DRY:
        case PARAM_INTENSITY: {
            double pct = 0.0;
            if (std::strchr(display, '%')) {
                if (std::sscanf(display, "%lf%%", &pct) == 1) {
                    *value = std::clamp(pct / 100.0, 0.0, 1.0);
                    return true;
                }
            } else {
                if (std::sscanf(display, "%lf", &pct) == 1) {
                    *value = std::clamp(pct, 0.0, 1.0);
                    return true;
                }
            }
            return false;
        }
        default:
            return false;
    }
}

static void clap_params_flush(const clap_plugin_t *plugin, const clap_input_events_t *in,
                              const clap_output_events_t *out) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (!in) return;
    const uint32_t num_events = in->size(in);
    for (uint32_t i = 0; i < num_events; ++i) {
        const clap_event_header_t *event = in->get(in, i);
        if (event->space_id == CLAP_CORE_EVENT_SPACE_ID && event->type == CLAP_EVENT_PARAM_VALUE) {
            const clap_event_param_value_t *param_event = (const clap_event_param_value_t *) event;
            switch (param_event->param_id) {
                case PARAM_BYPASS:
                    state->bypass.store(static_cast<float>(param_event->value), std::memory_order_release);
                    break;
                case PARAM_WET_DRY:
                    state->wetDry.store(static_cast<float>(param_event->value), std::memory_order_release);
                    break;
                case PARAM_INTENSITY:
                    state->intensity.store(static_cast<float>(param_event->value), std::memory_order_release);
                    break;
            }
        }
    }
    (void) out;
}

static const clap_plugin_params_t mamba_params_extension = {clap_params_count,         clap_params_get_info,
                                                            clap_params_get_value,     clap_params_value_to_text,
                                                            clap_params_text_to_value, clap_params_flush};

// -----------------------------------------------------------------------------
// Latency — Mamba-3 has NO conv1d. Zero latency.
// -----------------------------------------------------------------------------

static uint32_t clap_latency_get(const clap_plugin_t *) {
    return 0;
}

static const clap_plugin_latency_t mamba_latency_extension = {clap_latency_get};

// -----------------------------------------------------------------------------
// State Extension
// -----------------------------------------------------------------------------

#pragma pack(push, 1)
struct MambaPresetData {
    float bypass;
    float wetDry;
    float intensity;
};
#pragma pack(pop)

static bool mamba_state_save(const clap_plugin_t *plugin, const clap_ostream_t *stream) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    const int32_t version = 2;
    if (stream->write(stream, &version, sizeof(version)) < 0) return false;
    MambaPresetData data;
    data.bypass = state->bypass.load(std::memory_order_acquire);
    data.wetDry = state->wetDry.load(std::memory_order_acquire);
    data.intensity = state->intensity.load(std::memory_order_acquire);
    return stream->write(stream, &data, sizeof(data)) >= 0;
}

static bool mamba_state_load(const clap_plugin_t *plugin, const clap_istream_t *stream) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    int32_t version = 0;
    if (stream->read(stream, &version, sizeof(version)) < (int64_t) sizeof(version)) return false;
    if (version > 2) return false;
    MambaPresetData data;
    if (stream->read(stream, &data, sizeof(data)) < (int64_t) sizeof(data)) return false;
    state->bypass.store(data.bypass, std::memory_order_release);
    state->wetDry.store(data.wetDry, std::memory_order_release);
    state->intensity.store(data.intensity, std::memory_order_release);
    state->currentBypassValue = data.bypass;
    if (state->engineL) state->engineL->reset_states();
    if (state->engineR) state->engineR->reset_states();
    return true;
}

const clap_plugin_state_t mamba_state_extension = {mamba_state_save, mamba_state_load};

static bool mamba_state_context_save(const clap_plugin_t *plugin, const clap_ostream_t *stream, uint32_t context_type) {
    (void) context_type;
    return mamba_state_save(plugin, stream);
}

static bool mamba_state_context_load(const clap_plugin_t *plugin, const clap_istream_t *stream, uint32_t context_type) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    int32_t version = 0;
    if (stream->read(stream, &version, sizeof(version)) < (int64_t) sizeof(version)) return false;
    if (version > 2) return false;
    MambaPresetData data;
    if (stream->read(stream, &data, sizeof(data)) < (int64_t) sizeof(data)) return false;
    state->bypass.store(data.bypass, std::memory_order_release);
    state->wetDry.store(data.wetDry, std::memory_order_release);
    state->intensity.store(data.intensity, std::memory_order_release);
    state->currentBypassValue = data.bypass;
    if (context_type != CLAP_STATE_CONTEXT_FOR_PROJECT) {
        if (state->engineL) state->engineL->reset_states();
        if (state->engineR) state->engineR->reset_states();
    }
    return true;
}

const clap_plugin_state_context_t mamba_state_context_extension = {mamba_state_context_save, mamba_state_context_load};

// -----------------------------------------------------------------------------
// Render + Tail (unchanged)
// -----------------------------------------------------------------------------

static bool clap_render_has_hard_realtime(const clap_plugin_t *) {
    return true;
}
static bool clap_render_set_mode(const clap_plugin_t *, clap_plugin_render_mode mode) {
    return mode == CLAP_RENDER_REALTIME;
}
const clap_plugin_render_t mamba_render_extension = {clap_render_has_hard_realtime, clap_render_set_mode};

static uint32_t clap_tail_get(const clap_plugin_t *) {
    return 0;
}
const clap_plugin_tail_t mamba_tail_extension = {clap_tail_get};

// -----------------------------------------------------------------------------
// Plugin Lifecycle
// -----------------------------------------------------------------------------

static bool mamba_init(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state->modelPath.empty()) return false;
    if (!mambaFileExists(state->modelPath)) return false;
    return true;
}

static void mamba_destroy(const clap_plugin_t *plugin) {
    delete (MambaPluginState *) plugin->plugin_data;
}

static bool mamba_activate(const clap_plugin_t *plugin, double sample_rate, uint32_t, uint32_t max_frames_count) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    state->sampleRate = static_cast<float>(sample_rate);

    // Resolve model path: look for model.bin adjacent to the plugin binary
    if (state->modelPath.empty()) {
        state->modelPath = mambaResolveModulePath(nullptr, "model.bin");
    }

    size_t bufSize = std::max(static_cast<size_t>(max_frames_count), static_cast<size_t>(256));
    for (int ch = 0; ch < 2; ++ch) {
        state->chIn[ch].resize(bufSize);
        state->chOut[ch].resize(bufSize);
    }

    // Create engines (one per channel)
    auto engineL = std::make_unique<SsmInferenceCore>();
    auto engineR = std::make_unique<SsmInferenceCore>();

    if (!engineL->init(state->modelPath.c_str()) || !engineR->init(state->modelPath.c_str())) {
        std::fprintf(stderr, "[CLARA] ERROR: failed to load model: %s\n", state->modelPath.c_str());
        return false;
    }

    // Default conditioning: neutral (all zeros = no modulation)
    ConditioningGenerator::neutral(state->condBuffer, COND_DIM);
    engineL->set_conditioning(state->condBuffer);
    engineR->set_conditioning(state->condBuffer);

    state->enginePtrL.store(engineL.get(), std::memory_order_release);
    state->enginePtrR.store(engineR.get(), std::memory_order_release);
    state->engineL = std::move(engineL);
    state->engineR = std::move(engineR);

    state->crossfadeStep = 1.0f / (0.05f * static_cast<float>(sample_rate));
    return true;
}

static void mamba_deactivate(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    state->enginePtrL.store(nullptr, std::memory_order_release);
    state->enginePtrR.store(nullptr, std::memory_order_release);
    state->engineL.reset();
    state->engineR.reset();
}

static bool mamba_start_processing(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    return state->enginePtrL.load(std::memory_order_acquire) != nullptr;
}

static void mamba_stop_processing(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state && state->engineL) state->engineL->reset_states();
    if (state && state->engineR) state->engineR->reset_states();
}

static void mamba_reset(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state && state->engineL) state->engineL->reset_states();
    if (state && state->engineR) state->engineR->reset_states();
}

// -----------------------------------------------------------------------------
// Audio Processing
// -----------------------------------------------------------------------------

static clap_process_status mamba_process(const clap_plugin_t *plugin, const clap_process_t *process) {
    auto *state = (MambaPluginState *) plugin->plugin_data;

    if (process->audio_outputs_count == 0 || process->audio_inputs_count == 0) return CLAP_PROCESS_CONTINUE;

    const uint32_t numSamples = process->frames_count;
    if (numSamples == 0) return CLAP_PROCESS_CONTINUE;

    const uint32_t inChans = process->audio_inputs[0].channel_count;
    const uint32_t outChans = process->audio_outputs[0].channel_count;
    const uint32_t processChans = std::min({inChans, outChans, 2u});

    float wetDry = state->wetDry.load(std::memory_order_acquire);
    float targetBypass = state->bypass.load(std::memory_order_acquire);
    float intensityVal = state->intensity.load(std::memory_order_acquire);

    const float crossfadeStep =
        (targetBypass != state->currentBypassValue)
            ? ((targetBypass > state->currentBypassValue) ? state->crossfadeStep : -state->crossfadeStep)
            : 0.0f;

    auto *engineL = state->enginePtrL.load(std::memory_order_acquire);
    auto *engineR = state->enginePtrR.load(std::memory_order_acquire);

    if (!engineL || !engineR || state->modelReloadPending.load(std::memory_order_acquire)) {
        // Passthrough
        for (uint32_t ch = 0; ch < processChans; ++ch)
            std::memcpy(process->audio_outputs[0].data32[ch], process->audio_inputs[0].data32[ch],
                        numSamples * sizeof(float));
        for (uint32_t ch = processChans; ch < outChans; ++ch)
            std::memcpy(process->audio_outputs[0].data32[ch], process->audio_outputs[0].data32[0],
                        numSamples * sizeof(float));
        return CLAP_PROCESS_CONTINUE;
    }

    // Update conditioning: push intensity into user-knob slots [40-71]
    // ponytail: simple intensity mapping to first 2 knob dimensions
    float intensityKnob = intensityVal * 2.0f - 1.0f; // map [0,1] → [-1,1]
    state->condBuffer[COND_KNOB_OFFSET] = intensityKnob;
    state->condBuffer[COND_KNOB_OFFSET + 1] = wetDry * 2.0f - 1.0f;
    engineL->set_conditioning(state->condBuffer);
    engineR->set_conditioning(state->condBuffer);

    // Process L channel through engineL, R channel through engineR
    float *inputs[2] = {process->audio_inputs[0].data32[0],
                        processChans > 1 ? process->audio_inputs[0].data32[1] : process->audio_inputs[0].data32[0]};
    SsmInferenceCore *engines[2] = {engineL, engineR};

    for (uint32_t ch = 0; ch < 2; ++ch) {
        if (ch < processChans) engines[ch]->process_block(inputs[ch], state->chOut[ch].data(), numSamples);
        else
            std::memset(state->chOut[ch].data(), 0, numSamples * sizeof(float));
    }

    // Mix per-sample (shared bypass ramp across channels)
    float bypassVal = state->currentBypassValue;
    for (uint32_t s = 0; s < numSamples; ++s) {
        for (uint32_t ch = 0; ch < processChans; ++ch) {
            float dry = process->audio_inputs[0].data32[ch][s];
            float wet = state->chOut[ch][s];
            wet = dry + (wet - dry) * intensityVal;
            float mix = (1.0f - bypassVal) * (wet * wetDry + dry * (1.0f - wetDry)) + bypassVal * dry;
            process->audio_outputs[0].data32[ch][s] = mix;
        }
        bypassVal = std::clamp(bypassVal + crossfadeStep, 0.0f, 1.0f);
    }
    state->currentBypassValue = bypassVal;

    for (uint32_t ch = processChans; ch < outChans; ++ch)
        std::memcpy(process->audio_outputs[0].data32[ch], process->audio_outputs[0].data32[0],
                    numSamples * sizeof(float));

    return CLAP_PROCESS_CONTINUE;
}

// -----------------------------------------------------------------------------
// Audio Ports
// -----------------------------------------------------------------------------

static uint32_t clap_audio_ports_count(const clap_plugin_t *, bool is_input) {
    return is_input ? 1 : 1;
}
static bool clap_audio_ports_get(const clap_plugin_t *, uint32_t index, bool is_input, clap_audio_port_info_t *info) {
    if (index > 0) return false;
    std::memset(info, 0, sizeof(*info));
    info->id = 0;
    info->channel_count = 2;
    info->flags = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type = nullptr;
    std::strncpy(info->name, is_input ? "Stereo In" : "Stereo Out", sizeof(info->name));
    return true;
}

static const clap_plugin_audio_ports_t mamba_audio_ports_extension = {clap_audio_ports_count, clap_audio_ports_get};

// -----------------------------------------------------------------------------
// Extensions
// -----------------------------------------------------------------------------

const void *mamba_get_extension(const clap_plugin_t *, const char *extensionId) {
    if (strcmp(extensionId, CLAP_EXT_STATE) == 0) return &mamba_state_extension;
    if (strcmp(extensionId, CLAP_EXT_STATE_CONTEXT) == 0) return &mamba_state_context_extension;
    if (strcmp(extensionId, CLAP_EXT_PARAMS) == 0) return &mamba_params_extension;
    if (strcmp(extensionId, CLAP_EXT_LATENCY) == 0) return &mamba_latency_extension;
    if (strcmp(extensionId, CLAP_EXT_AUDIO_PORTS) == 0) return &mamba_audio_ports_extension;
    if (strcmp(extensionId, CLAP_EXT_RENDER) == 0) return &mamba_render_extension;
    if (strcmp(extensionId, CLAP_EXT_TAIL) == 0) return &mamba_tail_extension;
    return nullptr;
}

// -----------------------------------------------------------------------------
// Main Thread (deferred tasks)
// -----------------------------------------------------------------------------

static void mamba_on_main_thread(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    MambaPluginState::DeferredTask task;
    while (state->deferredTaskQueue.pop(task)) {
        switch (task) {
            case MambaPluginState::DeferredTask::ModelReload: {
                state->modelReloadPending.store(true, std::memory_order_release);
                std::atomic_thread_fence(std::memory_order_seq_cst);

                auto newL = std::make_unique<SsmInferenceCore>();
                auto newR = std::make_unique<SsmInferenceCore>();
                if (newL->init(state->modelPath.c_str()) && newR->init(state->modelPath.c_str())) {
                    newL->set_conditioning(state->condBuffer);
                    newR->set_conditioning(state->condBuffer);
                    state->enginePtrL.store(newL.get(), std::memory_order_release);
                    state->enginePtrR.store(newR.get(), std::memory_order_release);
                    state->engineL = std::move(newL);
                    state->engineR = std::move(newR);
                } else {
                    std::fprintf(stderr, "[CLARA] WARNING: model reload failed\n");
                }
                state->modelReloadPending.store(false, std::memory_order_release);
                break;
            }
            case MambaPluginState::DeferredTask::AnalysisQueueDrain:
                // Future: ARA bridge updates condBuffer from musical context
                break;
        }
    }
}

// -----------------------------------------------------------------------------
// Plugin VTable
// -----------------------------------------------------------------------------

const clap_plugin_t mamba_plugin = {&mamba_plugin_info,
                                    nullptr,
                                    mamba_init,
                                    mamba_destroy,
                                    mamba_activate,
                                    mamba_deactivate,
                                    mamba_start_processing,
                                    mamba_stop_processing,
                                    mamba_reset,
                                    mamba_process,
                                    mamba_get_extension,
                                    mamba_on_main_thread};

// -----------------------------------------------------------------------------
// Plugin Entry
// -----------------------------------------------------------------------------

static uint32_t clap_get_plugin_count(const clap_plugin_factory_t *) {
    return 1;
}
static const clap_plugin_descriptor_t *clap_get_plugin_descriptor(const clap_plugin_factory_t *, uint32_t index) {
    return index == 0 ? &mamba_plugin_info : nullptr;
}

static const clap_plugin_t *clap_create_plugin(const clap_plugin_factory_t *, const clap_host_t *host,
                                               const char *plugin_id) {
    if (strcmp(plugin_id, mamba_plugin_info.id) != 0) return nullptr;
    auto plugin = std::make_unique<clap_plugin_t>(mamba_plugin);
    auto state = std::make_unique<MambaPluginState>();
    state->host = host;
    state->modelPath = mambaResolveModulePath((void *) clap_create_plugin, "model.bin");
    plugin->plugin_data = state.release();
    return plugin.release();
}

static const clap_plugin_factory_t mamba_plugin_factory = {clap_get_plugin_count, clap_get_plugin_descriptor,
                                                           clap_create_plugin};

static const void *clap_get_factory(const char *factory_id) {
    if (strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID) == 0) return &mamba_plugin_factory;
    return nullptr;
}

extern "C" CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    CLAP_VERSION_INIT, [](const char *) -> bool { return true; }, []() -> void {}, clap_get_factory};
