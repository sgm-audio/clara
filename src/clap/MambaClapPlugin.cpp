#include "MambaClapPlugin.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <cstdio>
#include <clap/ext/params.h>
#include <clap/ext/latency.h>
#include <clap/ext/state.h>
#include <clap/ext/audio-ports.h>
#include <dlfcn.h>
#include <unistd.h>

// -----------------------------------------------------------------------------
// Model Path Resolution
// -----------------------------------------------------------------------------

static std::string resolveModelPath(const char* plugin_path, const char* model_filename) {
    std::string base(plugin_path ? plugin_path : "");
    auto sep = base.rfind('/');
    if (sep != std::string::npos)
        base.resize(sep + 1);
    else
        base = "./";
    return base + "models/" + model_filename;
}

// -----------------------------------------------------------------------------
// Plugin Info
// -----------------------------------------------------------------------------

const clap_plugin_descriptor_t mamba_plugin_info = {
    .clap_version = CLAP_VERSION_INIT,
    .id = "com.mamba.audio.suite",
    .name = "Mamba-SSM Processor",
    .vendor = "Elite DSP",
    .url = "https://example.com/mamba_audio_suite",
    .manual_url = "https://example.com/mamba_audio_suite/manual",
    .support_url = "https://example.com/mamba_audio_suite/support",
    .version = "1.0.0",
    .description = "Real-time Mamba-SSM audio processor with ARA2 integration",
    .features = (const char* const[]){CLAP_PLUGIN_FEATURE_AUDIO_EFFECT, CLAP_PLUGIN_FEATURE_UTILITY, nullptr},
};

// -----------------------------------------------------------------------------
// Parameters Extension
// -----------------------------------------------------------------------------

static uint32_t clap_params_count(const clap_plugin_t*) {
    return PARAM_COUNT;
}

static bool clap_params_get_info(const clap_plugin_t*, uint32_t param_index, clap_param_info_t* param_info) {
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

static bool clap_params_get_value(const clap_plugin_t* plugin, clap_id param_id, double* value) {
    auto* state = (MambaPluginState*)plugin->plugin_data;
    switch (param_id) {
    case PARAM_BYPASS:   *value = state->bypass.load(std::memory_order_acquire); return true;
    case PARAM_WET_DRY:  *value = state->wetDry.load(std::memory_order_acquire); return true;
    case PARAM_INTENSITY:*value = state->intensity.load(std::memory_order_acquire); return true;
    default: return false;
    }
}

static bool clap_params_value_to_text(const clap_plugin_t*, clap_id param_id, double value, char* display, uint32_t size) {
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

static bool clap_params_text_to_value(const clap_plugin_t*, clap_id param_id, const char* display, double* value) {
    switch (param_id) {
    case PARAM_BYPASS:
        *value = (std::strcmp(display, "On") == 0 || std::strcmp(display, "1") == 0) ? 1.0 : 0.0;
        return true;
    case PARAM_WET_DRY:
    case PARAM_INTENSITY: {
        double pct = 0.0;
        if (std::sscanf(display, "%lf%%", &pct) == 1 || std::sscanf(display, "%lf", &pct) == 1) {
            *value = std::clamp(pct / 100.0, 0.0, 1.0);
            return true;
        }
        return false;
    }
    default:
        return false;
    }
}

static void clap_params_flush(const clap_plugin_t* plugin, const clap_input_events_t* in, const clap_output_events_t*) {
    auto* state = (MambaPluginState*)plugin->plugin_data;
    if (!in) return;
    const uint32_t num_events = in->size(in);
    for (uint32_t i = 0; i < num_events; ++i) {
        const clap_event_header_t* event = in->get(in, i);
        if (event->space_id == CLAP_CORE_EVENT_SPACE_ID && event->type == CLAP_EVENT_PARAM_VALUE) {
            const clap_event_param_value_t* param_event = (const clap_event_param_value_t*)event;
            switch (param_event->param_id) {
            case PARAM_BYPASS:    state->bypass.store(static_cast<float>(param_event->value), std::memory_order_release); break;
            case PARAM_WET_DRY:   state->wetDry.store(static_cast<float>(param_event->value), std::memory_order_release); break;
            case PARAM_INTENSITY: state->intensity.store(static_cast<float>(param_event->value), std::memory_order_release); break;
            }
        }
    }
}

static const clap_plugin_params_t mamba_params_extension = {
    clap_params_count,
    clap_params_get_info,
    clap_params_get_value,
    clap_params_value_to_text,
    clap_params_text_to_value,
    clap_params_flush
};

// -----------------------------------------------------------------------------
// Latency Extension
// -----------------------------------------------------------------------------

static uint32_t clap_latency_get(const clap_plugin_t*) {
    return 4;
}

static const clap_plugin_latency_t mamba_latency_extension = {
    clap_latency_get
};

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

static bool mamba_state_save(const clap_plugin_t* plugin, const clap_ostream_t* stream) {
    auto* state = (MambaPluginState*)plugin->plugin_data;
    const int32_t version = 1;
    if (stream->write(stream, &version, sizeof(version)) < 0) return false;
    MambaPresetData data;
    data.bypass = state->bypass.load(std::memory_order_acquire);
    data.wetDry = state->wetDry.load(std::memory_order_acquire);
    data.intensity = state->intensity.load(std::memory_order_acquire);
    return stream->write(stream, &data, sizeof(data)) >= 0;
}

static bool mamba_state_load(const clap_plugin_t* plugin, const clap_istream_t* stream) {
    auto* state = (MambaPluginState*)plugin->plugin_data;

    int32_t version = 0;
    if (stream->read(stream, &version, sizeof(version)) < (int64_t)sizeof(version)) {
        return false;
    }
    if (version > 1) return false;

    MambaPresetData data;
    if (stream->read(stream, &data, sizeof(data)) < (int64_t)sizeof(data)) {
        return false;
    }
    state->bypass.store(data.bypass, std::memory_order_release);
    state->wetDry.store(data.wetDry, std::memory_order_release);
    state->intensity.store(data.intensity, std::memory_order_release);
    state->currentBypassValue = data.bypass;

    if (state->engine) {
        state->engine->resetState();
    }
    return true;
}

const clap_plugin_state_t mamba_state_extension = {
    mamba_state_save,
    mamba_state_load
};

// -----------------------------------------------------------------------------
// Plugin Implementation
// -----------------------------------------------------------------------------

static bool mamba_init(const clap_plugin_t*) {
    return true;
}

static void mamba_destroy(const clap_plugin_t* plugin) {
    auto* state = (MambaPluginState*)plugin->plugin_data;
    delete state;
}

static bool mamba_activate(const clap_plugin_t* plugin, double sample_rate, uint32_t, uint32_t max_frames_count) {
    auto* state = (MambaPluginState*)plugin->plugin_data;
    state->sampleRate = static_cast<float>(sample_rate);

    MambaInferenceEngine::Config config;
    if (state->modelPath.empty()) {
        state->modelPath = resolveModelPath(nullptr, "mamba_rt.onnx");
    }
    config.modelPath = state->modelPath;
    config.maxBlockSize = std::max(static_cast<size_t>(max_frames_count), static_cast<size_t>(256));

    for (int ch = 0; ch < 2; ++ch) {
        state->chIn[ch].resize(config.maxBlockSize);
        state->chOut[ch].resize(config.maxBlockSize);
    }

    state->engine = std::make_unique<MambaInferenceEngine>(config);
    try {
        state->engine->init();
    } catch (const std::exception&) {
        state->engine.reset();
        return false;
    }

    state->crossfadeStep = 1.0f / (0.05f * static_cast<float>(sample_rate));
    return true;
}

static void mamba_deactivate(const clap_plugin_t* plugin) {
    auto* state = (MambaPluginState*)plugin->plugin_data;
    state->engine.reset();
}

static bool mamba_start_processing(const clap_plugin_t*) {
    return true;
}

static void mamba_stop_processing(const clap_plugin_t*) {
}

static void mamba_reset(const clap_plugin_t* plugin) {
    auto* state = (MambaPluginState*)plugin->plugin_data;
    if (state && state->engine) {
        state->engine->resetState();
    }
}

static clap_process_status mamba_process(const clap_plugin_t* plugin, const clap_process_t* process) {
    auto* state = (MambaPluginState*)plugin->plugin_data;

    const uint32_t num_events = process->in_events->size(process->in_events);
    for (uint32_t i = 0; i < num_events; ++i) {
        const clap_event_header_t* event = process->in_events->get(process->in_events, i);
        if (event->space_id == CLAP_CORE_EVENT_SPACE_ID && event->type == CLAP_EVENT_PARAM_VALUE) {
            const clap_event_param_value_t* param_event = (const clap_event_param_value_t*)event;
            switch (param_event->param_id) {
            case PARAM_BYPASS:    state->bypass.store(static_cast<float>(param_event->value), std::memory_order_release); break;
            case PARAM_WET_DRY:   state->wetDry.store(static_cast<float>(param_event->value), std::memory_order_release); break;
            case PARAM_INTENSITY: state->intensity.store(static_cast<float>(param_event->value), std::memory_order_release); break;
            }
        }
    }

    if (process->audio_outputs_count == 0 || process->audio_inputs_count == 0) {
        return CLAP_PROCESS_CONTINUE;
    }

    const uint32_t numSamples = process->frames_count;
    if (numSamples == 0) return CLAP_PROCESS_CONTINUE;

    const uint32_t inChans = process->audio_inputs[0].channel_count;
    const uint32_t outChans = process->audio_outputs[0].channel_count;
    const uint32_t processChans = std::min({inChans, outChans, 2u});

    float wetDry = state->wetDry.load(std::memory_order_acquire);
    float targetBypass = state->bypass.load(std::memory_order_acquire);
    float intensityVal = state->intensity.load(std::memory_order_acquire);

    const float crossfadeStep = (targetBypass != state->currentBypassValue)
        ? ((targetBypass > state->currentBypassValue) ? state->crossfadeStep : -state->crossfadeStep)
        : 0.0f;

    for (uint32_t ch = 0; ch < processChans; ++ch) {
        float* input = process->audio_inputs[0].data32[ch];
        float* output = process->audio_outputs[0].data32[ch];

        std::memcpy(state->chIn[ch].data(), input, numSamples * sizeof(float));

        state->engine->processBlock(state->chIn[ch].data(), state->chOut[ch].data(), numSamples);

        for (uint32_t s = 0; s < numSamples; ++s) {
            float dry = input[s];
            float wet = state->chOut[ch][s];

            wet = dry + (wet - dry) * intensityVal;

            float mix = (1.0f - state->currentBypassValue) * (wet * wetDry + dry * (1.0f - wetDry))
                      + state->currentBypassValue * dry;

            state->currentBypassValue = std::clamp(state->currentBypassValue + crossfadeStep, 0.0f, 1.0f);

            output[s] = mix;
        }
    }

    for (uint32_t ch = processChans; ch < outChans; ++ch) {
        std::memcpy(process->audio_outputs[0].data32[ch],
                    process->audio_outputs[0].data32[0],
                    numSamples * sizeof(float));
    }

    return CLAP_PROCESS_CONTINUE;
}

static uint32_t clap_audio_ports_count(const clap_plugin_t*, bool is_input) {
    return (is_input) ? 1 : 1;
}

static bool clap_audio_ports_get(const clap_plugin_t*, uint32_t index, bool is_input, clap_audio_port_info_t* info) {
    if (index > 0) return false;
    std::memset(info, 0, sizeof(*info));
    info->id = 0;
    info->channel_count = 2;
    info->flags = CLAP_AUDIO_PORT_IS_MAIN;
    info->port_type = CLAP_PORT_STEREO;
    std::strncpy(info->name, is_input ? "Stereo In" : "Stereo Out", sizeof(info->name));
    return true;
}

static const clap_plugin_audio_ports_t mamba_audio_ports_extension = {
    clap_audio_ports_count,
    clap_audio_ports_get
};

static const void* mamba_get_extension(const clap_plugin_t*, const char* extensionId) {
    if (strcmp(extensionId, CLAP_EXT_STATE) == 0) return &mamba_state_extension;
    if (strcmp(extensionId, CLAP_EXT_PARAMS) == 0) return &mamba_params_extension;
    if (strcmp(extensionId, CLAP_EXT_LATENCY) == 0) return &mamba_latency_extension;
    if (strcmp(extensionId, CLAP_EXT_AUDIO_PORTS) == 0) return &mamba_audio_ports_extension;
    return nullptr;
}

static void mamba_on_main_thread(const clap_plugin_t*) {}

const clap_plugin_t mamba_plugin = {
    &mamba_plugin_info,
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
    mamba_on_main_thread
};

// -----------------------------------------------------------------------------
// Plugin Entry
// -----------------------------------------------------------------------------

static uint32_t clap_get_plugin_count(const clap_plugin_factory_t*) {
    return 1;
}

static const clap_plugin_descriptor_t* clap_get_plugin_descriptor(const clap_plugin_factory_t*, uint32_t index) {
    return index == 0 ? &mamba_plugin_info : nullptr;
}

static const clap_plugin_t* clap_create_plugin(const clap_plugin_factory_t*, const clap_host_t* host, const char* plugin_id) {
    if (strcmp(plugin_id, mamba_plugin_info.id) != 0) return nullptr;

    auto plugin = std::make_unique<clap_plugin_t>(mamba_plugin);
    auto state = std::make_unique<MambaPluginState>();
    state->host = host;

    Dl_info info;
    if (dladdr((void*)clap_create_plugin, &info) && info.dli_fname) {
        state->modelPath = resolveModelPath(info.dli_fname, "mamba_rt.onnx");
    }

    plugin->plugin_data = state.release();
    return plugin.release();
}

static const clap_plugin_factory_t mamba_plugin_factory = {
    clap_get_plugin_count,
    clap_get_plugin_descriptor,
    clap_create_plugin
};

static const void* clap_get_factory(const char* factory_id) {
    if (strcmp(factory_id, CLAP_PLUGIN_FACTORY_ID) == 0) {
        return &mamba_plugin_factory;
    }
    return nullptr;
}

extern "C" CLAP_EXPORT const clap_plugin_entry_t clap_entry = {
    CLAP_VERSION_INIT,
    [](const char*) -> bool { return true; },
    []() -> void {},
    clap_get_factory
};
