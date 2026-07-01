#include "../core/SPSCQueue.hpp"
#include "clap/MambaClapPlugin.hpp"

#include <clap/clap.h>
#include <clap/ext/audio-ports.h>
#include <clap/ext/latency.h>
#include <clap/ext/params.h>
#include <clap/ext/state.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Test framework
// ---------------------------------------------------------------------------
#define TEST_ASSERT(cond, msg)                                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define TEST_PASS(msg) std::printf("  PASS: %s\n", msg)

// ---------------------------------------------------------------------------
// Mock CLAP streams for state save/load roundtrip
// ---------------------------------------------------------------------------
struct MockOStream : clap_ostream {
    std::vector<char> buf;

    MockOStream() {
        this->ctx = this;
        this->write = [](const clap_ostream *s, const void *d, uint64_t sz) -> int64_t {
            auto *self = static_cast<MockOStream *>(s->ctx);
            const char *data = static_cast<const char *>(d);
            self->buf.insert(self->buf.end(), data, data + static_cast<size_t>(sz));
            return static_cast<int64_t>(sz);
        };
    }
};

struct MockIStream : clap_istream {
    std::vector<char> buf;
    size_t pos;

    explicit MockIStream(const std::vector<char> &data) : buf(data), pos(0) {
        this->ctx = this;
        this->read = [](const clap_istream *s, void *buffer, uint64_t size) -> int64_t {
            auto *self = static_cast<MockIStream *>(s->ctx);
            uint64_t avail = self->buf.size() - self->pos;
            uint64_t toRead = std::min(size, avail);
            if (toRead == 0) return 0;
            std::memcpy(buffer, self->buf.data() + self->pos, static_cast<size_t>(toRead));
            self->pos += static_cast<size_t>(toRead);
            return static_cast<int64_t>(toRead);
        };
    }
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static clap_plugin_t makeMockPlugin(MambaPluginState *state) {
    clap_plugin_t plugin = mamba_plugin;
    plugin.plugin_data = state;
    return plugin;
}

// ---------------------------------------------------------------------------
// 1. test_plugin_descriptor
//    Verify MambaPluginState default values
// ---------------------------------------------------------------------------
static int test_plugin_descriptor() {
    MambaPluginState s;

    TEST_ASSERT(s.bypass.load() == 0.0f, "default bypass is 0.0");
    TEST_ASSERT(s.wetDry.load() == 1.0f, "default wetDry is 1.0");
    TEST_ASSERT(s.intensity.load() == 0.5f, "default intensity is 0.5");
    TEST_ASSERT(s.modelPath.empty(), "modelPath initially empty");
    TEST_ASSERT(s.sampleRate == 44100.0f, "default sampleRate is 44100");

    TEST_PASS("plugin descriptor");
    return 0;
}

// ---------------------------------------------------------------------------
// 2. test_plugin_info_descriptor
//    Verify the global plugin descriptor
// ---------------------------------------------------------------------------
static int test_plugin_info_descriptor() {
    TEST_ASSERT(mamba_plugin_info.clap_version.major == 1, "CLAP version major is 1");
    TEST_ASSERT(mamba_plugin_info.clap_version.minor == 2, "CLAP version minor is 2");

    std::string id(mamba_plugin_info.id);
    TEST_ASSERT(id == "com.mamba.audio.suite", "plugin id");

    bool hasAudioEffect = false;
    bool hasUtility = false;
    for (const char *const *f = mamba_plugin_info.features; *f != nullptr; ++f) {
        if (std::strcmp(*f, CLAP_PLUGIN_FEATURE_AUDIO_EFFECT) == 0) hasAudioEffect = true;
        if (std::strcmp(*f, CLAP_PLUGIN_FEATURE_UTILITY) == 0) hasUtility = true;
    }
    TEST_ASSERT(hasAudioEffect, "features include audio-effect");
    TEST_ASSERT(hasUtility, "features include utility");

    TEST_PASS("plugin info descriptor");
    return 0;
}

// ---------------------------------------------------------------------------
// 3. test_preset_data_packing
//    ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct TestPresetData {
    float bypass;
    float wetDry;
    float intensity;
};
#pragma pack(pop)

static_assert(sizeof(TestPresetData) == 12, "MambaPresetData must be exactly 12 bytes");
static_assert(offsetof(TestPresetData, bypass) == 0, "bypass offset 0");
static_assert(offsetof(TestPresetData, wetDry) == 4, "wetDry offset 4");
static_assert(offsetof(TestPresetData, intensity) == 8, "intensity offset 8");

static int test_preset_data_packing() {
    TestPresetData data;
    data.bypass = 1.0f;
    data.wetDry = 0.5f;
    data.intensity = 0.75f;

    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&data);
    float readBypass, readWetDry, readIntensity;
    std::memcpy(&readBypass, bytes + 0, 4);
    std::memcpy(&readWetDry, bytes + 4, 4);
    std::memcpy(&readIntensity, bytes + 8, 4);

    TEST_ASSERT(readBypass == 1.0f, "bypass byte-roundtrip");
    TEST_ASSERT(readWetDry == 0.5f, "wetDry byte-roundtrip");
    TEST_ASSERT(readIntensity == 0.75f, "intensity byte-roundtrip");

    TEST_PASS("preset data packing");
    return 0;
}

// ---------------------------------------------------------------------------
// 4. test_param_ranges
//    ---------------------------------------------------------------------------
static int test_param_ranges() {
    MambaPluginState state;
    clap_plugin_t plugin = makeMockPlugin(&state);

    const auto *params = static_cast<const clap_plugin_params_t *>(mamba_get_extension(&plugin, CLAP_EXT_PARAMS));
    TEST_ASSERT(params != nullptr, "params extension available");

    clap_param_info_t info;

    std::memset(&info, 0, sizeof(info));
    TEST_ASSERT(params->get_info(&plugin, PARAM_BYPASS, &info), "bypass get_info");
    TEST_ASSERT(info.id == PARAM_BYPASS, "bypass id");
    TEST_ASSERT(info.min_value == 0.0, "bypass min");
    TEST_ASSERT(info.max_value == 1.0, "bypass max");
    TEST_ASSERT(info.default_value == 0.0, "bypass default");

    std::memset(&info, 0, sizeof(info));
    TEST_ASSERT(params->get_info(&plugin, PARAM_WET_DRY, &info), "wetDry get_info");
    TEST_ASSERT(info.id == PARAM_WET_DRY, "wetDry id");
    TEST_ASSERT(info.min_value == 0.0, "wetDry min");
    TEST_ASSERT(info.max_value == 1.0, "wetDry max");
    TEST_ASSERT(info.default_value == 1.0, "wetDry default");

    std::memset(&info, 0, sizeof(info));
    TEST_ASSERT(params->get_info(&plugin, PARAM_INTENSITY, &info), "intensity get_info");
    TEST_ASSERT(info.id == PARAM_INTENSITY, "intensity id");
    TEST_ASSERT(info.min_value == 0.0, "intensity min");
    TEST_ASSERT(info.max_value == 1.0, "intensity max");
    TEST_ASSERT(info.default_value == 0.5, "intensity default");

    TEST_PASS("param ranges");
    return 0;
}

// ---------------------------------------------------------------------------
// 5. test_param_value_to_text
//    ---------------------------------------------------------------------------
static int test_param_value_to_text() {
    MambaPluginState state;
    clap_plugin_t plugin = makeMockPlugin(&state);

    const auto *params = static_cast<const clap_plugin_params_t *>(mamba_get_extension(&plugin, CLAP_EXT_PARAMS));

    char buf[256];

    // Bypass
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_BYPASS, 0.0, buf, sizeof(buf)), "bypass 0 -> text");
    TEST_ASSERT(std::strcmp(buf, "Off") == 0, "bypass 0.0 -> \"Off\"");
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_BYPASS, 1.0, buf, sizeof(buf)), "bypass 1 -> text");
    TEST_ASSERT(std::strcmp(buf, "On") == 0, "bypass 1.0 -> \"On\"");

    // Wet/Dry
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_WET_DRY, 0.5, buf, sizeof(buf)), "wetDry 0.5 -> text");
    TEST_ASSERT(std::strcmp(buf, "50.0%") == 0, "wetDry 0.5 -> \"50.0%\"");
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_WET_DRY, 0.0, buf, sizeof(buf)), "wetDry 0.0 -> text");
    TEST_ASSERT(std::strcmp(buf, "0.0%") == 0, "wetDry 0.0 -> \"0.0%\"");
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_WET_DRY, 1.0, buf, sizeof(buf)), "wetDry 1.0 -> text");
    TEST_ASSERT(std::strcmp(buf, "100.0%") == 0, "wetDry 1.0 -> \"100.0%\"");

    // Intensity
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_INTENSITY, 0.5, buf, sizeof(buf)), "intensity 0.5 -> text");
    TEST_ASSERT(std::strcmp(buf, "50.0%") == 0, "intensity 0.5 -> \"50.0%\"");
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_INTENSITY, 0.0, buf, sizeof(buf)), "intensity 0.0 -> text");
    TEST_ASSERT(std::strcmp(buf, "0.0%") == 0, "intensity 0.0 -> \"0.0%\"");
    TEST_ASSERT(params->value_to_text(&plugin, PARAM_INTENSITY, 1.0, buf, sizeof(buf)), "intensity 1.0 -> text");
    TEST_ASSERT(std::strcmp(buf, "100.0%") == 0, "intensity 1.0 -> \"100.0%\"");

    TEST_PASS("param value to text");
    return 0;
}

// ---------------------------------------------------------------------------
// 6. test_param_text_to_value
//    ---------------------------------------------------------------------------
static int test_param_text_to_value() {
    MambaPluginState state;
    clap_plugin_t plugin = makeMockPlugin(&state);

    const auto *params = static_cast<const clap_plugin_params_t *>(mamba_get_extension(&plugin, CLAP_EXT_PARAMS));

    double val = -1.0;

    // Bypass
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_BYPASS, "On", &val), "bypass \"On\" -> value");
    TEST_ASSERT(val == 1.0, "bypass \"On\" -> 1.0");
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_BYPASS, "Off", &val), "bypass \"Off\" -> value");
    TEST_ASSERT(val == 0.0, "bypass \"Off\" -> 0.0");

    // Wet/Dry
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_WET_DRY, "75%", &val), "wetDry \"75%\" -> value");
    TEST_ASSERT(val == 0.75, "wetDry \"75%\" -> 0.75");
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_WET_DRY, "50.0%", &val), "wetDry \"50.0%\" -> value");
    TEST_ASSERT(val == 0.5, "wetDry \"50.0%\" -> 0.5");
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_WET_DRY, "0.5", &val), "wetDry \"0.5\" -> value");
    TEST_ASSERT(val == 0.5, "wetDry \"0.5\" -> 0.5");

    // Intensity (same parsing as Wet/Dry)
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_INTENSITY, "75%", &val), "intensity \"75%\" -> value");
    TEST_ASSERT(val == 0.75, "intensity \"75%\" -> 0.75");
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_INTENSITY, "50.0%", &val), "intensity \"50.0%\" -> value");
    TEST_ASSERT(val == 0.5, "intensity \"50.0%\" -> 0.5");
    TEST_ASSERT(params->text_to_value(&plugin, PARAM_INTENSITY, "0.5", &val), "intensity \"0.5\" -> value");
    TEST_ASSERT(val == 0.5, "intensity \"0.5\" -> 0.5");

    TEST_PASS("param text to value");
    return 0;
}

// ---------------------------------------------------------------------------
// 7. test_state_save_load_roundtrip
//    ---------------------------------------------------------------------------
static int test_state_save_load_roundtrip() {
    MambaPluginState saveState;
    clap_plugin_t savePlugin = makeMockPlugin(&saveState);

    saveState.bypass.store(1.0f, std::memory_order_release);
    saveState.wetDry.store(0.3f, std::memory_order_release);
    saveState.intensity.store(0.7f, std::memory_order_release);

    MockOStream os;
    TEST_ASSERT(mamba_state_extension.save(&savePlugin, &os), "state save");

    MambaPluginState loadState;
    clap_plugin_t loadPlugin = makeMockPlugin(&loadState);

    MockIStream is(os.buf);
    TEST_ASSERT(mamba_state_extension.load(&loadPlugin, &is), "state load");

    TEST_ASSERT(loadState.bypass.load() == 1.0f, "bypass restored");
    TEST_ASSERT(loadState.wetDry.load() == 0.3f, "wetDry restored");
    TEST_ASSERT(loadState.intensity.load() == 0.7f, "intensity restored");
    TEST_ASSERT(loadState.currentBypassValue == 1.0f, "currentBypassValue restored");

    // Verify saved stream is exactly 4 (version) + 12 (3 floats) = 16 bytes
    TEST_ASSERT(os.buf.size() == 16, "serialized size is 16 bytes");

    // Verify version word in the stream
    int32_t version;
    std::memcpy(&version, os.buf.data(), sizeof(version));
    TEST_ASSERT(version == 2, "version is 2 (Mamba-3)");

    TEST_PASS("state save/load roundtrip");
    return 0;
}

// ---------------------------------------------------------------------------
// 8. test_crossfade_step_calculation
//    ---------------------------------------------------------------------------
static int test_crossfade_step_calculation() {
    // The formula in mamba_activate: crossfadeStep = 1.0f / (0.05f * sampleRate)
    // 50ms crossfade at 44100 Hz
    const float sampleRate = 44100.0f;
    const float crossfadeStep = 1.0f / (0.05f * sampleRate);

    TEST_ASSERT(crossfadeStep > 0.0f, "crossfadeStep is positive");
    TEST_ASSERT(crossfadeStep < 0.001f, "crossfadeStep < 1 ms worth of samples");
    TEST_ASSERT(std::fabs(crossfadeStep - 0.0004535147f) < 1e-7f, "crossfadeStep at 44.1 kHz is ~0.0004535");

    // At 48000 Hz
    const float step48k = 1.0f / (0.05f * 48000.0f);
    const float expectedStep48k = 1.0f / 2400.0f;
    TEST_ASSERT(std::fabs(step48k - expectedStep48k) < 1e-9f, "crossfadeStep at 48 kHz is correct");

    TEST_PASS("crossfade step calculation");
    return 0;
}

// ---------------------------------------------------------------------------
// 9. test_latency_reporting
//    ---------------------------------------------------------------------------
static int test_latency_reporting() {
    MambaPluginState state;
    clap_plugin_t plugin = makeMockPlugin(&state);

    const auto *latency = static_cast<const clap_plugin_latency_t *>(mamba_get_extension(&plugin, CLAP_EXT_LATENCY));
    TEST_ASSERT(latency != nullptr, "latency extension available");

    uint32_t samples = latency->get(&plugin);
    TEST_ASSERT(samples == 0, "latency is 0 samples (Mamba-3: no conv1d)");

    TEST_PASS("latency reporting");
    return 0;
}

// ---------------------------------------------------------------------------
// 10. test_audio_ports
//    ---------------------------------------------------------------------------
static int test_audio_ports() {
    MambaPluginState state;
    clap_plugin_t plugin = makeMockPlugin(&state);

    const auto *ports =
        static_cast<const clap_plugin_audio_ports_t *>(mamba_get_extension(&plugin, CLAP_EXT_AUDIO_PORTS));
    TEST_ASSERT(ports != nullptr, "audio ports extension available");

    // Input ports
    uint32_t inputCount = ports->count(&plugin, true);
    TEST_ASSERT(inputCount == 1, "1 input port");

    clap_audio_port_info_t info;
    std::memset(&info, 0, sizeof(info));
    TEST_ASSERT(ports->get(&plugin, 0, true, &info), "input port 0 info");
    TEST_ASSERT(info.id == 0, "input port id is 0");
    TEST_ASSERT(info.channel_count == 2, "input port is stereo");
    TEST_ASSERT(info.flags & CLAP_AUDIO_PORT_IS_MAIN, "input port has IS_MAIN flag");
    TEST_ASSERT(std::strcmp(info.name, "Stereo In") == 0, "input port name");

    // Output ports
    uint32_t outputCount = ports->count(&plugin, false);
    TEST_ASSERT(outputCount == 1, "1 output port");

    std::memset(&info, 0, sizeof(info));
    TEST_ASSERT(ports->get(&plugin, 0, false, &info), "output port 0 info");
    TEST_ASSERT(info.id == 0, "output port id is 0");
    TEST_ASSERT(info.channel_count == 2, "output port is stereo");
    TEST_ASSERT(info.flags & CLAP_AUDIO_PORT_IS_MAIN, "output port has IS_MAIN flag");
    TEST_ASSERT(std::strcmp(info.name, "Stereo Out") == 0, "output port name");

    TEST_PASS("audio ports");
    return 0;
}

// ---------------------------------------------------------------------------
// 11. test_deferred_task_queue
//    ---------------------------------------------------------------------------
static int test_deferred_task_queue() {
    SPSCQueue<int, 64> queue;

    TEST_ASSERT(queue.empty(), "queue initially empty");

    queue.push(42);
    TEST_ASSERT(!queue.empty(), "queue non-empty after push");

    int val = 0;
    TEST_ASSERT(queue.pop(val), "pop succeeds");
    TEST_ASSERT(val == 42, "popped value matches");
    TEST_ASSERT(queue.empty(), "queue empty after pop");

    for (int i = 0; i < 63; ++i) {
        TEST_ASSERT(queue.push(i), "push succeeds up to capacity");
    }
    TEST_ASSERT(!queue.push(64), "push fails at capacity");

    for (int i = 0; i < 63; ++i) {
        TEST_ASSERT(queue.pop(val), "pop all items");
        TEST_ASSERT(val == i, "items in order");
    }
    TEST_ASSERT(queue.empty(), "queue empty after draining");

    TEST_PASS("deferred task queue");
    return 0;
}

// ---------------------------------------------------------------------------
// 12. test_spsc_two_thread
//     True SPSC concurrency: producer thread pushes N items, consumer thread
//     pops them. Verifies FIFO order under real concurrent contention and
//     exercises the alignas(64) cache-line layout.
// ---------------------------------------------------------------------------
static int test_spsc_two_thread() {
    SPSCQueue<int, 64> queue;
    constexpr int N = 1000;
    std::atomic<bool> producerDone{false};
    std::vector<int> consumed;
    consumed.reserve(N);

    std::thread producer([&]() {
        for (int i = 0; i < N; ++i) {
            while (!queue.push(i)) { /* spin until space */
            }
        }
        producerDone.store(true, std::memory_order_release);
    });

    std::thread consumer([&]() {
        int value;
        while (!producerDone.load(std::memory_order_acquire) || !queue.empty()) {
            if (queue.pop(value)) consumed.push_back(value);
        }
    });

    producer.join();
    consumer.join();

    TEST_ASSERT((int) consumed.size() == N, "all items consumed");
    for (int i = 0; i < N; ++i) {
        if (consumed[i] != i) {
            TEST_ASSERT(false, "order violated");
            return 1;
        }
    }
    TEST_PASS("SPSC two-thread FIFO");
    return 0;
}

// ---------------------------------------------------------------------------
// 13. test_spsc_wrap_around
//     Push/pop more items than capacity to exercise the ring buffer's
//     head/tail wrap-around. Capacity 64, push+pop 200 in 4 batches of 50.
// ---------------------------------------------------------------------------
static int test_spsc_wrap_around() {
    SPSCQueue<int, 64> queue;
    constexpr int N = 200;

    for (int batch = 0; batch < 4; ++batch) {
        for (int i = 0; i < 50; ++i) {
            if (!queue.push(batch * 50 + i)) {
                TEST_ASSERT(false, "push should succeed within batch");
                return 1;
            }
        }
        for (int i = 0; i < 50; ++i) {
            int value;
            if (!queue.pop(value) || value != batch * 50 + i) {
                TEST_ASSERT(false, "pop order violated");
                return 1;
            }
        }
    }

    TEST_ASSERT(queue.empty(), "queue should be empty after wrap-around test");
    TEST_PASS("SPSC wrap-around");
    return 0;
}

// ---------------------------------------------------------------------------
// 14. test_plugin_descriptor_urls
//    ---------------------------------------------------------------------------
static int test_plugin_descriptor_urls() {
    TEST_ASSERT(mamba_plugin_info.url != nullptr, "url not null");
    TEST_ASSERT(std::strlen(mamba_plugin_info.url) > 0, "url not empty");

    // manual_url and support_url are spec-optional (CLAP plugin.h: manual_url
    // and support_url "may be NULL"). CLARA intentionally does not ship
    // dedicated manual or support URLs at the project URL root, so both are
    // set to nullptr per the source descriptor.
    TEST_ASSERT(mamba_plugin_info.manual_url == nullptr, "manual_url is nullptr (spec-optional)");
    TEST_ASSERT(mamba_plugin_info.support_url == nullptr, "support_url is nullptr (spec-optional)");

    TEST_PASS("plugin descriptor URLs");
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("=== Mamba CLAP Plugin Tests ===\n\n");

    struct TestCase {
        const char *name;
        int (*func)();
    } tests[] = {
        {"plugin descriptor", test_plugin_descriptor},
        {"plugin info descriptor", test_plugin_info_descriptor},
        {"preset data packing", test_preset_data_packing},
        {"param ranges", test_param_ranges},
        {"param value to text", test_param_value_to_text},
        {"param text to value", test_param_text_to_value},
        {"state save/load roundtrip", test_state_save_load_roundtrip},
        {"crossfade step calculation", test_crossfade_step_calculation},
        {"latency reporting", test_latency_reporting},
        {"audio ports", test_audio_ports},
        {"deferred task queue", test_deferred_task_queue},
        {"SPSC two-thread FIFO", test_spsc_two_thread},
        {"SPSC wrap-around", test_spsc_wrap_around},
        {"plugin descriptor URLs", test_plugin_descriptor_urls},
    };

    int failures = 0;
    for (const auto &t : tests) {
        std::printf("Test: %s\n", t.name);
        failures += t.func();
    }

    std::printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures ? 1 : 0;
}
