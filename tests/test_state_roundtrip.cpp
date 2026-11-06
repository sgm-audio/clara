// State round-trip, text parsing and bypass tests for the CLAP plugin.
//
// These pin the failures clap-validator reports as state-reproducibility-*
// (values set through process() events did not survive save/reload) and
// param-conversions (text_to_value accepted NaN/Inf and read a bare number as
// a fraction) plus the bypass early-out.
//
// The streams here can move fewer bytes per call than asked, which the CLAP
// stream API allows and which is what state-reproducibility-buffered exercises.

#include "clap/MambaClapPlugin.hpp"

#include <clap/clap.h>
#include <clap/ext/params.h>
#include <clap/ext/state.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);                                     \
            ++g_failures;                                                                                              \
        } else {                                                                                                       \
            std::printf("  ok: %s\n", msg);                                                                            \
        }                                                                                                              \
    } while (0)

// ── Streams that move at most `chunk` bytes per call ────────────────────────

struct BufferOStream {
    clap_ostream base{};
    std::vector<char> data;
    size_t chunk;
};

static int64_t o_stream_write(const clap_ostream_t *stream, const void *buffer, uint64_t size) {
    auto *self = static_cast<BufferOStream *>(stream->ctx);
    const size_t n = std::min<size_t>(static_cast<size_t>(size), self->chunk);
    const char *p = static_cast<const char *>(buffer);
    self->data.insert(self->data.end(), p, p + n);
    return static_cast<int64_t>(n);
}

struct BufferIStream {
    clap_istream_t base{};
    const std::vector<char> *data;
    size_t pos{0};
    size_t chunk;
};

static int64_t i_stream_read(const clap_istream_t *stream, void *buffer, uint64_t size) {
    auto *self = static_cast<BufferIStream *>(stream->ctx);
    const size_t avail = self->data->size() - self->pos;
    const size_t n = std::min<size_t>({static_cast<size_t>(size), avail, self->chunk});
    if (n == 0) return 0;
    std::memcpy(buffer, self->data->data() + self->pos, n);
    self->pos += n;
    return static_cast<int64_t>(n);
}

// ── Instance helpers ────────────────────────────────────────────────────────

struct Instance {
    MambaPluginState state;
    clap_plugin_t plugin = mamba_plugin;
    const clap_plugin_params_t *params = nullptr;
    const clap_plugin_state_t *stateExt = nullptr;

    Instance() {
        plugin.plugin_data = &state;
        params = static_cast<const clap_plugin_params_t *>(mamba_get_extension(&plugin, CLAP_EXT_PARAMS));
        stateExt = static_cast<const clap_plugin_state_t *>(mamba_get_extension(&plugin, CLAP_EXT_STATE));
    }

    double get(clap_id id) const {
        double v = -1.0;
        return (params && params->get_value(&plugin, id, &v)) ? v : -1.0;
    }
};

// One param event, delivered at block start like a host would.
struct EventList {
    clap_event_param_value_t ev{};
};
static uint32_t event_size(const clap_input_events_t *) { return 1; }
static const clap_event_header_t *event_get(const clap_input_events_t *list, uint32_t) {
    return &static_cast<const EventList *>(list->ctx)->ev.header;
}

static void send_param(Instance &inst, clap_id id, double value) {
    EventList list{};
    list.ev.header.size = sizeof(clap_event_param_value_t);
    list.ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
    list.ev.header.type = CLAP_EVENT_PARAM_VALUE;
    list.ev.param_id = id;
    list.ev.value = value;
    clap_input_events_t in{&list, event_size, event_get};
    clap_process_t proc{};
    proc.in_events = &in;
    inst.plugin.process(&inst.plugin, &proc);
}

static std::vector<char> save_state(Instance &inst, size_t chunk) {
    BufferOStream out;
    out.chunk = chunk;
    out.base.ctx = &out;
    out.base.write = o_stream_write;
    inst.stateExt->save(&inst.plugin, &out.base);
    return out.data;
}

static bool load_state(Instance &inst, const std::vector<char> &bytes, size_t chunk) {
    BufferIStream in;
    in.data = &bytes;
    in.chunk = chunk;
    in.base.ctx = &in;
    in.base.read = i_stream_read;
    return inst.stateExt->load(&inst.plugin, &in.base);
}

// ── Mock host: records the param rescan request ─────────────────────────────

static int g_rescanCalls = 0;
static uint32_t g_lastRescanFlags = 0;

static void host_params_rescan(const clap_host_t *, uint32_t flags) {
    ++g_rescanCalls;
    g_lastRescanFlags = flags;
}

static const clap_host_params_t g_hostParams = {host_params_rescan, nullptr, nullptr};

static const void *host_get_extension(const clap_host_t *, const char *id) {
    if (std::strcmp(id, CLAP_EXT_PARAMS) == 0) return &g_hostParams;
    return nullptr;
}
static void host_request_restart(const clap_host_t *) {}
static void host_request_process(const clap_host_t *) {}
static void host_request_callback(const clap_host_t *) {}

// ── Tests ───────────────────────────────────────────────────────────────────

// Values set through process() events must come back after save/load, and a
// second save must produce the same bytes.
static void test_roundtrip(size_t chunk, const char *label) {
    std::printf("\n--- state round-trip, %s stream ---\n", label);

    const float bypass = 0.25f, wetDry = 0.1821f, intensity = 0.5452f;

    Instance a;
    send_param(a, PARAM_BYPASS, bypass);
    send_param(a, PARAM_WET_DRY, wetDry);
    send_param(a, PARAM_INTENSITY, intensity);

    CHECK(std::fabs(a.get(PARAM_WET_DRY) - wetDry) < 1e-6, "process() event reaches get_value (wet/dry)");
    CHECK(std::fabs(a.get(PARAM_INTENSITY) - intensity) < 1e-6, "process() event reaches get_value (intensity)");

    const std::vector<char> saved = save_state(a, chunk);
    CHECK(saved.size() == 16, "state blob is 16 bytes (version + 3 floats)");

    Instance b;
    clap_host_t host{};
    host.get_extension = host_get_extension;
    host.request_restart = host_request_restart;
    host.request_process = host_request_process;
    host.request_callback = host_request_callback;
    b.state.host = &host;
    g_rescanCalls = 0;
    g_lastRescanFlags = 0;

    CHECK(load_state(b, saved, chunk), "load() succeeds");
    CHECK(std::fabs(b.get(PARAM_BYPASS) - bypass) < 1e-6, "bypass survives save/load");
    CHECK(std::fabs(b.get(PARAM_WET_DRY) - wetDry) < 1e-6, "wet/dry survives save/load");
    CHECK(std::fabs(b.get(PARAM_INTENSITY) - intensity) < 1e-6, "intensity survives save/load");
    // CLAP requires this or the host keeps showing the pre-load values, which
    // is exactly what clap-validator's state-reproducibility-* reported.
    CHECK(g_rescanCalls == 1, "load requests one param rescan");
    CHECK((g_lastRescanFlags & CLAP_PARAM_RESCAN_VALUES) != 0, "rescan asks for values");

    const std::vector<char> resaved = save_state(b, chunk);
    CHECK(resaved == saved, "re-saved state is byte-identical");
}

// text_to_value: no NaN/Inf, and a bare number means percent, matching
// value_to_text ("50.0%").
static void test_text_to_value() {
    std::printf("\n--- text_to_value ---\n");
    Instance inst;
    const auto *p = inst.params;
    double v = 0.0;

    CHECK(!p->text_to_value(&inst.plugin, PARAM_WET_DRY, "nan", &v), "\"nan\" rejected");
    CHECK(!p->text_to_value(&inst.plugin, PARAM_WET_DRY, "inf", &v), "\"inf\" rejected");
    CHECK(!p->text_to_value(&inst.plugin, PARAM_WET_DRY, "-inf", &v), "\"-inf\" rejected");
    CHECK(!p->text_to_value(&inst.plugin, PARAM_WET_DRY, "1e999", &v), "\"1e999\" rejected (overflow to inf)");
    CHECK(!p->text_to_value(&inst.plugin, PARAM_WET_DRY, "", &v), "empty string rejected");
    CHECK(!p->text_to_value(&inst.plugin, PARAM_WET_DRY, "abc", &v), "\"abc\" rejected");

    CHECK(p->text_to_value(&inst.plugin, PARAM_WET_DRY, "50%", &v) && std::fabs(v - 0.5) < 1e-9, "\"50%\" -> 0.5");
    CHECK(p->text_to_value(&inst.plugin, PARAM_WET_DRY, "50", &v) && std::fabs(v - 0.5) < 1e-9,
          "\"50\" -> 0.5 (percent, same as value_to_text)");
    CHECK(p->text_to_value(&inst.plugin, PARAM_WET_DRY, "50.0%", &v) && std::fabs(v - 0.5) < 1e-9, "\"50.0%\" -> 0.5");
    CHECK(p->text_to_value(&inst.plugin, PARAM_WET_DRY, "150%", &v) && std::fabs(v - 1.0) < 1e-9, "\"150%\" clamps to 1");
    CHECK(p->text_to_value(&inst.plugin, PARAM_WET_DRY, "0", &v) && std::fabs(v) < 1e-9, "\"0\" -> 0");

    // Round-trip against value_to_text for the values a host might show.
    for (double val : {0.0, 0.1821, 0.5, 1.0}) {
        char display[64] = {};
        CHECK(p->value_to_text(&inst.plugin, PARAM_WET_DRY, val, display, sizeof(display)), "value_to_text");
        double back = -1.0;
        CHECK(p->text_to_value(&inst.plugin, PARAM_WET_DRY, display, &back), "text_to_value of its own text");
        CHECK(std::fabs(back - val) < 5e-3, "text round-trip within display precision");
    }
}

// clap_tail_get: 0 until activate() has loaded the weights, then the number of
// samples until the slowest zero-input pole has decayed 60 dB, clamped to 10 s.
// A host that asks before activation gets 0; the engine does not exist yet.
static void test_tail() {
    std::printf("\n--- tail ---\n");
    Instance inst;
    const auto *tailExt =
        static_cast<const clap_plugin_tail_t *>(mamba_get_extension(&inst.plugin, CLAP_EXT_TAIL));
    CHECK(tailExt != nullptr, "tail extension is exposed");
    if (!tailExt) return;
    CHECK(tailExt->get(&inst.plugin) == 0, "tail is 0 before the weights are loaded");

    inst.state.modelPath = std::string(TEST_MODEL_DIR) + "/model.bin";
    CHECK(inst.plugin.init(&inst.plugin) && inst.plugin.activate(&inst.plugin, 48000.0, 32, 512),
          "init() + activate() with the test model");
    if (g_failures) return;
    inst.state.sampleRate = 48000.0f;
    const uint32_t tail = tailExt->get(&inst.plugin);
    inst.state.sampleRate = 44100.0f;
    const uint32_t tail44 = tailExt->get(&inst.plugin);
    inst.state.sampleRate = 48000.0f;
    std::printf("  INFO: tail at 48 kHz = %u samples (%.2f ms), at 44.1 kHz = %u\n", tail,
                static_cast<double>(tail) * 1000.0 / 48000.0, tail44);
    CHECK(tail > 0 && tail <= 480000, "tail is in (0, 10 s] at 48 kHz");
    CHECK(tail44 > 0 && tail44 <= 441000, "tail is in (0, 10 s] at 44.1 kHz");
    // The recurrence runs per sample and dt carries no 1/sr, so the decay
    // length in samples does not depend on the sample rate (until the cap).
    CHECK(tail44 == tail, "tail in samples is rate-independent below the 10 s cap");
}

// Fully bypassed: output must be the input, bit for bit, and the engine must
// not run. The engine costs ~0.28 ms per sample on this machine (see
// test_perf_budget), so a 512-sample block that takes under 5 ms proves the
// forward pass was skipped.
static void test_bypass_early_out() {
    std::printf("\n--- bypass early-out ---\n");

    Instance inst;
    inst.state.modelPath = std::string(TEST_MODEL_DIR) + "/model.bin";
    // Not a skip: CMake only builds this test when a model exists
    // (-DCLARA_ALLOW_NO_MODEL=ON marks it DISABLED instead), so failing here
    // means the model or the loader broke.
    CHECK(inst.plugin.init(&inst.plugin), "init() with the test model");
    if (g_failures) return;
    CHECK(inst.plugin.activate(&inst.plugin, 48000.0, 32, 512) && inst.plugin.start_processing(&inst.plugin),
          "activate() + start_processing()");

    constexpr uint32_t N = 512;
    std::vector<float> inL(N), inR(N), outL(N, 0.0f), outR(N, 0.0f);
    for (uint32_t i = 0; i < N; ++i) {
        inL[i] = 0.5f * std::sin(2.0f * 3.14159265358979f * 440.0f * static_cast<float>(i) / 48000.0f);
        inR[i] = inL[i];
    }

    float *inPtrs[2] = {inL.data(), inR.data()};
    float *outPtrs[2] = {outL.data(), outR.data()};
    clap_audio_buffer_t inBuf{};
    inBuf.data32 = inPtrs;
    inBuf.channel_count = 2;
    clap_audio_buffer_t outBuf{};
    outBuf.data32 = outPtrs;
    outBuf.channel_count = 2;
    clap_process_t proc{};
    proc.frames_count = N;
    proc.audio_inputs = &inBuf;
    proc.audio_inputs_count = 1;
    proc.audio_outputs = &outBuf;
    proc.audio_outputs_count = 1;

    // First block with bypass off, to prove the engine path is the slow one.
    const auto t0 = std::chrono::steady_clock::now();
    inst.plugin.process(&inst.plugin, &proc);
    const auto t1 = std::chrono::steady_clock::now();
    const double wetMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Now bypass, already settled, and time the block.
    inst.state.bypass.store(1.0f, std::memory_order_release);
    inst.state.currentBypassValue = 1.0f;
    std::fill(outL.begin(), outL.end(), 0.0f);
    std::fill(outR.begin(), outR.end(), 0.0f);

    const auto t2 = std::chrono::steady_clock::now();
    inst.plugin.process(&inst.plugin, &proc);
    const auto t3 = std::chrono::steady_clock::now();
    const double dryMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

    CHECK(std::memcmp(outL.data(), inL.data(), N * sizeof(float)) == 0, "bypassed block is bit-identical to input");
    CHECK(std::memcmp(outR.data(), inR.data(), N * sizeof(float)) == 0, "bypassed right channel too");
    std::printf("  INFO: wet block %.2f ms, bypassed block %.3f ms\n", wetMs, dryMs);
    CHECK(dryMs * 3.0 < wetMs, "bypassed block is at least 3x cheaper than the wet block (engine skipped)");

    inst.plugin.deactivate(&inst.plugin);
}

// State load on the main thread while process() runs on the audio thread.
// This is the shape the production race (C9/C23) had; without ThreadSanitizer
// it only proves "no crash, no NaN", so the TSan run belongs on Linux CI.
static void test_state_load_stress() {
    std::printf("\n--- state load vs process() stress ---\n");

    Instance inst;
    inst.state.modelPath = std::string(TEST_MODEL_DIR) + "/model.bin";
    CHECK(inst.plugin.init(&inst.plugin), "init() with the test model");
    if (g_failures) return;
    CHECK(inst.plugin.activate(&inst.plugin, 48000.0, 32, 64) && inst.plugin.start_processing(&inst.plugin),
          "activate() + start_processing()");

    constexpr uint32_t N = 64;
    std::vector<float> inL(N, 0.25f), inR(N, 0.25f), outL(N), outR(N);
    float *inPtrs[2] = {inL.data(), inR.data()};
    float *outPtrs[2] = {outL.data(), outR.data()};
    clap_audio_buffer_t inBuf{};
    inBuf.data32 = inPtrs;
    inBuf.channel_count = 2;
    clap_audio_buffer_t outBuf{};
    outBuf.data32 = outPtrs;
    outBuf.channel_count = 2;
    clap_process_t proc{};
    proc.frames_count = N;
    proc.audio_inputs = &inBuf;
    proc.audio_inputs_count = 1;
    proc.audio_outputs = &outBuf;
    proc.audio_outputs_count = 1;

    const std::vector<char> blob = save_state(inst, 5);
    std::atomic<bool> stop{false};
    std::atomic<bool> sawNaN{false};

    std::thread audio([&] {
        while (!stop.load(std::memory_order_acquire)) {
            inst.plugin.process(&inst.plugin, &proc);
            for (uint32_t i = 0; i < N; ++i) {
                if (!mambaIsFinite(outL[i]) || !mambaIsFinite(outR[i])) sawNaN.store(true, std::memory_order_relaxed);
            }
        }
    });

    for (int i = 0; i < 200; ++i) load_state(inst, blob, 3);
    stop.store(true, std::memory_order_release);
    audio.join();

    CHECK(!sawNaN.load(), "200 state loads during process(): no NaN/Inf in the output");
    std::printf("  INFO: TSan not available in this MinGW build; run the same test under -fsanitize=thread on Linux\n");
    inst.plugin.deactivate(&inst.plugin);
}

int main() {
    std::printf("=== CLARA state / text / bypass tests ===\n");

    test_roundtrip(std::numeric_limits<size_t>::max(), "single-call");
    test_roundtrip(1, "1-byte");
    test_roundtrip(7, "7-byte");
    test_text_to_value();
    test_tail();
    test_bypass_early_out();
    test_state_load_stress();

    std::printf("\n=== %s (%d failure%s) ===\n", g_failures ? "FAILED" : "ALL TESTS PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
