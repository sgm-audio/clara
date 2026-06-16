// CLARA Mamba-3 Inference Core — verification test
//
// Phase 4: Loads model.bin, runs SsmInferenceCore, checks:
//   1. Init succeeds with valid weight file
//   2. process_block produces finite, non-trivial output
//   3. Conditioning changes output (non-zero cond ≠ zero cond)
//   4. State reset produces deterministic output
//   5. State continuity (consecutive blocks use accumulated state)

#include "core/MambaConfig.h"
#include "core/SsmInferenceCore.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace clara::mamba3;

// ── Helpers ────────────────────────────────────────────────────────

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::printf("  FAIL: %s\n", msg); \
        ++failures; \
    } else { \
        std::printf("  OK:   %s\n", msg); \
    } \
} while(0)

#define CHECK_CLOSE(a, b, tol, msg) do { \
    if (std::fabs((a) - (b)) > (tol)) { \
        std::printf("  FAIL: %s (%.6f vs %.6f)\n", msg, (double)(a), (double)(b)); \
        ++failures; \
    } else { \
        std::printf("  OK:   %s\n", msg); \
    } \
} while(0)

static bool all_finite(const float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) return false;
    }
    return true;
}

static float max_abs(const float* data, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(data[i]);
        if (a > m) m = a;
    }
    return m;
}

static float stddev(const float* data, size_t n) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += data[i];
    mean /= (float)n;
    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = data[i] - mean;
        var += d * d;
    }
    return std::sqrt(var / (float)n);
}

// ── Tests ──────────────────────────────────────────────────────────

static void test_init(const std::string& model_path) {
    std::printf("\n--- test_init ---\n");

    SsmInferenceCore engine;
    CHECK(engine.init(model_path.c_str()), "init succeeds");
    CHECK(engine.is_ready(), "is_ready after init");
}

static void test_process_silence(const std::string& model_path) {
    std::printf("\n--- test_process_silence ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 64;
    float input[N] = {};
    float output[N] = {};

    engine.process_block(input, output, N);

    CHECK(all_finite(output, N), "silence output is finite");

    // With nonzero weights and silence input, output should be near zero
    // (no input energy to excite the SSM)
    float m = max_abs(output, N);
    std::printf("  INFO: silence output max_abs = %.6f\n", (double)m);
    CHECK(m < 10.0f, "silence output not exploding");
}

static void test_process_sine(const std::string& model_path) {
    std::printf("\n--- test_process_sine ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 256;
    float input[N];
    for (size_t i = 0; i < N; ++i) {
        input[i] = std::sin(2.0f * 3.14159f * 440.0f * (float)i / 44100.0f);
    }
    float output[N] = {};

    engine.process_block(input, output, N);

    CHECK(all_finite(output, N), "sine output is finite");

    float stdv = stddev(output, N);
    std::printf("  INFO: sine output stddev = %.6f\n", (double)stdv);
    CHECK(stdv > 1e-8f, "sine output is non-constant");
}

static void test_conditioning(const std::string& model_path) {
    std::printf("\n--- test_conditioning ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 64;
    float input[N];
    for (size_t i = 0; i < N; ++i) {
        input[i] = (float)i / (float)N - 0.5f;  // ramp
    }

    // Run with zero conditioning
    engine.reset_states();
    float out_zero[N] = {};
    engine.process_block(input, out_zero, N);

    // Run with non-zero conditioning
    float cond[COND_DIM];
    for (size_t i = 0; i < COND_DIM; ++i) {
        cond[i] = (float)(i % 7) * 0.1f;
    }
    engine.set_conditioning(cond);
    engine.reset_states();
    float out_cond[N] = {};
    engine.process_block(input, out_cond, N);

    CHECK(all_finite(out_zero, N), "zero-cond output finite");
    CHECK(all_finite(out_cond, N), "non-zero-cond output finite");

    // Output should differ when conditioning changes
    bool differs = false;
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(out_zero[i] - out_cond[i]) > 1e-6f) {
            differs = true;
            break;
        }
    }
    CHECK(differs, "conditioning changes output");
}

static void test_deterministic(const std::string& model_path) {
    std::printf("\n--- test_deterministic ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 64;
    float input[N];
    for (size_t i = 0; i < N; ++i) input[i] = (float)i / (float)N;

    engine.reset_states();
    float out1[N] = {};
    engine.process_block(input, out1, N);

    engine.reset_states();
    float out2[N] = {};
    engine.process_block(input, out2, N);

    bool same = true;
    for (size_t i = 0; i < N; ++i) {
        if (out1[i] != out2[i]) { same = false; break; }
    }
    CHECK(same, "deterministic: two runs after reset produce same output");
}

static void test_state_continuity(const std::string& model_path) {
    std::printf("\n--- test_state_continuity ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 32;
    float input[N];
    for (size_t i = 0; i < N; ++i) input[i] = 0.1f;

    // One continuous block of 64 samples
    engine.reset_states();
    float cont[2 * N] = {};
    float full_input[2 * N];
    for (size_t i = 0; i < 2 * N; ++i) full_input[i] = 0.1f;
    engine.process_block(full_input, cont, 2 * N);

    // Two separate blocks of 32 samples (state carries across calls)
    engine.reset_states();
    float split[N] = {};
    engine.process_block(input, split, N);
    float split2[N] = {};
    engine.process_block(input, split2, N);

    // Second half of continuous should match first half of second split
    // (since both start with same accumulated state after N samples)
    // Actually: continuous[0..31] from fresh state, split[0..31] from fresh state
    // continuous[32..63] from N-sample warmed state, split2[0..31] from N-sample warmed state
    bool match = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(cont[N + i] - split2[i]) > 1e-6f) {
            match = false;
            break;
        }
    }
    CHECK(match, "state continuity: split blocks = continuous block");
}

static void test_long_stability(const std::string& model_path) {
    std::printf("\n--- test_long_stability ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    // Process 1024 samples, check no NaN/Inf
    constexpr size_t N = 1024;
    float input[N];
    for (size_t i = 0; i < N; ++i) {
        input[i] = std::sin(2.0f * 3.14159f * 440.0f * (float)i / 44100.0f) * 0.5f;
    }
    float output[N] = {};

    engine.process_block(input, output, N);

    CHECK(all_finite(output, N), "1024-sample run: output finite");
    float max_out = max_abs(output, N);
    std::printf("  INFO: 1024-sample max output = %.4f\n", (double)max_out);
    CHECK(max_out < 100.0f, "1024-sample output not exploding");
}

// ── Main ───────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string model_path = "models/model.bin";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::printf("=== Mamba-3 Inference Core Verification ===\n");
    std::printf("Model: %s\n", model_path.c_str());

    // Check if model file exists
    FILE* f = std::fopen(model_path.c_str(), "rb");
    if (!f) {
        std::printf("\nSKIP: model file not found: %s\n", model_path.c_str());
        std::printf("Generate it with:\n");
        std::printf("  python3 scripts/export_mamba_weights.py --output %s\n",
                    model_path.c_str());
        return 0;
    }
    std::fclose(f);

    test_init(model_path);
    test_process_silence(model_path);
    test_process_sine(model_path);
    test_conditioning(model_path);
    test_deterministic(model_path);
    test_state_continuity(model_path);
    test_long_stability(model_path);

    std::printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures ? 1 : 0;
}
