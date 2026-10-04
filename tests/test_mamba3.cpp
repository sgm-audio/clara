// CLARA Mamba-3 Inference Core — verification test
//
// Phase 4: Loads model.bin, runs SsmInferenceCore, checks:
//   1. Init succeeds with valid weight file
//   2. process_block produces finite, non-trivial output
//   3. Conditioning changes output (non-zero cond ≠ zero cond)
//   4. State reset produces deterministic output
//   5. State continuity (consecutive blocks use accumulated state)

#include "core/MambaConfig.h"
#include "core/Platform.hpp"
#include "core/SsmInferenceCore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace clara::mamba3;

// ── Helpers ────────────────────────────────────────────────────────

static int failures = 0;

#define CHECK(cond, msg)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::printf("  FAIL: %s\n", msg);                                                                          \
            ++failures;                                                                                                \
        } else {                                                                                                       \
            std::printf("  OK:   %s\n", msg);                                                                          \
        }                                                                                                              \
    } while (0)

#define CHECK_CLOSE(a, b, tol, msg)                                                                                    \
    do {                                                                                                               \
        if (std::fabs((a) - (b)) > (tol)) {                                                                            \
            std::printf("  FAIL: %s (%.6f vs %.6f)\n", msg, (double) (a), (double) (b));                               \
            ++failures;                                                                                                \
        } else {                                                                                                       \
            std::printf("  OK:   %s\n", msg);                                                                          \
        }                                                                                                              \
    } while (0)

static bool all_finite(const float *data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        // mambaIsFinite, not std::isfinite: Release builds use -ffast-math,
        // which folds std::isfinite() away and makes these checks vacuous.
        if (!mambaIsFinite(data[i])) return false;
    }
    return true;
}

static float max_abs(const float *data, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float a = std::fabs(data[i]);
        if (a > m) m = a;
    }
    return m;
}

static float stddev(const float *data, size_t n) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i)
        mean += data[i];
    mean /= (float) n;
    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = data[i] - mean;
        var += d * d;
    }
    return std::sqrt(var / (float) n);
}

// ── Tests ──────────────────────────────────────────────────────────

static void test_init(const std::string &model_path) {
    std::printf("\n--- test_init ---\n");

    SsmInferenceCore engine;
    CHECK(engine.init(model_path.c_str()), "init succeeds");
    CHECK(engine.is_ready(), "is_ready after init");
}

static void test_process_silence(const std::string &model_path) {
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
    std::printf("  INFO: silence output max_abs = %.6f\n", (double) m);
    CHECK(m < 10.0f, "silence output not exploding");
}

static void test_process_sine(const std::string &model_path) {
    std::printf("\n--- test_process_sine ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 256;
    float input[N];
    for (size_t i = 0; i < N; ++i) {
        input[i] = std::sin(2.0f * 3.14159f * 440.0f * (float) i / 44100.0f);
    }
    float output[N] = {};

    engine.process_block(input, output, N);

    CHECK(all_finite(output, N), "sine output is finite");

    float stdv = stddev(output, N);
    std::printf("  INFO: sine output stddev = %.6f\n", (double) stdv);
    CHECK(stdv > 1e-8f, "sine output is non-constant");
}

static void test_conditioning(const std::string &model_path) {
    std::printf("\n--- test_conditioning ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 64;
    float input[N];
    for (size_t i = 0; i < N; ++i) {
        input[i] = (float) i / (float) N - 0.5f; // ramp
    }

    // Run with zero conditioning
    engine.reset_states();
    float out_zero[N] = {};
    engine.process_block(input, out_zero, N);

    // Run with non-zero conditioning
    float cond[COND_DIM];
    for (size_t i = 0; i < COND_DIM; ++i) {
        cond[i] = (float) (i % 7) * 0.1f;
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

static void test_deterministic(const std::string &model_path) {
    std::printf("\n--- test_deterministic ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 64;
    float input[N];
    for (size_t i = 0; i < N; ++i)
        input[i] = (float) i / (float) N;

    engine.reset_states();
    float out1[N] = {};
    engine.process_block(input, out1, N);

    engine.reset_states();
    float out2[N] = {};
    engine.process_block(input, out2, N);

    bool same = true;
    for (size_t i = 0; i < N; ++i) {
        if (out1[i] != out2[i]) {
            same = false;
            break;
        }
    }
    CHECK(same, "deterministic: two runs after reset produce same output");
}

static void test_state_continuity(const std::string &model_path) {
    std::printf("\n--- test_state_continuity ---\n");

    SsmInferenceCore engine;
    if (!engine.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    constexpr size_t N = 32;
    float input[N];
    for (size_t i = 0; i < N; ++i)
        input[i] = 0.1f;

    // One continuous block of 64 samples
    engine.reset_states();
    float cont[2 * N] = {};
    float full_input[2 * N];
    for (size_t i = 0; i < 2 * N; ++i)
        full_input[i] = 0.1f;
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

static void test_long_stability(const std::string &model_path) {
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
        input[i] = std::sin(2.0f * 3.14159f * 440.0f * (float) i / 44100.0f) * 0.5f;
    }
    float output[N] = {};

    engine.process_block(input, output, N);

    CHECK(all_finite(output, N), "1024-sample run: output finite");
    float max_out = max_abs(output, N);
    std::printf("  INFO: 1024-sample max output = %.4f\n", (double) max_out);
    CHECK(max_out < 100.0f, "1024-sample output not exploding");
}

// ── Bit-Exact Python↔C++ Comparison ─────────────────────────────

// Length-prefixed float32 fixture: uint32 N followed by N floats.
// Returns false (and prints why) if the file is missing, truncated, or its
// declared length disagrees with its actual size. Never sizes a buffer from the
// header alone — an unresolved Git LFS pointer or a corrupt file used to make
// this test allocate ~7.7 GB and abort with std::bad_alloc.
static bool read_fixture(const std::string &path, std::vector<float> &out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;

    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    long file_size = std::ftell(f);
    if (file_size < 4 || std::fseek(f, 0, SEEK_SET) != 0) {
        std::printf("  FAIL: %s is too small to be a fixture\n", path.c_str());
        std::fclose(f);
        return false;
    }

    uint32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1) {
        std::fclose(f);
        return false;
    }

    const long expected_size = 4 + static_cast<long>(n) * static_cast<long>(sizeof(float));
    if (expected_size != file_size) {
        std::printf("  FAIL: %s declares %u samples (%ld bytes) but is %ld bytes — "
                    "unresolved Git LFS pointer or truncated file?\n",
                    path.c_str(), n, expected_size, file_size);
        std::fclose(f);
        return false;
    }

    out.resize(n);
    size_t got = (n == 0) ? 0 : std::fread(out.data(), sizeof(float), n, f);
    std::fclose(f);
    if (got != n) {
        std::printf("  FAIL: %s short read\n", path.c_str());
        return false;
    }
    return true;
}

// True if the file starts with the Git LFS pointer header ("version https://...").
static bool is_lfs_pointer(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    char buf[8] = {};
    size_t got = std::fread(buf, 1, sizeof(buf), f);
    std::fclose(f);
    return got == sizeof(buf) && std::memcmp(buf, "version ", sizeof(buf)) == 0;
}

// ── Test: Python reference equivalence (R=4 time-domain) ──────────
//
// Compares the shipping engine against python/reference_forward.py, which
// implements the same 8-step pipeline. Fixtures are generated by:
//   python3 python/tools/dump_engine_reference.py --model <model.bin> --output-dir models/
//
// Earlier this section was called "bit_exact" and compared the R=4 time-domain
// engine against the R=8 spectral PerTrackProcessor exporter — different
// architectures, so it could never pass. See docs/BLUEPRINT_COMPARISON.md.
static void test_python_reference(const std::string &model_path) {
    std::printf("\n--- test_python_reference (python/reference_forward.py) ---\n");

    // Reference files live next to the model
    std::string input_path = model_path;
    size_t pos = input_path.rfind('/');
    std::string dir = (pos != std::string::npos) ? input_path.substr(0, pos + 1) : "";
    std::string ref_input = dir + "input_ref.bin";
    std::string ref_output = dir + "output_ref.bin";

    if (is_lfs_pointer(model_path)) {
        std::printf("  FAIL: %s is an unresolved Git LFS pointer, not a model\n", model_path.c_str());
        ++failures;
        return;
    }
    if (std::fopen(ref_input.c_str(), "rb") == nullptr || std::fopen(ref_output.c_str(), "rb") == nullptr) {
        std::printf("  SKIP: reference fixtures not found in %s — generate with:\n", dir.c_str());
        std::printf("        python3 python/tools/dump_engine_reference.py --model %s\n", model_path.c_str());
        return;
    }

    // Load the model under test (not a separate reference model)
    SsmInferenceCore engine;
    CHECK(engine.init(model_path.c_str()), "load model for reference comparison");

    std::vector<float> input, ref_out;
    if (!read_fixture(ref_input, input) || !read_fixture(ref_output, ref_out)) {
        ++failures;
        return;
    }
    const uint32_t N = static_cast<uint32_t>(input.size());
    std::printf("  INFO: reference input: %u samples\n", N);

    // Set conditioning (same as the fixture generator: cond[0]=0.5, cond[40]=0.3)
    float cond[COND_DIM] = {};
    cond[0] = 0.5f;
    cond[40] = 0.3f;
    engine.set_conditioning(cond);

    // Run the C++ forward pass
    std::vector<float> cpp_output(N);
    engine.process_block(input.data(), cpp_output.data(), N);

    CHECK(all_finite(cpp_output.data(), N), "C++ output finite");
    CHECK(ref_out.size() == N, "reference output matches sample count");
    if (ref_out.size() != N) return;

    // Per-sample comparison. The tolerance is loose enough for float32 vs
    // float32 (different summation orders) but tight enough to catch any real
    // divergence in the 8-step pipeline. Measured max diff with the shipped
    // fixture is ~1.5e-3 (the number this test prints); the previous threshold
    // here was 0.05, which would have accepted an entirely wrong model.
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = std::fabs(cpp_output[i] - ref_out[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = (int) i;
        }
    }

    // Relative error against the reference peak, for an interpretable number
    float ref_peak = 0.0f;
    for (size_t i = 0; i < N; ++i)
        ref_peak = std::max(ref_peak, std::fabs(ref_out[i]));
    const float rel_err = (ref_peak > 0.0f) ? max_diff / ref_peak : max_diff;

    std::printf("  INFO: max sample diff = %.3e at index %d (%.3e relative to peak %.6f)\n", (double) max_diff,
                max_diff_idx, (double) rel_err, (double) ref_peak);
    std::printf("  INFO: C++ std=%.6f Python std=%.6f\n", (double) stddev(cpp_output.data(), N),
                (double) stddev(ref_out.data(), N));

    // Thresholds: measured agreement with the shipped fixture is 1.44e-3 max
    // abs / 6.75e-4 relative over 512 samples. The two implementations are
    // op-for-op equivalent, but both are float32 and the recurrence amplifies
    // rounding: a float64 oracle of the same pipeline differs from BOTH by
    // ~2% of peak at 512 samples (see test_numeric_sensitivity). A tighter
    // bound here would be measuring float32 rounding, not correctness.
    CHECK(max_diff < 5e-3f, "Python↔C++ max abs diff < 5e-3");
    CHECK(rel_err < 2e-3f, "Python↔C++ relative error < 2e-3");
    CHECK(stddev(cpp_output.data(), N) > 1e-4f, "C++ output has non-trivial variance");
}

// ── Test: numerical sensitivity / drift guard ─────────────────────
//
// The R=4 recurrence amplifies float32 rounding. Measured 2026-10-04 with the
// shipped fixture: perturbing ONE input sample by 1 ULP diverges the output by
// ~0.05% of peak within 85 ms, and a float64 version of the same pipeline
// differs from float32 by ~2% of peak within 12 ms and ~7% within 93 ms.
//
// This section records those numbers in CTest output and fails only on genuine
// breakage (non-finite output, or divergence beyond a sanity bound), so the
// drift is visible without making CI flaky. Tracked as stability work in
// docs/TODO.md — the fix is either a smaller/less saturating state or double
// precision accumulation, not a test tweak.
static void test_numeric_sensitivity(const std::string &model_path) {
    std::printf("\n--- test_numeric_sensitivity (1-ULP drift guard) ---\n");

    SsmInferenceCore engine_a, engine_b;
    if (!engine_a.init(model_path.c_str()) || !engine_b.init(model_path.c_str())) {
        std::printf("  SKIP: init failed\n");
        return;
    }

    const size_t N = 4096;
    std::vector<float> in(N), in_perturbed(N), out_a(N), out_b(N);
    for (size_t i = 0; i < N; ++i) {
        in[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * (double) i / 48000.0);
    }
    in_perturbed = in;
    in_perturbed[64] = std::nextafterf(in_perturbed[64], 1e9f); // 1 ULP

    float cond[COND_DIM] = {};
    cond[0] = 0.5f;
    cond[40] = 0.3f;
    engine_a.set_conditioning(cond);
    engine_b.set_conditioning(cond);

    engine_a.process_block(in.data(), out_a.data(), N);
    engine_b.process_block(in_perturbed.data(), out_b.data(), N);

    CHECK(all_finite(out_a.data(), N), "output finite under 1-ULP input perturbation");

    float peak = 0.0f;
    for (size_t i = 0; i < N; ++i)
        peak = std::max(peak, std::fabs(out_a[i]));

    float worst = 0.0f;
    for (size_t i = 0; i < N; ++i)
        worst = std::max(worst, std::fabs(out_a[i] - out_b[i]));

    std::printf("  INFO: output peak = %.4f\n", (double) peak);
    std::printf("  INFO: 1-ULP input perturbation -> max divergence %.3e (%.3f%% of peak) over %zu samples\n",
                (double) worst, 100.0 * (double) worst / (double) peak, N);
    std::printf("  INFO: (drift is a known property of the float32 recurrence — see docs/TODO.md)\n");

    CHECK(worst < 0.10f * peak, "1-ULP divergence stays below 10% of peak (sanity bound)");
}

// ── Main ───────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    std::string model_path = "models/model.bin";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::printf("=== Mamba-3 Inference Core Verification ===\n");
    std::printf("Model: %s\n", model_path.c_str());

    // Check if model file exists. An unresolved Git LFS pointer is a broken
    // checkout (see .gitattributes), not a skip — report it as a failure so it
    // cannot silently pass.
    FILE *f = std::fopen(model_path.c_str(), "rb");
    if (!f) {
        std::printf("\nSKIP: model file not found: %s\n", model_path.c_str());
        std::printf("Generate one with:\n");
        std::printf("  python3 python/tools/export_mamba_weights.py --output %s --seed 42\n", model_path.c_str());
        // 77 is CTest's SKIP code (SKIP_RETURN_CODE is set for this test in
        // tests/CMakeLists.txt). Returning 0 here reported a missing model as a
        // *passing* test, which hid a broken checkout.
        return 77;
    }
    std::fclose(f);
    if (is_lfs_pointer(model_path)) {
        std::printf("\nFAIL: %s is a Git LFS pointer, not a model.\n", model_path.c_str());
        std::printf("Weights are stored in Git directly since 2026-10-04; re-clone the repo,\n");
        std::printf("or regenerate fixtures: python3 python/tools/export_mamba_weights.py --output %s\n",
                    model_path.c_str());
        return 1;
    }

    test_init(model_path);
    test_process_silence(model_path);
    test_process_sine(model_path);
    test_conditioning(model_path);
    test_deterministic(model_path);
    test_state_continuity(model_path);
    test_long_stability(model_path);
    test_python_reference(model_path);
    test_numeric_sensitivity(model_path);

    std::printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures ? 1 : 0;
}
