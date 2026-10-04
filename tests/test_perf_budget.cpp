// CLARA engine compute budget — regression guard for real-time cost.
//
// Why this test exists: the engine is currently far too slow for real-time use
// (see docs/BLUEPRINT_COMPARISON.md). Without a budget check in CI, a change that
// doubles the per-sample cost is invisible — the numerical tests still pass.
//
// The budget is expressed as a RATIO against a calibration loop measured on the
// same machine in the same process, so it is meaningful on fast and slow CI
// runners alike. The absolute ceiling is a second, coarser guard.
//
// Measured cost per sample of one stream:
//   48 kHz mono  = ns_per_sample * 48000 / 1e9 * 100 % of one core
//
// Budget rationale: the engine performs ~2.03e6 MACs per sample (in_proj 206k +
// out_proj 33k + FiLM/Δ 34k + SSM recurrence ~66k, per layer × 6). The
// calibration loop measures the machine's plain-loop MAC rate, so a ratio of 1.0
// means "the engine costs exactly one calibration unit per sample". Measured
// value with the blocked GEMM kernels is ~0.8-1.0 on the development machine.

#include "core/MambaConfig.h"
#include "core/SsmInferenceCore.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace clara::mamba3;
using Clock = std::chrono::steady_clock;

// Relative ceiling: engine cost vs. the machine's own MAC throughput.
static constexpr double BUDGET_RATIO = 3.0;
// Absolute ceiling: 5 ms per sample is beyond unusable on any hardware.
static constexpr double BUDGET_NS_PER_SAMPLE = 5.0e6;

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : "models/model.bin";

    SsmInferenceCore engine;
    if (!engine.init(model_path)) {
        std::printf("SKIP: cannot load model %s\n", model_path);
        return 77; // CTest SKIP
    }

    // ── Calibration: plain-loop float MACs ──
    // The loop shape mirrors the engine's inner products (two contiguous rows).
    constexpr size_t CAL_MACS = 2000000;
    constexpr size_t VEC = 1024;
    std::vector<float> a(VEC), b(VEC);
    for (size_t i = 0; i < VEC; ++i) {
        a[i] = 0.001f * (float) (i % 13);
        b[i] = 0.002f * (float) (i % 7);
    }

    // Four independent accumulators: a single accumulator would measure FMA
    // *latency* (a serial dependency chain) rather than the machine's MAC
    // throughput, understating the calibration by ~3x on the dev machine and
    // making the ratio budget meaninglessly lenient.
    auto run_calibration = [&]() {
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        size_t done = 0;
        while (done < CAL_MACS) {
            for (size_t i = 0; i + 4 <= VEC; i += 4) {
                acc0 += a[i] * b[i];
                acc1 += a[i + 1] * b[i + 1];
                acc2 += a[i + 2] * b[i + 2];
                acc3 += a[i + 3] * b[i + 3];
            }
            done += VEC;
        }
        return acc0 + acc1 + acc2 + acc3;
    };

    volatile float sink = 0.0f;
    sink = run_calibration(); // warmup
    sink = run_calibration();

    auto t0 = Clock::now();
    sink = run_calibration();
    auto t1 = Clock::now();
    const double cal_sec = std::chrono::duration<double>(t1 - t0).count();
    const double cal_ns = cal_sec * 1e9;

    // ── Engine cost per sample ──
    constexpr size_t BLOCK = 512;
    constexpr int ROUNDS = 4;
    std::vector<float> in(BLOCK), out(BLOCK);
    for (size_t i = 0; i < BLOCK; ++i) {
        in[i] = 0.25f * std::sin(2.0 * M_PI * 440.0 * (double) i / 48000.0);
    }

    for (int w = 0; w < 32; ++w) {
        engine.process_block(in.data(), out.data(), BLOCK);
    }

    std::vector<double> per_round;
    for (int r = 0; r < ROUNDS; ++r) {
        auto s0 = Clock::now();
        engine.process_block(in.data(), out.data(), BLOCK);
        auto s1 = Clock::now();
        per_round.push_back(std::chrono::duration<double>(s1 - s0).count() / (double) BLOCK);
    }
    std::sort(per_round.begin(), per_round.end());
    const double ns_per_sample = per_round[per_round.size() / 2] * 1e9;

    const double mono_core_pct = ns_per_sample * 1e-9 * 48000.0 * 100.0;
    const double stereo_core_pct = mono_core_pct * 2.0;
    const double ratio = ns_per_sample / cal_ns;

    std::printf("\n--- CLARA compute budget ---\n");
    std::printf("  calibration (%zu MACs):       %10.1f ns\n", CAL_MACS, cal_ns);
    std::printf("  engine per sample:           %10.1f ns  (%.3f ms)\n", ns_per_sample, ns_per_sample / 1e6);
    std::printf("  ratio (engine/calibration):   %10.3f     (budget < %.1f)\n", ratio, BUDGET_RATIO);
    std::printf("  cost of 1 stream @48k mono:   %10.1f%% of one core\n", mono_core_pct);
    std::printf("  cost of 1 stream @48k stereo: %10.1f%% of one core\n", stereo_core_pct);
    std::printf("  real-time factor (1 core):    %10.3fx  (1.0x = exactly real time)\n",
                1.0 / (ns_per_sample * 1e-9 * 48000.0));
    std::printf("  NOTE: real-time needs stereo < 100%%, i.e. ~%.0fx faster than today.\n",
                stereo_core_pct / 100.0);

    int failures = 0;
    if (ratio >= BUDGET_RATIO) {
        std::printf("  FAIL: engine cost ratio %.3f exceeds budget %.1f\n", ratio, BUDGET_RATIO);
        ++failures;
    } else {
        std::printf("  OK:   engine cost ratio within budget\n");
    }
    if (ns_per_sample >= BUDGET_NS_PER_SAMPLE) {
        std::printf("  FAIL: %.0f ns/sample exceeds absolute ceiling %.0f\n", ns_per_sample, BUDGET_NS_PER_SAMPLE);
        ++failures;
    } else {
        std::printf("  OK:   per-sample cost below absolute ceiling\n");
    }

    return failures ? 1 : 0;
}
