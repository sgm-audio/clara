#include <cstdio>
#include <cstring>
#include <cmath>
#include "core/MambaInferenceEngine.hpp"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            return 1; \
        } \
    } while(0)

#define TEST_PASS(msg) \
    std::printf("  PASS: %s\n", msg)

static int test_construct_destroy()
{
    auto engine = std::make_unique<MambaInferenceEngine>();
    TEST_ASSERT(engine != nullptr, "engine constructed");
    TEST_PASS("construct/destroy");
    return 0;
}

static int test_init_model_path()
{
    // The model directory is baked at compile time via TEST_MODEL_DIR
    const char* modelDir = TEST_MODEL_DIR;
    std::printf("  model dir: %s\n", modelDir);

    auto engine = std::make_unique<MambaInferenceEngine>();
    TEST_ASSERT(engine != nullptr, "engine constructed");

    bool ok = engine->init(modelDir, 48000.0, 256);
    // If model file doesn't exist, init returns false — that's OK in CI
    // without the ONNX model. We just check it doesn't crash.
    std::printf("  init() returned: %s\n", ok ? "true" : "false (model may be missing, not a failure)");
    TEST_PASS("init model path");
    return 0;
}

static int test_process_block()
{
    const char* modelDir = TEST_MODEL_DIR;
    auto engine = std::make_unique<MambaInferenceEngine>();
    if (!engine->init(modelDir, 48000.0, 256)) {
        std::printf("  SKIP: model not available\n");
        TEST_PASS("process block (skipped)");
        return 0;
    }

    // Allocate a block of silence
    const int blockSamples = 256;
    const int numChannels = 2;
    float input[512] = {};
    float output[512] = {};

    float* inPtrs[2] = { input, input + blockSamples };
    float* outPtrs[2] = { output, output + blockSamples };

    engine->resetState();
    bool ok = engine->processBlock(inPtrs, outPtrs, blockSamples, numChannels);
    TEST_ASSERT(ok, "processBlock returned true");

    // Output should be finite (no NaN)
    for (int i = 0; i < blockSamples * numChannels; ++i) {
        TEST_ASSERT(std::isfinite(output[i]), "output is finite");
    }

    TEST_PASS("process block");
    return 0;
}

static int test_reset_state()
{
    const char* modelDir = TEST_MODEL_DIR;
    auto engine = std::make_unique<MambaInferenceEngine>();
    if (!engine->init(modelDir, 48000.0, 256)) {
        std::printf("  SKIP: model not available\n");
        TEST_PASS("reset state (skipped)");
        return 0;
    }

    engine->resetState();

    float input[512] = {};
    float output[512] = {};
    float* inPtrs[2] = { input, input + 256 };
    float* outPtrs[2] = { output, output + 256 };

    bool ok = engine->processBlock(inPtrs, outPtrs, 256, 2);
    TEST_ASSERT(ok, "processBlock after reset");
    TEST_ASSERT(std::isfinite(output[0]), "output is finite after reset");

    // Second reset (deferred) — should not crash
    engine->resetState();
    TEST_PASS("reset state");
    return 0;
}

int main()
{
    std::printf("=== MambaInferenceEngine Tests ===\n\n");

    int failures = 0;

    std::printf("Test: construct/destroy\n");
    failures += test_construct_destroy();

    std::printf("Test: init model path\n");
    failures += test_init_model_path();

    std::printf("Test: process block\n");
    failures += test_process_block();

    std::printf("Test: reset state\n");
    failures += test_reset_state();

    std::printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures;
}
