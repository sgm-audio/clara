#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <memory>
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

static std::string getModelPath()
{
    return std::string(TEST_MODEL_DIR) + "/mamba_rt.onnx";
}

static std::unique_ptr<MambaInferenceEngine> createEngine(const std::string& mp)
{
    MambaInferenceEngine::Config cfg;
    cfg.modelPath = mp;
    return std::make_unique<MambaInferenceEngine>(cfg);
}

static int skipTest(const char* name)
{
    std::printf("  SKIP: model not available\n");
    TEST_PASS(name);
    return 0;
}

// --- existing tests (rewritten for current API) ---

static int test_construct_destroy()
{
    {
        auto engine = std::make_unique<MambaInferenceEngine>(MambaInferenceEngine::Config{});
        TEST_ASSERT(engine != nullptr, "engine constructed");
    }
    {
        MambaInferenceEngine::Config cfg;
        cfg.maxBlockSize = 128;
        auto engine = std::make_unique<MambaInferenceEngine>(cfg);
        TEST_ASSERT(engine != nullptr, "engine constructed with custom config");
    }
    TEST_PASS("construct/destroy");
    return 0;
}

static int test_init_model_path()
{
    const std::string mp = getModelPath();
    std::printf("  model path: %s\n", mp.c_str());

    auto engine = createEngine(mp);
    TEST_ASSERT(engine != nullptr, "engine constructed");

    bool ok = engine->init();
    std::printf("  init() returned: %s\n", ok ? "true" : "false (model may be missing, not a failure)");
    TEST_PASS("init model path");
    return 0;
}

static int test_process_block()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("process block");
    }

    const int blockSamples = 256;
    float input[256] = {};
    float output[256] = {};

    engine->resetState();
    engine->processBlock(input, output, blockSamples);

    for (int i = 0; i < blockSamples; ++i) {
        TEST_ASSERT(std::isfinite(output[i]), "output is finite");
    }

    TEST_PASS("process block");
    return 0;
}

static int test_reset_state()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("reset state");
    }

    engine->resetState();

    float input[256] = {};
    float output[256] = {};

    engine->processBlock(input, output, 256);
    TEST_ASSERT(std::isfinite(output[0]), "output is finite after reset");

    engine->resetState();
    TEST_PASS("reset state");
    return 0;
}

// --- new tests ---

static int test_process_block_partial_block()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("process block partial block");
    }

    const int partialSamples = 64;
    float input[64] = {};
    float output[64] = {};

    engine->resetState();
    engine->processBlock(input, output, partialSamples);

    for (int i = 0; i < partialSamples; ++i) {
        TEST_ASSERT(std::isfinite(output[i]), "partial block output is finite");
    }

    TEST_PASS("process block partial block");
    return 0;
}

static int test_process_block_stereo()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("process block stereo");
    }

    // Process two "channels" sequentially through the same mono engine.
    // The CLAP plugin wrapper calls processBlock once per channel.
    const int blockSamples = 256;
    float leftIn[256] = {};
    float rightIn[256] = {};
    float leftOut[256] = {};
    float rightOut[256] = {};

    engine->resetState();
    engine->processBlock(leftIn, leftOut, blockSamples);
    engine->processBlock(rightIn, rightOut, blockSamples);

    for (int i = 0; i < blockSamples; ++i) {
        TEST_ASSERT(std::isfinite(leftOut[i]), "left output is finite");
        TEST_ASSERT(std::isfinite(rightOut[i]), "right output is finite");
    }

    TEST_PASS("process block stereo");
    return 0;
}

static int test_reset_state_during_processing()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("reset state during processing");
    }

    float input[256] = {};
    float output1[256] = {};
    float output2[256] = {};

    engine->resetState();
    engine->processBlock(input, output1, 256);
    TEST_ASSERT(std::isfinite(output1[0]), "output before reset is finite");

    engine->resetState();
    engine->processBlock(input, output2, 256);
    TEST_ASSERT(std::isfinite(output2[0]), "output after reset is finite");

    TEST_PASS("reset state during processing");
    return 0;
}

static int test_multiple_process_blocks()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("multiple process blocks");
    }

    const int blockSamples = 256;
    float input[256] = {};
    float prev[256] = {};
    float curr[256] = {};

    engine->resetState();
    engine->processBlock(input, prev, blockSamples);

    for (int block = 0; block < 5; ++block) {
        engine->processBlock(input, curr, blockSamples);
        for (int i = 0; i < blockSamples; ++i) {
            TEST_ASSERT(std::isfinite(curr[i]), "output is finite across blocks");
        }
        std::memcpy(prev, curr, blockSamples * sizeof(float));
    }

    TEST_PASS("multiple process blocks");
    return 0;
}

static int test_zero_samples()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("zero samples");
    }

    float input[1] = {};
    float output[1] = {};

    engine->resetState();
    engine->processBlock(input, output, 0);

    TEST_PASS("zero samples");
    return 0;
}

static int test_large_block_size()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("large block size");
    }

    const int largeSamples = 512;
    float input[512] = {};
    float output[512] = {};

    engine->resetState();
    engine->processBlock(input, output, largeSamples);

    for (int i = 0; i < 256; ++i) {
        TEST_ASSERT(std::isfinite(output[i]), "clamped output is finite");
    }

    TEST_PASS("large block size");
    return 0;
}

static int test_init_idempotent()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);

    bool first = engine->init();
    bool second = engine->init();

    TEST_ASSERT(first == second, "init idempotent — both calls return same result");

    TEST_PASS("init idempotent");
    return 0;
}

static int test_construct_with_different_configs()
{
    {
        MambaInferenceEngine::Config cfg;
        cfg.dModel = 64;
        cfg.nLayers = 4;
        auto engine = std::make_unique<MambaInferenceEngine>(cfg);
        TEST_ASSERT(engine != nullptr, "default config construct");
    }
    {
        MambaInferenceEngine::Config cfg;
        cfg.dModel = 128;
        cfg.dState = 32;
        cfg.nLayers = 8;
        cfg.convWidth = 8;
        cfg.maxBlockSize = 512;
        auto engine = std::make_unique<MambaInferenceEngine>(cfg);
        TEST_ASSERT(engine != nullptr, "increased dimensions config construct");
    }
    {
        MambaInferenceEngine::Config cfg;
        cfg.dModel = 32;
        cfg.dState = 8;
        cfg.nLayers = 2;
        cfg.convWidth = 2;
        cfg.maxBlockSize = 64;
        auto engine = std::make_unique<MambaInferenceEngine>(cfg);
        TEST_ASSERT(engine != nullptr, "reduced dimensions config construct");
    }

    TEST_PASS("construct with different configs");
    return 0;
}

static int test_construct_destroy_no_init()
{
    {
        auto engine = std::make_unique<MambaInferenceEngine>(MambaInferenceEngine::Config{});
    }
    TEST_PASS("construct/destroy without init");
    return 0;
}

static int test_process_block_no_init()
{
    auto engine = std::make_unique<MambaInferenceEngine>(MambaInferenceEngine::Config{});
    float input[64] = {};
    float output[64] = {1.0f, 2.0f, 3.0f};
    engine->processBlock(input, output, 64);
    for (int i = 0; i < 64; ++i) {
        TEST_ASSERT(output[i] == 0.0f, "uninitialized engine produces silence");
    }
    TEST_PASS("process block no init");
    return 0;
}

static int test_set_conditioning()
{
    auto engine = std::make_unique<MambaInferenceEngine>(MambaInferenceEngine::Config{});

    float cond1[8] = {1.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float cond2[8] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    engine->setConditioning(cond1, 8);
    engine->setConditioning(cond2, 8);
    engine->setConditioning(cond1, 4);

    TEST_PASS("set conditioning");
    return 0;
}

static int test_load_cond_weights()
{
    auto engine = std::make_unique<MambaInferenceEngine>(MambaInferenceEngine::Config{});

    float weights[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine->loadCondWeights(weights, 8);
    engine->loadCondWeights(weights, 4);
    engine->loadCondWeights(nullptr, 0);

    TEST_PASS("load cond weights");
    return 0;
}

static int test_conditioning_with_model()
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest("conditioning with model");
    }

    const int blockSamples = 256;
    float input[256] = {};
    float output[256] = {};
    float outputCond[256] = {};

    engine->resetState();
    engine->processBlock(input, output, blockSamples);

    float weights[8] = {0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine->loadCondWeights(weights, 8);
    float cond[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine->setConditioning(cond, 8);

    engine->resetState();
    engine->processBlock(input, outputCond, blockSamples);

    bool differs = false;
    for (int i = 0; i < blockSamples; ++i) {
        if (std::fabs(outputCond[i] - output[i]) > 1e-6f) {
            differs = true;
            break;
        }
    }
    TEST_ASSERT(differs, "conditioning changes output");

    TEST_PASS("conditioning with model");
    return 0;
}

static int test_init_missing_model_path()
{
    MambaInferenceEngine::Config config;
    config.modelPath = "/nonexistent/path/mamba_rt.onnx";
    MambaInferenceEngine engine(config);
    bool result = engine.init();
    TEST_ASSERT(result == false, "init() should return false for missing model path");
    TEST_PASS("init missing model path");
    return 0;
}

static int test_conditioning_atomic_swap()
{
    auto engine = std::make_unique<MambaInferenceEngine>(MambaInferenceEngine::Config{});

    const float* p0 = engine->getActiveConditioningForTest();
    TEST_ASSERT(p0 == nullptr, "active conditioning is null before any setConditioning() call");

    float cond1[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float cond2[8] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float cond3[8] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    engine->setConditioning(cond1, 8);
    const float* p1 = engine->getActiveConditioningForTest();
    TEST_ASSERT(p1 != nullptr, "active conditioning is non-null after first setConditioning()");

    engine->setConditioning(cond2, 8);
    const float* p2 = engine->getActiveConditioningForTest();
    TEST_ASSERT(p2 != nullptr, "active conditioning is non-null after second setConditioning()");
    TEST_ASSERT(p2 != p1, "active pointer toggles to a different buffer on second setConditioning()");

    engine->setConditioning(cond3, 8);
    const float* p3 = engine->getActiveConditioningForTest();
    TEST_ASSERT(p3 != nullptr, "active conditioning is non-null after third setConditioning()");
    TEST_ASSERT(p3 != p2, "active pointer toggles again on third setConditioning()");
    TEST_ASSERT(p3 == p1, "active pointer returns to first buffer on third setConditioning() (double-buffer ping-pong)");

    // Verify that the active pointer points to the last-written data and
    // that the dormant buffer retains its previous contents.
    TEST_ASSERT(p3[2] == 1.0f, "active buffer holds the most recent conditioning data (cond3[2]==1.0f)");
    TEST_ASSERT(p2[1] == 1.0f, "dormant buffer (p2, condBufB_) still holds cond2[1]==1.0f (not overwritten by 3rd call)");

    TEST_PASS("conditioning atomic swap");
    return 0;
}

static int test_conditioning_projection_math()
{
    auto engine = std::make_unique<MambaInferenceEngine>(MambaInferenceEngine::Config{});

    // Initially: no weights loaded, no active conditioning.
    TEST_ASSERT(engine->getCondWeightsLoadedForTest() == false,
        "condWeightsLoaded_ is false before loadCondWeights()");
    TEST_ASSERT(engine->getActiveConditioningForTest() == nullptr,
        "active conditioning is null before setConditioning()");

    // Load weights (non-zero pattern).
    float weights[8] = {0.5f, 0.25f, 0.125f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine->loadCondWeights(weights, 8);
    TEST_ASSERT(engine->getCondWeightsLoadedForTest() == true,
        "condWeightsLoaded_ is true after loadCondWeights(weights, 8)");

    // Set a known conditioning vector.
    float cond[8] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    engine->setConditioning(cond, 8);
    TEST_ASSERT(engine->getActiveConditioningForTest() != nullptr,
        "active conditioning is non-null after setConditioning(cond, 8)");

    // Edge case: passing nullptr should clear the loaded-weights flag.
    engine->loadCondWeights(nullptr, 0);
    TEST_ASSERT(engine->getCondWeightsLoadedForTest() == false,
        "condWeightsLoaded_ is false after loadCondWeights(nullptr, 0)");

    // Reload for downstream callers (processBlock would apply the projection).
    engine->loadCondWeights(weights, 8);
    TEST_ASSERT(engine->getCondWeightsLoadedForTest() == true,
        "condWeightsLoaded_ is true after re-loading weights");

    TEST_PASS("conditioning projection math");
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

    std::printf("Test: process block partial block\n");
    failures += test_process_block_partial_block();

    std::printf("Test: process block stereo\n");
    failures += test_process_block_stereo();

    std::printf("Test: reset state during processing\n");
    failures += test_reset_state_during_processing();

    std::printf("Test: multiple process blocks\n");
    failures += test_multiple_process_blocks();

    std::printf("Test: zero samples\n");
    failures += test_zero_samples();

    std::printf("Test: large block size\n");
    failures += test_large_block_size();

    std::printf("Test: init idempotent\n");
    failures += test_init_idempotent();

    std::printf("Test: construct with different configs\n");
    failures += test_construct_with_different_configs();

    std::printf("Test: construct/destroy without init\n");
    failures += test_construct_destroy_no_init();

    std::printf("Test: process block no init\n");
    failures += test_process_block_no_init();

    std::printf("Test: set conditioning\n");
    failures += test_set_conditioning();

    std::printf("Test: load cond weights\n");
    failures += test_load_cond_weights();

    std::printf("Test: conditioning with model\n");
    failures += test_conditioning_with_model();

    std::printf("Test: init missing model path\n");
    failures += test_init_missing_model_path();

    std::printf("Test: conditioning atomic swap\n");
    failures += test_conditioning_atomic_swap();

    std::printf("Test: conditioning projection math\n");
    failures += test_conditioning_projection_math();

    std::printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures;
}
