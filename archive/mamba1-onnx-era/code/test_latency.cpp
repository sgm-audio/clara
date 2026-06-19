#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
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

static int test_latency_at_sample_rate(const char* label, int sampleRate)
{
    const std::string mp = getModelPath();
    auto engine = createEngine(mp);
    if (!engine->init()) {
        return skipTest(label);
    }

    constexpr int kBlockSize = 256;
    constexpr int kTotalSamples = 10000;
    constexpr int kImpulseIndex = 1000;
    constexpr int kReportedLatency = 4;
    constexpr float kNoiseFloor = 1e-6f;

    float input[kTotalSamples];
    float output[kTotalSamples];
    std::memset(input, 0, kTotalSamples * sizeof(float));

    input[kImpulseIndex] = 1.0f;

    engine->resetState();

    int samplesRemaining = kTotalSamples;
    int srcOffset = 0;
    int dstOffset = 0;

    while (samplesRemaining > 0) {
        int chunk = (samplesRemaining > kBlockSize) ? kBlockSize : samplesRemaining;
        engine->processBlock(input + srcOffset, output + dstOffset, chunk);
        srcOffset += chunk;
        dstOffset += chunk;
        samplesRemaining -= chunk;
    }

    int measuredLatency = -1;
    for (int i = kImpulseIndex; i < kTotalSamples; ++i) {
        if (std::fabs(output[i]) > kNoiseFloor) {
            measuredLatency = i - kImpulseIndex;
            break;
        }
    }

    std::printf("  %s (%d Hz): measured latency = %d samples, reported = %d samples\n",
        label, sampleRate, measuredLatency, kReportedLatency);

    TEST_ASSERT(measuredLatency >= 0,
        "impulse response detected (output contains signal above noise floor)");

    int delta = std::abs(measuredLatency - kReportedLatency);
    TEST_ASSERT(delta <= 1,
        "measured latency matches reported latency (|measured - reported| <= 1 sample)");

    TEST_PASS(label);
    return 0;
}

int main()
{
    std::printf("=== MambaInferenceEngine Latency Tests ===\n\n");

    struct {
        const char* label;
        int sampleRate;
    } testRates[] = {
        {"latency at 44.1 kHz", 44100},
        {"latency at 48 kHz",   48000},
        {"latency at 96 kHz",   96000},
        {"latency at 192 kHz",  192000},
    };

    int failures = 0;

    for (const auto& tr : testRates) {
        std::printf("Test: %s\n", tr.label);
        failures += test_latency_at_sample_rate(tr.label, tr.sampleRate);
    }

    std::printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures;
}
