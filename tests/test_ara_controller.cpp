#include <cstdio>
#include <cstring>
#include <cstddef>
#include <atomic>
#include <string>
#include "core/Platform.hpp"
#include "ara/MambaAraController.hpp"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            return 1; \
        } \
    } while(0)

#define TEST_PASS(msg) \
    std::printf("  PASS: %s\n", msg)

static int test_model_dir_resolution_real()
{
    // Use ARAGetFactory (exported from MambaAraController.cpp, declared
    // in MambaAraController.hpp) as a real function address. dladdr /
    // GetModuleHandleExA needs a real address to resolve the module path;
    // passing nullptr skips the platform path resolution and returns the
    // default "./models/" stub.
    std::string dir = mambaResolveModelDir((void*)&ARAGetFactory);
    TEST_ASSERT(!dir.empty(), "model dir is not empty");

    TEST_ASSERT(dir.find("models") != std::string::npos,
                "model dir contains 'models'");
    char last = dir[dir.size() - 1];
    TEST_ASSERT(last == '/' || last == '\\',
                "model dir ends with a path separator");

    // Sanity: the resolved path should NOT be the "./models/" stub that
    // mambaResolveModelDir returns when funcAddr is nullptr. The stub is
    // exactly "./models/" (9 chars); any real module path is longer.
    TEST_ASSERT(dir.size() > 16,
                "resolved path is longer than the nullptr stub ('./models/')");

    TEST_PASS("model dir resolution");
    return 0;
}

static int test_factory_config_methods()
{
    MambaFactoryConfig config;

    const char* factoryID = config.getFactoryID();
    TEST_ASSERT(factoryID != nullptr, "factoryID is not null");
    TEST_ASSERT(std::strcmp(factoryID, "com.mamba.audio.suite.ara") == 0,
                "factoryID == 'com.mamba.audio.suite.ara'");

    const char* pluginName = config.getPlugInName();
    TEST_ASSERT(pluginName != nullptr, "pluginName is not null");
    TEST_ASSERT(std::strlen(pluginName) > 0, "pluginName is not empty");

    const char* manufacturer = config.getManufacturerName();
    TEST_ASSERT(manufacturer != nullptr, "manufacturer is not null");
    TEST_ASSERT(std::strlen(manufacturer) > 0, "manufacturer is not empty");

    const char* version = config.getVersion();
    TEST_ASSERT(version != nullptr, "version is not null");
    TEST_ASSERT(std::strcmp(version, "1.1.0") == 0, "version == '1.1.0'");

    const char* archiveID = config.getDocumentArchiveID();
    TEST_ASSERT(archiveID != nullptr, "documentArchiveID is not null");
    TEST_ASSERT(std::strlen(archiveID) > 0, "documentArchiveID is not empty");

    TEST_PASS("factory config methods");
    return 0;
}

static int test_archive_params_size()
{
    struct Params {
        float bypass;
        float wetDry;
        float intensity;
    };

    TEST_ASSERT(sizeof(Params) == 12, "3 floats = 12 bytes");
    TEST_ASSERT(offsetof(Params, bypass) == 0, "bypass at offset 0");
    TEST_ASSERT(offsetof(Params, wetDry) == 4, "wetDry at offset 4");
    TEST_ASSERT(offsetof(Params, intensity) == 8, "intensity at offset 8");

    TEST_PASS("archive params size");
    return 0;
}

static int test_atomic_defaults()
{
    {
        std::atomic<float> bypass{0.0f};
        std::atomic<float> wetDry{1.0f};
        std::atomic<float> intensity{0.5f};
        std::atomic<bool> contentDirty{false};

        TEST_ASSERT(bypass.load() == 0.0f, "atomic<float> bypass default = 0.0");
        TEST_ASSERT(wetDry.load() == 1.0f, "atomic<float> wetDry default = 1.0");
        TEST_ASSERT(intensity.load() == 0.5f, "atomic<float> intensity default = 0.5");
        TEST_ASSERT(contentDirty.load() == false, "atomic<bool> contentDirty default = false");
    }

    {
        std::atomic<float> value{0.0f};
        value.store(0.5f, std::memory_order_release);
        TEST_ASSERT(value.load(std::memory_order_acquire) == 0.5f,
                    "atomic store/load with acquire/release");

        value.store(1.0f);
        TEST_ASSERT(value.load() == 1.0f,
                    "atomic store/load with default seq_cst");
    }

    TEST_PASS("atomic defaults");
    return 0;
}

static int test_information_url()
{
    MambaFactoryConfig config;
    const char* url = config.getInformationURL();
    TEST_ASSERT(url != nullptr, "information URL is not null");
    TEST_ASSERT(std::strlen(url) > 0, "information URL is not empty");
    TEST_ASSERT(std::strncmp(url, "https://", 8) == 0, "information URL starts with https://");
    
    TEST_PASS("information URL");
    return 0;
}

int main()
{
    std::printf("=== MambaARA Controller Tests ===\n\n");

    int failures = 0;

    std::printf("Test: model dir resolution (real symbol address)\n");
    failures += test_model_dir_resolution_real();

    std::printf("Test: factory config methods\n");
    failures += test_factory_config_methods();

    std::printf("Test: archive params size\n");
    failures += test_archive_params_size();

    std::printf("Test: atomic defaults\n");
    failures += test_atomic_defaults();

    std::printf("Test: information URL\n");
    failures += test_information_url();

    std::printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures;
}
