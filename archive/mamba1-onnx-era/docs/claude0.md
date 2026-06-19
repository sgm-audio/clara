# CLARA — Mamba Audio Suite (CLAP + ARA2)

> Complete project context for Claude agents. Covers PRD, ADRs, source code, build system, training pipeline, CI/CD, and roadmap.

---

## 1. Project PRD

### Product
CLARA is a real-time Mamba SSM (Selective State Space Model) audio processor distributed as a CLAP plugin and ARA2 shared library. Both plugins share a single inference core (`MambaCore` static library + `MambaInferenceEngine`).

### Target Users
- Music producers using CLAP-compatible DAWs (Bitwig, REAPER, Cubase 13+, FL Studio)
- Post-production engineers using ARA2-compatible DAWs (Studio One, REAPER, Cubase, Logic Pro via VST3/ARA bridge)
- Developers wanting a reference Mamba SSM C++ implementation with ONNX Runtime

### Key Features
- Real-time Mamba SSM inference (d_model=64, n_layers=4, ~52K params, random weights currently)
- 3 exposed parameters: bypass (stepped), wet/dry, intensity (all automatable)
- Click-free bypass via 50ms linear crossfade (per-sample ramp, shared across channels)
- 4 samples of reported latency (pure signal processing pipeline, not lookahead)
- Stereo I/O processing (mono model applied per channel)
- Preset save/load via CLAP_EXT_STATE (16-byte binary format, versioned)
- ARA2 archive persistence of same 3 float parameters
- Deferred atomic state reset (SSM/conv state clears on audio thread, safe to call from any thread)
- Zero heap allocations on audio thread (thread_local Ort::Value vectors with reserve())
- ONNX Runtime CPU provider (no GPU dependency)
- Cross-platform: Phase 0 (portability scaffolding) complete; macOS and Windows in progress

### Future Features (blocked on training pipeline + GUI framework)
- GUI (CLAP_EXT_GUI, ARA EditorRenderer/EditorView)
- Trained model weights (replace current random noise output)
- Tempo-sync or key-aware processing
- ARA analysis features (content-dirty signaling exists, no analysis yet)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Host DAW                              │
├──────────────────────┬──────────────────────────────────┤
│    CLAP Plugin        │        ARA 2.0 Library           │
│  MambaSuiteClap.clap  │     MambaSuiteARA.so/.dll/.dylib  │
│                      │                                  │
│  ┌────────────────┐  │  ┌────────────────────────────┐  │
│  │ MambaPluginState│  │  │ MambaDocumentController   │  │
│  │  - 3 atomics   │  │  │  - 3 atomics (mirrored)   │  │
│  │  - engine ptr  │  │  │  - contentDirty_ flag     │  │
│  │  - deferred    │  │  │  - MambaPlaybackRenderer  │  │
│  │    tasks       │  │  │  - EditorRenderer stub    │  │
│  └───────┬────────┘  │  └────────┬───────────────────┘  │
│          │            │           │                      │
│  ┌───────┴────────┐  │  ┌────────┴───────────────────┐  │
│  │ MambaCore      │  │  │ MambaCore                  │  │
│  │ (static lib)   │  │  │ (static lib, same code)    │  │
│  │                │  │  │                            │  │
│  │ ┌────────────┐ │  │  │ ┌────────────────────────┐ │  │
│  │ │MambaInfer..│ │  │  │ │MambaInferenceEngine    │ │  │
│  │ │ Engine     │ │  │  │ │ (maxBlockSize=1024)    │ │  │
│  │ │(maxBlock   │ │  │  │ └────────────────────────┘ │  │
│  │ │=256)       │ │  │  └────────────────────────────┘  │
│  │ └────────────┘ │  │                                  │
│  └────────────────┘  └──────────────────────────────────┘
│                      │                                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │  ONNX Runtime (libonnxruntime.so/.dylib/.dll)      │ │
│  │  CPU provider only                                 │ │
│  └────────────────────────────────────────────────────┘ │
│                      │                                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Model: models/mamba_rt.onnx + .onnx.data          │ │
│  │  9 inputs → 3 outputs, dynamic seq_len             │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Data Flow (CLAP process block)

```
audio input (stereo)
  → memcpy to chIn[2] buffers
  → engine->processBlock(chIn[ch], chOut[ch], numSamples) per channel
  → intensity scaling: wet = dry + (wet - dry) * intensity
  → wet/dry blend: mix = wet*wetDry + dry*(1-wetDry)
  → bypass crossfade: output = (1-bypass)*mix + bypass*dry
  → copy to output buffers
  → duplicate ch0 to extra output channels
```

### ONNX Model Contract

**9 Inputs:**
| Input | Shape | Description |
|-------|-------|-------------|
| `audio_chunk` | [1, seq_len, 1] | Mono audio samples |
| `conv0`..`conv3` | [1, 64, 4] | Per-layer conv state |
| `ssm0`..`ssm3` | [1, 64, 16] | Per-layer SSM state |

**3 Outputs:**
| Output | Shape | Description |
|--------|-------|-------------|
| `processed_audio` | [1, seq_len, 1] | Processed audio |
| `new_conv_state` | [1, 4, 64, 4] | Stacked conv states |
| `new_ssm_state` | [1, 4, 64, 16] | Stacked SSM states |

**Dynamic axes:** seq_len only (state tensors are fixed shape).

### Model Architecture

```
audio_chunk [B, seq_len, 1]
  → InputProjection (Linear 1→64)
  → MambaLayer 0 (Conv1d + S6 scan + SiLU gate + out_proj)
  → MambaLayer 1
  → MambaLayer 2
  → MambaLayer 3
  → LayerNorm (64)
  → OutputProjection (Linear 64→1)
  → processed_audio [B, seq_len, 1]
```

Each MambaLayer:
1. `in_proj`: Linear 64→128 (split: 64 for SSM path, 64 for gate)
2. `conv1d`: depthwise Conv1d(64→64, width=4, groups=64, padding=3), causal via state prepend
3. SiLU activation: `conv_out * SiLU(gate)`
4. `ssm_proj`: Linear 64→48 (dt_rank=16 + 2*d_state=32) → `dt_proj`(16→64, bias) + Softplus → dt; B, C from split
5. `scan_step`: Selective scan loop over seq_len with discretised dA/dB
6. `out_proj`: Linear 64→64

SSM discretisation:
- A = -exp(log_A_real) — always negative-definite
- dA = exp(dt * A) — zero-order hold
- dB = dt * B — Euler
- h' = dA * h + dB * x
- y = C * h' + D * x

### Key Design Constraints

1. `Ort::Value` has NO default constructor — use `thread_local std::vector` with `reserve()` instead of `std::array`
2. `ARA::PlugIn::PlugInEntry::getPlugInEntry<F, D>()` creates static singletons — cannot inject custom constructor args
3. ARA 2.0 `PlaybackRendererInterface` has no `renderSamples()` method; audio processing is companion CLAP plugin only
4. Plugins resolve model path relative to their own binary; models must be in `models/` subdirectory adjacent to plugin
5. `libonnxruntime.so` must be deployed alongside plugins (resolved via `$ORIGIN` RUNPATH)
6. `dladdr` usage replaced with `mambaResolveModulePath()` in Platform.hpp for cross-platform support

---

## 3. Architecture Decision Records

### ADR-001: Shared ONNX Runtime Inference Engine
**Status:** Accepted | **Date:** 2026-05-30

**Context:** Need RT-safe inference engine shared by CLAP and ARA plugins.
**Decision:** `MambaCore` static library with `MambaInferenceEngine` class wrapping ONNX Runtime.
- Pre-allocated all RT buffers in `init()`, zero heap allocations in `processBlock()`
- Exception handling: catch in `init()` and `processBlock()`, never cross C boundary
- Dynamic seq_len prevents SSM state corruption from zero-padding
- Chosen ONNX Runtime over LibTorch (~1GB, too heavy), RTNeural (no S6 scan), native C++ Mamba (too complex for v1)
- CPU-only inference (no GPU dependency)
- min RT buffer size 256 samples (~5.8ms at 44.1kHz); ARA uses 1024 for offline

### ADR-002: Deferred Atomic State Reset
**Status:** Accepted | **Date:** 2026-05-30

**Context:** SSM state (8 buffers, ~20KB) must be reset from multiple threads without locks on audio thread.
**Decision:** Single `std::atomic<bool> resetRequested_` with release-store / acq_rel-exchange.
- `resetState()` stores `true` with `memory_order_release` — never blocks
- `processBlock()` uses `exchange(false, std::memory_order_acq_rel)` — atomic read-and-clear
- Zeroing (bounded ~1us) happens on audio thread, deferred by at most one block
- Rejected: mutex (RT violation), SPSC queue (heavyweight), immediate zero with fence (UB)

### ADR-003: Click-Free Bypass via 50ms Linear Crossfade
**Status:** Accepted | **Date:** 2026-05-30

**Context:** Bypass toggle causes audible pop without crossfade.
**Decision:** 50ms linear crossfade, sample-rate-independent step calculation.
- CrossfadeStep = 1.0f / (0.05f * sampleRate)
- Step accumulation outside per-channel loop (fixes stereo aliasing bug)
- Block-spanning transition via `currentBypassValue` persisted across `processBlock()` calls
- Mix chain: intensity(wet) → wetDry blend → bypass crossfade → output
- Model runs during bypass (SSM state stays current for un-bypass)
- Rejected: instantaneous switch (pop), exponential ramp (non-linear), equal-power cosine (unnecessary for simple bypass)

### ADR-004: CLAP + ARA Dual-Target Build
**Status:** Accepted | **Date:** 2026-05-30

**Context:** Need reproducible build for both CLAP and ARA from same source.
**Decision:** CMake + FetchContent for all 3 dependencies (CLAP SDK `29ffcc27`, ARA SDK `1ca0b0aa`, ONNX Runtime v1.17.1).
- No system packages, no vcpkg, no git submodules
- `MambaCore` STATIC with `-fPIC` / `POSITION_INDEPENDENT_CODE ON`
- `MambaSuiteClap` as MODULE with SUFFIX `.clap`
- `MambaSuiteARA` as MODULE with platform-conditional SUFFIX (`.so`/`.dylib`/`.dll`)
- Models copied at build time → `build/plugins/models/`
- ONNX Runtime deployed adjacent to plugins
- Component-based install: `--component clap` / `--component ara`
- Cross-platform RPATH handling: `$ORIGIN` (Linux), `@loader_path` (macOS), none needed (Windows)

### ADR-005: thread_local + reserve() for Ort::Value Vectors
**Status:** Accepted | **Date:** 2026-05-30

**Context:** `Ort::Value` has no default constructor — `std::array` is not possible.
**Decision:** Two `thread_local std::vector<Ort::Value>` with `clear()` + `reserve()` + `push_back()`.
- After first call per thread, all subsequent calls are allocation-free
- First-call warm-up: ~1-5us heap allocation (acceptable for audio thread)
- Rejected: `std::array` (no default ctor), `std::optional` array (extra overhead), raw pointer placement new (error-prone)

### ADR-006: dladdr-Based Model Path Resolution (Now Platform.hpp)
**Status:** Accepted | **Date:** 2026-05-30 | **Updated:** 2026-05-31 (cross-platform migration)

**Context:** Plugins need to locate `models/mamba_rt.onnx` relative to their own binary at runtime.
**Decision (original):** `dladdr()` to get plugin filesystem path, resolve `models/` subdirectory relative to it.
**Decision (updated):** Platform abstraction layer in `src/core/Platform.hpp` provides:
- `MAMBA_EXPORT` — `__declspec(dllexport)` on Windows, `__attribute__((visibility("default")))` on POSIX
- `mambaResolveModulePath(void* funcAddr, const char* modelFile)` — `GetModuleHandleExA`+`GetModuleFileNameA` on Windows, `dladdr` on POSIX
- `mambaResolveModelDir(void* funcAddr)` — same logic, returns directory path
- `mambaFileExists(const std::string& path)` — `std::ifstream::good()` (portable, no access() on Windows)
- Model path resolution in CLAP: `clap_create_plugin` → `mambaResolveModulePath((void*)clap_create_plugin, "mamba_rt.onnx")`
- Model path resolution in ARA: `MambaPlaybackRenderer` constructor → `mambaResolveModulePath((void*)&MambaPlaybackRenderer, "mamba_rt.onnx")`

---

## 4. Source Code

### Directory Structure

```
CLARA/
├── claude0.md                  ← THIS FILE
├── AGENTS.md                   ← Agent reference (build, arch, conventions)
├── TODO.md                     ← Master task tracker (32/34 closed)
├── TODO_PROGRESS.md            ← Progress summary
├── CHANGELOG.md                ← Release notes
├── CLAUDE_OPUS_PROMPT.md       ← Legacy onboarding
├── README.md                   ← Project README
├── CMakeLists.txt              ← Root build (313 lines)
├── export_mamba_onnx.py        ← ONNX model export/verify (386 lines)
│
├── src/
│   ├── core/
│   │   ├── Platform.hpp        ← Cross-platform portability (85 lines)
│   │   ├── MambaInferenceEngine.hpp  ← Engine header (60 lines)
│   │   └── MambaInferenceEngine.cpp  ← Engine impl (135 lines)
│   ├── clap/
│   │   ├── MambaClapPlugin.hpp ← CLAP plugin header (48 lines)
│   │   └── MambaClapPlugin.cpp ← CLAP plugin impl (449 lines)
│   └── ara/
│       ├── MambaAraController.hpp   ← ARA controller header (84 lines)
│       └── MambaAraController.cpp   ← ARA controller impl (252 lines)
│
├── tests/
│   ├── CMakeLists.txt          ← Test build (74 lines)
│   ├── test_engine.cpp         ← 12 engine tests
│   ├── test_clap_plugin.cpp    ← 10 CLAP plugin tests
│   └── test_ara_controller.cpp ← 4 ARA controller tests
│
├── docs/
│   ├── parameter-reference.md  ← Full parameter/model reference (379 lines)
│   └── training-pipeline.md    ← Training plan for A100 (1165 lines)
│
├── models/
│   ├── mamba_rt.onnx           ← ONNX model graph
│   └── mamba_rt.onnx.data      ← External weight data
│
├── .github/workflows/
│   └── ci.yml                  ← CI pipeline (137 lines)
│
├── ADR-001 through ADR-006     ← Architecture decisions
└── ADR-WORKFLOW.md             ← ADR workflow tracking
```

### Key Source Files (contents embedded)

#### `src/core/Platform.hpp` (85 lines)

```cpp
#pragma once
#include <string>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#define MAMBA_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#include <dlfcn.h>
#include <link.h>
#define MAMBA_EXPORT __attribute__((visibility("default")))
#endif

inline std::string mambaResolveModulePath(void* funcAddr, const char* modelFile) {
    (void)funcAddr;
    std::string base;
#if defined(_WIN32)
    char path[MAX_PATH];
    HMODULE hm = nullptr;
    if (funcAddr) {
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)funcAddr, &hm);
    }
    if (hm && GetModuleFileNameA(hm, path, MAX_PATH)) { base = path; }
#else
    if (funcAddr) {
        Dl_info info;
        if (dladdr(funcAddr, &info) && info.dli_fname) { base = info.dli_fname; }
    }
#endif
    auto sep = base.rfind('/');
    if (sep == std::string::npos) sep = base.rfind('\\');
    if (sep != std::string::npos) base.resize(sep + 1);
    else base = "./";
    return base + "models/" + modelFile;
}

inline bool mambaFileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

inline std::string mambaResolveModelDir(void* funcAddr) {
    // Same pattern as mambaResolveModulePath but returns ".../models/" directory
    std::string base;
#if defined(_WIN32)
    // ... GetModuleHandleExA + GetModuleFileNameA pattern ...
#else
    // ... dladdr pattern ...
#endif
    // ... path extraction logic ...
    return base + "models/";
}
```

#### `src/core/MambaInferenceEngine.hpp` (60 lines)

```cpp
#pragma once
#include <vector>
#include <memory>
#include <string>
#include <atomic>
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <array>
#include <algorithm>

class MambaInferenceEngine {
public:
    struct Config {
        std::string modelPath;
        size_t dModel = 64;
        size_t dState = 16;
        size_t nLayers = 4;
        size_t convWidth = 4;
        size_t maxBlockSize = 256;
    };
    explicit MambaInferenceEngine(const Config& config);
    ~MambaInferenceEngine();
    bool init();
    void resetState();
    void processBlock(const float* input, float* output, int numSamples);
private:
    Config config_;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "MambaInference"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memoryInfo_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    static constexpr size_t kMaxLayers = 4;
    std::atomic<bool> resetRequested_{false};
    std::vector<int64_t> audioChunkShape_, processedAudioShape_, convLayerShape_, ssmLayerShape_, convStackShape_, ssmStackShape_;
    std::vector<float> inputBuffer_;
    std::vector<float> convStateIn_[kMaxLayers];
    std::vector<float> ssmStateIn_[kMaxLayers];
    std::vector<float> processedAudioOut_, convStateOutStack_, ssmStateOutStack_;
    static constexpr size_t kNumInputs = 9;
    static constexpr size_t kNumOutputs = 3;
    std::array<const char*, kNumInputs> inputNames_;
    std::array<const char*, kNumOutputs> outputNames_;
    bool initialized_{false};
};
```

#### `src/core/MambaInferenceEngine.cpp` (135 lines)

Core engine: constructor initializes shapes/names, init() creates session + allocates buffers, processBlock() does deferred reset check → build input/output tensors → session->Run() → unstack states.

Key details:
- `thread_local std::vector<Ort::Value> inputTensors_` with `reserve(9)` + `clear()` + `push_back()`
- `thread_local std::vector<Ort::Value> outputTensors_` with `reserve(3)` + `clear()` + `push_back()`
- Dynamic shape `{1, blockSamples, 1}` for audio (not padded to maxBlockSize)
- Exception handling: catches `Ort::Exception`, writes silence, returns early
- State unstacking: `memcpy` per-layer from output stacks into per-layer input buffers

#### `src/clap/MambaClapPlugin.hpp` (48 lines)

```cpp
#pragma once
#include <clap/clap.h>
#include <clap/ext/state.h>
#include "../core/MambaInferenceEngine.hpp"
#include <atomic>
#include <vector>
#include <memory>

enum ParamId { PARAM_BYPASS = 0, PARAM_WET_DRY = 1, PARAM_INTENSITY = 2, PARAM_COUNT };

struct MambaPluginState {
    std::unique_ptr<MambaInferenceEngine> engine;
    const clap_host_t* host{nullptr};
    std::string modelPath;
    std::atomic<float> bypass{0.0f};
    std::atomic<float> wetDry{1.0f};
    std::atomic<float> intensity{0.5f};
    float sampleRate{44100.0f};
    float currentBypassValue{0.0f};
    float crossfadeStep{0.001f};
    std::vector<float> chIn[2];
    std::vector<float> chOut[2];
    enum class DeferredTask : uint8_t { None = 0, ModelReload, AnalysisQueueDrain };
    std::atomic<DeferredTask> pendingDeferredTask{DeferredTask::None};
};

extern "C" const clap_plugin_descriptor_t mamba_plugin_info;
extern "C" const clap_plugin_state_t mamba_state_extension;
extern const clap_plugin_t mamba_plugin;
const void* mamba_get_extension(const clap_plugin_t* plugin, const char* extensionId);
```

#### `src/clap/MambaClapPlugin.cpp` (449 lines)

Key lifecycle functions:
- `mamba_init`: checks modelPath not empty + file exists via mambaFileExists()
- `mamba_activate`: allocates per-channel buffers, creates engine, calls init(), calculates crossfadeStep
- `mamba_deactivate`: engine.reset()
- `mamba_start_processing`: returns engine != nullptr
- `mamba_reset`: calls engine->resetState() (deferred atomic flag)
- `mamba_process`: processes events, runs engine per channel, applies intensity→wetDry→bypass crossfade chain
- CLAP extensions: params (count/info/get/save/text/flush), latency (4), state (16-byte), audio-ports (stereo in/out)
- Entry: `clap_create_plugin` uses `mambaResolveModulePath((void*)clap_create_plugin, "mamba_rt.onnx")`
- Factory: `clap_get_factory` returns mamba_plugin_factory
- Export: `CLAP_EXPORT const clap_plugin_entry_t clap_entry`

#### `src/ara/MambaAraController.hpp` (84 lines)

```cpp
class MambaPlaybackRenderer : public ARA::PlugIn::PlaybackRenderer {
public:
    explicit MambaPlaybackRenderer(ARA::PlugIn::DocumentController* documentController);
    ~MambaPlaybackRenderer() noexcept override;
private:
    std::unique_ptr<MambaInferenceEngine> rtEngine_;
};

class MambaDocumentController : public ARA::PlugIn::DocumentController {
public:
    using ARA::PlugIn::DocumentController::DocumentController;
    std::atomic<float> bypass{0.0f};
    std::atomic<float> wetDry{1.0f};
    std::atomic<float> intensity{0.5f};
    // ... all doCreate/doDestroy overrides ...
private:
    std::atomic<bool> contentDirty_{false};
};

class MambaFactoryConfig : public ARA::PlugIn::FactoryConfig { /* factory methods */ };
MAMBA_EXPORT const ARA::ARAFactory* getMambaARAFactory() noexcept;
extern "C" MAMBA_EXPORT const ARA::ARAFactory* ARAGetFactory();
```

#### `src/ara/MambaAraController.cpp` (252 lines)

- MambaPlaybackRenderer constructor: uses `mambaResolveModulePath((void*)&MambaPlaybackRenderer, "mamba_rt.onnx")`, creates engine with maxBlockSize=1024
- ARA content stubs: `doUpdateMusicalContextContent` and `doUpdateAudioSourceContent` set `contentDirty_` flag
- `doRestoreObjectsFromArchive`: reads 12 bytes (3 floats), stores with release semantics
- `doStoreObjectsToArchive`: reads 3 atomics with acquire semantics, writes 12 bytes
- EditorRenderer/EditorView: return nullptr with TODO comments (deferred GUI)
- Factory: `ARAGetFactory()` → `getMambaARAFactory()` → static singleton via `PlugInEntry::getPlugInEntry<MambaFactoryConfig, MambaDocumentController>()`
- Export: `extern "C" MAMBA_EXPORT const ARA::ARAFactory* ARAGetFactory()`

### Tests

#### `tests/test_engine.cpp` (12 tests)
1. construct/destroy (2 sub-tests)
2. init model path
3. process block
4. reset state
5. process block partial block
6. process block stereo
7. reset state during processing
8. multiple process blocks
9. zero samples
10. large block size
11. init idempotent
12. construct with different configs

#### `tests/test_clap_plugin.cpp` (10 tests)
1. plugin descriptor (MambaPluginState defaults)
2. plugin info descriptor (CLAP version, IDs, features)
3. preset data packing (byte layout, static_asserts)
4. param ranges (bypass/wetDry/intensity min/max/default)
5. param value to text
6. param text to value
7. state save/load roundtrip (mock streams, 16-byte serialization)
8. crossfade step calculation (44.1kHz and 48kHz)
9. latency reporting (4 samples)
10. audio ports (stereo in/out, IS_MAIN flag, names)

#### `tests/test_ara_controller.cpp` (4 tests)
1. model dir resolution (ends with `models/` or `models\\`)
2. factory config methods (ID, name, manufacturer, version, archive ID)
3. archive params size (3 floats = 12 bytes, static_asserts)
4. atomic defaults (bypass=0.0, wetDry=1.0, intensity=0.5, contentDirty=false; acquire/release works)

---

## 5. Build System

### `CMakeLists.txt` (313 lines)

Min CMake 3.14, C++17. Dual-target build with 3 FetchContent dependencies.

**Compiler flags:**
- MSVC: `/W4`, Release=`/O2 /fp:fast /DNDEBUG /GR-`, Debug=`/Od /Zi /DDEBUG`
- GCC/Clang: `-Wall -Wextra -Werror=return-type`, Release=`-O3 -ffast-math -DNDEBUG -fno-rtti`, Debug=`-O0 -g -DDEBUG`

**RPATH configuration:**
- Linux: INSTALL_RPATH="$ORIGIN", BUILD_RPATH="$ORIGIN", BUILD_WITH_INSTALL_RPATH=ON
- macOS: INSTALL_RPATH="@loader_path", BUILD_RPATH="@loader_path", BUILD_WITH_INSTALL_RPATH=ON
- Windows: no RPATH (DLL search path)

**Platform-conditional settings:**
- ONNX Runtime URL: win-x64 zip / osx-universal2 tgz / linux-x64 tgz
- ARA suffix: `.dll` (Win) / `.dylib` (macOS) / `.so` (Linux)
- CLAP bundle dir: `x86_64-windows` / `MacOS` / `linux-x86-64`
- Strip: Unix `LINK_FLAGS "-s"` for Release
- Linked libs: `pthread dl` on Linux (not macOS or Windows)

**Build targets:**
- `MambaCore` (STATIC): MambaInferenceEngine.cpp, links ONNX Runtime
- `MambaSuiteClap` (MODULE, suffix .clap): MambaClapPlugin.cpp, links MambaCore
- `MambaSuiteARA` (MODULE, platform suffix): MambaAraController.cpp + ARAPlug.cpp, links MambaCore

**Deployment:**
- `file(COPY)` models from source to build/plugins/models/
- `file(COPY)` ONNX Runtime lib to build/plugins/
- Install rules for CLAP bundle, bare CLAP, ARA library, runtime deps, and models

### `tests/CMakeLists.txt` (74 lines)

- `test_engine`: links engine.cpp, ONNX Runtime, pthread+dl (Linux)
- `test_clap_plugin`: links MambaClapPlugin.cpp + engine.cpp, CLAP include, ONNX Runtime, pthread+dl
- `test_ara_controller`: links MambaAraController.cpp + ARAPlug.cpp + engine.cpp, ARA SDK, ONNX Runtime, pthread+dl
- All pass TEST_MODEL_DIR compile definition for model path

---

## 6. Model Export (`export_mamba_onnx.py`)

**386 lines** of Python. Implements full Mamba SSM in PyTorch:

- `SelectiveScanStep`: single timestep S6 scan
- `MambaSSMCore`: projects input to dt/B/C, returns per-timestep values
- `MambaLayer`: Conv1d + SSM + SiLU gating
- `MambaRTModel`: full model with InputProjection → 4×MambaLayer → LayerNorm → OutputProjection
- `export_mamba_to_onnx`: traces with torch.jit.trace, exports with dynamic_axes on audio_chunk/processed_audio
- `verify_onnx_model`: loads model via ONNX Runtime, checks shapes, non-zero output, finite values
- Self-verifies state continuity across two chunks (proves state propagation)
- Parameters: d_model=64, d_state=16, n_layers=4, conv_width=4, dt_rank=16, seq_len=256 (trace) / 64 (verify)
- Opset 18, external data format

Run: `python3 export_mamba_onnx.py` → produces `models/mamba_rt.onnx` + `models/mamba_rt.onnx.data`

---

## 7. Training Pipeline (`docs/training-pipeline.md`)

**1165 lines** — complete RunPod-based training plan:

**Model:** 72K params, 4-layer Mamba, ~0.72 GFLOPS forward pass
**Hardware:** RunPod A100 80GB (PCIe or SXM), ~$2/hr
**Setup:** tmux + aria2c + RAM caching for dataset

**Training script architecture:**
- PyTorch Lightning recommended (but raw PyTorch alternative provided)
- DataModule: loads audio, chunks to 256-sample blocks, normalizes
- Loss: L1 + multi-resolution STFT + optional feature loss
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Batch size: 512-1024 (depends on A100 memory vs. dataset size)

**Speed optimizations:**
- torch.compile (Inductor backend, dynamic=False)
- AMP (automatic mixed precision, bf16 on A100)
- cuDNN benchmarking enabled
- DataLoader with num_workers=8, pin_memory, prefetch_factor=4
- Expected: ~50K steps/day (100M samples at batch 512)

**Post-training:**
- Load checkpoint, export to ONNX via export_mamba_onnx.py
- Verify shapes, R  ontinuity
- Deploy as models/mamba_rt.onnx + .onnx.data
- Versioned checkpoints (YYYY-MM-DD format)

---

## 8. CI/CD (`.github/workflows/ci.yml`)

**Two jobs:**

1. **build** (ubuntu-22.04):
   - Install deps (cmake, g++, python3, torch, onnx, onnxruntime, numpy)
   - Configure + build Release
   - Verify symbols: `clap_entry`, `ARAGetFactory` via readelf
   - Verify RPATH: `$ORIGIN` via readelf
   - Verify ONNX Runtime deploy: `libonnxruntime.so.1.17.1` + symlink
   - Verify model files: `mamba_rt.onnx` + `.onnx.data`
   - Download + run clap-validator (continue-on-error: true — may fail without GUI)
   - Verify ARA library: file type, symbol, RPATH
   - Verify stripped: check .symtab presence
   - Verify Python model export + shapes
   - Upload artifact: `.clap`, `.so`, `libonnxruntime.so*`, `models/`

2. **test** (ubuntu-22.04, needs build):
   - Install cmake, g++
   - Configure Debug with BUILD_TESTING=ON
   - Build tests
   - Run: `ctest --output-on-failure`

---

## 9. C++ Conventions

- C++17, no RTTI in release (`-fno-rtti`); exceptions allowed (ORT API throws `Ort::Exception`)
- RT-safe: no allocations on audio thread — `thread_local std::vector` with `reserve()`, deferred atomic reset
- `-Wall -Wextra -Werror=return-type` always on (GCC/Clang)
- Symbols hidden (`CXX_VISIBILITY_PRESET hidden`), `MAMBA_EXPORT` for public symbols
- `init()` returns bool (never throws); callers use `if (!engine->init())` pattern
- Release builds stripped on Unix via `LINK_FLAGS "-s"`
- No comments in source files (code is the documentation)
- No emojis in code or docs

---

## 10. Cross-Platform Migration Status

### Phase 0: Portability Scaffolding ✅ COMPLETE
- Created `src/core/Platform.hpp` with:
  - `MAMBA_EXPORT` macro (Windows dllexport / POSIX visibility)
  - `mambaResolveModulePath()` (GetModuleHandleExA / dladdr)
  - `mambaResolveModelDir()` (same pattern)
  - `mambaFileExists()` (std::ifstream)
- Removed `dladdr()` calls from MambaClapPlugin.cpp and MambaAraController.cpp
- Removed `__attribute__((visibility("default")))` from MambaAraController.hpp
- Updated CMakeLists.txt with platform-conditional guards for all OS-specific logic
- Updated tests/CMakeLists.txt with conditional pthread+dl linking
- Updated test_ara_controller.cpp for cross-platform path separator support
- Zero behavior change on Linux (Phase 0 is purely scaffolding)

### Phase 1: macOS (IN PROGRESS — NOT STARTED)
- CMake preset for macOS + Apple Clang
- @loader_path RPATH (already in CMakeLists.txt)
- .dylib suffix (already in CMakeLists.txt)
- Universal binary build (x86_64 + arm64)
- CI: macos-14 runner, clap-validator-macos, brew deps

### Phase 2: Windows (IN PROGRESS — NOT STARTED)
- MSVC CMake preset
- .dll suffix (already in CMakeLists.txt)
- DLL search path deployment
- CI: windows-2022 runner, ONNX Runtime zip extraction, MSVC flags

### Phase 3: CI Matrix + Docs (FUTURE)
- Expand CI to `[ubuntu-22.04, macos-14, windows-2022]`
- Platform-specific artifact naming
- Update README/AGENTS.md/ADRs to remove "Linux x64 only" language

---

## 11. Known Gaps

| Gap | Severity | Notes |
|-----|----------|-------|
| Models untrained (random weights → noise) | P2 | Training pipeline doc exists; needs RunPod A100 run |
| No GUI (CLAP_EXT_GUI) | P1 | Deferred until trained model exists |
| No ARA EditorRenderer/EditorView | P1 | Part of GUI deferral |
| CLAP validator may fail without GUI | P4 | continue-on-error: true in CI |
| Training not yet executed | P2 | Pipeline documented, script ready, needs A100 time |
| ADR-WORKFLOW.md still lists ADR-003..006 as DRAFT | P3 | Needs status update |
| AGENTS.md says "Linux x64 only" | P3 | Needs update after cross-platform complete |
| README.md says "Linux x64 only" | P3 | Needs update after cross-platform complete |
| ADR-001/ADR-006 mention dladdr without cross-platform | P3 | Needs ADR-007 |

---

## 12. TODOs (from TODO.md)

**32/34 closed, 2 remaining:**

P0: Engine/Plugin stability — ALL CLOSED
P1: ARA correctness — ALL CLOSED  
P2: Testing — ALL CLOSED
P3: Code quality — ALL CLOSED
P4: CI quality — ALL CLOSED
P5: Documentation — ALL CLOSED

**Remaining:**
1. GUI (CLAP_EXT_GUI + ARA EditorRenderer/EditorView) — deferred
2. Cross-platform migration (Phases 1-3) — in progress

---

## 13. Session Memory (for Agent Continuity)

Key decisions captured from latest session:
1. Cross-platform migration is the current priority (Phase 0 done, Phases 1-3 remaining)
2. Platform.hpp is the single source of truth for OS-specific code (4 inline functions)
3. No build tools on current Windows dev machine — all verification deferred to CI
4. 10-subagent parallel approach worked well for Phase 0
5. ADR-007 must document the cross-platform strategy
6. CMakePresets.json should be added for Phase 1/2
7. macOS is lower-friction than Windows (same Clang, POSIX APIs)
8. AGENTS.md, README.md, ADR-*.md all need cross-platform updates (Phase 3)
