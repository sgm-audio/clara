# CLARA — Mamba Audio Suite (CLAP + ARA2)

C++17 audio plugin running a Mamba SSM (Selective State Space Model) via ONNX Runtime. Supports Linux (stable), macOS and Windows (in progress). Produces a CLAP plugin and an ARA2 shared library sharing one inference core.

See `ADR-001` through `ADR-008` in `ADR-*.md` for architectural decisions and `ADR-WORKFLOW.md` for the ADR drafting backlog and system prompts.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

With tests:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build . --parallel && ctest --output-on-failure
```

Cross-platform builds via CMake presets:
```bash
cmake --preset linux-release      # Linux
cmake --preset macos-release      # macOS
cmake --preset windows-release    # Windows
cmake --build --preset <preset>
```

Install (default prefix `/usr/local`):
```bash
cmake --install .          # both CLAP + ARA components
cmake --install . --component clap   # CLAP bundle + bare .clap only
cmake --install . --component ara    # ARA .so only
```

Outputs go to `build/plugins/`:
- `MambaSuiteClap.clap` — CLAP audio effect
- `MambaSuiteARA.so` — ARA2 library
- `models/` — ONNX models copied from source
- `libonnxruntime.so.1.17.1` — deployed alongside plugins, resolved via `$ORIGIN` RPATH

Three dependencies via CMake FetchContent (no manual install):
- CLAP SDK (`29ffcc27`)
- ARA SDK (`1ca0b0aa`)
- ONNX Runtime v1.17.1 (prebuilt Linux x64 tarball)

## Model Export

```bash
python3 export_mamba_onnx.py
```

Exports `models/mamba_rt.onnx` + `.onnx.data` (external data sidecar) using `torch.jit.trace`, opset 18. Requires `torch`, `onnxruntime`, `onnx`, `numpy`. The script self-verifies: runs ONNX inference, checks shapes, and validates state continuity across chunks. Re-export only if architecture changes.

## Architecture

`src/core/MambaInferenceEngine` — RT-safe ONNX wrapper. Uses `thread_local std::vector<Ort::Value>` with `reserve()` for zero per-call heap allocs in `processBlock()`. Output states use dynamic sequence length (`{1, blockSamples, 1}` shape) to avoid zero-padding SSM state corruption. Deferred state reset via `resetRequested_` atomic flag. Both CLAP and ARA link it via `MambaCore` static library.

`src/clap/MambaClapPlugin` — Full CLAP plugin. Registers extensions: params, state, latency (4 samples), audio-ports (stereo). 3 atomics: bypass (stepped), wet/dry, intensity. Click-free bypass via 50ms linear crossfade (per-sample ramp, shared across channels to prevent stereo aliasing). Uses `mambaResolveModulePath()` (Platform.hpp) for model path resolution relative to `.clap` binary. Uses `std::make_unique` + `.release()` for C-ABI lifetime.

`src/ara/MambaAraController` — DocumentController + PlaybackRenderer + `extern "C" ARAGetFactory()` with `visibility("default")`. ARA 2.0 has NO `renderSamples()` — audio processing is exclusively through the companion CLAP plugin. Content stubs documented with proper `enable` handling in `willEnableAudioSourceSamplesAccess`. Archive state persists 3 floats (bypass, wetDry, intensity) via `HostArchiveReader/Writer`.

`src/core/Platform.hpp` — cross-platform abstractions for module path resolution (`mambaResolveModulePath()`), symbol visibility (`MAMBA_EXPORT`/`MAMBA_IMPORT`), and thread naming. Each target platform (Linux, macOS, Windows) provides a conforming implementation via `#ifdef` guards.

ONNX model: 10 inputs (`audio_chunk` + 4×conv state + 4×ssm state + `cond`) → 3 outputs (`processed_audio`, stacked conv state, stacked ssm state). The engine unstacks output states per-layer each block. See ADR-001 §3 for the full I/O contract.

Conditioning vector `cond[COND_DIM]` (`COND_DIM=8`): double-buffered with atomic pointer swap for RT-safe updates from non-RT threads. Passed as the 10th ONNX input (FiLM) and additionally applied as external scalar gain modulation via `combinedWeights_` (`deltaWeights + bWeights`). See ADR-008.

`src/core/SPSCQueue.hpp` — fixed-capacity lock-free SPSC ring buffer for deferred task communication between audio thread and main thread. Capacity 64 entries, supports push from any thread, pop from main thread. Used for `ModelReload` and `AnalysisQueueDrain` deferred tasks.

## C++ Conventions

- C++17, no RTTI in release (`-ffast-math -O3 -DNDEBUG -fno-rtti`); exceptions allowed (ORT C++ API `Ort::Session::Run()` throws `Ort::Exception`)
- RT-safe: no allocations on audio thread — `thread_local std::vector` with `reserve()` in `processBlock()`; `resetState()` uses deferred atomic clear
- `-Wall -Wextra -Werror=return-type` always on
- Symbols hidden (`CXX_VISIBILITY_PRESET hidden`), `$ORIGIN` RPATH only (no build-tree leak via `BUILD_WITH_INSTALL_RPATH ON`)
- `init()` returns bool (never throws); callers use `if (!engine->init())` pattern
- Release builds stripped via `LINK_FLAGS "-s"`

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml`:
- Builds Release on `ubuntu-22.04`, `macos-14`, and `windows-2022` with all three FetchContent dependencies
- Verifies symbol exports (`clap_entry`, `ARAGetFactory`), RPATH / install name / DLL search path, model deployment, onnxruntime deploy
- Runs `clap-validator` against built `.clap` (continues on error — validator may fail without GUI)
- Re-exports and verifies ONNX model via `python3 export_mamba_onnx.py`
- Uploads plugin artifact (`.clap`, `.so`/`.dylib`/`.dll`, `libonnxruntime.*`, `models/`)
- Test job builds Debug on all three platforms with `BUILD_TESTING=ON` and runs CTest

## Verification

```bash
# Build and test (requires model files in models/)
cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

Tests cover: construct/destroy lifecycle, `init()` with model path, `processBlock()` with silence input (checking finite output), and `resetState()` thread safety (deferred clear). Python model export includes inline ONNX verification. For runtime testing, load `build/plugins/MambaSuiteClap.clap` in a CLAP host (Bitwig, Carla, clap-host); models must be in `models/` adjacent to the plugin binary.

## Known Gaps

| Gap | Severity | Notes |
|-----|----------|-------|
| Models are untrained (random weights → noise) | P2 | Requires training pipeline — no training script exists yet |
| No GUI (`CLAP_EXT_GUI`) | P1 | Deferred until model training is complete and JUCE/pugl integration is planned |
| No CLAP validator passing in CI | P4 | Validator downloaded and run; may fail without GUI — `continue-on-error: true` |
| Model training pipeline | P2 | Need training data, loss function, and export pipeline |
| Conditioning vector is zeros — no ARA integration to populate it | P2 | `COND_DIM=8` infrastructure exists but never set from musical context |
| ARA deferred analysis pipeline is a stub | P2 | `contentDirty_` flag set but never consumed — no analysis loop |
| `mamba_on_main_thread` case bodies empty (ModelReload, AnalysisQueueDrain) — queue drains but no real work | P2 | `MambaClapPlugin.cpp:410-417`; ARA `didEnableAudioSourceSamplesAccess` cannot push without host bridge |

## Session Logging — Mandatory at Session Close

A `SESSION_LOG.md` file is maintained in this project's root directory. It is an
**append-only log** of every OpenCode session. At the END OF EVERY SESSION (when
the user says goodbye, wrap up, or when no more tasks remain), the agent MUST:

1. Load the `session-log` skill (`skill` tool)
2. Use it to write a new entry to `SESSION_LOG.md`
3. Update `AGENTS.md` if the logging section is missing
4. Save key findings to agentmemory

The user should be able to start a session by reading ONLY `SESSION_LOG.md` —
no need to ask "what did we do last time?".

## Key Design Constraints

- `Ort::Value` has NO default constructor — use `thread_local std::vector` with `reserve()` instead of `std::array`
- `ARA::PlugIn::PlugInEntry::getPlugInEntry<F, D>()` creates static singletons — cannot inject custom constructor args
- ARA 2.0 `PlaybackRendererInterface` has no `renderSamples()` method; audio processing is companion CLAP plugin only
- Plugins resolve model path via `mambaResolveModulePath()` (Platform.hpp abstraction over `dladdr`/`GetModuleFileName`/`_NSGetExecutablePath`) relative to their own binary; models must be in `models/` subdirectory adjacent to plugin
- `MAMBA_EXPORT` macro (Platform.hpp) handles symbol visibility across platforms: ELF hidden/visibility(default), DLL export, macOS weak import
- Runtime library (`libonnxruntime.so`, `onnxruntime.dll`, `libonnxruntime.dylib`) must be deployed alongside plugins (resolved via platform-specific search path)
