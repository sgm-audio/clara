# CLARA — Mamba Audio Suite (CLAP + ARA2)

C++17 audio plugin running a Mamba SSM (Selective State Space Model) via ONNX Runtime. Linux x64 only. Produces a CLAP plugin and an ARA2 shared library sharing one inference core.

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

`src/clap/MambaClapPlugin` — Full CLAP plugin. Registers extensions: params, state, latency (4 samples), audio-ports (stereo). 3 atomics: bypass (stepped), wet/dry, intensity. Click-free bypass via 50ms linear crossfade (per-sample ramp, shared across channels to prevent stereo aliasing). Uses `dladdr` for model path resolution relative to `.clap` binary. Uses `std::make_unique` + `.release()` for C-ABI lifetime.

`src/ara/MambaAraController` — DocumentController + PlaybackRenderer + `extern "C" ARAGetFactory()` with `visibility("default")`. ARA 2.0 has NO `renderSamples()` — audio processing is exclusively through the companion CLAP plugin. Content stubs documented with proper `enable` handling in `willEnableAudioSourceSamplesAccess`. Archive state persists 3 floats (bypass, wetDry, intensity) via `HostArchiveReader/Writer`.

ONNX model: 9 inputs (`audio_chunk` + 4×conv state + 4×ssm state) → 3 outputs (`processed_audio`, stacked conv state, stacked ssm state). The engine unstacks output states per-layer each block.

## C++ Conventions

- C++17, no RTTI in release (`-ffast-math -O3 -DNDEBUG -fno-rtti`); exceptions allowed (ORT C++ API `Ort::Session::Run()` throws `Ort::Exception`)
- RT-safe: no allocations on audio thread — `thread_local std::vector` with `reserve()` in `processBlock()`; `resetState()` uses deferred atomic clear
- `-Wall -Wextra -Werror=return-type` always on
- Symbols hidden (`CXX_VISIBILITY_PRESET hidden`), `$ORIGIN` RPATH only (no build-tree leak via `BUILD_WITH_INSTALL_RPATH ON`)
- `init()` returns bool (never throws); callers use `if (!engine->init())` pattern
- Release builds stripped via `LINK_FLAGS "-s"`

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml`:
- Builds Release on `ubuntu-22.04` with all three FetchContent dependencies
- Verifies symbol exports (`clap_entry`, `ARAGetFactory`), RPATH (`$ORIGIN`), model deployment, onnxruntime deploy
- Runs `clap-validator` against built `.clap` (continues on error — validator may fail without GUI)
- Re-exports and verifies ONNX model via `python3 export_mamba_onnx.py`
- Uploads plugin artifact (`.clap`, `.so`, `libonnxruntime.so*`, `models/`)
- Test job builds Debug with `BUILD_TESTING=ON` and runs CTest

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
| No `clap-validator` passing in CI | P4 | Downloaded and run but may fail — blocker for merge confidence |

## Key Design Constraints

- `Ort::Value` has NO default constructor — use `thread_local std::vector` with `reserve()` instead of `std::array`
- `ARA::PlugIn::PlugInEntry::getPlugInEntry<F, D>()` creates static singletons — cannot inject custom constructor args
- ARA 2.0 `PlaybackRendererInterface` has no `renderSamples()` method; audio processing is companion CLAP plugin only
- Plugins resolve model path via `dladdr()` relative to their own binary; models must be in `models/` subdirectory adjacent to plugin
- `libonnxruntime.so` must be deployed alongside plugins (resolved via `$ORIGIN` RUNPATH)
