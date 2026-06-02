# CLARA — Mamba Audio Suite

C++17 audio plugin that runs a Mamba SSM (Selective State Space Model) via ONNX Runtime. Produces a CLAP effect plugin and an ARA 2.0 extension library that share a single inference core. Supports Linux (stable), macOS and Windows (in progress).

## Features

- Real-time Mamba SSM inference via ONNX Runtime CPU provider (~274K params)
- CLAP plugin with stereo I/O, bypass (50ms click-free crossfade), wet/dry blend, and intensity control
- ARA 2.0 library with document controller, playback renderer, and host archive state persistence
- Thread-safe deferred state reset via `std::atomic` — zero locks on the audio thread
- Zero heap allocations on the audio thread after warm-up (`thread_local` + `reserve()` pattern)
- Model path resolved automatically via `dladdr()` — no environment variables or config files
- Self-contained deployment: plugin + ONNX Runtime shared lib + model in one directory
- All dependencies fetched at CMake configure time (no system package installation)

## Architecture

```
Mamba SSM            →   ONNX Runtime         →   CLAP Plugin
(d_model=64,             (CPU provider,             (MambaSuiteClap.clap)
 n_layers=4,              v1.17.1,
 d_state=16,              graph opt level all)        ARA Library
 conv_width=4)                                          (MambaSuiteARA.so)
                              ↕
                       MambaCore (static lib)
                       MambaInferenceEngine
                       - RT-safe ONNX wrapper
                       - Pre-allocated buffers in init()
                       - thread_local Ort::Value vectors
                       - Deferred atomic state reset
```

The inference engine lives in `src/core/` as the `MambaCore` static library, compiled once and linked into both the CLAP and ARA targets.

**ONNX model I/O contract:** 9 inputs (`audio_chunk` + 4x conv state + 4x SSM state) → 3 outputs (`processed_audio`, stacked conv state, stacked SSM state). The engine unstacks output states per-layer each block.

## Build

Dependencies fetched automatically via `FetchContent`: CLAP SDK (1.2.7), ARA SDK, ONNX Runtime v1.17.1 (prebuilt packages for Linux/macOS/Windows). No manual dependency installation required.

Release build:
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

Debug build with tests:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

Install (default prefix `/usr/local`):
```bash
cmake --install .                         # both CLAP + ARA
cmake --install . --component clap        # CLAP only
cmake --install . --component ara         # ARA only
```

Outputs in `build/plugins/`:
- `MambaSuiteClap.clap` — CLAP plugin
- `MambaSuiteARA.so` / `MambaSuiteARA.dylib` / `MambaSuiteARA.dll` — ARA library
- `models/` — ONNX model files
- `libonnxruntime.so.1.17.1` / `onnxruntime.dll` / `libonnxruntime.1.17.1.dylib` — resolved via platform RPATH / search path

## Running

Load `build/plugins/MambaSuiteClap.clap` in a CLAP-compatible host (Bitwig Studio, Carla, clap-host). The ONNX model and runtime library must be present in `models/` adjacent to the plugin binary — the build system deploys them automatically.

ARA hosts (REAPER, Cubase) discover `MambaSuiteARA.so`. Audio processing is handled by the companion CLAP plugin (ARA 2.0 has no `renderSamples()` method).

The plugin exposes three parameters: bypass (stepped, click-free via 50ms linear crossfade), wet/dry blend, and intensity.

## Model

Architecture based on Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023):

| Parameter    | Value | Notes                                    |
|--------------|-------|------------------------------------------|
| d_model      | 64    | Embedding dimension                      |
| n_layers     | 4     | Stacked SSM layers                       |
| d_state      | 16    | SSM state dimension per channel          |
| conv_width   | 4     | Causal convolution kernel width          |
| dt_rank      | 16    | Discretization projection rank           |
| Total params | ~274K | Small enough for single-core RT at 48kHz |

Export:
```bash
python3 export_mamba_onnx.py
```

Exports `models/mamba_rt.onnx` + `mamba_rt.onnx.data` (external data sidecar) using `torch.jit.trace` with opset 18. The script self-verifies — runs ONNX inference, checks shapes, and validates state continuity across chunks. Requires `torch`, `onnx`, `onnxruntime`, `numpy`.

The model uses dynamic `seq_len` to avoid zero-padding corruption of SSM recurrent state. Default block size is 256 samples (~5.3ms at 48kHz); the CLAP plugin adapts to the host's block size (minimum 256).

## Project Structure

```
├── CMakeLists.txt                    # Build system (FetchContent, dual-target)
├── AGENTS.md                         # Build/architecture reference
├── export_mamba_onnx.py             # Model definition + ONNX export
├── TODO.md                           # Project status and gaps
├── ADR-*.md                          # Architectural Decision Records
├── LICENSE                           # Apache 2.0
├── .github/workflows/ci.yml         # CI pipeline
├── models/                           # ONNX model files
├── src/
│   ├── core/
│   │   ├── MambaInferenceEngine.hpp  # Engine interface
│   │   └── MambaInferenceEngine.cpp  # RT-safe ONNX wrapper
│   ├── clap/
│   │   ├── MambaClapPlugin.hpp       # CLAP plugin state
│   │   └── MambaClapPlugin.cpp       # CLAP entry point + processing
│   └── ara/
│       ├── MambaAraController.hpp    # ARA controller interface
│       └── MambaAraController.cpp    # ARA document controller + renderer
└── tests/
    ├── CMakeLists.txt                # Test build config
    └── test_engine.cpp               # Inference engine smoke tests
```

## Testing

```bash
cmake -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Test targets

| CTest name           | Scope                                                                                |
|----------------------|--------------------------------------------------------------------------------------|
| `test_engine`        | Construct/destroy lifecycle, `init()` with model path, `processBlock()` silence, `resetState()` thread safety |
| `test_clap_plugin`   | CLAP plugin descriptor, lifecycle, parameter registration, buffer handling           |
| `test_ara_controller`| ARA document controller creation/destruction, archive round-trip                      |
| `clap_validator`     | Runs [clap-validator](https://github.com/free-audio/clap-validator) against the built `.clap`. Required categories: `audio-ports`, `params`, `state`, `latency`. Returns SKIP (77) if the validator binary is not available. |

The `clap_validator` test is resolved in this order:

1. `find_program(clap-validator)` — local install on `PATH`
2. `cargo install clap-validator --locked` — fetches and builds from crates.io (requires Rust toolchain)
3. Otherwise the test is registered but exits 77 (SKIP) at runtime; CTest reports it as skipped rather than failed.

Tests require model files in `models/`. The CLAP plugin test and the validator test additionally require `MambaSuiteClap.clap` to be present at `build/plugins/MambaSuiteClap.clap` (built by the same `cmake --build` invocation).

To build and run the full test suite (including the validator) in a single step:

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --target clara_test
```

The target name `clara_test` (not `test`, which is reserved by CTest) builds the CLAP and ARA plugins first, then invokes `ctest --output-on-failure`. The clap-validator test is automatically skipped (logged as `[SKIPPED]`) if the validator binary is unavailable.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Builds Release on `ubuntu-22.04` with all FetchContent dependencies
- Verifies symbol exports (`clap_entry`, `ARAGetFactory`) and `$ORIGIN` RPATH
- Validates model deployment and ONNX Runtime adjacency
- Runs `clap-validator` against the built `.clap` (continues on error; may fail without GUI)
- Re-exports and verifies the ONNX model via `export_mamba_onnx.py`
- Uploads plugin artifact (`.clap`, `.so`, `libonnxruntime.so*`, `models/`)
- Debug build with `BUILD_TESTING=ON` runs CTest

## License

Apache 2.0. See `LICENSE`.

## Status

Current phase: **functional prototype** — all core CLAP and ARA scaffolding is implemented and builds. Known gaps (see `TODO.md`):

- **Models are untrained** (random weights produce noise) — no training pipeline exists (P2)
- **No GUI** — `CLAP_EXT_GUI` deferred until model training is underway (P1)
- **No CLAP validator passing in CI** — validator runs but may fail without GUI (P4)
- **Several ARA stubs** — `doUpdateMusicalContextContent`, `doUpdateAudioSourceContent`, `didEnableAudioSourceSamplesAccess` are no-ops (P2)
- **No editor views** — `doCreateEditorView` / `doCreateEditorRenderer` return nullptr (P2)
- **CLAP init and start_processing** — `mamba_init` and `mamba_start_processing` are no-ops (P2)

## Cross-Platform Migration

Cross-platform support is in progress. Key changes:
- `Platform.hpp` in `src/core/` abstracts platform-specific APIs: module path resolution (`mambaGetModulePath()`), symbol visibility (`MAMBA_EXPORT`/`MAMBA_IMPORT`), thread naming, and filesystem operations.
- macOS: `dlopen`/`dlsym` with `.dylib` extension, Mach-O visibility attributes, `@rpath` install names.
- Windows: `LoadLibrary`/`GetProcAddress` with `.dll` extension, `__declspec(dllexport/import)`, `SetThreadDescription` for thread naming.
- ONNX Runtime prebuilt packages are downloaded per-platform via FetchContent URL selection in `CMakeLists.txt`.
- See `ADR-007-cross-platform-portability.md` for full details.
