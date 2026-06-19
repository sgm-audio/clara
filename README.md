# CLARA — Mamba Audio Suite

C++17 audio plugin that runs a Mamba-3 SSM (Selective State Space Model) via a custom C++ inference engine. Produces dedicated CLAP effect plugins — one per trained task — and an ARA 2.0 extension library that share a single inference core. Supports Linux (stable), macOS and Windows (in progress).

No ONNX Runtime is used in the plugins. Weights are stored as a flat binary file (`model.bin`) and loaded via mmap at init.

## Features

- Real-time Mamba-3 SSM inference via custom C++ engine (~757K params)
- Dedicated CLAP plugins per effect model (Saturator, Compressor, Denoiser, Transparent, etc.)
- ARA 2.0 library with document controller, playback renderer, and host archive state persistence
- Thread-safe deferred state reset via `std::atomic` — zero locks on the audio thread
- Zero heap allocations on the audio thread after warm-up (`thread_local` + `reserve()` pattern)
- Model path resolved automatically via `dladdr()` / `GetModuleFileName()` / `_NSGetExecutablePath()` — no environment variables or config files
- Click-free bypass via 50ms linear crossfade (per-sample ramp, shared across channels)
- Self-contained deployment: plugin + shared engine + model in one directory
- All C++ dependencies fetched at CMake configure time (no system package installation)

## Architecture

```
DAW Host
│
├── CLAP Plugin (one per model)
│   e.g. "Mamba Saturator.clap", "Mamba Compressor.clap"
│   │
│   ├── Audio In → SsmInferenceCore → Audio Out
│   │              (6 layers, MIMO rank 4)
│   │
│   └── Conditioning Vector (128 floats)
│       [0-7: DAW context] [8-39: CLAP embedding]
│       [40-71: user knobs] [72-127: reserved]
│
└── ARA Extension (MambaSuiteARA.so)
    Reads tempo/key/time signature/section markers
    Writes cond vector slots 0-7 in real-time

Main Thread (non-RT) — Phase 2/3 features
├── CLAP Text Encoder  → 32-float embedding → cond slots 8-39  (planned)
├── CLAP Audio Encoder → 32-float embedding → cond slots 8-39  (planned)
└── Preset system      → precomputed embeddings → cond slots 8-39  (planned)
```

The inference engine lives in `src/core/` as the `Mamba3Core` static library, linked into both the CLAP and ARA targets. Each plugin loads its own `model.bin` (dedicated weights per task). CLAP text + audio encoders and the shared encoder library are Phase 2/3 features not yet implemented.

**Flat binary model contract:** one input file (`model.bin`, version 2) with a 64-byte header followed by 13 float32 arrays per layer. The engine reads weights at init and constructs `SSMBlock` objects with placement-new. Recurrent state is stack-allocated — no heap allocations on the audio thread.

## Build

Dependencies fetched automatically via `FetchContent`: CLAP SDK (`29ffcc27`) and ARA SDK (`1ca0b0aa`). No manual dependency installation required. ONNX Runtime is **not** used.

Release build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Debug build with tests:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel && ctest --test-dir build --output-on-failure
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
cmake --install build                         # both CLAP + ARA
cmake --install build --component clap        # CLAP only
cmake --install build --component ara         # ARA only
```

Outputs in `build/plugins/`:

- `Mamba<Effect>.clap` — CLAP audio effect (one per model)
- `MambaSuiteARA.so` / `MambaSuiteARA.dylib` / `MambaSuiteARA.dll` — ARA2 library
- `libMamba3Core.a` — static C++ engine
- `models/*.bin` — flat binary weights

## Running

Load `build/plugins/Mamba<Effect>.clap` in a CLAP-compatible host (Bitwig Studio, Carla, clap-host). The flat binary model must be present in `models/` adjacent to the plugin binary — the build system deploys it automatically.

ARA hosts (REAPER, Cubase) discover `MambaSuiteARA.so`. Audio processing is handled by the companion CLAP plugin (ARA 2.0 has no `renderSamples()` method).

Each plugin exposes three parameters: bypass (stepped, click-free via 50ms linear crossfade), wet/dry blend, and intensity.

## Model

Architecture: Mamba-3 with multi-head MIMO R=4.

| Parameter    | Value | Notes                                            |
|--------------|-------|--------------------------------------------------|
| d_model      | 128   | Embedding dimension                              |
| n_layers     | 6     | Stacked SSM layers                               |
| d_state      | 32    | SSM state dimension per channel                  |
| nheads       | 4     | Parallel heads                                   |
| headdim      | 64    | Per-head dimension (= d_inner / nheads)          |
| mimo_rank    | 4     | Slots per head — sub/bass/mids/highs             |
| d_inner      | 256   | 2× d_model expansion                             |
| cond_dim     | 128   | DAW(8) + CLAP(32) + user knobs(32) + reserved(56)|
| Total params | ~757K | Small enough for single-core RT at 48kHz         |

Each `SSMBlock` implements an 8-step per-sample pipeline: FiLM conditioning, fused `in_proj`, A/Δ preprocessing, RMSNorm on B/C, RoPE rotation, multi-head MIMO SSM recurrence (trapezoidal discretization), SiLU z-gating, output projection + D skip + residual, and periodic state normalization every 4096 samples.

### Training

See [`docs/TRAINING.md`](docs/TRAINING.md) for the complete training pipeline and multi-stage curriculum. The quick path:

```bash
# Stage 0: identity pretraining
python train_clara_mamba3.py --data ~/datasets/musdb18hq/train --device cuda --steps 100000

# Export to flat binary weights
python train_clara_mamba3.py --export checkpoints/final.pt --output models/model.bin
```

The only supported training script is `train_clara_mamba3.py`. Older scripts (`train_clara_parallel.py`, `train_clara_export.py`, `export_mamba_onnx.py`, `train_clara_now.py`) produce incompatible architectures and should not be used.

## Project Structure

```
├── CMakeLists.txt                    # Build system (FetchContent, dual-target)
├── AGENTS.md                         # Build/architecture reference
├── MAMBA3_MIGRATION_PLAN.md          # Complete architecture & migration plan
├── MI300X_TRAINING_PLAN.md           # Cloud training launch reference
├── TRAINING_QUICKSTART.md            # One-command quick start
├── train_clara_mamba3.py             # Mamba-3 training script
├── reference_forward.py              # PyTorch bit-exact gold standard
├── scripts/
│   ├── export_mamba_weights.py       # Flat binary weight export
│   ├── validate_export.py            # Python↔C++ round-trip validation
│   ├── precompute_features.py        # Mel/STFT precomputation
│   └── setup_cloud.sh                # MI300X one-launch setup
├── ADR-*.md                          # Architectural Decision Records
├── LICENSE                           # Apache 2.0
├── .github/workflows/ci.yml          # CI pipeline
├── models/                           # Flat binary model files
├── src/
│   ├── core/                         # Mamba3Core: SsmInferenceCore, SsmBlock, WeightLoader
│   ├── clap/                         # CLAP plugin entry + extensions
│   ├── ara/                          # ARA document controller + playback renderer
│   └── common/                       # Conditioning, Platform abstractions
├── tests/                            # CTest suite
├── docs/
│   └── TRAINING.md                   # Unified Mamba-3 training reference
└── archive/mamba1-onnx-era/          # Deprecated Mamba-1 / ONNX code, docs, models
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
| `test_mamba3`        | Python↔C++ numerical equivalence, state continuity, long-run stability                |
| `clap_validator`     | Runs [clap-validator](https://github.com/free-audio/clap-validator) against the built `.clap`. Required categories: `audio-ports`, `params`, `state`, `latency`. Returns SKIP (77) if the validator binary is not available. |

The `clap_validator` test is resolved in this order:

1. `find_program(clap-validator)` — local install on `PATH`
2. `cargo install clap-validator --locked` — fetches and builds from crates.io (requires Rust toolchain)
3. Otherwise the test is registered but exits 77 (SKIP) at runtime; CTest reports it as skipped rather than failed.

Tests require model files in `models/`. The CLAP plugin test and the validator test additionally require `Mamba<Effect>.clap` to be present at `build/plugins/Mamba<Effect>.clap` (built by the same `cmake --build` invocation).

To build and run the full test suite (including the validator) in a single step:

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --target clara_test
```

The target name `clara_test` (not `test`, which is reserved by CTest) builds the CLAP and ARA plugins first, then invokes `ctest --output-on-failure`. The clap-validator test is automatically skipped (logged as `[SKIPPED]`) if the validator binary is unavailable.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

- Builds Release on `ubuntu-22.04`, `macos-14`, and `windows-2022` with all FetchContent dependencies
- Verifies symbol exports (`clap_entry`, `ARAGetFactory`), RPATH / install name / DLL search path, model deployment
- Runs `clap-validator` against the built `.clap` (continues on error; may fail without GUI)
- Validates model round-trip via `python3 scripts/validate_export.py`
- Uploads plugin artifact (`.clap`, `.so`/`.dylib`/`.dll`, `libmamba_core.*`, `models/`)
- Debug build with `BUILD_TESTING=ON` runs CTest on all three platforms

## License

Apache 2.0. See `LICENSE`.

## Status

Current phase: **Mamba-3 engine complete, training pipeline ready.** All core CLAP and ARA scaffolding is implemented and builds. See [`MAMBA3_MIGRATION_PLAN.md`](MAMBA3_MIGRATION_PLAN.md) for the full roadmap and [`docs/TRAINING.md`](docs/TRAINING.md) for the training stages.

Known gaps:

- **No DAW load test** — engine works in tests, needs DAW validation
- **No GUI** — `CLAP_EXT_GUI` deferred until model training is underway
- **macOS/Windows builds** — defined in CMake presets but not fully tested
- **Install/packaging** — no DEB, installer, or distro-export script yet
