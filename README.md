# CLARA — Mamba Audio Suite

[![CI](https://github.com/SGM-Studios/clara/actions/workflows/ci.yml/badge.svg)](https://github.com/SGM-Studios/clara/actions/workflows/ci.yml)

C++17 audio plugin that runs a Mamba-3 SSM (Selective State Space Model) via a custom C++ inference engine. Ships as a CLAP plugin (`MambaSuiteClap.clap`) and an ARA 2.0 extension library (`MambaSuiteARA.so`), both sharing a single inference core. Supports Linux (stable), macOS and Windows (in progress).

No ONNX Runtime is used. Weights are stored as a flat binary file (`model.bin`) and loaded via mmap at init.

**Start here:** [Build the plugin](#build) · [Train a model](#training) ([`docs/TRAINING.md`](docs/TRAINING.md) reference, [`docs/TRAIN_GUIDE.md`](docs/TRAIN_GUIDE.md) for the R=8 target arch) · [Run in a DAW](#running)

## Status — what works right now

This is what an incubator leader testing the plugin will see:

| Capability | Status |
|---|---|
| Builds on Linux | ✅ `cmake -B build && cmake --build build` |
| VST3 for Ableton Live | ✅ `MambaSuite.vst3` built via clap-wrapper |
| CLAP for Bitwig/Carla | ✅ `MambaSuiteClap.clap` |
| ARA 2.0 extension | ✅ `MambaSuiteARA.so` (scaffolding, no real-time conditioning yet) |
| Processes stereo audio | ✅ 3 params: bypass, wet/dry, intensity |
| RT-safe (zero heap alloc on audio thread) | ✅ |
| Click-free bypass (50ms linear crossfade) | ✅ |
| Trained model | ✅ Identity autoencoder — briefly trained, audible effect |
| GUI | ❌ Deferred (CLAP_EXT_GUI) — generic host editor only |
| Per-effect model variants | ❌ Planned (Saturator, Compressor, Denoiser) |
| macOS/Windows builds | 🟡 CMake presets defined, not fully tested |
| ARA real-time conditioning | 🟡 Scaffolding only — no host content reader yet |

## Architecture

```
DAW Host
│
├── CLAP Plugin (MambaSuiteClap.clap) / VST3 (MambaSuite.vst3)
│   │
│   ├── Audio In → SsmInferenceCore → Audio Out
│   │              (6 layers, MIMO rank 4, time-domain)
│   │
│   └── Conditioning Vector (128 floats)
│       [0-127: reserved for future DAW context / CLAP embedding]
│
└── ARA Extension (MambaSuiteARA.so)
    Scaffolding only — content reader not yet implemented
```

**Current engine:** Mamba-3 R=4 time-domain — RoPE rotation, trapezoidal SSM, multi-head MIMO rank 4, d_model=128, d_state=32, 6 layers, ~1.66M trainable params. Per-sample processing (no STFT in the model — STFT is used only in the training loss).

**Next milestone:** PerTrackProcessor R=8 spectral-domain with STFT frontend, complex-pole SSM, CREPE f0 conditioning, cross-track attention, and ISTFT decoder (~4.26M params). See [`docs/MAMBA3_MIGRATION_PLAN.md`](docs/MAMBA3_MIGRATION_PLAN.md) for the full roadmap.

The inference engine lives in `src/core/` as the `Mamba3Core` static library, linked into both the CLAP and ARA targets. The plugin loads `model.bin` (version 2 flat binary weights).

## Build

Dependencies fetched automatically via `FetchContent`: CLAP SDK (`29ffcc27`), ARA SDK (`releases/2.3.0`), and clap-wrapper (`v0.9.1`). No manual dependency installation required. ONNX Runtime is **not** used.

Release build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default build produces the CLAP plugin and ARA library. The VST3 wrapper is a separate target (it pulls in the VST3 SDK):

```bash
cmake --build build --target MambaSuite_as_vst3 --parallel
```

Debug build with tests:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel && ctest --test-dir build --output-on-failure
```

Cross-platform builds via CMake presets:

```bash
cmake --preset default             # Linux (Release)
cmake --preset macos-release       # macOS
cmake --preset windows-msvc-release  # Windows
cmake --build --preset <preset>
```

Install (default prefix `/usr/local`):

```bash
cmake --install build                         # both CLAP + ARA
cmake --install build --component clap        # CLAP only
cmake --install build --component ara         # ARA only
```

Outputs:

- `build/plugins/MambaSuiteClap.clap` — CLAP audio effect
- `build/Release/MambaSuite.vst3` — VST3 bundle (loadable in Ableton Live, Cubase, REAPER, etc.; built by the `MambaSuite_as_vst3` target)
- `build/plugins/MambaSuiteARA.so` / `.dylib` / `.dll` — ARA2 library
- `build/libMamba3Core.a` — static C++ engine
- `build/plugins/models/*.bin` — flat binary weights

## Running

**Ableton Live (VST3):** Copy the `build/Release/MambaSuite.vst3` bundle to your VST3 plugin folder:
- Linux: `~/.vst3/` or `/usr/lib/vst3/`
- macOS: `~/Library/Audio/Plug-Ins/VST3/`
- Windows: `C:\Program Files\Common Files\VST3\`

**CLAP hosts (Bitwig Studio, Carla, clap-host):** Load `build/plugins/MambaSuiteClap.clap` directly.

The flat binary model must be present in `models/` adjacent to the plugin binary. The build deploys it automatically for the CLAP (`build/plugins/models/`); the VST3 bundle does **not** yet embed it — copy `build/plugins/models/` next to the `.so` inside the bundle (`Contents/x86_64-linux/`) after installing.

ARA hosts (REAPER, Cubase) discover `MambaSuiteARA.so`. Audio processing is handled by the companion CLAP plugin (ARA 2.0 has no `renderSamples()` method).

Each plugin exposes three parameters: bypass (stepped, click-free via 50ms linear crossfade), wet/dry blend, and intensity.

## Model

**Current shipping model:** Mamba-3 R=4 time-domain (multi-head MIMO, RoPE, trapezoidal SSM).

| Parameter    | Value | Notes                                            |
|--------------|-------|--------------------------------------------------|
| d_model      | 128   | Embedding dimension                              |
| n_layers     | 6     | Stacked SSM layers                               |
| d_state      | 32    | SSM state dimension per channel                  |
| nheads       | 4     | Parallel heads                                   |
| headdim      | 64    | Per-head dimension (= d_inner / nheads)          |
| mimo_rank    | 4     | MIMO rank (time-domain)                          |
| d_inner      | 256   | 2× d_model expansion                             |
| cond_dim     | 128   | Conditioning vector (reserved for future use)    |
| Total params | ~1.66M | 13 arrays/layer × 6 layers                       |

**Target architecture:** PerTrackProcessor R=8 spectral-domain (~4.26M params). See [`docs/MAMBA3_MIGRATION_PLAN.md`](docs/MAMBA3_MIGRATION_PLAN.md).

### Training

The R=4 time-domain model is trained with `python/training/train_local_radeon.py`:

```bash
# Stage 0: identity pretraining (local ROCm GPU)
python python/training/train_local_radeon.py \
  --data ~/datasets/musdb18hq/train \
  --steps 50000 --batch 4 --grad-accum 4 \
  --chunk-samples 2048

# Export to flat binary weights
python python/training/train_local_radeon.py \
  --export checkpoints_clara/best.pt \
  --output models/model.bin
```

The canonical PerTrackProcessor R=8 training script is `python/training/train_clara.py` (spectral-domain, complex poles). It is the target architecture; the C++ engine rewrite to match it is the next milestone.

## Project Structure

```
├── CMakeLists.txt                    # Build system (FetchContent, dual-target)
├── AGENTS.md                         # Build/architecture reference
├── docs/
│   ├── adr/                          # Architectural Decision Records (ADR-001..013)
│   ├── MAMBA3_MIGRATION_PLAN.md      # Complete architecture & migration plan
│   ├── TRAINING.md                   # Unified Mamba-3 training reference
│   ├── TRAINING_QUICKSTART.md        # One-command quick start
│   └── ...
├── python/
│   ├── reference_forward.py          # PyTorch bit-exact gold standard
│   ├── training/
│   │   ├── train_clara.py          # PerTrackProcessor R=8 training (target arch)
│   │   ├── train_local_radeon.py   # R=4 time-domain training (current engine)
│   │   ├── dataset.py               # StemDataset + DataLoader
│   │   └── loss.py                   # MR-STFT + Mel-L1 + SI-SDR losses
│   └── tools/
│       ├── export_mamba_weights.py   # Flat binary weight export (V2)
│       ├── validate_export.py        # Python↔C++ round-trip validation
│       └── ...
├── scripts/
│   ├── run_train.sh                  # Training launcher (ROCm)
│   ├── run_test.sh                   # Smoke test launcher
│   └── setup_local.sh                # Local ROCm env setup
├── LICENSE                           # Apache 2.0
├── .github/workflows/ci.yml          # CI pipeline
├── models/                           # Flat binary weights: model.bin = briefly-trained identity
│                                     #   autoencoder; *_ref.bin = seed-42 test fixtures (regenerated
│                                     #   by tests/CMakeLists.txt if deleted)
├── src/
│   ├── core/                         # Mamba3Core: SsmInferenceCore, SsmBlock, WeightLoader
│   ├── clap/                         # CLAP plugin entry + extensions
│   └── ara/                          # ARA document controller + playback renderer
├── tests/                            # CTest suite
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
| `test_mamba3`        | Python↔C++ numerical equivalence, state continuity, long-run stability                |
| `test_clap_plugin`   | CLAP plugin descriptor, lifecycle, parameter registration, buffer handling           |
| `test_ara_controller`| ARA document controller creation/destruction, archive round-trip                      |
| `clap_validator`     | Runs [clap-validator](https://github.com/free-audio/clap-validator) against the built `.clap`. Required categories: `audio-ports`, `params`, `state`, `latency`. Returns SKIP (77) if the validator binary is not available. |

The `clap_validator` test is resolved in this order:

1. `find_program(clap-validator)` — local install on `PATH`
2. `cargo install --git https://github.com/free-audio/clap-validator --locked` — builds from git master (the last tagged release doesn't compile on current Rust; requires Rust toolchain)
3. Otherwise the test is registered but exits 77 (SKIP) at runtime; CTest reports it as skipped rather than failed.

Tests require model files in `models/`. The CLAP plugin test and the validator test additionally require `MambaSuiteClap.clap` to be present at `build/plugins/MambaSuiteClap.clap` (built by the same `cmake --build` invocation).

To build and run the full test suite (including the validator) in a single step:

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --target clara_test
```

The target name `clara_test` (not `test`, which is reserved by CTest) builds the CLAP and ARA plugins first, then invokes `ctest --output-on-failure`. The clap-validator test is automatically skipped (logged as `[SKIPPED]`) if the validator binary is unavailable.

## Code Quality

This project uses [pre-commit](https://pre-commit.com) to enforce consistent formatting and catch common issues before they reach CI:

```bash
# Install hooks (one-time)
pre-commit install

# Run on all files (after clone or adding new hooks)
pre-commit run --all-files
```

Hooks include:
- **clang-format** — C++ formatting (`.clang-format`)
- **black** — Python formatting (100-char lines)
- **ruff** — Python linting (auto-fix enabled)
- **trailing-whitespace**, **end-of-file-fixer**, **mixed-line-ending** — file hygiene
- **check-added-large-files**, **check-merge-conflict**, **check-case-conflict** — safety checks

The `.clang-format` config uses LLVM style with 4-space indent, 120-column limit, right-aligned pointers/references, and C++17 standard.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

- Builds Release on `ubuntu-22.04`, `macos-14`, and `windows-2022` with all FetchContent dependencies
- Verifies symbol exports (`clap_entry`, `ARAGetFactory`), RPATH / install name / DLL search path, model deployment
- Runs `clap-validator` against the built `.clap` (continues on error; may fail without GUI)
- Validates model round-trip via `python3 python/tools/validate_export.py`
- Uploads plugin artifacts (`.clap`, `.vst3`, `.so`/`.dylib`/`.dll`, `models/`)
- Debug build with `BUILD_TESTING=ON` runs CTest on all three platforms

## License

Apache 2.0. See `LICENSE`.

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| Current | R=4 time-domain engine + CLAP/VST3/ARA | ✅ Shipping |
| Next | C++ engine rewrite to R=8 PerTrackProcessor (spectral, complex poles) | 🟡 Planned (see `docs/MAMBA3_MIGRATION_PLAN.md`) |
| Phase 2 | Per-effect model variants (Saturator, Compressor, Denoiser) | 🟡 Planned |
| Phase 3 | CLAP text/audio encoders, preset system | 🟡 Planned |
| Phase 4 | ARA real-time conditioning, content reader | 🟡 Planned |
