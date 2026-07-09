# Changelog

## v0.1.0 — 2026-07-09

First tagged release. Version numbering starts at 0.1.0: the earlier internal
"1.x" numbers (preserved in [`docs/CHANGELOG.md`](docs/CHANGELOG.md)) predated
any release and overstated maturity.

### What this is

A working, Linux-stable audio plugin suite running a custom Mamba-3 SSM
inference engine — no ONNX Runtime, pure C++ over mmap'd flat binary weights.
The shipping model is an identity autoencoder, briefly trained, with an
audible effect: a proof of the engine and pipeline, not a finished product.

### Engine

- Mamba-3 R=4 time-domain SSM: RoPE rotation, trapezoidal integration,
  multi-head MIMO rank 4 (d_model=128, d_state=32, 6 layers, ~1.66M params)
- Per-sample processing, zero heap allocation on the audio thread
- Flat binary weight format (V2) with header validation and mmap loading
  (heap read on Windows)
- Bit-exact Python↔C++ equivalence test plus 60s long-run stability test

### Plugin targets

- `MambaSuiteClap.clap` — CLAP audio effect (bypass, wet/dry, intensity;
  click-free 50ms bypass crossfade)
- `MambaSuite.vst3` — VST3 via clap-wrapper (loads in Ableton Live)
- `MambaSuiteARA.so` — ARA 2.0 extension library (scaffolding; no real-time
  conditioning yet)

### Training pipeline (Python)

- `python/training/train_local_radeon.py` — trains the current R=4 engine
  (local ROCm), exports `model.bin`
- `python/training/train_clara.py` — R=8 PerTrackProcessor target
  architecture (spectral-domain; C++ engine rewrite is the next milestone)
- MR-STFT + Mel-L1 + SI-SDR losses, round-trip export validation

### Infrastructure

- CMake + FetchContent (CLAP SDK, ARA SDK 2.3.0, clap-wrapper v0.9.1),
  presets for Linux/macOS/Windows
- CI on ubuntu-22.04 / macos-14 / windows-2022: build, symbol checks,
  clap-validator, model round-trip validation, CTest
- pre-commit: clang-format, black, ruff
- Mamba-1/ONNX era fully retired to `archive/mamba1-onnx-era/`

### Known limitations

- GUI deferred — generic host editor only
- macOS/Windows builds compile in CI but are not hand-tested in a DAW
- ARA content reader not implemented
- Single briefly-trained model; per-effect variants (Saturator, Compressor,
  Denoiser) are planned, not built
