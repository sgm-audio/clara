# Changelog

## v0.1.2 — 2026-10-04

Build portability. A fresh clone previously could not be configured at all in an
environment that permits `git clone` but blocks GitHub *release-asset* downloads.

### Fixed

- **Configure failed fetching `CPM.cmake`.** clap-wrapper bootstraps CPM by
  downloading `CPM.cmake` from a release asset (`file(DOWNLOAD … EXPECTED_HASH)`,
  `SSL connect error 35` here). CLARA now fetches the same VST3 SDK revision by
  git (`v3.7.6_build_18`) and passes it as `VST3_SDK_ROOT`; clap-wrapper skips CPM
  entirely when that is set. `-DVST3_SDK_ROOT=/path/to/sdk` reuses an existing
  checkout instead of fetching.
- **`libx11-dev` is no longer required.** The native editor includes
  `X11/Xlib.h` directly, so a headless machine could not build the CLAP plugin at
  all (and the MinGW cross-build in the release checklist could never work). CMake
  now probes for X11 and defines `CLARA_NO_NATIVE_GUI` when it is missing; the
  plugin compiles without the editor and hosts fall back to their generic UI.
- **New `-DCLARA_BUILD_VST3=OFF`** builds the CLAP plugin, ARA library and tests
  without clap-wrapper or the VST3 SDK.

### Verification

- Fresh clone, clean container, no flags beyond the CMake 4.x policy shim:
  configure, build and `ctest` all succeed — **4/4 tests pass**, including
  `test_clap_plugin`, which now genuinely builds and instantiates the plugin
  (previously it could not compile here and was reported as *Not Run*).
- All three plugin artefacts build: `MambaSuiteClap.clap`, `MambaSuiteARA.so`,
  `MambaSuite.vst3`.
- Zero-allocation guarantee re-proved on the refactored engine: 0 heap
  allocations across 4,096 `process_block()` samples (`LD_PRELOAD` counter).

## v0.1.1 — 2026-10-04

Correctness, reproducibility and performance pass, after an external review that
built the project, ran its tests and benchmarked the engine
([`docs/BLUEPRINT_COMPARISON.md`](docs/BLUEPRINT_COMPARISON.md)).

### Fixed — the repository could not be used from a fresh clone

- **Model weights were unreachable.** `models/*.bin` were Git LFS pointers and the
  account's LFS budget was exceeded (HTTP 403 for every object). A clone therefore
  contained 132-byte text files: the plugin failed to load and `test_mamba3`
  aborted with `std::bad_alloc`. LFS is removed for these files and the model is
  committed directly; `models/README.md` records the lost blobs and their oids.
- **`test_mamba3` aborted on malformed input** instead of failing. Fixtures are now
  length-validated before allocation, and `WeightLoader` rejects any file whose
  size does not exactly match the declared payload.
- **The "bit-exact" test could never pass.** It compared the R=4 time-domain engine
  against fixtures from the R=8 spectral `PerTrackProcessor` — different models. It
  is replaced by `test_python_reference`, which runs the engine's own model through
  `python/reference_forward.py`; agreement is 1.0–1.8e-3 abs / 5–9e-4 relative.
- **`python/reference_forward.py` was a `SyntaxError`** (multi-line generator in an
  f-string) and had never been importable. It also disagreed with the engine in five
  places — RoPE pairing, `B_bias`/`C_bias`, Δ conditioning, a broadcast bug in the
  D-skip, and state-normalization scope. All five now match the shipping engine.
- **CI was red on `main`**: the clap-validator download pointed at an asset that has
  never existed in any release. Pinned to the published 0.4.1 ubuntu-22.04 asset;
  `libx11-dev` added (the native GUI needs Xlib); model-pointer checks added on
  Linux and Windows.

### Changed — engine execution (no model change)

- The host block is processed layer-major in chunks of 128 samples: FiLM/Δ
  conditioning projections are evaluated once per chunk, and `in_proj`/`out_proj`
  run as register-blocked GEMMs (`gemm_blocked<BJ,BI,K>`). The SSM recurrence
  remains per-sample because it is causal. Scratch is engine-owned — the audio path
  still performs zero heap allocations.
- RoPE now computes `sin`/`cos` once per (head, angle) instead of once per
  (head, rank, angle) — 4x fewer transcendental calls per layer per sample.
- **Performance: 0.379 → 0.262 ms/sample mono (-31%), 0.419 → 0.259 ms stereo.**
  Still far from real time (~1,250% of one core per 48 kHz mono stream); see below.
- New `PROCESS_CHUNK_SIZE` (128, scheduling) is distinct from `MAX_CHUNK_SIZE`
  (64, model semantics, must match training).

### Added

- `test_perf_budget` — per-sample cost against a same-machine calibration loop,
  failing CI if the ratio exceeds 3.0. Real-time cost now has a guard; previously
  every numerical test could pass while the engine got slower.
- `test_numeric_sensitivity` — reports drift under a 1-ULP input perturbation.
- `python/tools/dump_engine_reference.py` — generates the Python↔C++ fixtures from
  the model the engine actually loads, with provenance in `ref_meta.txt`;
  `dump_reference_output.py` is now a deprecated shim.
- `models/README.md`, `docs/BLUEPRINT_COMPARISON.md`.

### Known issues carried forward

- **Not real time.** ~12–24x too slow for stereo; the remaining lever is model size
  (`d_model`, `in_proj`, MIMO rank), not kernel tuning — the GEMM is already at the
  CPU's floating-point issue limit.
- **Numerical drift.** The float32 recurrence diverges from a float64 oracle by
  2.0% of peak within 12 ms and 7.3% within 93 ms; peak state saturates the
  4096-sample clamp, so different CPUs will produce different audio.
- **`models/model.bin` is an untrained fixture**, not the trained autoencoder the
  v0.1.0 notes described (the trained blob was byte-identical to the fixture by oid,
  and is now unavailable).


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
