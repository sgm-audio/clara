# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CLARA is a C++17 real-time audio plugin (CLAP + ARA2) running a custom Mamba-3
SSM inference engine, plus a Python training/export pipeline. For architecture,
ADRs, conditioning-vector layout, and the migration/phase plan, read
`AGENTS.md` and `docs/MAMBA3_MIGRATION_PLAN.md` first — this file only goes
deep on build and test mechanics.

## Build

All C++ dependencies (CLAP SDK, ARA SDK) are fetched at configure time via
`FetchContent` — no system package installation needed. There is no ONNX
Runtime dependency anywhere in this repo; inference is pure C++ over flat
binary weights (`model.bin`, mmap'd at init).

Plain CMake, no presets:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
# with tests:
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build --parallel
```

Via `CMakePresets.json` (preset names are host-gated — `default`/`debug`
resolve on Linux, `macos-*` on Darwin, `windows-*` on Windows; the *only*
platform this repo currently builds and tests on is Linux, the rest are
defined but unverified per `docs/TODO.md` gap G12):

```bash
cmake --preset default     # Linux Release -> build/
cmake --preset debug       # Linux Debug + BUILD_TESTING=ON -> build/
cmake --build --preset <preset-name-if-defined>   # only windows-msvc-debug has one; otherwise: cmake --build build --parallel
```

`linux-release` / `windows-release` are back-compat aliases for
`default` / `windows-msvc-release`.

Outputs land in `build/plugins/`:
- `MambaSuiteClap.clap` — CLAP effect
- `MambaSuiteARA.so` — ARA2 library
- `libMamba3Core.a` — static inference engine
- `models/*.bin` — weights copied alongside the plugin (plugin resolves this
  path at runtime relative to its own binary — model files MUST sit next to
  the plugin, not in an arbitrary location)

Install: `cmake --install build` (or `--component clap` / `--component ara`
to install just one plugin format). Default prefix `/usr/local`.

## Test

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run a single test by name (all `add_test` names are exact, no globbing needed
but `-R` accepts regex):

```bash
ctest --test-dir build -R test_mamba3 --output-on-failure
ctest --test-dir build -R test_clap_plugin --output-on-failure
ctest --test-dir build -R test_ara_controller --output-on-failure
ctest --test-dir build --print-labels     # list available test labels
```

Four CTest targets, defined in `tests/CMakeLists.txt`:

| Test | Covers |
|---|---|
| `test_mamba3` | Engine construct/destroy, `init()` with a model path, `processBlock()` on silence (finite-output check), `resetState()` thread safety, **bit-exact Python↔C++ numerical equivalence**, 60s @ 44.1kHz long-run stability (no NaN/Inf) |
| `test_clap_plugin` | CLAP plugin lifecycle, params/state/latency/audio-ports extensions, bypass crossfade |
| `test_ara_controller` | ARA DocumentController + PlaybackRenderer + archive state persistence (bypass/wetDry/intensity round-trip) |
| `clap_validator` | External `clap-validator` run against the built `.clap` (see below) |

**Test-time codegen (automatic, happens on first `cmake --build`):**
`tests/CMakeLists.txt` shells out to `python3` to generate fixtures if they
don't already exist:
- `python/tools/export_mamba_weights.py --seed 42` → `models/model.bin` (used by `test_mamba3`)
- `python/tools/dump_reference_output.py --seed 42` → `models/input_ref.bin`,
  `models/output_ref.bin`, `models/model_ref.bin` — the Python reference
  forward pass that `test_mamba3` diffs the C++ engine's output against
  bit-exactly. This is the mechanism that catches Python↔C++ divergence; if
  you change the engine's math you MUST also update/regenerate these.

If `python3` isn't found on `PATH`, CMake emits a `WARNING` and the affected
tests degrade (missing `model.bin` means `test_mamba3` has nothing to load).
Delete `models/model.bin`, `models/input_ref.bin`, `models/output_ref.bin`,
`models/model_ref.bin` to force regeneration (e.g. after changing the export
format or the engine's forward pass).

**`clap_validator` test** (external tool, optional):
- Resolution order: `clap-validator`/`clap_validator` on `PATH` → else
  `cargo install --git https://github.com/free-audio/clap-validator --locked`
  (from git master, not crates.io — the last tagged release doesn't compile
  on current Rust) → else the test is registered as **SKIP** (exit code 77),
  not a failure.
- Default is **quick mode**: only the four required categories
  (audio-ports, params, state, latency) at the structural/lifecycle level,
  filtering out slow cases that invoke `process()` (fuzz, param-fuzz,
  state-reproducibility, etc.) — a full pass takes many minutes.
- `-DCLAP_VALIDATOR_FULL=ON` at configure time runs the complete, slow suite
  (intended for nightly CI or manual debugging, not routine local runs).

**Manual/DAW testing** (no automated coverage — gap G5 in `docs/TODO.md`):
load `build/plugins/MambaSuiteClap.clap` in a CLAP host (Bitwig, Carla,
`clap-host`); `models/` must be adjacent to the plugin binary.

## CI

`.github/workflows/ci.yml` builds Release on `ubuntu-22.04`, `macos-14`, and
`windows-2022`, verifies exported symbols (`clap_entry`, `ARAGetFactory`) and
RPATH/install-name/DLL-search-path correctness, runs `clap-validator`
(continue-on-error — it can fail without a GUI), validates the model
round-trip via `python3 python/tools/validate_export.py`, and uploads plugin
artifacts. A separate job builds Debug with `BUILD_TESTING=ON` on all three
platforms and runs CTest.

## Python side (training / export)

Canonical training entry point — **all other training scripts
(`train_clara_parallel.py`, `train_clara_export.py`, `export_mamba_onnx.py`,
`train_clara_now.py`, `train_local_radeon.py`) are dead/legacy, do not use
them**:

```bash
python python/training/train_clara.py --data ~/datasets/musdb18hq/train --device cuda --steps 100000
python python/training/train_clara.py --export checkpoints/final.pt --output models/model.bin
```

Smoke-test the model without a full training run:

```bash
scripts/run_test.sh          # activates venv_rocm, sets HSA_OVERRIDE_GFX_VERSION, runs python/tools/test_model.py
```
`python/tools/test_model.py` is a 7-section check: ROCm env, model build,
forward pass, bf16 AMP, backward pass, memory, chunked-vs-full equivalence.

`scripts/setup_local.sh` creates the `venv_rocm` virtualenv and installs
ROCm PyTorch + deps for local (non-cloud) work; `scripts/bootstrap_local.sh`
is the documented ROCm cloud-training launch path. There is no
`pyproject.toml`/`setup.py` — everything is run as a script against
`venv_rocm`, not installed as a package.

## Code quality gate

```bash
pre-commit install
pre-commit run --all-files
```
Runs `clang-format` (C++, LLVM style, 4-space, 120-col, `.clang-format`),
`black` (Python, 100-char) and `ruff` (Python lint, auto-fix via `ruff.toml`),
plus whitespace/EOL/large-file/merge-conflict hooks. Run this before
committing — CI does not currently re-run formatting checks itself, `ctest`
does not check style.
