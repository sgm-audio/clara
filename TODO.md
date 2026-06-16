# CLARA — TODO (2026-06-16)

**Mamba-1 is dead. Mamba-3 migration in progress.** See `MAMBA3_MIGRATION_PLAN.md` for the full architectural plan. This TODO reflects the current Mamba-3 migration status only.

**Build status:** ✅ Working (CMake 4.3.3 + GCC 16, with workarounds)
**Tests:** ✅ 3/4 pass (`test_latency` fails — known: conv_width=4 model has 0 actual latency)
**Current phase:** Phase 0 — Configuration & Types

---

## Phase 0: Configuration & Types (✅ DONE)

| # | Task | Status |
|---|------|--------|
| P0.1 | Create `src/core/MambaConfig.h` — all compile-time dimension constants | ✅ |
| P0.2 | Create `src/core/MambaConfig.cpp` — validation + serialization (minimal stub) | ✅ |
| P0.3 | Fix CMake build: CMake 4.3.3 + GCC 16 workarounds (C_COMPILE_OBJECT, -fPIC) | ✅ |
| P0.4 | HEADDIM=64 (=D_INNER/NHEADS), state = 16,960 floats/layer, 101,760 total | ✅ |

## Phase 1: Python Reference Forward (✅ DONE)

| # | Task | Status |
|---|------|--------|
| P1.1 | Create `reference_forward.py` — full 8-step per-sample pipeline | ✅ |
| P1.2 | All 6 verification checks pass: finite, non-constant, cond-sensitive, deterministic, stable | ✅ |
| P1.3 | LayerNorm per-layer prevents value explosion across 6 layers | ✅ |

## Phase 2: Weight Export Pipeline

| # | Task | Status |
|---|------|--------|
| P2.1 | Create `scripts/export_mamba_weights.py` — Version 2 flat binary format | ⬜ |
| P2.2 | Create `scripts/validate_export.py` — round-trip validation | ⬜ |
| P2.3 | Test: export random model → validate → load in Python → same output | ⬜ |

## Phase 3: C++ SsmBlock Implementation

| # | Task | Status |
|---|------|--------|
| P3.1 | Create `src/core/SsmBlock.h` — SSMState, SSMWeights, SSMBlock class | ⬜ |
| P3.2 | Create `src/core/SsmBlock.cpp` — per-sample 8-step pipeline | ⬜ |
| P3.3 | Create `src/core/WeightLoader.h/cpp` — mmap-based version 2 loader | ⬜ |
| P3.4 | **Bit-exact test**: C++ SsmBlock output == Python reference (same random weights) | ⬜ |

## Phase 4: Multi-Layer Engine

| # | Task | Status |
|---|------|--------|
| P4.1 | Create `src/core/SsmInferenceCore.h/cpp` — 6-layer wrapper, state management | ⬜ |
| P4.2 | Create `src/core/ConditionProcessor.h/cpp` — FiLM per-layer conditioning | ⬜ |
| P4.3 | Block processing: 64-sample chunks, state carry-over | ⬜ |

## Phase 5: CLAP Plugin Adaptation

| # | Task | Status |
|---|------|--------|
| P5.1 | Replace `MambaInferenceEngine` with `SsmInferenceCore` in `MambaClapPlugin.hpp` | ⬜ |
| P5.2 | Update `init()` / `processBlock()` / `resetState()` to use new engine | ⬜ |
| P5.3 | Update `ClapLatency.cpp` — return 0 samples (no conv1d) | ⬜ |
| P5.4 | Update `Platform.hpp` — model path resolution for `.bin` files | ⬜ |
| P5.5 | Expand `ConditioningGenerator` from 8-dim to 128-dim | ⬜ |

## Phase 6+: ARA Bridge, CLAP Encoders, Training (later phases)

See `MAMBA3_MIGRATION_PLAN.md` sections J.6–J.10 for detailed tasks.

---

## Known Issues

| Issue | Severity | Fix |
|-------|----------|-----|
| `test_latency` fails (measured=0, reported=4) | 🟡 Low | Fixed when Mamba-3 engine replaces ONNX (conv1d is gone) |
| CMake 4.3.3 + GCC 16 requires `CMAKE_C_COMPILE_OBJECT` workaround | 🟡 Med | Guard in CMakeLists.txt. Remove when CMake ≥ 4.3.4 |
| ARA `ARADebug.c` needs explicit `-fPIC` on Debug builds | 🟢 Low | Added to `CMAKE_C_FLAGS`. Upstream fix preferred. |
| No Mamba-3 checkpoints exist (Mamba-1 checkpoints are incompatible) | 🔴 Blocking | Blocked on Phase 8 (training pipeline) |
| `IN_PROJ_DIM` = 1612 (Python) vs migration plan's 419 | 🟡 Doc | Migration plan number is approximate/outdated. Our computed value is correct. |

---

## Deleted (Mamba-1 Artifacts)

These files exist in the repo but are dead code per the migration plan.
Delete them when the new engine ships:

- `src/core/MambaInferenceEngine.{hpp,cpp}` — ONNX Runtime wrapper
- `train_clara_parallel.py`, `train_clara_now.py`, `train_clara_export.py` — Mamba-1 training
- `export_mamba_onnx.py` — ONNX export (incompatible)
- `models/*.onnx` — ONNX model files (replaced by `.bin`)
- All ONNX Runtime FetchContent/link/deploy in CMakeLists.txt
