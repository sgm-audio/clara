# CLARA TODO

## 2026-10-04 STATUS — read this first

The phases below are the **historical** 2026-07-02 audit-remediation list. Most of
Phase 0 and Phase 1 are done; the file was never pruned. Current open work is the
list immediately below, and the measurements behind it are in
`docs/BLUEPRINT_COMPARISON.md`.

### P0 — reproducibility ✅ (done 2026-10-04)

- [x] Weights committed to Git, Git LFS removed (`models/README.md` documents the lost oids)
- [x] `WeightLoader` rejects files whose size does not exactly match the declared payload
- [x] `test_mamba3` validates fixture sizes before allocating (no more `std::bad_alloc`)
- [x] CI: pinned clap-validator asset, `libx11-dev`, model-pointer check
- [x] **Fresh clone builds with no manual steps** (verified 2026-10-04): the VST3
      SDK is fetched by git and handed to clap-wrapper as `VST3_SDK_ROOT`, so
      clap-wrapper's CPM bootstrap — which downloads `CPM.cmake` from a GitHub
      *release asset* and breaks in sandboxes that only allow `git` — never runs.
      `-DCLARA_BUILD_VST3=OFF` skips the VST3 path entirely.
- [x] **X11 no longer mandatory**: the native editor is compiled out
      (`CLARA_NO_NATIVE_GUI`) when the X11 headers are missing, so plugins and all
      four tests build on a headless container; the MinGW/X11 worry is gone too.
- [x] `test_python_reference` replaces the impossible "bit-exact" comparison
- [x] README / AGENTS / INCUBATOR_CHECKLIST synced with reality

### P1 — blocking for real time

- [ ] **Shrink the model to fit a CPU budget.** Measured cost is ~1,250% of one
      core per 48 kHz mono stream (0.262 ms/sample); real time needs < 100% for
      stereo, so ~12–24x. `in_proj` (1612×128) is 61% of the MACs per sample.
      Levers: `d_model` 128 → 32–48, `in_proj` width, MIMO rank 4 → 1–2, 6 → 4
      layers. Kernel tuning is exhausted (1.45x total, and the GEMM is at the
      CPU's issue limit) — this is a modelling decision, not a code one.
- [ ] **Fix the float32 drift.** 2.0% of peak divergence vs a float64 oracle within
      12 ms, 7.3% within 93 ms; peak `|ssm_real|` saturates the 4096-sample clamp.
      Options: double accumulation in the recurrence, or fix the dynamics so the
      clamp never fires. Add a drift assertion once it is fixed.
- [ ] **Trained model.** `models/model.bin` is an untrained fixture. The trained
      blobs were lost with the LFS budget; retrain and regenerate the reference
      fixtures (`python/tools/dump_engine_reference.py`).

### P1.5 — packaging & CI (found during the 2026-10-04 review)

- [ ] **CI is not running.** Every run created since 2026-10-04 (pushes,
      `pull_request`, scheduled CodeQL) concludes `startup_failure` with 0 jobs
      and `path=BuildFailed`; the last run that executed anything was
      2026-07-10. `.github/workflows/ci.yml` parses as YAML, is registered
      `active` and is byte-identical to `main`, so this is a repository/account
      level condition (Actions spending limit/billing, or Actions disabled) —
      needs someone with admin rights, not a workflow edit. Until then every CI
      claim in `CHANGELOG.md`/README is unverified.
- [ ] **VST3 wrapper cannot find the CLAP as built.** clap-wrapper derives the
      CLAP file name from the wrapper binary stem (`MambaSuite` →
      `MambaSuite.clap`); the build produces `MambaSuiteClap.clap`. Reproduced:
      `GetPluginFactory()` returns `nullptr` with only the built name present,
      and works as soon as the same binary is also reachable as
      `MambaSuite.clap`. Fix options: rename the CLAP target/output to
      `MambaSuite.clap`, or install a copy/symlink under the expected name in
      the VST3 bundle.
- [ ] **Install components are broken** (see README *Install*):
      `--component clap` exits 1 (the bare `.clap` file collides with the
      `.clap` bundle dir it is installed next to), and `--component vst3`
      ignores `--prefix` and writes into the build tree. `--component ara`
      works.
- [ ] **`-ffast-math` and NaN/Inf checks.** Release builds compile with
      `-ffast-math` (`-ffinite-math-only`), so `std::isfinite`/`std::isnan` are
      folded to constants; the param entry points and the test suite now use the
      bit-level `mambaIsFinite` (`src/core/Platform.hpp`) instead. If more
      non-finite guards are added, they must use that helper.
- [ ] **Stale personal files at the repository root** (proposed): `SESSION_LOG.md`
      (104 KB), `omo-config.json` (unreferenced agent-tool config) and the
      tracked-but-ignored `.agent_context/nrinstruct.md` (session notes from
      2026-06-04 describing the old architecture). Move them under `docs/` or
      untrack them.

### P2 — product evidence

- [ ] One flagship effect, trained, with a same-loudness A/B in a real DAW
- [ ] DAW load test (never performed on this codebase)
- [ ] CPU% and latency numbers printed by the plugin itself
- [ ] Presets, installer, manual, demo limitation

### P3 — optional / after P1

- [ ] Vectorize the SSM recurrence by reordering state to `[h][s][d]` (~34% of
      per-sample cost; the inner loop currently strides and cannot vectorize)
- [ ] H3: R=8 PerTrackProcessor engine rewrite — **reconcile with P1 first**: at
      4.26M params it is further from real time than the current 1.66M model
- [ ] Per-effect variants, CLAP text/audio encoders
- [ ] macOS/Windows verification

---

## Historical: 2026-07-02 portfolio audit remediation

**Archived context — kept for traceability.** Many boxes below are stale.

**Created:** 2026-07-02 (from comprehensive audit)
**Architecture:** R=8 PerTrackProcessor (spectral, complex poles, ~4.26M params) — canonical
**C++ engine:** R=4 time-domain Mamba-3 — needs rewrite to R=8 (H3)

---

## PHASE 0: Rename canonical training script + update all references [IN PROGRESS]

### 0.1 Rename files
- [ ] `python/training/train_clara.py` → `python/training/train_clara.py`
- [ ] `python/tools/train_clara_kaggle.py` → `python/tools/train_clara_kaggle.py`

### 0.2 Update references in documentation (34 refs across 10+ files)
- [ ] `README.md` (5 refs)
- [ ] `AGENTS.md` (7 refs)
- [ ] `docs/TRAINING.md` (14 refs)
- [ ] `docs/TRAINING_QUICKSTART.md` (2 refs)
- [ ] `docs/TRAINING_SETUP_CONTEXT.md` (5 refs)
- [ ] `docs/MAMBA3_MIGRATION_PLAN.md` (3 refs)
- [ ] `docs/PHASE2_ROADMAP.md` (2 refs)
- [ ] `docs/CLARA_OVERHAUL_PLAN.md` (3 refs)
- [ ] `docs/ARCHITECTURE_COMPARISON.md` (2 refs)
- [ ] `docs/MAMBA3_DRIVE_AUDIT.md` (5 refs)
- [ ] `docs/SESSION_SUMMARY_2026-06-21.md` (1 ref)

### 0.3 Update shell scripts
- [ ] `scripts/run_train.sh`
- [ ] `scripts/bootstrap_local.sh`

### 0.4 Update Python internal references
- [ ] `python/tools/test_model.py` (any import refs)
- [ ] `python/tools/validate_export.py` (any refs)
- [ ] `python/training/loss.py` (import refs)
- [ ] `python/training/dataset.py` (import refs)

---

## PHASE 1: Documentation accuracy — fix every stale claim

### 1.1 Fix parameter count (757K → ~4.26M PerTrackProcessor)
- [ ] `README.md` lines 9, 112
- [ ] `AGENTS.md` lines 16, 72, 77, 149, 155, 160
- [ ] `docs/MAMBA3_MIGRATION_PLAN.md` lines 117, 284, 357
- [ ] `docs/ARCHITECTURE_COMPARISON.md` line 37, 228
- [ ] `docs/CLARA_OVERHAUL_PLAN.md` line 188
- [ ] `docs/SESSION_SUMMARY_2026-06-21.md` line 20

### 1.2 Fix CMake preset names in docs
- [ ] `README.md`: `linux-release` → `default`, `windows-release` → `windows-msvc-release`
- [ ] `AGENTS.md`: same
- [ ] Add `linux-release`/`windows-release` aliases to `CMakePresets.json`

### 1.3 Fix plugin naming (Mamba<Effect>.clap → MambaSuiteClap.clap)
- [ ] `README.md` lines 3, 10, 24, 85, 92, 190
- [ ] `AGENTS.md` lines 12, 59, 141
- [ ] `docs/MAMBA3_MIGRATION_PLAN.md` lines 16, 353, 406-418, 687-709
- [ ] `docs/CLARA_OVERHAUL_PLAN.md` lines 1602, 1607

### 1.4 Fix migration plan stale references (MAMBA3_MIGRATION_PLAN.md)
- [ ] `libmamba_core.so` (shared lib) → `libMamba3Core.a` (static)
- [ ] `libmamba_clap.so` (600MB) → remove (CLAP encoder is future work)
- [ ] Install layout with per-plugin dirs → update to single-plugin layout
- [ ] `RingBuffer.h`, `DoubleBuffer.h` → `SPSCQueue.hpp`
- [ ] `src/common/ConditioningGenerator.hpp` → `src/core/ConditionProcessor.h`
- [ ] Training cost estimates → mark as estimated

### 1.5 Clean stale ONNX references in C++ comments
- [ ] `src/core/SsmInferenceCore.h:4`
- [ ] `src/clap/MambaClapPlugin.cpp:3`
- [ ] `src/ara/MambaAraController.cpp:3`

### 1.6 Fix HEADDIM docstring in reference_forward.py
- [ ] `python/reference_forward.py` line 8-9: `headdim=32` → `headdim=64`

---

## PHASE 2: ADR cleanup — mark obsolete, write replacements

### 2.1 Mark as HISTORICAL/SUPERSEDED
- [ ] `docs/adr/ADR-005-thread-local-reserve-ort-value.md` → HISTORICAL (ONNX Runtime removed)
- [ ] `docs/adr/ADR-008-conditioning-vector.md` → HISTORICAL (Mamba-1, COND_DIM=8)
- [ ] `docs/adr/ADR-009.md` → HISTORICAL (ONNX sub-block processing) + add descriptive filename
- [ ] `docs/adr/ADR-010.md` → HISTORICAL (ONNX dual-path conditioning) + add descriptive filename

### 2.2 Partial updates
- [ ] `docs/adr/ADR-002` — replace MambaInferenceEngine refs with SsmInferenceCore/PerBandSSM
- [ ] `docs/adr/ADR-004` — update ONNX Runtime deployment section
- [ ] `docs/adr/ADR-001` — add note pointing to ADR-011

### 2.3 Write new ADRs for PerTrackProcessor era
- [ ] `docs/adr/ADR-011-per-track-processor-architecture.md` — R=8, spectral, complex poles
- [ ] `docs/adr/ADR-012-per-track-processor-weight-format.md` — V3, 18 arrays/layer
- [ ] `docs/adr/ADR-013-condition-vector-v2.md` — d_cond=128, FiLM via CondMLP

### 2.4 Update ADR-WORKFLOW.md
- [ ] Fix filename mismatches for ADR-004, ADR-005
- [ ] Remove ONNX Runtime references
- [ ] Remove MambaInferenceEngine/MambaCore references
- [ ] Add ADR-011, 012, 013 to backlog
- [ ] Remove completed drafting prompts for ADR-003, 004, 005

---

## PHASE 3: C++ engine update to R=8 PerTrackProcessor [BLOCKING]

### 3.1 `src/core/MambaConfig.h`
- [ ] MAX_MIMO_RANK: 4 → 8
- [ ] Add MAX_COMPLEX_POLES, MAX_R8_BANDS constants
- [ ] WEIGHT_VERSION: 2 → 3
- [ ] WEIGHTS_PER_LAYER: 13 → 18
- [ ] Update MAX_IN_PROJ_DIM for R=8
- [ ] Add STFT config constants (n_fft=2048, hop_length=512, n_mels=128)
- [ ] Update HEADER layout comment for v3
- [ ] Update static_assert checks

### 3.2 `src/core/WeightLoader.h/cpp`
- [ ] New header struct for v3 (R8 extractor weights, n_fft, hop_length)
- [ ] 18 arrays per layer layout
- [ ] Frontend weights (R8FeatureExtractor)
- [ ] Update validate_header() for R=8, v3 dimensions
- [ ] Update setup_layer_pointers()
- [ ] Windows portability (#ifdef _WIN32 CreateFileMapping/MapViewOfFile)
- [ ] Verify macOS mmap path

### 3.3 `src/core/SsmBlock.h/cpp`
- [ ] Add complex pole state (log_a_real, a_imag per head per band)
- [ ] Rewrite SSM recurrence: complex trapezoidal
- [ ] Remove RoPE rotation
- [ ] PerBandSSM: iterate over R=8 bands
- [ ] Update SSMState struct

### 3.4 `src/core/SsmInferenceCore.h/cpp`
- [ ] Add STFT frontend (Hann window, FFT)
- [ ] Add mel filterbank
- [ ] Add R8FeatureExtractor
- [ ] Add ISTFT output
- [ ] Frame-based processing (remove per-sample)
- [ ] Update latency: 0 → n_fft/2

### 3.5 Tests
- [ ] Update test_mamba3.cpp for R=8 spectral domain
- [ ] Generate new model_ref.bin in v3 format
- [ ] Generate new input_ref.bin, output_ref.bin
- [ ] Update test_clap_plugin.cpp expected latency

### 3.6 Model regeneration
- [ ] Export v3 weights: `python train_clara.py --export checkpoints/best.pt --output models/model.bin`
- [ ] Smoke test: build + ctest pass

---

## PHASE 4: C++ code quality

### 4.1 Thread safety fix (H1)
- [ ] Add retirement queue for old engines during model reload
- [ ] Atomic counter for audio thread completion signaling
- [ ] Remove redundant seq_cst fence (L5)

### 4.2 Remove dead code
- [ ] Delete `src/core/ConditioningFrame.h` (entire file, never used)
- [ ] Delete `src/core/MambaConfig.cpp:10-16` (validate_config stub, never called)
- [ ] Delete `src/core/Platform.hpp:68-106` (mambaResolveModelDir, 39 lines, never called)
- [ ] Delete empty for-loop in SsmBlock.cpp:141-148
- [ ] Delete or wire up SSMBlock::process_block() stub

### 4.3 Portability fixes
- [ ] `using namespace` in headers → explicit qualification (MambaClapPlugin.hpp, MambaAraController.hpp)
- [ ] `strcmp` → `std::strcmp` (6 occurrences in MambaClapPlugin.cpp)
- [ ] `strncpy` → add explicit null-termination (7 occurrences)
- [ ] WeightLoader POSIX-only → add #ifdef _WIN32 path

### 4.4 Version unification
- [ ] CMakeLists.txt: VERSION 1.1.0 → 1.2.0 (or generate version.h via configure_file)
- [ ] Verify CLAP descriptor and ARA factory match

### 4.5 Miscellaneous
- [ ] Fix "64 bytes" comment in WeightLoader.cpp (actual is 52)
- [ ] Fix chunked-vs-per-sample comment in SsmInferenceCore.cpp
- [ ] Verify MAMBA_PROJECT_URL in Platform.hpp

---

## PHASE 5: Python code cleanup

### 5.1 Remove duplicates
- [ ] Delete `python/tools/train_local_radeon.py` (keep `python/training/train_local_radeon.py`)

### 5.2 Fix missing dependencies
- [ ] `CachedMelDataset` — implement or remove import from train_local_radeon.py, train_clara_kaggle.py
- [ ] `psutil` — add to requirements or wrap in try/except with warning

### 5.3 Fix validate_export.py
- [ ] Docstring says "Version 2" → "Version 3" (code already expects V3)

### 5.4 Clean stale artifacts
- [ ] `find . -type d -name __pycache__ -exec rm -rf {} +`
- [ ] `rm -rf scripts/checkpoints/`

---

## PHASE 6: Build system & CI

### 6.1 CMake presets
- [ ] Add `linux-release` → inherits `default` (backwards compat)
- [ ] Add `windows-release` → inherits `windows-msvc-release` (backwards compat)

### 6.2 CI improvements
- [ ] Add macos-14 build job
- [ ] Add macos-14 test job
- [ ] Test job: download build artifact instead of rebuilding
- [ ] test-windows: add Python validation (validate_export.py)
- [ ] Pin clap-validator to a version

### 6.3 CMakeLists.txt fixes
- [ ] generate_reference_outputs: remove ALL target, make test dependency
- [ ] test_clap_plugin: link Mamba3Core static lib instead of recompiling source
- [ ] Reconsider -ffast-math for test builds (may break bit-exact equivalence)

---

## PHASE 7: Repo hygiene

### 7.1 .gitignore additions
- [ ] Add `.env*` and `.env.*`
- [ ] Add `.agent_context/`

### 7.2 Cleanup
- [ ] Commit or discard docs/CHANGELOG.md unstaged changes
- [ ] `rm -rf build/` (stale ONNX Runtime artifacts)
- [ ] Rebuild from clean: `cmake -B build -DBUILD_TESTING=ON`

---

## PHASE 8: Final verification

- [ ] `grep -r "train_clara.py" --include="*.md" --include="*.py" --include="*.sh"` returns ZERO results
- [ ] `grep -r "757K\|757,032" --include="*.md" --include="*.py"` returns ZERO results (except historical ADRs)
- [ ] `grep -r "linux-release" --include="*.md"` returns ZERO results (except preset alias docs)
- [ ] `cmake --preset default` builds successfully
- [ ] `cmake -B build -DBUILD_TESTING=ON && cmake --build build --parallel && ctest --test-dir build` passes
- [ ] Python smoke test passes
- [ ] train_clara.py --export produces valid v3 model.bin
- [ ] C++ engine loads exported model.bin
- [ ] No `using namespace` in any .h/.hpp file
- [ ] No dead code files (ConditioningFrame.h, validate_config, mambaResolveModelDir)
- [ ] CI passes on ubuntu-22.04, macos-14, windows-2022
- [ ] validate_export.py models/model.bin passes
