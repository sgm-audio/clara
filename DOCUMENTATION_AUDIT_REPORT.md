# CLARA — Documentation Audit & ADR Completeness Report

**Generated:** 2026-06-07  
**Scope:** All ADR-*.md, AGENTS.md, README.md, SESSION_LOG.md, TODO.md, CHANGELOG.md, docs/*.md, source code cross-reference

---

## 1. Missing ADRs

Architectural decisions in code that have **no ADR**:

| # | Decision | Where in Code | Severity |
|---|----------|--------------|----------|
| 1 | **Sub-block processing (64-sample chunks)** | `MambaInferenceEngine.hpp:64`: `kSubBlockSize = 64`; `MambaInferenceEngine.cpp:188-232`: while-loop processing across sub-blocks with state carry-over | **P1** — Major departure from ADR-001's 256-sample seq_len design. The export model is seq_len=64, the engine loops sub-blocks, and dynamic seq_len is fixed within each sub-block. No ADR documents why 64 was chosen or the tradeoffs. |
| 2 | **`ORT_ENABLE_BASIC` not `ORT_ENABLE_ALL`** | `MambaInferenceEngine.cpp:24`: `SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC)` | **P2** — ADR-001 §7 claims `ORT_ENABLE_ALL`. Code uses BASIC. Unknown whether this was an intentional change or drift. |
| 3 | **Conditioning folded into ONNX as 10th input** | `MambaInferenceEngine.hpp:81`: `kNumInputs = 10`; inputNames includes `"cond"` | **P1** — ADR-008 explicitly says "does not change ONNX I/O contract" and "9 inputs / 3 outputs". Code added the cond tensor as 10th input. External gain modulation (`combinedWeights_`, `loadCondWeights()`) also still exists alongside the ONNX-integrated cond. This dual-path conditioning is undocumented. |
| 4 | **`ConditioningFrame.h` (not `ConditioningGenerator.hpp`) used by the engine** | `MambaInferenceEngine.hpp:11` includes `"ConditioningFrame.h"`; `ConditioningGenerator.hpp` is a parallel class used only by CLAP/ARA for encoding | **P3** — Two conditioning types (`ConditioningFrame` and `ConditioningGenerator`) coexist. Their relationship and division of responsibility is undocumented. |
| 5 | **Double-buffered `ConditioningFrame` with atomic pointer swap (not index swap)** | `MambaInferenceEngine.hpp:87-89`: `frameA_`/`frameB_` + `std::atomic<ConditioningFrame*> activeFrame_` | **P3** — ADR-008 §2 describes `std::atomic<float*> activeCond_` pointing to a float buffer. The actual implementation points to a `ConditioningFrame` struct (containing `std::array<float, 8>` + atomic sequence counter). Different type, same pattern. |
| 6 | **ModelReload implemented (not empty stub)** | `MambaClapPlugin.cpp:408-434` creates new engine, inits, swaps with `modelReloadPending` fence | **P2** — AGENTS.md Known Gaps says "case bodies empty" but ModelReload is fully implemented including engine creation, conditioning setup, and failure handling. |
| 7 | **Training moved to `train_clara_now.py` monolithic script** | Only 5 training files exist: `train_clara_now.py`, `train_clara_model.py`, `train_clara_export.py`, `train_clara.ps1`, `train_clara_requirements.txt` | **P2** — TODO.md claims 10 modular training files (config, dataset, dsp, loss, etc.), but the modular pipeline was replaced by a single `train_clara_now.py`. The `verify_training_pipeline.py` still references the old modular API. No ADR covers the consolidation. |
| 8 | **Chunked parallel SSM scan (C=64 chunks)** | `train_clara_now.py:94-136`: chunked scan replacing Heinsen log-cumsum | **P3** — This is documented in SESSION_LOG but has no ADR. The numerical stability problem (NaN at step 0) and chunked solution is architecture-significant. |
| 9 | **ROCm-specific workarounds** | `clip_seconds=0.5` OOM mitigation, `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `num_workers=0` to prevent deadlock, `amp=False` default | **P3** — These are RTFM-level details scattered across SESSION_LOG. A consolidated "ROCm GPU Training Environment" doc is missing. |
| 10 | **ONNX manifest validation at init()** | `MambaInferenceEngine.cpp:106-148`: `validateManifest()` reads `manifest.json`, cross-checks architecture params | **P2** — Referenced in CHANGELOG (CLARA-08) and TODO but no ADR. The manifest format (schema v2), validation behavior (warnings only, not errors), and contract are undocumented architectural decisions. |

---

## 2. Stale ADRs

ADRs that describe things no longer matching the code:

| ADR | Claim | Reality | Severity |
|-----|-------|---------|----------|
| **ADR-001 §3** | "9 inputs → 3 outputs" | `kNumInputs = 10` (includes `cond`) | **P0** — Core architectural contract is wrong. References throughout: AGENTS.md line 65, README.md line 35, `docs/parameter-reference.md` line 189, `docs/training-pipeline.md` line 44 all repeat the 9-input claim. |
| **ADR-001 §3** | "`{1, seq_len, 1}` shape" with dynamic seq_len | Engine processes in fixed 64-sample sub-blocks (`kSubBlockSize = 64`). Dynamic seq_len at export time is `seq_len=64`, not variable at inference. | **P1** |
| **ADR-001 §4** | "maxBlockSize = 256 (default)" | `kSubBlockSize = 64` is the actual processing granularity. `config_.maxBlockSize` is still 256 but only controls the outer `totalSamples` clamp, not ORT inference. | **P2** |
| **ADR-001 §6** | "GraphOptimizationLevel ORT_ENABLE_ALL" | Code uses `ORT_ENABLE_BASIC` | **P2** |
| **ADR-001 line 147** | `libonnxruntime.so.1.17.1` deployment contract shows `$ORIGIN` RPATH only | Code has platform-conditional RPATH (`$ORIGIN` on Linux, `@loader_path` on macOS, none on Windows). ADR-001 deployment contract is Linux-only. | **P2** |
| **ADR-006** | `dladdr`-based resolution | Code now uses `Platform.hpp`'s `mambaResolveModulePath()` which wraps `GetModuleHandleExA` on Windows, `dladdr` on POSIX. ADR-006 is Linux-only, superseded by ADR-007 Phase 0. | **P2** |
| **ADR-008 §1** | `COND_DIM = 8` | Both `ConditioningGenerator::COND_DIM = 8` and `ConditioningFrame::kCondDim = 8` exist. | **P4** (consistent, just redundant) |
| **ADR-008 §4** | "External application — does not change ONNX I/O contract" | Code added `cond` as 10th ONNX input AND kept external gain modulation (`loadCondWeights()`, `combinedWeights_`). | **P0** — ADR-008 contradicts itself with implementation reality. |
| **ADR-008 §3** | "Projection weights loaded at `activate()`, immutable in RT" | `loadCondWeights()` is called from `init()` (line 50), which may be called from `activate()` via the CLAP plugin. But the weights live in config (`deltaWeights`, `bWeights`), not loaded from a separate source. AGENTS.md says "`COND_DIM` scalars" but code has two separate vectors combined element-wise. | **P2** |
| **ADR-008 Compliance** | "External gain modulation loop removed" | `combinedWeights_` is still kept in class definition (`MambaInferenceEngine.hpp:94`) with comment "Applied as: output *= (1.0f + sum(cond[d] * combinedWeights_[d]))". Whether this is actually applied in `processBlock()` needs line-level verification. | **P1** |

---

## 3. ADR Gaps — Decisions That Should Be Documented

| # | Decision | Rationale for ADR |
|---|----------|-------------------|
| 1 | **Sub-block processing (64 samples)** | The ORT session creation timeout at 256-step unrolled graph was a major architectural constraint. Moving to seq_len=64 with loop-based sub-block processing changes the computational model, latency guarantees, and ONNX export strategy. |
| 2 | **Conditioning dual-path (ONNX FiLM + external gain)** | If both paths exist, why? Is one deprecated? What happens when both are active? |
| 3 | **Graph optimization level change (ALL → BASIC)** | ORT_ENABLE_BASIC vs ORT_ENABLE_ALL has different memory/compute tradeoffs. Was this intentional or accidental? |
| 4 | **Manifest validation strategy** | Warnings-only approach, schema versioning, which fields are checked vs ignored. |
| 5 | **Training pipeline consolidation** | Why the modular 10-file pipeline was replaced by `train_clara_now.py`. |
| 6 | **`ConditioningFrame` vs `ConditioningGenerator` design** | Two classes, different lifetimes, different consumers. Why two? |
| 7 | **`train_clara_model.py` dual model (parallel scan training, sequential scan export)** | Architecture-significant: training uses parallel scan, export uses sequential. Weights are compatible. |
| 8 | **ModelReload lifecycle** | Engine swap with atomic pending flag, fallback on failure, conditioning preservation. |

---

## 4. Documentation Gaps

### 4.1 Missing from README.md
- **Cross-platform output filenames**: Lists only `MambaSuiteARA.so` but outputs vary by OS (`.dylib`, `.dll`)
- **Windows build instructions**: Only Linux-style `mkdir -p build && cd build`. Windows MSVC build requires different commands.
- **ROCm/GPU training**: Not mentioned at all. The README is purely about the C++ plugin.
- **10-input ONNX contract**: Says 9 inputs
- **Parameter count is stale**: README says ~274K params. Code and training docs say ~77K or ~171K depending on which model.

### 4.2 Missing from AGENTS.md
- **10-input ONNX contract**: Says 9 inputs (line 65)
- **[FIXED] `mambaGetModulePath()` → `mambaResolveModulePath()`**: Lines 63 and 131 in AGENTS.md now use the correct function name.
- **Graph optimization level**: Not documented anywhere in AGENTS.md
- **Sub-block processing**: Not mentioned. Architecture section still implies single-pass processing at maxBlockSize.
- **`train_clara_now.py`**: Not referenced. Training commands reference non-existent scripts.
- **`ConditioningFrame` class**: Not referenced. AGENTS.md mentions only `ConditioningGenerator`.

### 4.3 Stale Parameter Counts Across All Docs

| Document | Model Size Claim | Reality |
|----------|-----------------|---------|
| README.md | ~274K params | ~77K (current ONNX model) |
| ADR-001 §1 | 274K params | ~77K |
| `docs/parameter-reference.md` | ~52K | ~77K |
| `docs/training-pipeline.md` | ~72K | ~77K |
| `docs/training-plan.md` | ~77K (with cond FiLM) | ~77K (correct) |
| SESSION_LOG 2026-06-04 | 76,865 params | ~77K (correct) |
| SESSION_LOG 2026-06-06 | 171,395 params | Different model (`train_clara_now.py` has extra layers) |

### 4.4 Missing Documentation Types
- **No API docs**: No Doxygen, no header comments catalog, no function-level documentation for the public API.
- **No user docs**: No plugin manual, no parameter guide beyond the internal `docs/parameter-reference.md`.
- **No developer onboarding doc**: No "how to contribute," no code walkthrough, no architecture diagram (beyond the ASCII art in README).
- **No training runbook**: `training_commands.md` exists but `TRAINING_COMMANDS.md` (referenced in SESSION_LOG) doesn't. The existing `training_commands.md` references modular pipeline scripts that don't exist (`train_clara_launcher.py`, `train_clara_config.py`, etc.).
- **No model versioning strategy doc**: How ONNX models are versioned, how the manifest works, compatibility guarantees.
- **No ROCm/GPU environment doc**: `HSA_OVERRIDE_GFX_VERSION`, venv setup, Windows-specific caveats scattered across 3+ SESSION_LOG entries.

---

## 5. Known Gaps Table Accuracy (AGENTS.md lines 99-109)

| Gap | AGENTS Claim | Reality Check | Verdict |
|-----|-------------|---------------|---------|
| "Models are untrained (random weights → noise)" P2 | "Requires training pipeline — no training script exists yet" | Training pipeline files exist (`train_clara_now.py`, `train_clara_model.py`, `train_clara_export.py`, `verify_training_pipeline.py`). Phase 1 identity autoencoder training was launched per SESSION_LOG. | **STALE** — Should say "Training pipeline exists but models not yet trained to usable quality" or differentiate which checkpoints exist. |
| "No GUI (`CLAP_EXT_GUI`)" P1 | "Deferred until model training is complete" | `docs/ui-polish-design.md` has a complete 850+ line design. | **ACCURATE** — No code, but design exists. |
| "No CLAP validator passing in CI" P4 | "Validator downloaded and run; may fail without GUI" | Quick-mode validator passes (8 pass / 0 fail / 4 skip). Full mode has process-related failures. | **PARTIALLY ACCURATE** — Should note quick-mode passes. |
| "Model training pipeline" P2 | "Need training data, loss function, and export pipeline" | Training scripts, datasets (MUSDB18-HQ fully downloaded, MAESTRO partial), loss functions, export pipeline all exist and are verified (56/56 checks PASS). | **STALE** — Training pipeline exists and is verified. Gap is now "no trained checkpoint deployed to models/" not "no pipeline". |
| "Conditioning vector is zeros" P2 | "COND_DIM=8 infrastructure exists but never set from musical context" | `ConditioningGenerator::defaultContext()` sets non-zero neutral values in `mamba_activate()`. ARA `doUpdateMusicalContextContent` calls `setConditioning()` with neutral values. No live ARA host bridge yet. | **PARTIALLY ACCURATE** — The cond vector is set to defaults (not zeros), but real musical context from ARA is still missing. |
| "ARA deferred analysis pipeline is a stub" P2 | "contentDirty_ flag set but never consumed" | Still true. `contentDirty_` is set in `doUpdateMusicalContextContent`, `doUpdateAudioSourceContent`, `didEnableAudioSourceSamplesAccess` but never read to trigger analysis. | **ACCURATE** |
| "mamba_on_main_thread case bodies empty (ModelReload, AnalysisQueueDrain)" P2 | "MambaClapPlugin.cpp:410-417" | `ModelReload` is fully implemented (engine creation, init, cond setup, failure fallback). `AnalysisQueueDrain` is still a stub with comments. | **PARTIALLY STALE** — ModelReload is no longer empty. Line references (410-417) are wrong (code is at 408-442). |
| (Missing entry) | — | `mamba_init` and `mamba_start_processing` were no-ops per README but now have validation logic (check model path, check engine). | **MISSING** — These were fixed but the gap was never recorded. |

---

## 6. Key Design Constraints Accuracy (AGENTS.md lines 125-132)

| Constraint | AGENTS.md Text | Code Reality | Verdict |
|-----------|---------------|-------------|---------|
| "`Ort::Value` has NO default constructor" | "use `thread_local std::vector` with `reserve()` instead of `std::array`" | Code matches. `thread_local std::vector<Ort::Value>` with `clear()` + `reserve()` pattern in `processBlock()`. | **ACCURATE** |
| "`ARA::PlugIn::PlugInEntry::getPlugInEntry` creates static singletons" | "cannot inject custom constructor args" | Code matches. Factory uses `ARAPlug.h` wrappers. | **ACCURATE** |
| "ARA 2.0 `PlaybackRendererInterface` has no `renderSamples()`" | "audio processing is companion CLAP plugin only" | Code matches. `MambaPlaybackRenderer` constructor just creates engine, no renderSamples. | **ACCURATE** |
| "Plugins resolve model path via `mambaResolveModulePath()`" | "(Platform.hpp abstraction over `dladdr`/`GetModuleFileName`/`_NSGetExecutablePath`)" | Function name corrected. `_NSGetExecutablePath` fallback added to Platform.hpp for macOS. | **[FIXED]** — Function name and macOS backend corrected. |
| "`MAMBA_EXPORT` macro handles symbol visibility" | "ELF hidden/visibility(default), DLL export, macOS weak import" | Code shows `MAMBA_EXPORT` as `__declspec(dllexport)` (Windows) or `__attribute__((visibility("default")))` (GCC/Clang). No macOS weak import variant. | **PARTIALLY ACCURATE** — "weak import" is inaccurate; it's `visibility("default")`. |
| "Runtime library must be deployed alongside plugins" | "resolved via platform-specific search path" | Code matches. `$ORIGIN` on Linux, `@loader_path` on macOS, side-by-side on Windows. | **ACCURATE** |

---

## 7. Architecture Section Accuracy (AGENTS.md lines 55-69)

| Claim | Line | Reality | Verdict |
|-------|------|---------|---------|
| "Output states use dynamic sequence length" | 57 | Engine processes in fixed 64-sample sub-blocks. Dynamic seq_len is no longer the primary processing mode. | **PARTIALLY ACCURATE** — Dynamic at export, fixed 64 at inference. |
| "latency (4 samples)" | 59 | Code `clap_latency_get` returns 4. `docs/latency-architecture.md` and `tests/test_latency.cpp` verify. | **ACCURATE** |
| "Uses `dladdr` for model path resolution" | 59 | Code uses `mambaResolveModulePath()` from Platform.hpp, which wraps `dladdr` on POSIX and `GetModuleHandleExA` on Windows. | **PARTIALLY ACCURATE** — Implementation detail, not `dladdr` directly. |
| "Platform.hpp abstractions for module path resolution (`mambaResolveModulePath()`)" | 63 | Function name corrected in AGENTS.md. | **[FIXED]** |
| "9 inputs" | 65 | 10 inputs. | **INACCURATE** |
| "Projection weights (`COND_DIM` scalars) loaded at `activate()`" | 67 | `loadCondWeights()` called from `init()`, weights come from Config (two vectors: `deltaWeights`, `bWeights`). | **PARTIALLY ACCURATE** — Calling context and weight source are wrong. |
| "Applied as output gain modulation in `processBlock()`" | 67 | Both ONNX-integrated (cond tensor input) AND external gain modulation exist. Might be one or both. | **PARTIALLY ACCURATE** — Incomplete description. |
| "SPSCQueue.hpp — fixed-capacity lock-free SPSC ring buffer... Capacity 64" | 69 | Code matches. Template `SPSCQueue<T, Capacity=64>` with power-of-2 requirement. | **ACCURATE** |

---

## 8. Code Comments Quality

**Findings from source survey:**

| File | Comment Quality | Notes |
|------|----------------|--------|
| `MambaInferenceEngine.hpp` | **Good** | Member comments explain double-buffering. Public API methods have brief docstrings. `validateManifest()` is undocumented. |
| `MambaInferenceEngine.cpp` | **Sparse** | `processBlock()` has no inline comments explaining the sub-block loop or conditioning application. The `validateManifest()` function and `jsonStringValue`/`jsonIntValue` helpers have no comments. The `resetState()` / `processBlock()` atomic exchange is undocumented in code (only in ADR-002). |
| `MambaClapPlugin.cpp` | **Mixed** | `mamba_on_main_thread` ModelReload case has good comments. Mix chain (intensity→wetDry→bypass) is undocumented inline. Crossfade logic has no comments. |
| `MambaAraController.cpp` | **Good** | ARA callback stubs have clear intent comments. Conditioning calls are documented. |
| `Platform.hpp` | **Minimal** | Functions have no docstrings. Platform guards are clear from `#if` names. |
| `SPSCQueue.hpp` | **Good** | Cache line alignment documented. Push/pop logic clear. Power-of-2 requirement documented. |
| `ConditioningGenerator.hpp` | **Good** | Slot layout thoroughly documented with field descriptions. |
| `ConditioningFrame.h` | — | Not inspected but exists and is included by engine. |

**Critical paths lacking comments:**
- `processBlock()` sub-block loop (~60 lines, no comments)
- `mamba_process()` mix chain operations
- Crossfade implementation (no reference to ADR-003 rationale)
- `train_clara_now.py` parallel scan (comments exist but don't explain chunking rationale)

---

## 9. ADR-WORKFLOW.md Status

| ADR | Workflow Status | File Exists | Content Status |
|-----|----------------|-------------|----------------|
| ADR-001 | `[ACCEPTED]` | Yes | **Stale** — 9→10 inputs, sub-block processing |
| ADR-002 | `[ACCEPTED]` | Yes | Accurate |
| ADR-003 | `[ACCEPTED]` | Yes | Accurate |
| ADR-004 | `[ACCEPTED]` | Yes | Mostly accurate (Linux-centric install paths) |
| ADR-005 | `[ACCEPTED]` | Yes | Accurate |
| ADR-006 | `[ACCEPTED]` | Yes | **Superseded** — Replaced by ADR-007 Phase 0 Platform.hpp |
| ADR-007 | `[ACCEPTED]` | Yes | Accurate. Windows CI not yet implemented (Phase 2 incomplete). |
| ADR-008 | `[ACCEPTED]` | Yes | **Stale** — Claims external-only conditioning but code has dual-path (ONNX + external). Claims "removed `loadCondWeights`" but code retains it. |
| No ADR | N/A | — | Sub-block processing, ModelReload implementation, manifest validation, training consolidation, dual-path conditioning |

**ADR-WORKFLOW.md draft status:**
- No `[DRAFT]` entries — the workflow backlog is empty
- No `[REVIEW]` entries
- All 8 ADRs are `[ACCEPTED]`
- **Gap:** The workflow file hasn't been updated to reflect that ADR-001, ADR-006, and ADR-008 are stale/superseded

---

## 10. Summary of Critical Issues

### P0 — Must Fix (core architectural contract wrong)
1. **ADR-001 I/O contract is wrong**: 9 inputs should be 10 in ADR, README, docs/parameter-reference.md, docs/training-pipeline.md, and AGENTS.md
2. **ADR-008 contradicts implementation**: Claims external-only conditioning but code has dual-path

### P1 — High Priority
3. **No ADR for sub-block processing**: The 64-sample sub-block architecture is a fundamental design change
4. **`GraphOptimizationLevel` discrepancy**: ALL vs BASIC — which is correct?
5. **Dual-path conditioning not documented**: If both ONNX FiLM and external gain apply, what's the interaction?
6. **Known Gaps table is stale**: 4 of 7 entries have inaccurate claims
7. **Training file inventory mismatch**: TODO.md claims 10 modular files but only 5 consolidated files exist
8. **ModelReload is implemented but AGENTS says it's empty**

### P2 — Should Fix
9. **Parameter count inconsistencies across all docs**: Values of ~52K, ~72K, ~77K, ~171K, ~274K
10. ~~**`mambaGetModulePath()` referenced but doesn't exist** in 2 places in AGENTS.md~~ **[FIXED]**
11. **ADR-006 is superseded** by ADR-007 but not marked as such
12. **Documentation references stale file names**: `TRAINING_COMMANDS.md` vs `training_commands.md`
13. **No ROCm environment documentation** for training

### P3 — Nice to Have
14. **No developer onboarding guide**
15. **No API documentation (Doxygen, etc.)**
16. **No model versioning strategy documented**
17. **`ConditioningFrame` vs `ConditioningGenerator` relationship undocumented**
18. **Line number references in ADRs are fragile** and likely stale after code changes

---

## 11. Recommended Actions

1. **Update ADR-001** to reflect 10-input contract and 64-sample sub-block processing
2. **Write ADR-009** for sub-block processing architecture
3. **Write ADR-010** for conditioning dual-path (FiLM + external gain) — or resolve which path is canonical and remove the other
4. **Update Known Gaps** in AGENTS.md to match reality
5. ~~**Fix `mambaGetModulePath()` → `mambaResolveModulePath()`** in AGENTS.md~~ **[FIXED]**
6. **Update parameter count** to a single canonical value across README, ADR-001, and docs
7. **Mark ADR-006 as superseded** by ADR-007
8. **Update ADR-008** to reflect actual implementation (10th ONNX input + possible external gain)
9. **Update ADR-WORKFLOW.md** with stale/superseded ADR status and new draft entries
10. **Normalize training documentation** — either adopt `train_clara_now.py` as canonical or restore the modular pipeline
11. **Create ROCm environment setup doc** from scattered SESSION_LOG entries
12. **Add ADR for `ORT_ENABLE_BASIC` change** or revert to `ORT_ENABLE_ALL` if accidental
13. **CONFIRMED: Dual-path conditioning is active.** `MambaInferenceEngine.cpp:224-235` feeds cond tensor as 10th ONNX input (CondFiLM), AND lines 260-273 apply external scalar gain modulation via `combinedWeights_`. Both paths use the same `activeFrame_` pointer. Whether the ONNX model actually contains CondFiLM layers depends on which `export_mamba_onnx.py` version generated `models/mamba_rt.onnx`. If the ONNX model also has FiLM, the cond vector is applied twice (FiLM inside ONNX + external scalar gain post-ONNX). This needs clarification. **Solution: Either make ADR-008 accurate (cond is ONNX-integrated) and remove the external gain path, or document that external gain is the only active path and the cond tensor input to ONNX is dead code.**
