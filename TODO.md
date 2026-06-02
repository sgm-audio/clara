# CLARA — Master TODO

**Generated:** 2026-06-02 | **Last update:** 2026-06-02 | **Status key:** `⬜ pending` `🔄 in progress` `✅ done` `❌ cancelled`

---

## 🔴 P0 — Missing Architecture (fix immediately)

| # | File | Issue | Status |
|---|------|-------|--------|
| 1 | `src/core/MambaInferenceEngine.hpp/.cpp` | **Conditioning vector `cond[COND_DIM]`** — double-buffered with atomic pointer swap, projection weights loaded at `activate()`, immutable in RT. (See ADR-008) | ✅ |
| 2 | `src/core/SPSCQueue.hpp` | **SPSC ring buffer** — lock-free queue for deferred tasks (ModelReload, AnalysisQueueDrain) | ✅ |

## 🟠 P1 — Functional Gaps

| # | File | Issue | Status |
|---|------|-------|--------|
| 3 | `src/clap/MambaClapPlugin.cpp:377-389` | `mamba_on_main_thread` — `ModelReload` and `AnalysisQueueDrain` cases are empty no-ops | ⬜ |
| 4 | `src/clap/MambaClapPlugin.hpp:33-39` | `DeferredTask` enum `ModelReload`/`AnalysisQueueDrain` never pushed — atomic stays at `None` | ⬜ |
| 5 | `src/ara/MambaAraController.cpp:44-51` | `doUpdateMusicalContextContent` — sets `contentDirty_` but never triggers re-analysis | ⬜ |
| 6 | `src/ara/MambaAraController.cpp:73-80` | `doUpdateAudioSourceContent` — same pattern, no re-analysis | ⬜ |
| 7 | `src/ara/MambaAraController.cpp:93-100` | `didEnableAudioSourceSamplesAccess` — comment says trigger analysis, body is no-op | ⬜ |
| 8 | `src/ara/MambaAraController.cpp:132-139` | `doCreateEditorRenderer` — returns nullptr | ⬜ |
| 9 | `src/ara/MambaAraController.cpp:141-145` | `doDestroyEditorRenderer` — no-op, would leak if renderer existed | ⬜ |
| 10 | `src/ara/MambaAraController.cpp:147-153` | `doCreateEditorView` — returns nullptr | ⬜ |
| 11 | `src/ara/MambaAraController.cpp:155-159` | `doDestroyEditorView` — no-op, would leak | ⬜ |

## 🟡 P2 — Correctness & Technical Debt

| # | File | Issue | Status |
|---|------|-------|--------|
| 12 | `src/core/MambaInferenceEngine.cpp:56` | FIXME: Silent failure — `!initialized_` path memset zeros with no error indication | ⬜ |
| 13 | `src/clap/MambaClapPlugin.cpp:283-292` | Duplicate param flush — `mamba_process` and `clap_params_flush` both handle `CLAP_EVENT_PARAM_VALUE` | ⬜ |
| 14 | `src/clap/MambaClapPlugin.cpp:317` | Missing engine health check — `mamba_start_processing` now validates engine before `processBlock` | ✅ |
| 25 | `CMakeLists.txt:203-298` | Multi-config generators (VS/Xcode/Ninja MC) put plugin at `build/plugins/$<CONFIG>/`; old `file(COPY ...)` landed models at `build/plugins/models/`. Replaced with 8× `add_custom_command(TARGET ... POST_BUILD)` using `$<TARGET_FILE_DIR>` and `${CMAKE_COMMAND} -E copy_directory` (Debug, Release, RelWithDebInfo, MinSizeRel × {clap, ara}) | ✅ |
| 26 | `tests/CMakeLists.txt:99-122`, `tests/clap_validator.cmake:60-95` | Added `option(CLAP_VALIDATOR_FULL "Run the full clap-validator suite (slow)" OFF)` (default OFF) and invert-filter regex excluding tests which call `process()`. Quick mode: 12 tests, 8 passed, 0 failed, 4 skipped in 0.10s. Full mode available via `-DCLAP_VALIDATOR_FULL=ON` for nightly CI | ✅ |
| 27 | `tests/test_clap_plugin.cpp:261` | Pre-existing wetDry `text_to_value` parser mismatch — source accepts `"N%"` format, test sends raw `"0.5"`. Unrelated to URL changes. Filed as separate issue (do not refactor here per AGENTS.md) | ⬜ |

## 🟢 P3 — Test Coverage

| # | File | Issue | Status |
|---|------|-------|--------|
| 15 | `tests/test_ara_controller.cpp:20-42` | Model dir resolution test now uses real symbol address (test executable) to exercise the actual `mambaGetModulePath` code path on Windows | ✅ |
| 16 | `tests/test_engine.cpp:407` | `test_conditioning_atomic_swap()` added — verifies double-buffer atomic pointer swap is RT-safe | ✅ |
| 17 | `tests/test_clap_plugin.cpp:434` | `test_spsc_two_thread()` added — producer/consumer across two threads for SPSC FIFO | ✅ |
| 18 | `tests/test_engine.cpp:396` | `test_init_missing_model_path()` added — engine gracefully fails when model file absent | ✅ |
| 19 | `tests/test_clap_plugin.cpp` | No test for conditioning activation or deferred task wiring | ⬜ |

## 🔵 P4 — Cosmetic / Documentation

| # | File | Issue | Status |
|---|------|-------|--------|
| 20 | `src/clap/MambaClapPlugin.cpp:28-30`, `src/core/Platform.hpp:6` | `plugin_url` → `MAMBA_PROJECT_URL` macro; `manual_url` and `support_url` → `nullptr` (spec-allowed per CLAP plugin.h:15-16) | ✅ |
| 21 | `src/ara/MambaAraController.cpp:232` | `getInformationURL()` now returns `MAMBA_PROJECT_URL` (was nullptr) | ✅ |
| 22 | `ADR-008-conditioning-vector.md` | ADR-008 drafted — conditioning vector architecture (double-buffer + projection weights) | ✅ |
| 23 | `CHANGELOG.md:20-24` | Added `Fixes` subsection between `Changed` and `Infrastructure` recording this session's changes | ✅ |
| 24 | `AGENTS.md:107-109` | Known Gaps table updated — removed duplicate validator row, added new P2 row for multi-config model deploy fix | ✅ |

---

## Previous Sessions — Completed

### 🔴 P0 — Runtime Bugs (3/3)
| # | Item | Status |
|---|------|--------|
| — | `init()` return value ignored — plugin returns true even when init fails | ✅ |
| — | ARA constructor marked `noexcept` but calls `std::make_unique` | ✅ |
| — | Dead try/catch around `engine->init()` | ✅ |

### 🟠 P1 — Dead Code & Cleanup (6/9)
| # | Item | Status |
|---|------|--------|
| — | `engineValid_` dead member removed | ✅ |
| — | `audioSourceAccessEnabled_` dead member removed | ✅ |
| — | 4 PlaybackRegion delegation stubs removed | ✅ |
| — | Placeholder URLs → TODO comments | ✅ |
| — | `getInformationURL()` returns nullptr → TODO comment | ✅ |
| — | Phase 0: cross-platform portability scaffolding (Platform.hpp) | ✅ |

### 🟡 P2 — Missing Functionality (10/11)
| # | Item | Status |
|---|------|--------|
| — | `mamba_on_main_thread` deferred task scaffolding | ✅ |
| — | `mamba_init` validates model path | ✅ |
| — | `mamba_start_processing` checks engine | ✅ |
| — | EditorRenderer/EditorView stubs with TODO docs | ✅ |
| — | `doUpdateMusicalContextContent` sets `contentDirty_` | ✅ |
| — | `doUpdateAudioSourceContent` sets `contentDirty_` | ✅ |
| — | `didEnableAudioSourceSamplesAccess` documented, resets dirty flag | ✅ |
| — | Training pipeline doc created (`docs/training-pipeline.md`) | ✅ |
| — | CI enhanced with model shape verify + ARA checks | ✅ |

### 🟢 P3 — Documentation Gaps (7/7)
| # | Item | Status |
|---|------|--------|
| — | ADR-003, ADR-004, ADR-005, ADR-006 created | ✅ |
| — | README.md created | ✅ |
| — | `docs/parameter-reference.md` created | ✅ |
| — | LICENSE (Apache 2.0) | ✅ |

### 🔵 P4 — Stale/Outdated Docs (5/5)
| # | Item | Status |
|---|------|--------|
| — | TODO_PROGRESS.md stale refs fixed | ✅ |
| — | CHANGELOG.md stale refs fixed | ✅ |
| — | CLAUDE_OPUS_PROMPT.md stale refs fixed | ✅ |

### ⚪ P5 — Testing & CI (3/3)
| # | Item | Status |
|---|------|--------|
| — | 3 test files created (engine: 12, clap: 10, ara: 4) | ✅ |
| — | CI enhanced — ARA verify, model shape verify, stripped check | ✅ |
| — | ARA library verification step in CI | ✅ |

---

## Phase 1+2 Session 2026-06-02 — Completed

Platform: Windows MSVC, Visual Studio 18 2026 generator, `BUILD_TESTING=ON`. All builds clean. All tests pass except pre-existing wetDry parser issue (now P2 #27).

| # | Deliverable | Files | Notes |
|---|-------------|-------|-------|
| Subagent 1 | Multi-config POST_BUILD model deploy fix | `CMakeLists.txt:203-298` (8 POST_BUILD blocks) | Fixes pre-existing bug where multi-config generators (VS/Xcode/Ninja MC) put plugin at `build/plugins/$<CONFIG>/` but `file(COPY)` landed models at `build/plugins/models/`. `mambaFileExists` pre-check then failed and validator reported "clap_plugin::init() returned false" for 11/21 sub-cases |
| Subagent 2 | Source cleanups | `src/core/Platform.hpp:6`; `src/clap/MambaClapPlugin.cpp:28-30`; `src/ara/MambaAraController.cpp:86,98,232,237` | `MAMBA_PROJECT_URL` macro; `manual_url`/`support_url` → nullptr (spec-allowed); corrected `AudioSourceReader` → `HostAudioReader` comments; factory version 1.0.0 → 1.1.0 |
| Subagent 3 | Docs sync | `ADR-WORKFLOW.md:246-257`; `ADR-008-conditioning-vector.md:72,83`; `AGENTS.md:107-109`; `CHANGELOG.md:20-24` | Appended ADR-008 section header; fixed stale line ref `:232` → `:263` (loadCondWeights call site moved); updated Known Gaps; added Fixes subsection |
| Subagent 4 | Engine tests | `tests/test_engine.cpp:396,407` | `test_init_missing_model_path` (graceful failure when model absent) + `test_conditioning_atomic_swap` (RT-safe double-buffer swap) |
| Subagent 5 | SPSC + ARA tests | `tests/test_clap_plugin.cpp:434,475`; `tests/test_ara_controller.cpp:20-42` | `test_spsc_two_thread` + `test_spsc_wrap_around`; replaced vacuous `test_model_dir_resolution` with one using a real symbol address (the test exe) to exercise real `mambaGetModulePath` on Windows |
| Post-session | Quick-mode clap-validator + test updates | `tests/CMakeLists.txt:99-122`; `tests/clap_validator.cmake:60-95`; `tests/test_clap_plugin.cpp:507-512`; `tests/test_ara_controller.cpp:65` | `CLAP_VALIDATOR_FULL=OFF` default with invert-filter regex; 12 tests / 8 passed / 0 failed / 4 skipped in 0.10s. Test updates: descriptor URL test expects nullptr for manual/support with comment; ARA test version assertion → `"1.1.0"` |

---

## Quick Stats

| Category | Completed | Remaining |
|----------|-----------|-----------|
| 🔴 P0 — Missing Architecture | 2 / 2 | 0 |
| 🟠 P1 — Functional Gaps | 0 / 9 | 9 |
| 🟡 P2 — Correctness & Debt | 3 / 6 | 3 |
| 🟢 P3 — Test Coverage | 4 / 5 | 1 |
| 🔵 P4 — Cosmetic / Docs | 5 / 5 | 0 |
| Previous sessions | 34 / 34 | 0 |
| **Total** | **48 / 61** | **13** |
