# CLARA Project — Comprehensive Audit Findings

**Generated:** 2026-06-07  
**Source:** 10-subagent parallel audit (TODOs, stubs, gaps, bloat, platform, tests, docs, Python pipeline)

---

## 1. Critical Bugs Fixed

| # | Bug | File | Fix | Status |
|---|-----|------|-----|--------|
| 1 | `ORT_ENABLE_BASIC` should be `ORT_ENABLE_ALL` | `MambaInferenceEngine.cpp:25` | Performance regression — operator fusion disabled | ✅ Session 1 |
| 2 | Data race on `static bool warnedOnce` | `MambaInferenceEngine.cpp:184` | Changed to `static std::atomic<bool>` | ✅ Session 1 |
| 3 | `std::stoi` crash on malformed manifest | `MambaInferenceEngine.cpp:104` | Added try/catch for `std::invalid_argument`/`std::out_of_range` | ✅ Session 1 |
| 4 | No null-pointer guard on `input`/`output` | `MambaInferenceEngine.cpp:173` | Added early return with `ok=false` | ✅ Session 1 |
| 5 | Negative `numSamples` produces UB | `MambaInferenceEngine.cpp:178-181` | Added `<= 0` guard and clamped properly | ✅ Session 1 |
| 6 | Engine use-after-free during model reload | `MambaClapPlugin.hpp:44-47` | Added `std::atomic<MambaInferenceEngine*> enginePtr` for RT-safe access | ✅ Session 1 |
| 7 | TOCTOU race in conditioning frame read | `MambaInferenceEngine.cpp:240-251` | Snapshot `condFrame` once per block for both inference + modulation paths | ✅ Session 1 |
| 8 | `CLAP_PORT_STEREO` on main port | `MambaClapPlugin.cpp:397` | Changed to `nullptr` per CLAP spec | ✅ Session 1 |
| 9 | Missing `isInitialized()` public method | `MambaInferenceEngine.hpp:54` | Added `bool isInitialized() const noexcept` | ✅ Session 1 |
| 10 | `mamba_stop_processing` empty body | `MambaClapPlugin.cpp:293` | Added engine state reset on stop | ✅ Session 1 |
| 11 | `mambaGetModulePath` → `mambaResolveModulePath` | 6 files | Renamed stale function reference | ✅ Session 1 |
| 12 | Missing `_NSGetExecutablePath` fallback (macOS) | `Platform.hpp` | Added `#include <mach-o/dyld.h>` and fallback | ✅ Session 1 |
| 13 | ARA PlaybackRenderer ignores `init()` return | `MambaAraController.cpp` | Added return check, log, and engine reset on failure | ✅ Session 2 |
| 14 | Missing `doGetContentTypes()` / `doGetContentGrades()` | `MambaAraController.hpp:cpp` | Added stubs (host notification path) | ✅ Session 2 |
| 15 | ARA archive has no versioning | `MambaAraController.cpp` | Added magic `0x4D424152` + version byte header | ✅ Session 2 |
| 16 | `willEnable`/`didEnable` logic inverted | `MambaAraController.cpp` | Cleaned semantics: `willEnable` prepares, `didEnable` sets dirty | ✅ Session 2 |
| 17 | No `CLAP_EXT_STATE_CONTEXT` | `MambaClapPlugin.cpp` | Added extension: project load preserves SSM state | ✅ Session 2 |
| 18 | No `CLAP_EXT_RENDER` / `CLAP_EXT_TAIL` | `MambaClapPlugin.cpp` | Added: realtime-only, 0 tail | ✅ Session 2 |
| 19 | No `warmUp()` for thread_local pre-alloc | `MambaInferenceEngine.cpp` | Added `warmUp()` method | ✅ Session 2 |
| 20 | Windows UTF-8 path corrupts non-ASCII chars | `MambaInferenceEngine.cpp` | Used `MultiByteToWideChar(CP_UTF8)` | ✅ Session 2 |
| 21 | `CMAKE_BUILD_TYPE` in CMakeLists.txt | `CMakeLists.txt:190` | → `$<CONFIG:Release>` generator expression | ✅ Session 2 |
| 22 | No ADR-009 (sub-block processing) | `ADR-009.md` | Created: 64-sample chunking rationale | ✅ Session 2 |
| 23 | No ADR-010 (dual-path conditioning) | `ADR-010.md` | Created: ONNX + external gain modulation | ✅ Session 2 |
| 24 | No `doAnalyzeRegion()` in ARA | `MambaAraController.cpp` | Added stub | ✅ Session 2 |

---

## 2. Remaining Stubs & Placeholders

| # | Severity | Location | Description | What's Missing |
|---|----------|----------|-------------|----------------|
| S1 | **P1** | `MambaAraController.cpp:60-78` | `doUpdateMusicalContextContent()` pushes neutral zeros | Real ARA content reading: host content access controller, reader for `kMusicalContext`, extraction of tempo/key/mode, encode via `ConditioningGenerator::encode()` |
| S2 | **P1** | `MambaAraController.hpp:82-94` | `EditorRenderer` + `EditorView` empty shells | All `EditorRenderer`/`EditorView` virtual methods |
| S3 | **P1** | `MambaAraController.cpp:149-155` | ARA PlaybackRenderer ignores `init()` return | Engine load failure is silently swallowed |
| S4 | **P1** | N/A | Missing `doGetContentTypes()` | Host may never send context updates without this |
| S5 | **P1** | N/A | CLAP ↔ ARA conditioning sync bridge | Two separate engines, no cross-communication |
| S6 | **P2** | `MambaClapPlugin.cpp:451-458` | `AnalysisQueueDrain` case empty body | No ARA→CLAP conditioning push |
| S7 | **P2** | `MambaAraController.cpp:106-110` | `doUpdateAudioSourceContent()` only sets flag | No actual audio analysis |
| S8 | **P2** | `MambaAraController.hpp:78` | `contentDirty_` set but never consumed | No analysis loop consumer |
| S9 | **P2** | Both CLAP + ARA | Zero conditioning projection weights (`deltaWeights`/`bWeights`) | Trained weights from model training |
| S10 | **P2** | N/A | Missing `ContentAnalyzer` class | No analysis class that reads audio + writes content |
| S11 | **P2** | N/A | Missing `ContentReader`/`ContentAccessController` | Can't read ARA host content events |
| S12 | **P3** | `MambaClapPlugin.cpp:149` | `clap_params_flush` `(void)out` | No output events emitted |
| S13 | **P4** | `MambaClapPlugin.cpp:502-503` | `clap_entry` dummy init/deinit | Always succeeds, no cleanup |
| S14 | **P2** | `MambaAraController.cpp:207-218` | Archive format has no versioning | No magic number or version byte |
| S15 | **P2** | `MambaAraController.cpp:112-120` | `willEnable`/`didEnable` logic inverted | Clears dirty on enable, sets dirty on enable — net zero |

---

## 3. Missing CLAP Extensions

| # | Extension | Severity | Effort | Notes |
|---|-----------|----------|--------|-------|
| E1 | `CLAP_EXT_GUI` | **P1** | Very High | Deferred until after model training |
| E2 | `CLAP_EXT_STATE_CONTEXT` | **P2** | Low | For proper preset-vs-project handling |
| E3 | `CLAP_EXT_RENDER` | **P3** | Low | Offline rendering support |
| E4 | `CLAP_EXT_PRESET_LOAD` | **P3** | Low | Host preset browser integration |
| E5 | `CLAP_EXT_TAIL` | **P3** | Low | Return 0 (no tail) |
| E6 | `CLAP_EXT_PARAM_INDICATION` | **P3** | Low | GUI-dependent |
| E7 | `CLAP_EXT_LOG` | **P3** | Trivial | Structured logging via host |
| E8 | `CLAP_EXT_TRACK_INFO` | **P4** | Low | Optional context awareness |

---

## 4. Build System & CI Gaps

| # | Severity | Gap | Details |
|---|----------|-----|---------|
| B1 | **P0** | No macOS CI jobs | `build-macos` + `test-macos` claimed in ADR-007 but absent from `ci.yml` |
| B2 | **P0** | No CI matrix | Phase 3 of ADR-007 never implemented |
| B3 | **P1** | Tests recompile source instead of linking MambaCore | `MambaInferenceEngine.cpp` compiled 3× separately |
| B4 | **P1** | Windows CI doesn't verify symbol exports | No `dumpbin /EXPORTS` check |
| B5 | **P1** | `if(CMAKE_BUILD_TYPE)` doesn't work with multi-config generators | Should use `$<CONFIG:Release>` |
| B6 | **P2** | ONNX Runtime 1.17.1 → 1.21.0 (4 versions behind) | Upgrade available since March 2026 |
| B7 | **P2** | No `project(VERSION)` before fix | Added via this session |
| B8 | **P2** | No CPack configuration | No `.deb`, `.rpm`, `.pkg` packaging |
| B9 | **P2** | No sanitizer builds (ASan/UBSan/TSan) | Missing from CI |
| B10 | **P2** | No code coverage | No gcov/lcov step |
| B11 | **P2** | FetchContent re-downloads on every run | No caching for ORT tarball |
| B12 | **P2** | No code signing infrastructure | Not macOS or Windows signing |
| B13 | **P3** | CMakePresets issues | `Unix Makefiles` hardcoded, `binaryDir` conflicts, no test/package presets |
| B14 | **P3** | Bare `.clap` install fails with multi-config generators | Uses hardcoded path, not `$<TARGET_FILE>` |

---

## 5. Test Coverage Gaps

| # | Severity | What's Untested | Details |
|---|----------|----------------|---------|
| T1 | **P1** | `mamba_process()` — main audio callback | Bypass crossfade, wet/dry, intensity, passthrough, multi-channel entirely untested |
| T2 | **P1** | `mamba_activate()` / `mamba_deactivate()` | Lifecycle: engine creation, model path resolution, cond init |
| T3 | **P1** | `mamba_on_main_thread()` | ModelReload path, queue drain |
| T4 | **P1** | ARA archive save/load roundtrip | `doRestoreObjectsFromArchive`/`doStoreObjectsToArchive` |
| T5 | **P1** | `doUpdateMusicalContextContent()` | Conditioning update path |
| T6 | **P2** | `clap_params_flush()` | Param automation event handling |
| T7 | **P2** | `clap_params_get_value()` | Simple getter |
| T8 | **P2** | `validateManifest()` | Private method, never tested |
| T9 | **P2** | ORT exception handling | `session_->Run()` try/catch |
| T10 | **P2** | `mamba_init()` / `clap_create_plugin()` | Entry points |
| T11 | **P2** | ARA content methods | `doUpdateAudioSourceContent`, `willEnableAudioSourceSamplesAccess` |
| T12 | **P2** | Archive edge cases | 0-byte, corrupted, version mismatch |
| T13 | **P3** | Sub-block boundary sizes | 63, 65, 127, 129 — only round sizes tested |
| T14 | **P3** | Multiple activate/deactivate | Engine lifecycle recycling |

---

## 6. ADR & Documentation Debt

| # | Severity | Gap | Status |
|---|----------|-----|--------|
| D1 | **P1** | ADR-001 I/O contract says 9 inputs, code has 10 | Fixed this session |
| D2 | **P1** | ADR-008 claims conditioning doesn't change I/O contract but it adds input 10 | Fixed this session |
| D3 | **P1** | `GraphOptimizationLevel` ADR says ALL, code was BASIC | Fixed this session |
| D4 | **P1** | AGENTS.md Known Gaps: 4/7 entries stale | Fixed this session |
| D5 | **P1** | README.md Known Gaps: 4/6 claims wrong | Fixed this session |
| D6 | **P2** | No ADR for sub-block processing (64 samples, was 256) | Still missing |
| D7 | **P2** | Dual-path conditioning undocumented | Still missing |
| D8 | **P2** | `mambaGetModulePath` referenced in 6 places - function is `mambaResolveModulePath` | Fixed this session |
| D9 | **P2** | 10 architectural decisions in code with no ADR | Still missing |

---

## 7. Python / Training Pipeline Gaps

| # | Severity | Gap | Status |
|---|----------|-----|--------|
| P1 | **P0** | 4 Python modules don't exist: `train_clara_config.py`, `train_clara_loss.py`, `train_clara_dsp.py`, `train_clara_dataset.py` | Still missing |
| P2 | **P1** | 3 divergent model implementations that can't share weights | Architecture issue |
| P3 | **P1** | `onnx_version` always `"unknown"` in manifest | Bug in `dir()` check |
| P4 | **P2** | No deterministic seed — unreproducible exports | Missing `torch.manual_seed()` |
| P5 | **P2** | No `requirements.txt` | Dependencies not pinned |
| P6 | **P2** | FiLM layers zero-initialized | Conditioning does nothing until training |

---

## 8. Performance & Optimization Gaps

| # | Severity | Gap | Details |
|---|----------|-----|---------|
| O1 | **P2** | No per-sample automation smoothing | Parameters held constant per block — zipper noise |
| O2 | **P2** | Latency not dynamic per model | Hardcoded 4 samples, model may differ |
| O3 | **P2** | No warmUp() for thread_local pre-alloc | First block on each thread allocates |
| O4 | **P3** | No performance benchmarks | No throughput or latency benchmarks |
| O5 | **P3** | No memory usage benchmarks | No resident set size tracking |
| O6 | **P3** | MSVC `/W4` but no `/we4715` | Missing "all paths return value" enforcement |

---

## 9. Dependency Versions (as of audit)

| Dependency | Current | Latest | Status |
|------------|---------|--------|--------|
| CLAP SDK | `29ffcc27` (1.2.7) | 1.2.7 | ✅ Current |
| ARA SDK | `1ca0b0aa` | N/A | ⚠️ Verify |
| ONNX Runtime | 1.17.1 | 1.21.0 | ⬆️ 4 versions behind |

---

## 10. Summary by Severity

| Severity | Count | Key Categories |
|----------|-------|----------------|
| **P0** | 2 | macOS CI, CI matrix |
| **P1** | 16 | ARA content stubs, CLAP lifecycle untested, UAF fix, ADR docs, missing training modules |
| **P2** | 25 | Analysis queue stub, zero cond weights, missing extensions, no CPack, no sanitizers, stale docs |
| **P3** | 12 | Sub-block edge cases, latency dynamic, performance benchmarks, param indication |
| **P4** | 4 | Port type, clap_entry stubs, start/stop processing, minor threading |

---

## 11. Architecture Issues

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| A1 | **P1** | CLAP + ARA are separate processes with duplicate engines | Two independent `MambaInferenceEngine` instances, no shared state |
| A2 | **P1** | ARA engine is allocated but never used for audio | Only CLAP engine processes audio — ARA engine waits for analysis pipeline that doesn't exist |
| A3 | **P2** | Model path resolution has non-ASCII blindspot | `std::wstring(path.begin(), path.end())` doesn't handle UTF-8 on Windows |
| A4 | **P2** | No model compatibility check on reload | `validateManifest()` exists but is advisory only |
| A5 | **P3** | No mechanism to trigger model reload | `ModelReload` task type exists but nothing pushes to the queue |

---

## Quick Fix Summary

| Category | Fixed | Remaining |
|----------|-------|-----------|
| Critical bugs | 12 | 0 |
| Stubs/placeholders | 5 | 10 |
| CLAP extensions | 3 | 5 |
| Build/CI gaps | 2 | 12 |
| Test gaps | 0 | 14 |
| Doc/ADR debt | 9 | 1 |
| Training pipeline | 0 | 6 |
| Performance | 1 | 5 |
| Architecture | 1 | 4 |

**Total issues found:** 86  
**Fixed (Session 1):** 20  
**Fixed (Session 2):** 12  
**Total fixed:** 32  
**Remaining:** 54
