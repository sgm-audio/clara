# ADR Workflow — Drafting System

**Updated:** 2026-07-02 — Mamba-3 PerTrackProcessor era. See ADR-011 through ADR-013 for current architecture.

This file serves as both a todo list and a prompting system for documenting architectural decisions in CLARA. Each ADR follows the lifecycle **Draft → Review → Revise → Accepted**. Historical ADRs from the Mamba-1 / ONNX Runtime era (ADR-005, ADR-008, ADR-009, ADR-010) are marked HISTORICAL.

## Status Legend

| Badge | Meaning |
|-------|---------|
| `[DRAFT]` | Not yet written. Needs creation. |
| `[REVIEW]` | Written, awaiting approval. |
| `[REVISE]` | Feedback received, being updated. |
| `[ACCEPTED]` | Finalized and implemented. |

---

## ADR-001: Shared ONNX Runtime Inference Engine

**Status:** `[ACCEPTED]` — **Historical / Superseded by Mamba-3**
**File:** `ADR-001-shared-onnx-inference-engine.md`

**Context:** CLARA needs to run Mamba SSM inference in both CLAP (RT) and ARA2 (offline) contexts via a shared inference core. This ADR now describes the deprecated Mamba-1 / ONNX Runtime era; the current architecture uses the custom C++ engine documented in `MAMBA3_MIGRATION_PLAN.md`.

**Key decisions documented:**
- ONNX Runtime v1.17.1 on CPU (not LibTorch, RTNeural, or native C++)
- `MambaCore` static library compiled once, linked into both targets
- 9-input / 3-output ONNX tensor contract with dynamic `seq_len`
- Pre-allocated buffers in `init()`, zero heap allocs in `processBlock()`
- Model architecture sizing: d_model=64, n_layers=4, d_state=16, conv_width=4, dt_rank=16
- CPU-only (no GPU dependency), `-ffast-math` risk documented
- Deployment contract with `$ORIGIN` RPATH

---

## ADR-002: Deferred Atomic State Reset

**Status:** `[ACCEPTED]`
**File:** `ADR-002-deferred-atomic-state-reset.md`

**Context:** SSM state (8 buffers, ~20 KB) must be zeroed on transport seek, preset load, or bypass. `resetState()` can be called from any thread. Direct zeroing races with `processBlock()`.

**Key decisions documented:**
- Single `std::atomic<bool> resetRequested_` with release-store / acq_rel-exchange
- Zeroing deferred to audio thread at next `processBlock()` boundary
- Memory ordering analysis: happens-before relationship, `exchange` vs `load`+`store` TOCTOU
- All three call sites analyzed (transport seek, preset load, ARA constructor)
- Initial state guaranteed by `init()` resize-to-zero

---

## ADR-003: Click-Free Bypass via 50ms Linear Crossfade

**Status:** `[ACCEPTED]`
**File:** `ADR-003-click-free-bypass-linear-crossfade.md`

**Context:** Bypass parameter transitions must be click-free. Instantaneous switching between dry/wet paths produces a DC discontinuity (pop). A smooth crossfade is required.

**Key points to cover:**
- Why 50ms duration (psychoacoustic: fast enough to be imperceptible, slow enough to suppress clicks across the audible spectrum)
- Linear vs equal-power (logarithmic) crossfade — why linear chosen for deterministic per-sample math
- Per-sample ramp, step accumulated **outside** per-channel loop (the stereo aliasing bug fix from the audit — the original version had step accumulation inside the channel loop, causing left/right channels to drift apart during the transition)
- Sample-rate-independent: `crossfadeStep = 1.0f / (0.05f * sampleRate)`, computed at `activate()`
- `currentBypassValue` state variable tracks ramp position across blocks — bypass transitions can span multiple `processBlock()` calls
- Interaction with `wetDry` and `intensity` atomics — bypass is applied last in the mix chain
- `bypass` is a `CLAP_PARAM_IS_STEPPED` parameter with `CLAP_PARAM_IS_BYPASS` flag — host may send discrete 0/1, but the DSP side ramps smoothly

**Files to reference:**
- `src/clap/MambaClapPlugin.cpp:263` — `crossfadeStep` initialization at activate
- `src/clap/MambaClapPlugin.cpp:317-345` — crossfade ramp + mix in `mamba_process`
- `src/clap/MambaClapPlugin.cpp:339-341` — the mix equation: `(1-bypass)*(wet*dryMix + dry*(1-dryMix)) + bypass*dry`
- `src/clap/MambaClapPlugin.hpp:27-28` — `currentBypassValue` and `crossfadeStep` members

**Prompt for drafting agent:**

```
Draft ADR-003: Click-Free Bypass via 50ms Linear Crossfade

Cover these topics in detail:
1. Why instantaneous bypass causes audible pops (DC discontinuity analysis)
2. Why 50ms linear crossfade was chosen over alternatives:
   - Exponential ramp (perceptually non-linear, harder to tune)
   - Step-synchronous (requires lookahead, adds latency)
   - Equal-power cosine (more complex math, marginal benefit for binary bypass)
3. The stereo aliasing bug: step accumulation was inside the per-channel loop,
   causing left/right channels to drift. Fix: move step accumulation outside.
4. Block-spanning transitions: crossfade may span multiple processBlock() calls.
   currentBypassValue persists across calls.
5. Mix equation derivation and order of operations:
   intensity(wet) → wetDry blend → bypass crossfade
6. Sample rate independence: crossfadeStep derived from sampleRate at activate().
7. CLAP parameter flags: CLAP_PARAM_IS_STEPPED + CLAP_PARAM_IS_BYPASS interaction
   with the smooth ramp.

Format: Follow ADR-001 and ADR-002 structure exactly.
```

---

## ADR-004: CLAP + ARA Dual-Target Build Architecture

**Status:** `[ACCEPTED]`
**File:** `ADR-004-clap-ara-dual-target-build-architecture.md`

**Context:** Two shared library targets (`MambaSuiteClap.clap`, `MambaSuiteARA.so`) share one inference core (`MambaCore` static library). The build system must support this without duplication.

**Key points to cover:**
- CMake `FetchContent` for all three dependencies — no system packages required
- `MambaCore` STATIC library with `-fPIC` (`POSITION_INDEPENDENT_CODE ON`)
- `MambaSuiteClap` as MODULE library with `.clap` suffix and `PREFIX ""`
- `MambaSuiteARA` as MODULE library with `.so` suffix and `PREFIX ""`
- `COMMON_INCLUDES` list shared between both targets
- Both targets linked against `pthread` and `dl` for `dladdr` and threading
- Release builds stripped via `LINK_FLAGS "-s"`
- Model copy: `file(COPY ...)` at build time from source `models/` → `build/plugins/models/`
- `libonnxruntime.so.1.17.1` deploy from `_deps/` → `build/plugins/` at build time
- Install rules for CLAP bundle directory (`lib/clap/MambaSuiteClap.clap/Contents/linux-x86-64/`) + bare `.clap` fallback + ARA `.so`
- Component-based install: `cmake --install . --component clap` / `--component ara`
- `BUILD_WITH_INSTALL_RPATH ON` ensures `$ORIGIN` resolution during development
- `cmake_minimum_required(VERSION 3.14)` because FetchContent + URL download support

**Files to reference:**
- `CMakeLists.txt` (entire file, 212 lines)

**Prompt for drafting agent:**

```
Draft ADR-004: CLAP + ARA Dual-Target Build Architecture

Cover these topics:
1. Why FetchContent over find_package or vcpkg — zero system deps for users,
   pinned versions for reproducibility.
2. Static library (MambaCore) vs compiling into each target — rationale for
   compilation-once with -fPIC.
3. MODULE library semantics vs SHARED — MODULE is correct for dlopen()-loaded
   plugins; CMake does not add the "lib" prefix.
4. The .clap suffix trick — CMake has no native CLAP support, we set
   SUFFIX ".clap" and PREFIX "" on MambaSuiteClap.
5. RPATH strategy — BUILD_WITH_INSTALL_RPATH ON + CMAKE_INSTALL_RPATH "$ORIGIN"
   means the build tree layout matches the install layout. No LD_LIBRARY_PATH
   or rpath patching needed.
6. Install directory structure documented inline in CMakeLists.txt (lines 138-157).
7. Component-based install allows selective deployment (CLAP-only, ARA-only, or both).
8. Model and ORT runtime deployment: copied at build time for development,
   installed at install time for distribution.
9. Minimum CMake 3.14 requirement — needed for FetchContent URL downloads with
   DOWNLOAD_EXTRACT_TIMESTAMP.

Format: Follow ADR-001 and ADR-002 structure exactly.
```

---

## ADR-005: `thread_local` + `reserve()` for Ort::Value Vectors

**Status:** `[ACCEPTED]`
**File:** `ADR-005-thread-local-reserve-ort-value-vectors.md`

**Context:** ONNX Runtime's `Ort::Value` has no default constructor — it can only be created via static factory methods like `Ort::Value::CreateTensor<T>(...)`. This prevents using `std::array<Ort::Value, N>` as class members, because array elements would need default construction.

**Key points to cover:**
- `Ort::Value` API constraint: no default constructor, no move constructor (pre-C++17 ORT versions), only factory-created
- Attempted approaches that failed:
  - `std::array<Ort::Value, 9>` as class member — rejected by compiler (no default ctor)
  - `std::array<std::optional<Ort::Value>, 9>` — functional but adds `std::optional` overhead and requires `emplace()` per call
  - Raw `Ort::Value*` array with placement new — error-prone, manual destruction
  - `std::vector<Ort::Value>` member with `reserve()` + `clear()` — persisting across calls means the vector carries capacity across the RT/non-RT boundary, which is safe but the member is accessed from only the audio thread (no contention)
- Chosen: `thread_local std::vector<Ort::Value>` — per-thread singleton, `reserve()` on first call to prevent reallocation, `clear()` per call (does not free memory, only resets size)
- Two vectors: `inputTensors_` (capacity 9) and `outputTensors_` (capacity 3), both `thread_local`
- `clear()` preserves capacity — subsequent `push_back()` does not reallocate
- Thread safety: `thread_local` means no sharing between threads. CLAP and ARA run on different threads with independent vectors.
- First-call warm-up: the first `processBlock()` call allocates capacity. All subsequent calls are allocation-free.
- Must be `thread_local` because `processBlock()` can be called from multiple audio threads (CLAP thread pool extension, or multiple plugin instances).

**Files to reference:**
- `src/core/MambaInferenceEngine.cpp:75-77` — `thread_local inputTensors_` with `clear()` + `reserve(9)`
- `src/core/MambaInferenceEngine.cpp:94-96` — `thread_local outputTensors_` with `clear()` + `reserve(3)`

**Prompt for drafting agent:**

```
Draft ADR-005: thread_local + reserve() for Ort::Value Vectors

Cover these topics:
1. The fundamental constraint: Ort::Value has no default constructor. Show
   compiler error and explain why.
2. Approaches tried and rejected:
   - std::array<Ort::Value, N> (no default ctor, compiler error)
   - std::array<std::optional<Ort::Value>, N> (works but adds overhead)
   - raw pointer + placement new (lifetime management burden)
   - member std::vector<Ort::Value> (safe but disturbs class invariants)
3. Why thread_local:
   - CLAP thread pool extension may call processBlock() from multiple threads
   - Multiple plugin instances each have their own engine
   - Zero contention, zero synchronization
4. Why reserve() then clear():
   - clear() preserves capacity, does not free memory
   - push_back() after clear() reuses existing storage
   - First call allocates, subsequent calls are allocation-free
5. Verify allocation-free: document that after first call on each thread,
   all subsequent calls have zero heap allocations.
6. The tradeoff: first-call latency is higher (warm-up allocation).
   Acceptable — happens once per thread per session.
7. Alternative: use a member std::vector with thread-safe double-buffering.
   Rejected because the vector is only accessed from one thread at a time
   (each processBlock() call is serialized by the host).

Format: Follow ADR-001 and ADR-002 structure exactly.
```

---

## ADR-006: `dladdr`-Based Model Path Resolution

**Status:** `[ACCEPTED]`
**File:** `ADR-006-dladdr-model-path-resolution.md`

**Context:** Both plugins must locate `models/mamba_rt.onnx` at runtime relative to their own binary. No hardcoded paths, environment variables, or configuration files should be required. The model directory must work identically in development (build tree) and deployment (install tree).

**Key decisions documented:**
- `dladdr()` resolves the address of a known function to the shared object's filesystem path.
- CLAP: `dladdr((void*)clap_create_plugin, &info)` in factory entry point, stored in `MambaPluginState::modelPath`.
- ARA: `resolveModelDirOnce()` uses `dladdr` on itself, cached in `static const std::string` via `getMambaModelDir()`.
- Comparison of 6 alternatives (argv[0], getcwd, /proc/self/exe, hardcoded, env var, dlinfo) — all rejected.
- Fallback to `"./models/"` if `dladdr` fails (defense-in-depth).
- Both `mamba_rt.onnx` and `mamba_rt.onnx.data` must be present in `models/` directory.
- `$ORIGIN` RPATH is independent of dladdr-based model resolution.

---

## ADR-007: Cross-Platform Portability Strategy for CLAP + ARA Plugins

**Status:** `[ACCEPTED]`
**File:** `ADR-007-cross-platform-portability.md`

**Context:** Project was Linux x86_64 only (dladdr, __attribute__((visibility)), $ORIGIN RPATH, .so suffix, access(), pthread/dl). Needed Windows (MSVC) and macOS (Apple Clang) support.

**Key decisions documented:**
- Four-phase approach: Phase 0 (Platform.hpp scaffolding, zero behavior change on Linux), Phase 1 (macOS), Phase 2 (Windows), Phase 3 (CI matrix + docs).
- Single `Platform.hpp` header with inline functions: `MAMBA_EXPORT`, `mambaResolveModulePath()`, `mambaResolveModelDir()`, `mambaFileExists()`.
- CMake platform-conditional guards for compiler flags, RPATH/loader_path, ONNX Runtime URL/lib, ARA suffix, bundle paths, deploy/install rules.
- macOS: @loader_path, .dylib, Apple Clang (same flags as GCC), dladdr available.
- Windows: GetModuleHandleExA + GetModuleFileNameA for module path, no RPATH (side-by-side DLL), MSVC flags (/W4, /fp:fast, /GR-).
- CI: macos-14 builds exist, windows-2022 builds needed.
- Files modified: Platform.hpp (new), CMakeLists.txt, MambaClapPlugin.cpp, ci.yml.
- Rejected alternatives: preprocessor-ifdef everywhere, platform-specific compilation units, vcpkg/Hunter, separate fork per platform.

---

## ADR-008: Conditioning Vector for Context-Aware Mamba SSM

**Status:** [ACCEPTED]
**File:** ADR-008-conditioning-vector.md
**Context:** SSM processes audio identically regardless of musical context. A conditioning vector lets ARA musical-context data (tempo, key) modulate output without changing the ONNX I/O contract.
**Key decisions documented:**
- `COND_DIM = 8`, double-buffered (`condBufA_`/`condBufB_`) with atomic `float*` pointer swap
- `setConditioning()` from any thread, `loadCondWeights()` from `activate()` only
- External gain modulation in `processBlock()` post-ONNX — zero cond = bit-identical to unconditioned
- Backward-compat: ONNX I/O stays 9/3; no model re-export needed
