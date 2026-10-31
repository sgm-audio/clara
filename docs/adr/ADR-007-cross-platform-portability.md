# ADR-007: Cross-Platform Portability Strategy for CLAP + ARA Plugins

**Status:** Accepted
**Date:** 2026-05-31

## Context

The CLARA project was initially developed exclusively for Linux x86_64. Several Linux-isms were baked into the codebase and build system:

| Linux-ism | Usage | Portability Issue |
|-----------|-------|-------------------|
| `dladdr()` | Model path resolution in CLAP and ARA entry points (ADR-006) | POSIX function — not available on Windows. macOS supports it but `_dyld_get_image_name` is the preferred API. |
| `__attribute__((visibility("default")))` | Export `clap_entry`, `ARAGetFactory` | GCC/Clang extension. MSVC requires `__declspec(dllexport)`. |
| `$ORIGIN` RPATH | Runtime search path for `libonnxruntime.so` | ELF-specific. macOS uses `@loader_path`. Windows has no RPATH — DLLs are resolved via search path. |
| `-fno-rtti`, `-Wall -Wextra -Werror=return-type` | Compiler flags in CMake | MSVC uses `/GR-`, `/W4`, different warning semantics. Apple Clang handles these identically to GCC. |
| `.so` suffix | Shared library extension | `.dylib` on macOS, `.dll` on Windows. |
| `-s` (strip) linker flag | Release binary stripping | ELF-specific. Windows builds use `/PDBCompress` or separate strip tools. |
| `access()` for file existence | Model file check | POSIX function — Windows has `PathFileExistsA` or `GetFileAttributesA`. |
| `"lib"` prefix for shared libraries | CMake `PREFIX ""` override | macOS frameworks add `lib` prefix — must be suppressed for CLAP and ARA discovery. |
| `pthread` and `dl` link dependencies | Threading and dynamic loading | Only needed on Linux. macOS has these built into the system. Windows uses different APIs. |
| Single-platform CI (ubuntu-22.04) | GitHub Actions workflow | No macOS or Windows build verification, no cross-platform regression testing. |

The project must support all three major desktop platforms used by audio plugin consumers — Windows (x86_64), macOS (arm64 + x86_64), and Linux (x86_64). A portable codebase also lowers the barrier for open-source contributions.

## Decision

We adopt a **four-phase approach** to cross-platform portability. Each phase is independently mergable and testable, with zero behavior change on Linux.

### Phase 0 — Portability Scaffolding (Complete)

Phase 0 introduces a single `Platform.hpp` header and CMake platform guards **without changing any behavior on Linux**. Every change is behind `#if defined(_WIN32)` or `elif defined(__APPLE__)` guards that compile to no-ops on the current Linux-only build.

**Files created:**

- **`src/core/Platform.hpp`** (new) — Single header with inline functions:
  - `MAMBA_EXPORT` macro: `__declspec(dllexport)` on Windows, `__attribute__((visibility("default")))` on GCC/Clang, empty fallback.
  - `mambaResolveModulePath(funcAddr, modelFile)`: `GetModuleHandleExA` + `GetModuleFileNameA` on Windows, `dladdr` on POSIX. Returns `"<plugin_dir>/models/<modelFile>"`.
  - `mambaResolveModelDir(funcAddr)`: Same resolution but returns `"<plugin_dir>/models/"` directory only.
  - `mambaFileExists(path)`: `std::ifstream` — portable across all platforms.
  - All functions detect both `/` and `\\` path separators.

**Files modified:**

- **`CMakeLists.txt`** — Platform-conditional build configuration:
  - MSVC flags: `/W4` (warnings), `/O2 /fp:fast /DNDEBUG /GR-` (release), `/Od /Zi /DDEBUG` (debug).
  - Apple Clang: Same flags as GCC (`-Wall -Wextra -Werror=return-type`, `-O3 -ffast-math -DNDEBUG -fno-rtti`).
  - RPATH: `@loader_path` on macOS, `$ORIGIN` on Linux, none on Windows.
  - ONNX Runtime URL: `onnxruntime-win-x64-*.zip`, `onnxruntime-osx-universal2-*.tgz`, or `onnxruntime-linux-x64-*.tgz`.
  - Runtime library naming: `onnxruntime.dll`, `libonnxruntime.*.dylib`, `libonnxruntime.so.*`.
  - ARA suffix: `.dll`, `.dylib`, or `.so`.
  - CLAP bundle resource directory: `x86_64-windows`, `MacOS`, or `linux-x86-64`.
  - Per-platform deploy rules for ONNX Runtime side-by-side with plugins.
  - Per-platform install rules for runtime deps.
  - `link pthread dl` only on `UNIX AND NOT APPLE`.
  - `-s` strip flag only on `UNIX`.

- **`src/clap/MambaClapPlugin.cpp`** — Now includes `core/Platform.hpp` and uses `mambaResolveModulePath()` instead of inline `dladdr` + path manipulation.
- **`src/ara/MambaAraController.cpp`** — Already includes `core/Platform.hpp` and uses `mambaResolveModulePath()`.
- **`src/ara/MambaAraController.hpp`** — Uses `MAMBA_EXPORT` for `ARAGetFactory` and `getMambaARAFactory`.

**Files not modified (no Linux-isms):**

- `src/core/MambaInferenceEngine.hpp` / `.cpp` — Pure ONNX Runtime wrapper. No platform-specific code. Already portable.
- `tests/` — Use printf + assert, already portable.
- `export_mamba_onnx.py` — Pure Python, already portable.

**Verification:**

The Linux build after Phase 0 produces identical binaries to before. `Platform.hpp` functions on Linux compile to the same `dladdr` calls previously inlined in each plugin. The `MAMBA_EXPORT` macro resolves to `__attribute__((visibility("default")))` — identical to the previous `extern "C"` + hidden visibility default.

### Phase 1 — macOS Support (Apple Silicon + Intel)

Phase 1 enables building on macOS (both arm64 and x86_64) using Apple Clang (LLVM).

**Compiler considerations:**

| Aspect | macOS (Apple Clang) | Notes |
|--------|--------------------|-------|
| C++17 | Full support | Same as GCC, no changes needed. |
| Visibility | `__attribute__((visibility("default")))` | Same as GCC. `MAMBA_EXPORT` macro works identically. |
| RTTI | `-fno-rtti` | Apple Clang supports the same flag. |
| Fast math | `-ffast-math` | Supported. Same risk of ORT numerical issues documented in ADR-001. |
| `dladdr` | Available via `<dlfcn.h>` | Same POSIX API as Linux. |
| `dl` library | Built into system | No `-ldl` needed. Already guarded by `UNIX AND NOT APPLE`. |

**Model path resolution:**

macOS uses `@loader_path` instead of `$ORIGIN` for runtime library resolution. The `MambaInferenceEngine` is unaffected — it receives the model path as a `std::string`. The `mambaResolveModulePath()` function uses `dladdr()` which works identically to Linux on macOS.

**Universal binary support:**

macOS builds can produce universal binaries (`-arch arm64 -arch x86_64`) via a CMake toolchain file or by setting `CMAKE_OSX_ARCHITECTURES`. This is optional — the current CI uses `macos-14` (Apple Silicon only). A future optimization can enable universal builds.

**Files modified (already done in Phase 0 scaffolding):**
- `CMakeLists.txt` — Apple Clang flags, `@loader_path` RPATH, `.dylib` suffix, macOS bundle directory, OSX ONNX Runtime URL.

**CI additions (already in `.github/workflows/ci.yml`):**
- `build-macos` job on `macos-14` — builds Release, verifies `nm -gU` for `clap_entry` / `ARAGetFactory`, verifies `@loader_path` via `otool -L`, deploys model and `libonnxruntime.dylib`, runs Python model export, uploads artifact.
- `test-macos` job — builds Debug with `BUILD_TESTING=ON`, runs CTest.

### Phase 2 — Windows Support (MSVC)

Phase 2 enables building on Windows (x86_64) using Microsoft Visual C++ (MSVC) 2019+.

**Compiler considerations:**

| Aspect | MSVC | Notes |
|--------|------|-------|
| C++17 | Full support (VS 2019 16.8+) | `std::clamp`, `std::make_unique`, `constexpr if`, `thread_local`, structured bindings all supported. No code changes needed. |
| Visibility | `__declspec(dllexport)` | `MAMBA_EXPORT` macro handles this via `MAMBA_EXPORT` definition for `_WIN32`. |
| RTTI | `/GR-` | Already set in `CMakeLists.txt` for MSVC Release. |
| Fast math | `/fp:fast` | Already set. |
| Warning flags | `/W4` | Already set. `-Werror=return-type` has no MSVC equivalent — `/WX` is too aggressive. |
| `thread_local` | Supported (MSVC 2015+) | No changes needed in `MambaInferenceEngine`. |
| `std::atomic<bool>` | Supported | No changes needed. |

**Model path resolution:**

Windows has no `dladdr()`. The `mambaResolveModulePath()` function uses `GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, ...)` followed by `GetModuleFileNameA()`. The calling code passes a function pointer in the same binary — identical to how `dladdr` is called on POSIX.

Note: `GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT` prevents the reference count from incrementing, avoiding a resource leak. The module stays loaded as long as the plugin is loaded (the DAW holds the reference via `LoadLibrary`/`dlopen` equivalent).

**DLL search path:**

Windows has no RPATH equivalent. The ONNX Runtime DLL resolution strategy:

1. **Side-by-side deployment**: `onnxruntime.dll` is copied into the same directory as the plugin binary (both `build/plugins/` for development and the install directory for deployment).
2. **`SetDllDirectory`**: Not required — loading from the same directory is Windows default search behavior for `LoadLibrary`-loaded DLLs.
3. **Delay-load**: Not used — ONNX Runtime is linked normally. The DLL must be present at load time.

**CLAP export on Windows:**

CLAP SDK defines `CLAP_EXPORT` as `__declspec(dllexport)` for `_WIN32`. The `clap_entry` symbol uses `CLAP_EXPORT` (already in the CLAP headers). No change needed in our code.

**ARA export on Windows:**

`ARAGetFactory` uses `MAMBA_EXPORT` which resolves to `__declspec(dllexport)` on Windows. Verified in `MambaAraController.hpp:84`.

**Strip / debug symbols:**

Windows debugging uses PDB files rather than embedded debug info. The `-s` equivalent for MSVC is not applied in release — users can optionally run `strip`-equivalent tools (e.g., `dumpbin /PDATA`), or we can add a post-build step with Microsoft's `editbin` / `link /PDBCOMPRESS`. For now, release builds on Windows keep debug symbols in separate PDB files generated by the linker.

**Files modified (already done in Phase 0 scaffolding):**
- `CMakeLists.txt` — MSVC flags, no RPATH, `.dll` ARA suffix, Windows bundle directory, Windows ONNX Runtime URL, runtime DLL deploy rules.

**CI additions (not yet in `.github/workflows/ci.yml`):**
- `build-windows` job on `windows-2022` — build Release, verify `clap_entry` / `ARAGetFactory` via `dumpbin /EXPORTS`, verify model deployment, run Python model export, upload artifact.
- `test-windows` job — build Debug with `BUILD_TESTING=ON`, run CTest.

### Phase 3 — CI Matrix + Documentation

Phase 3 consolidates the CI into a build matrix and updates project documentation.

**CI matrix structure:**

```yaml
strategy:
  matrix:
    os: [ubuntu-22.04, macos-14, windows-2022]
```

Each OS entry:
- Installs system dependencies (cmake, compiler, python).
- Installs Python model deps (torch, onnx, onnxruntime).
- Configures with CMake.
- Builds Release.
- Verifies exported symbols (readelf / nm / dumpbin).
- Verifies RPATH / search path (readelf / otool / dumpbin).
- Verifies ONNX Runtime deployment.
- Verifies model file deployment.
- Runs Python model export verification.
- Uploads plugin artifact (with OS-specific suffix in artifact name).

**Test jobs:**

Each OS gets its own test job or uses a `needs` dependency chain. Tests are OS-agnostic — the same `TEST_ASSERT` / `TEST_PASS` framework works everywhere.

**Documentation updates:**

- `README.md` — Update build instructions to mention Windows (MSVC) and macOS (Xcode CLT) prerequisites.
- `README.md` — Add cross-platform build badge to CI status.

## Files Modified — Summary

| File | Phase | Change |
|------|-------|--------|
| `src/core/Platform.hpp` | 0 | **New file.** `MAMBA_EXPORT` macro, `mambaResolveModulePath()`, `mambaResolveModelDir()`, `mambaFileExists()`. Portable across Windows/POSIX. |
| `CMakeLists.txt` | 0 | Platform-conditional flags, RPATH, ONNX Runtime URL/library, ARA suffix, bundle path, runtime deploy, install rules. |
| `src/clap/MambaClapPlugin.cpp` | 0 | Include `core/Platform.hpp`, use `mambaResolveModulePath()` instead of inline `dladdr`. File existence check via `mambaFileExists()`. |
| `src/ara/MambaAraController.cpp` | 0 | Already uses `core/Platform.hpp` and `mambaResolveModulePath()`. |
| `src/ara/MambaAraController.hpp` | 0 | Already uses `MAMBA_EXPORT` for `ARAGetFactory`. |
| `.github/workflows/ci.yml` | 1, 2, 3 | macOS build + test jobs exist. Windows build + test jobs needed. Matrix consolidation in Phase 3. |
| `tests/CMakeLists.txt` | 0 | Already uses `UNIX AND NOT APPLE` guard for `pthread dl`. No changes needed — `std::ifstream` for file checks is portable. |
| `tests/test_ara_controller.cpp` | 0 | Already handles both `/` and `\` path separators in `test_model_dir_resolution()`. |
| `src/core/MambaInferenceEngine.hpp` | — | No changes — pure ONNX Runtime wrapper, no platform-specific code. Portable as-is. |
| `src/core/MambaInferenceEngine.cpp` | — | No changes — `std::thread_local`, `std::atomic`, `std::vector`, `std::memcpy` are all standard C++17. |
| `export_mamba_onnx.py` | — | No changes — pure Python, portable across all platforms. |

## Consequences

### Positive

- **Single codebase, three platforms** — no forks, no platform-specific compilation units, no `#ifdef` spaghetti in business logic.
- **`Platform.hpp` is the only platform-aware header** — all OS differences are abstracted behind inline functions in one file. The rest of the codebase is portable C++17.
- **Zero behavior change on Linux** — all Phase 0 changes are behind platform guards or compile to identical code on Linux.
- **macOS support is already building in CI** — build-macos and test-macos jobs exist and pass.
- **Minimal Windows CI effort** — the CMakeLists.txt already has Windows-conditional flags, paths, and deploy rules. Only the CI workflow YAML needs new jobs.
- **`MAMBA_EXPORT` macro works correctly on all three platforms** — MSVC gets `__declspec(dllexport)`, GCC/Clang get `__attribute__((visibility("default")))`, featureless compilers get empty.
- **ONNX Runtime ships prebuilt binaries for all three platforms** — no need to build ORT from source.
- **Test framework is already portable** — printf + assert works identically on all platforms.
- **C++17 standard is well-supported** — MSVC 2019+, Apple Clang 10+, GCC 8+ all have complete C++17 support.

### Negative

- **Windows DLL search path is implicit** — `onnxruntime.dll` must be in the same directory as the plugin binary. If the DAW modifies the DLL search path (e.g., via `SetDllDirectory`), resolution may fail. This is the standard deployment model for Windows audio plugins and is widely supported.
- **No Windows CI yet** — Phase 2 (Windows CI jobs) is not yet implemented. Manual Windows builds are possible but not verified in CI.
- **macOS code signing and notarization** — macOS audio plugins must be signed (Apple Developer Program) and notarized for distribution without gatekeeper warnings. This requires a paid developer account and CI integration. Not addressed in this ADR.
- **ARM (Windows ARM64 / macOS ARM64 native) not covered** — ONNX Runtime ships ARM64 binaries for macOS, but Windows ARM64 support is not yet addressed. macOS arm64 builds work via Rosetta 2 or native universal binaries.
- **`/WX` not equivalent to `-Werror=return-type`** — MSVC's "treat warnings as errors" is all-or-nothing. The project uses `-Werror=return-type` on Linux/macOS to catch missing return values in C-ABI callbacks. MSVC does not have a precise equivalent, so some return-value bugs may go undetected on Windows.
- **PDB files are not stripped** — Windows release builds ship with PDB files (debug information) by default. A post-build step or install rule is needed to exclude PDBs from distribution artifacts.
- **ONNX Runtime version must be bumped independently per platform** — ORT releases are published simultaneously for all platforms, but the download URLs differ. The version string `1.17.1` is shared in CMake, but platform-specific URLs must be kept in sync.
- **Test CMakeLists.txt has OS-specific link dependencies** — `pthread dl` on Linux only. If a future test needs a Windows-specific library (e.g., `winmm` for performance counters), the CMake must be updated with another platform guard.

### Constraints

- `Platform.hpp` must remain a **single header** — no `.cpp` file. Inline functions avoid adding a build dependency. If the header grows significantly (currently 85 lines), consider splitting into `Platform.hpp` (exports/resolution) and `Platform.cpp` (heavier utility functions).
- `mambaResolveModulePath(nullptr, ...)` must fall back to `"./models/<file>"` on all platforms (defense-in-depth). Already implemented.
- All platform functions in `Platform.hpp` use `#if defined(_WIN32)` / `#elif defined(__APPLE__)` / `#else` (POSIX). No platform identity is assumed — `__APPLE__` is detected before `__GNUC__` because Apple Clang defines both.
- The CLAP plugin entry point (`clap_entry`) uses `CLAP_EXPORT` from the CLAP SDK, not `MAMBA_EXPORT`. The ARA entry point (`ARAGetFactory`) uses `MAMBA_EXPORT`. Both resolve to `__declspec(dllexport)` on Windows — this is correct per the respective SDK conventions.
- ONNX Runtime must remain **side-by-side** with the plugin binary on all platforms. No system-wide installation, no `PATH` modification, no registry keys. This is already the deployment strategy on all three platforms.

## Compliance

- Verified: `Platform.hpp` defines `MAMBA_EXPORT` via `_WIN32` / `__GNUC__ || __clang__` guards.
- Verified: `Platform.hpp` `mambaResolveModulePath()` uses `GetModuleHandleExA` + `GetModuleFileNameA` on Windows, `dladdr` on POSIX.
- Verified: `Platform.hpp` `mambaResolveModelDir()` same pattern.
- Verified: `Platform.hpp` `mambaFileExists()` uses `std::ifstream` — portable.
- Verified: All path resolution detects both `/` and `\\` separators.
- Verified: `CMakeLists.txt` has `if(MSVC)`, `elseif(APPLE)`, `else()` blocks for flags, RPATH, ONNX Runtime URL, runtime lib names, ARA suffix, bundle paths, deploy rules, install rules.
- Verified: `CMakeLists.txt` guards `pthread dl` with `if(UNIX AND NOT APPLE)`.
- Verified: `CMakeLists.txt` guards `-s` strip with `if(UNIX AND CMAKE_BUILD_TYPE STREQUAL "Release")`.
- Verified: `CMakeLists.txt` `if(WIN32)` / `elseif(APPLE)` / `else()` for ONNX Runtime deploy and install rules.
- Verified: macOS CI (`build-macos`) exists in `.github/workflows/ci.yml`, builds Release, verifies symbols and `@loader_path`.
- Verified: macOS CI (`test-macos`) exists, builds Debug with tests, runs CTest.
- Verified: `MambaClapPlugin.cpp` includes `core/Platform.hpp`, uses `mambaResolveModulePath()` (not raw `dladdr`).
- Verified: `MambaClapPlugin.cpp` uses `mambaFileExists()` (not `access()`).
- Verified: `MambaAraController.hpp` uses `MAMBA_EXPORT` for both `getMambaARAFactory()` and `ARAGetFactory()`.
- Verified: `tests/test_ara_controller.cpp` `test_model_dir_resolution()` checks for both `/` and `\\` path separators.
- Verified: `tests/CMakeLists.txt` guards `pthread dl` with `if(UNIX AND NOT APPLE)`.
- Verified: `src/core/MambaInferenceEngine.hpp` / `.cpp` contain zero platform-specific code.
- Verified: `export_mamba_onnx.py` contains zero platform-specific code.

## Related

- `src/core/Platform.hpp` — Single portability header (85 lines)
- `CMakeLists.txt` — Platform-conditional build configuration throughout
- `src/clap/MambaClapPlugin.cpp` — Uses `Platform.hpp` instead of inline `dladdr`
- `src/ara/MambaAraController.hpp` — Uses `MAMBA_EXPORT` for factory entry point
- `src/ara/MambaAraController.cpp` — Uses `Platform.hpp` for model path resolution
- `src/core/MambaInferenceEngine.hpp` — No changes needed (portable C++17)
- `src/core/MambaInferenceEngine.cpp` — No changes needed (`thread_local` + `std::vector`)
- `tests/test_ara_controller.cpp:22-41` — Handles both `/` and `\` path separators
- `tests/CMakeLists.txt` — Platform-conditional link dependencies
- `.github/workflows/ci.yml` — macOS CI exists, Windows CI needed
- ADR-006: `dladdr`-Based Model Path Resolution — Evolution from `dladdr` to portable `Platform.hpp`
- ADR-004: CLAP + ARA Dual-Target Build Architecture — RPATH strategy, `$ORIGIN` → platform-conditional
- ADR-001: Shared ONNX Inference Engine — ONNX Runtime dependency, deployment contract
- `README.md` — Build instructions, model export, architecture summary
