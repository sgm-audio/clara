# ADR-004: CLAP + ARA Dual-Target Build Architecture

**Status:** Accepted (implemented) — see the update note below
**Date:** 2026-05-30

> **Update 2026-10-04 (historical record otherwise unchanged).** The plugin file
> was renamed `MambaSuite.clap` → **`MambaSuite.clap`** so clap-wrapper's
> VST3 name match succeeds, and the install layout on Linux is now a *bare*
> `.clap` plus a sibling `models/` rather than a bundle (a bundle directory and a
> same-named bare file cannot coexist). The links and layout in the body of this
> ADR describe the tree as it was in May 2026, when the engine was ONNX-based.

## Context

CLARA produces two shared library binaries from a single CMake project:

1. **`MambaSuite.clap`** — A CLAP audio effect plugin, loaded by DAWs via `dlopen()` on the `.clap` file.
2. **`MambaSuiteARA.so`** — An ARA 2.0 library, loaded by DAWs via the ARA factory entry point.

Both share one inference core (`MambaCore` static library) that encapsulates `MambaInferenceEngine` and ONNX Runtime. The build system must compile this core once, link it into both targets, and deploy runtime dependencies (ONNX Runtime shared library, model files) alongside each plugin — without requiring the user to install any system packages or set environment variables.

### Dependency Strategy Options

| Approach | Effort | Reproducibility | User Friction | Verdict |
|----------|--------|-----------------|---------------|---------|
| **System packages** (`apt install libonnxruntime-dev`) | Low | None — depends on distro version | High (user must install deps) | Rejected |
| **vcpkg** | Medium | Good (vcpkg.json manifest) | High (user must install vcpkg, run bootstrap) | Rejected |
| **Conan** | Medium | Good (conanfile.txt) | High (user must install Conan, configure profile) | Rejected |
| **git submodules** | Low | Good (pinned commits) | Medium (user must init/update submodules) | Rejected |
| **FetchContent** (chosen) | Low | High (pinned commits / URLs) | None — automatic fetch at cmake configure | **Chosen** |

## Decision

### 1. FetchContent for All Three Dependencies — No System Packages

All dependencies are fetched automatically at CMake configure time. The user only needs CMake and a C++17 compiler. No `apt install`, no `vcpkg install`, no submodule init.

**CLAP SDK** (pinned to commit `29ffcc27`, corresponds to CLAP 1.2.7):
```cmake
FetchContent_Declare(
    clap
    GIT_REPOSITORY https://github.com/free-audio/clap.git
    GIT_TAG 29ffcc273be7c7c651f6c9953b99e69700e2387a
)
```

**ARA SDK** (pinned to commit `1ca0b0aa`):
```cmake
FetchContent_Declare(
    ara_sdk
    GIT_REPOSITORY https://github.com/Celemony/ARA_SDK.git
    GIT_TAG 1ca0b0aa65a73e932daa4a9402387a8afa5afd17
)
```

**ONNX Runtime v1.17.1** (prebuilt Linux x64 tarball):
```cmake
FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/.../onnxruntime-linux-x64-1.17.1.tgz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
```

ONNX Runtime uses URL download rather than git because Microsoft publishes prebuilt binary tarballs. `DOWNLOAD_EXTRACT_TIMESTAMP TRUE` is required for reproducibility with CMake 3.24+ and is safe because we specify a fixed version URL.

**Why not `find_package` (system packages)?**

ONNX Runtime is not consistently packaged across Linux distros. Ubuntu 22.04 ships no `libonnxruntime-dev`. Fedora ships an older version. ARA SDK and CLAP SDK are not packaged at all. Requiring system packages would limit the project to developers who can install them — a barrier for open-source contributions.

**Why not vcpkg?**

vcpkg is a valid alternative but adds a bootstrapping step (`./bootstrap-vcpkg.sh`) and requires the user to understand vcpkg's triplet and integration model. FetchContent is simpler: one `cmake ..` invocation fetches everything. The tradeoff is that FetchContent does not cache built artifacts across projects (vcpkg does), but for a single-project build this is irrelevant.

**Why not git submodules?**

Submodules require explicit `git submodule update --init` and become stale when the parent repository is cloned without `--recursive`. FetchContent automatically fetches at configure time, ensuring the developer always gets the pinned version.

### 2. Pinned Versions for Reproducibility

Every dependency is pinned to a specific git tag (CLAP, ARA) or release URL (ONNX Runtime). No floating `main` branches or `latest` tags. This ensures:

- A checkout of the project at any point in time builds with identical dependencies.
- Two developers building from the same commit produce the same binary (modulo compiler version differences).
- Breaking changes in an upstream SDK do not surprise the team mid-development.

Upgrading a dependency is a deliberate commit that changes a single hash or URL.

### 3. `MambaCore` Static Library — Compiled Once, Linked Twice

```cmake
add_library(MambaCore STATIC src/core/MambaInferenceEngine.cpp)
target_include_directories(MambaCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(MambaCore PUBLIC ${ONNXRUNTIME_LIB})
set_target_properties(MambaCore PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)
```

**Why static library?**

| Alternative | Issue |
|-------------|-------|
| **Compile `.cpp` into each target** (list the source file in both `add_library` calls) | Duplicates compilation time. A fix to `MambaInferenceEngine.cpp` recompiles twice. Object code is duplicated in both binaries. Single point of failure if one target's source list drifts out of sync. |
| **Header-only** (`inline` functions or `.hpp`-only) | ORT headers are large; two translation units per target would bloat compile time. Just enough ORT API surface is needed to justify a `.cpp` file. |
| **Shared library** (`.so` for MambaCore) | Adds a third `.so` to deploy alongside both plugins. The inference engine has no reason to be independently versioned or hot-swappable. On Linux, linking a shared library from two other shared libraries adds complexity to the RUNPATH chain. |
| **Static library** (current) | Compiled once, linked into both targets. Linker extracts only referenced symbols — minimal binary size increase. No extra deployment artifact. Single fix point. |

**Why `POSITION_INDEPENDENT_CODE ON`?**

On Linux, a static library linked into a shared library (`MODULE` or `SHARED`) must be compiled with `-fPIC`. CMake's `POSITION_INDEPENDENT_CODE ON` property ensures this. Without it, the linker would reject `MambaCore` with a relocation error.

### 4. MODULE Library Semantics — Correct for `dlopen()`-Loaded Plugins

Both targets use `add_library(... MODULE ...)`. CMake's MODULE library type is specifically designed for plugins loaded by `dlopen()`:

| Library Type | `lib` prefix? | Suffix | Output Location | Link Type |
|-------------|---------------|--------|-----------------|-----------|
| `SHARED` | Yes (`libfoo.so`) | `.so` | `CMAKE_LIBRARY_OUTPUT_DIRECTORY` | Standard dynamic library |
| `MODULE` | No (`foo.so`) | `.so` | `CMAKE_LIBRARY_OUTPUT_DIRECTORY` | `dlopen()`-only — cannot be used as a link dependency |
| `STATIC` | N/A | `.a` | `CMAKE_ARCHIVE_OUTPUT_DIRECTORY` | Archive for link-time inclusion |

MODULE libraries are not linkable by other targets (no `target_link_libraries(otherTarget MambaSuiteClap)`). They cannot have `SONAME` set. CMake does not add the `lib` prefix. These are exactly the properties needed for audio plugins.

**CLAP target (`MambaSuiteClap`):**
```cmake
add_library(MambaSuiteClap MODULE src/clap/MambaClapPlugin.cpp)
set_target_properties(MambaSuiteClap PROPERTIES
    PREFIX ""
    SUFFIX ".clap"
    ...
)
```

**ARA target (`MambaSuiteARA`):**
```cmake
add_library(MambaSuiteARA MODULE src/ara/MambaAraController.cpp ...)
set_target_properties(MambaSuiteARA PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    ...
)
```

### 5. The `.clap` Suffix Trick

CLAP plugins use the `.clap` extension for discovery. CMake has no built-in CLAP plugin target type. We abuse MODULE library properties:

- `PREFIX ""` — Prevents CMake from adding `lib` (output is `MambaSuite.clap`, not `libMambaSuite.clap`).
- `SUFFIX ".clap"` — Overrides the default `.so` suffix with `.clap`.

This is a simple but important configuration detail. ADAW scanning for `.clap` files will not recognize a file named `libMambaSuiteClap.so`.

The ARA target keeps `SUFFIX ".so"` and `PREFIX ""` (no `lib` prefix) for the same reason — ARA libraries are discovered by filename convention.

### 6. Common Include Directories

```cmake
set(COMMON_INCLUDES
    ${clap_SOURCE_DIR}/include
    ${ara_sdk_SOURCE_DIR}
    ${ara_sdk_SOURCE_DIR}/ARA_Library/Dispatch
    ${ONNXRUNTIME_INCLUDE_DIR}
)
```

Both plugins need access to the CLAP headers (for the entry point), ARA headers (for the factory and controller), and ONNX Runtime headers (for the inference engine via `MambaCore`). Rather than duplicating the list across both `target_include_directories` calls, we define `COMMON_INCLUDES` once.

Note the asymmetry: `MambaCore` does NOT need the CLAP or ARA headers — it's purely an inference wrapper. Only the plugin targets need them. So `COMMON_INCLUDES` is applied PRIVATE to each plugin target, not PUBLIC to `MambaCore`.

### 7. Output Directory — Both Targets in `build/plugins/`

```cmake
set_target_properties(MambaSuiteClap PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    ...
)
set_target_properties(MambaSuiteARA PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/plugins
    ...
)
```

Both targets write to `build/plugins/`. This directory mirrors the install layout: runtime dependencies (`libonnxruntime.so`, `models/`) are also copied there. The `$ORIGIN` RPATH means everything just works when run from the build tree.

### 8. System Library Linking — `pthread` and `dl`

```cmake
target_link_libraries(MambaSuiteClap PRIVATE MambaCore pthread dl)
target_link_libraries(MambaSuiteARA PRIVATE MambaCore pthread dl)
```

- **`pthread`**: Required for the PLAP plugin's `std::thread` usage (if any in future) and by ORT's internal thread pool (when intranode threading is enabled).
- **`dl`**: Required for `dladdr()` in both plugins (see ADR-006). CLAP uses `dladdr` to resolve the model path relative to the `.clap` binary; ARA uses it relative to the `.so`.

### 9. Release Build — Strip Symbols

```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(MambaSuiteClap MambaSuiteARA PROPERTIES
        LINK_FLAGS "-s"
    )
endif()
```

`-s` strips all symbol table and debug information from the final binary. Audio plugins benefit from reduced file size (faster DAW scan) and obfuscation (harder to reverse-engineer internal function names). Debug builds omit `-s`, preserving symbols for profiling and crash analysis.

### 10. Model and Runtime Deployment at Build Time

**Models (`models/` directory):**
```cmake
if(EXISTS ${CMAKE_SOURCE_DIR}/models)
    file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR}/plugins)
endif()
```

Copies the entire `models/` directory (containing `mamba_rt.onnx` and `mamba_rt.onnx.data`) into `build/plugins/models/`. The `EXISTS` guard means the build does not fail if the models directory is absent — useful for CI environments that may not have trained models.

**ONNX Runtime shared library:**
```cmake
set(ONNXRUNTIME_SO ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.17.1)
if(EXISTS ${ONNXRUNTIME_SO})
    file(COPY ${ONNXRUNTIME_SO} DESTINATION ${CMAKE_BINARY_DIR}/plugins)
endif()
```

Copies `libonnxruntime.so.1.17.1` from the FetchContent download directory to `build/plugins/`. The `$ORIGIN` RPATH (see next section) ensures the runtime linker finds it adjacent to the plugin binaries.

### 11. RPATH Strategy — `$ORIGIN` at Build and Install

```cmake
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")
set_target_properties(MambaSuiteClap PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
    ...
)
set_target_properties(MambaSuiteARA PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
    ...
)
```

**What `$ORIGIN` means:** The dynamic linker resolves `libonnxruntime.so.1.17.1` relative to the directory containing the running plugin binary. If `MambaSuite.clap` is in `build/plugins/`, the linker looks for `build/plugins/libonnxruntime.so.1.17.1`.

**`BUILD_WITH_INSTALL_RPATH ON`** is critical: it tells CMake to use the install RPATH settings (`$ORIGIN`) during development builds too. Without this, CMake would embed an absolute build-tree path in the plugin's RPATH, and the plugin would fail to find `libonnxruntime.so` when loaded from the install tree (because the embedded path would point to the build tree).

**Why it works identically in both trees:**
- **Build tree:** `build/plugins/` contains the plugin, `libonnxruntime.so`, and `models/`. All are siblings via `$ORIGIN`.
- **Install tree:** The CLAP bundle structure (see §12) places the plugin binary, runtime lib, and models in the same directory. The ARA install places them in a flat `lib/ara/` directory. Both resolve via `$ORIGIN`.

No `LD_LIBRARY_PATH` manipulation, no `patchelf --set-rpath`, no wrapper scripts.

### 12. Install Directory Structure

```
# CLAP bundle (for host that discover bundles):
${CMAKE_INSTALL_PREFIX}/lib/clap/MambaSuite.clap/
  Contents/
    linux-x86-64/
      MambaSuite.clap        # the plugin binary
      libonnxruntime.so.1.17.1    # ONNX Runtime
      libonnxruntime.so           # symlink for SONAME resolution
      models/
        mamba_rt.onnx
        mamba_rt.onnx.data

# Bare CLAP binary (for hosts that don't support bundles):
${CMAKE_INSTALL_PREFIX}/lib/clap/MambaSuite.clap

# ARA standalone:
${CMAKE_INSTALL_PREFIX}/lib/ara/
  MambaSuiteARA.so
  libonnxruntime.so.1.17.1
  libonnxruntime.so
  models/
    mamba_rt.onnx
    mamba_rt.onnx.data
```

The CLAP bundle structure follows the CLAP specification for bundle-based discovery: the plugin binary lives inside a `.clap` directory at `Contents/linux-x86-64/`. The bare `.clap` file (also installed) is a flat copy for hosts that scan individual files.

### 13. Component-Based Install

```cmake
# CLAP-only:
install(TARGETS MambaSuiteClap LIBRARY DESTINATION ... COMPONENT clap)
install(FILES ... DESTINATION ... COMPONENT clap)  # bare .clap
install(FILES ... DESTINATION ... COMPONENT clap)  # runtime deps
install(DIRECTORY ... DESTINATION ... COMPONENT clap)  # models

# ARA-only:
install(TARGETS MambaSuiteARA LIBRARY DESTINATION ... COMPONENT ara)
install(FILES ... DESTINATION ... COMPONENT ara)  # runtime deps
install(DIRECTORY ... DESTINATION ... COMPONENT ara)  # models
```

Components enable selective installation:

```bash
cmake --install . --component clap   # CLAP bundle + bare .clap only
cmake --install . --component ara    # ARA .so only
cmake --install .                    # both (default)
```

This is useful for distribution packaging, CI/CD artifact separation, and users who only need one plugin type.

Each component is self-contained: the CLAP install includes `libonnxruntime.so` and models alongside the `.clap` file. The ARA install includes the same for the `.so`. No cross-component dependencies.

### 14. Minimum CMake 3.14

`cmake_minimum_required(VERSION 3.14)` is the floor. The specific constraint: `FetchContent_Declare` with `URL` and `DOWNLOAD_EXTRACT_TIMESTAMP` requires CMake 3.14+ (the `DOWNLOAD_EXTRACT_TIMESTAMP` option was added in 3.24 but is harmless on 3.14 — older versions ignore it). CMake 3.14 is available in Ubuntu 20.04 LTS and all newer LTS releases.

The project would build with CMake 3.11+ (adding `FetchContent_MakeAvailable` shim required), but we choose 3.14 to avoid compatibility workarounds.

### 15. Test Infrastructure

```cmake
add_subdirectory(tests)
```

Tests are opt-in via `-DBUILD_TESTING=ON`. When enabled, CMake descends into the `tests/` directory. This minimal incantation keeps the main CMakeLists.txt focused on the build architecture.

## Consequences

### Positive
- **Zero system dependency installation** — single `cmake .. && cmake --build` workflow.
- **Pinned dependencies** ensure reproducible builds across machines and time.
- **Compiled-once `MambaCore`** avoids duplicate compilation and object code.
- **MODULE library semantics** produce correct `dlopen()`-ready binaries without `lib` prefix.
- **`.clap` suffix** makes CLAP discovery work without post-build renaming.
- **`$ORIGIN` RPATH** works identically in build tree and install tree — no environment variable or post-install patching.
- **Component-based install** allows selective deployment.
- **Self-contained install components** — each plugin directory includes everything needed at runtime.

### Negative
- **FetchContent downloads on every clean build** (unless cached) — adds ~40MB for ONNX Runtime tarball on first configure.
- **FetchContent does not cache across projects** — unlike vcpkg, each project fetches its own copy of dependencies.
- **No system package fallback** — if a system already has the dependencies installed, FetchContent still downloads them.
- **`BUILD_WITH_INSTALL_RPATH ON` means the build tree is tied to `$ORIGIN` semantics** — cannot relocate `build/` without breaking RPATH resolution.
- **CMake 3.14 minimum** excludes older LTS distributions (Ubuntu 18.04 ships CMake 3.10).
- **ONNX Runtime from URL, not git** — no easy way to diff local changes against the upstream version.

### Constraints
- `MambaCore` must always be compiled with `-fPIC` (`POSITION_INDEPENDENT_CODE ON`) for static link into shared libraries.
- The `COMMON_INCLUDES` list must be kept in sync if a dependency's include path changes.
- The `.clap` target is named `MambaSuiteClap`; the install rules reference the output filename directly — renaming the target requires updating install rules.
- Build-tree `$ORIGIN` assumption: `build/plugins/` layout must contain plugin + ORT runtime + models. Any deviation breaks runtime resolution.
- `-s` strip flag is Linux-specific — cross-platform builds will need a conditional.

## Compliance

- Verified: `FetchContent_Declare` for all three dependencies in `CMakeLists.txt:18-41`.
- Verified: `MambaCore` is `STATIC` with `POSITION_INDEPENDENT_CODE ON` (`CMakeLists.txt:57-66`).
- Verified: `MambaSuiteClap` is `MODULE` with `PREFIX ""` and `SUFFIX ".clap"` (`CMakeLists.txt:69-85`).
- Verified: `MambaSuiteARA` is `MODULE` with `PREFIX ""` and `SUFFIX ".so"` (`CMakeLists.txt:93-109`).
- Verified: Both targets link `pthread` and `dl` (`CMakeLists.txt:82-85`, `105-108`).
- Verified: Release builds use `LINK_FLAGS "-s"` (`CMakeLists.txt:112-116`).
- Verified: Models copied to `build/plugins/models/` at build time (`CMakeLists.txt:119-121`).
- Verified: `libonnxruntime.so.1.17.1` copied to `build/plugins/` at build time (`CMakeLists.txt:124-128`).
- Verified: Install rules for CLAP bundle, bare `.clap`, and ARA `.so` with component separation (`CMakeLists.txt:159-208`).
- Verified: `BUILD_WITH_INSTALL_RPATH ON` for both plugin targets (`CMakeLists.txt:77`, `101`).
- Verified: `CMAKE_INSTALL_RPATH "$ORIGIN"` set globally and per-target (`CMakeLists.txt:12-13`, `75-76`, `99-100`).

## Related

- `CMakeLists.txt` — entire file, 212 lines
- `src/core/MambaInferenceEngine.cpp` — compiled into `MambaCore`
- `src/clap/MambaClapPlugin.cpp` — compiled into `MambaSuiteClap`
- `src/ara/MambaAraController.cpp` — compiled into `MambaSuiteARA`
- ADR-001: Shared ONNX Runtime Inference Engine (MambaCore context, deployment contract)
- ADR-006: `dladdr`-Based Model Path Resolution (runtime dependency on `$ORIGIN` layout)
- README.md — build instructions, verification commands
