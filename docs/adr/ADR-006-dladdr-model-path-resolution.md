# ADR-006: `dladdr`-Based Model Path Resolution

**Status:** Accepted (implemented) — see the update note below

> **Update 2026-10-04.** The download-path part of this ADR is ONNX-era history
> (`mamba_rt.onnx` and its sidecar), but two things moved: the ARA engine now
> passes a real symbol address to `mambaResolveModulePath()` instead of
> `nullptr` (`MambaAraController.cpp`, which had left it resolving against the
> host's working directory), and on Linux the install layout is a bare
> `lib/clap/MambaSuite.clap` with a sibling `models/` rather than a bundle —
> `lib/clap/<stem>.clap/Contents/...` and a same-named bare file cannot coexist.
> The plugin was renamed `MambaSuiteClap.clap` → `MambaSuite.clap` so the VST3
> wrapper finds it.
**Date:** 2026-05-30

## Context

Both CLARA plugins (`MambaSuiteClap.clap`, `MambaSuiteARA.so`) must locate the ONNX model file (`models/mamba_rt.onnx`) and its external data sidecar (`mamba_rt.onnx.data`) at runtime. The model directory must work identically in the build tree and the install tree without hardcoded paths, environment variables, or user configuration.

### Deployment Layout

The expected runtime directory structure (identical for development in `build/plugins/` and deployment under `/usr/local/lib/clap/`):

```
plugin_binary(.clap or .so)/
  libonnxruntime.so.1.17.1    # resolved via $ORIGIN RPATH
  models/
    mamba_rt.onnx
    mamba_rt.onnx.data         # external data sidecar (weights)
```

The challenge: plugins are loaded dynamically (`dlopen`) by the DAW. Standard approaches for locating self-referencing files fail in the DAW plugin environment:

| Approach | Failure Mode |
|----------|-------------|
| `argv[0]` | DAWs typically do not set `argv[0]` for plugin processes, or it points to the DAW executable. Unreliable. |
| `getcwd()` / `/proc/self/cwd` | Working directory is the DAW's (e.g., `~/.config/renoise` or the DAW's install directory). Not the plugin's directory. |
| `/proc/self/exe` | Resolves to the DAW executable path, not the plugin shared object. |
| Hardcoded path | Breaks on different install prefixes (`/usr/local` vs `/opt` vs user home). Violates deployment contract. |
| Environment variable | Requires user setup (`MAMBA_MODEL_PATH`). Fragile, poor UX, not discoverable. |
| `dlinfo()` with `RTLD_DI_ORIGIN` | Linux-specific. Not available on musl or older glibc. Only returns origin of the library, not a specific calling context. |

### Why `dladdr`

`dladdr()` is a POSIX function (available on Linux, macOS, BSD) that takes a function pointer and returns the filesystem path of the shared object containing that function. By passing a known function in our own binary, we get the exact path of the plugin's `.clap` or `.so` file.

```c
// dladdr signature:
int dladdr(void* addr, Dl_info* info);
// On success: info->dli_fname points to the path of the shared object
```

Key properties:
- **POSIX-standard** — available on all Linux libc implementations (glibc, musl, uclibc).
- **Path-accurate** — returns the exact path used by `dlopen`, not a resolved symlink or `$ORIGIN` expansion.
- **No user setup** — works automatically regardless of install location.
- **Identical results in build and install trees** — because the plugin binary is always loaded from its deployed location.

## Decision

### 1. CLAP Plugin: `dladdr` on `clap_create_plugin`, Instance Storage

In `clap_create_plugin()` (`MambaClapPlugin.cpp:415-429`), the factory function resolves the model path once and stores it in `MambaPluginState::modelPath`:

```cpp
// MambaClapPlugin.cpp:422-425
Dl_info info;
if (dladdr((void*)clap_create_plugin, &info) && info.dli_fname) {
    state->modelPath = resolveModelPath(info.dli_fname, "mamba_rt.onnx");
}
```

The `resolveModelPath()` helper (`MambaClapPlugin.cpp:18-26`) extracts the directory portion of the plugin path and appends `models/<filename>`:

```cpp
static std::string resolveModelPath(const char* plugin_path, const char* model_filename) {
    std::string base(plugin_path ? plugin_path : "");
    auto sep = base.rfind('/');
    if (sep != std::string::npos)
        base.resize(sep + 1);
    else
        base = "./";
    return base + "models/" + model_filename;
}
```

If `dladdr` fails (e.g., statically linked or unusual loading scenario), `plugin_path` is `nullptr`, and `resolveModelPath` falls back to `"./models/mamba_rt.onnx"` (relative to the DAW's working directory). This is a best-effort fallback.

The resolved path is stored per-instance in `MambaPluginState::modelPath` because the CLAP plugin could theoretically support multiple instances with different model paths. This is also consistent with the CLAP lifecycle — `modelPath` is referenced in `mamba_activate()` (`MambaClapPlugin.cpp:245-247`), where it feeds into the `MambaInferenceEngine::Config`.

### 2. ARA Plugin: `dladdr` on `resolveModelDirOnce`, Static Local Cache

In `MambaAraController.cpp:9-26`, the ARA library uses a static helper function that resolves the model directory once per process lifetime:

```cpp
static std::string resolveModelDirOnce() {
    Dl_info info;
    if (dladdr((void*)resolveModelDirOnce, &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        auto sep = path.rfind('/');
        if (sep != std::string::npos)
            path.resize(sep + 1);
        else
            return "./";
        return path + "models/";
    }
    return "./models/";
}

const std::string& getMambaModelDir() {
    static const std::string dir = resolveModelDirOnce();
    return dir;
}
```

Key differences from the CLAP approach:

| Aspect | CLAP | ARA |
|--------|------|-----|
| **Resolution pointer** | `clap_create_plugin` (factory function) | `resolveModelDirOnce` (helper itself) |
| **Storage** | Instance member `MambaPluginState::modelPath` | `static const std::string` in `getMambaModelDir()` |
| **Lifetime** | Per-instance | Process-global (`static local` initialized once) |
| **Granularity** | Full model filename path | Directory only (caller appends `"mamba_rt.onnx"`) |

The difference in storage strategy reflects the different constraints:
- CLAP: `clap_create_plugin` can be called multiple times (once per plugin instance). Each instance stores its own `modelPath`. This allows each instance to theoretically use a different model — though currently all instances use `"mamba_rt.onnx"`.
- ARA: The factory is a singleton enforced by `ARA::PlugIn::PlugInEntry::getPlugInEntry<F, D>()`, which creates a single static `PlugInEntry` instance. There is no per-instance factory call. The model directory is resolved once and cached in a `static const std::string` for the process lifetime.

Thread safety of the ARA cache: In C++11, initialization of function-scope `static` variables is guaranteed to be thread-safe (the compiler emits a guard variable with double-checked locking). The first call to `getMambaModelDir()` from any thread will safely initialize `dir` exactly once.

### 3. Model Path Usage in the ARA PlaybackRenderer

The ARA `MambaPlaybackRenderer` constructor (`MambaAraController.cpp:32-40`) retrieves the cached directory and constructs the full model path:

```cpp
MambaInferenceEngine::Config config;
config.modelPath = getMambaModelDir() + "mamba_rt.onnx";
config.maxBlockSize = 1024;
rtEngine_ = std::make_unique<MambaInferenceEngine>(config);
engineValid_ = rtEngine_->init();
```

Note: ARA uses `maxBlockSize = 1024` (vs CLAP's `std::max(host_max, 256)`) because ARA rendering is offline — latency is not a concern, and larger blocks improve throughput.

### 4. Deployment Contract: Two Required Files

The model file `mamba_rt.onnx` is exported with ONNX external data format, which produces a sidecar file `mamba_rt.onnx.data`. Both files must be present in the `models/` directory:

- `mamba_rt.onnx` — Contains the model graph and metadata. Small file (~few KB).
- `mamba_rt.onnx.data` — Contains the actual weight tensors. Large file (~few MB for 274K params as float32).

If `mamba_rt.onnx.data` is missing, `Ort::Session::Run()` throws an `Ort::Exception` with an error like "external data file not found". This is caught by the engine's exception handlers — `init()` returns `false`, and the plugin deactivates gracefully.

The `models/` directory is deployed adjacent to the plugin binary in both:
- **Build tree**: `file(COPY ...)` in `CMakeLists.txt` copies `source/models/` → `build/plugins/models/`
- **Install tree**: Install rules deploy `models/` alongside the plugin under `lib/clap/MambaSuiteClap.clap/...` or `lib/`

The `$ORIGIN` RPATH is set on both targets via:

```cmake
set_target_properties(MambaSuiteClap MambaSuiteARA PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)
```

This ensures `libonnxruntime.so.1.17.1` is resolved relative to the plugin binary's directory. The model resolution via `dladdr` complements this — both use the same deployment-relative approach (one for the runtime library, one for the model file).

### 5. Fallback Behavior

If `dladdr` returns failure (impossible under normal DAW loading, but defense-in-depth), both implementations fall back to `"./models/"`:

- CLAP: `resolveModelPath(nullptr, ...)` returns `"./models/mamba_rt.onnx"`
- ARA: `resolveModelDirOnce()` returns `"./models/"`

This fallback is intentionally relative to the DAW's current working directory, which will fail in most hosts but provides a last-resort search path for development scenarios (e.g., running the plugin from a script that `cd`s to the build directory first).

## Consequences

### Positive
- **Zero-configuration deployment** — plugins find their own model automatically regardless of install location.
- **Identical behavior in build and install trees** — the same code path works for development and distribution.
- **No environment variables, config files, or registry keys.** Plugin is truly self-contained.
- **POSIX-standard** — no Linux-specific APIs or conditional compilation. Same code works on macOS if ported.
- **Graceful fallback** — if `dladdr` fails, the plugin can still attempt to find its model via `"./models/"`.

### Negative
- **`dladdr` is a Linux-ism in practice** — it's available on macOS (`_dyld_get_image_name` is the preferred API) and BSD, but not on Windows. A Windows port would need `GetModuleFileName` + `GetModuleHandleEx` (passing the address of a known function). The `#ifdef` burden for cross-platform support is manageable but non-zero.
- **Fragile if plugin is statically linked or embedded** — if the plugin binary is not a `dlopen`-loaded shared object, `dladdr` on its own function may return the path of the executable that loaded it (e.g., the DAW), not the plugin. This is not a concern for the current deployment model (CLAP/ARA shared libraries loaded via `dlopen`), but would break if the code were reused in a different context.
- **ARA static cache is process-global** — if a DAW hosts multiple ARA plugin instances from different plugin directories, the first `getMambaModelDir()` call wins. In practice, DAWs (e.g., REAPER, Cubase) load ARA plugins once per factory, and the factory is a singleton — this scenario cannot occur.
- **`models/` directory name is hardcoded** — cannot be changed without code modification. If this becomes a requirement, the model directory should be parameterized (e.g., via `$MAMBA_MODEL_DIR` environment variable with `dladdr` as default).

### Constraints
- Both `mamba_rt.onnx` and `mamba_rt.onnx.data` must exist in the `models/` subdirectory relative to the plugin binary.
- The model path must be set in `MambaInferenceEngine::Config::modelPath` before calling `init()`. In both CLAP and ARA, this is guaranteed by the resolution happening before engine construction.
- `$ORIGIN` RPATH must be set on both plugin targets (already done in `CMakeLists.txt`) for `libonnxruntime.so` resolution. The model resolution via `dladdr` is independent of `$ORIGIN` — it uses `dlinfo`-level path information.

## Compliance

- Verified: `MambaClapPlugin.cpp:422-425` — `dladdr((void*)clap_create_plugin, &info)` in factory entry point, stored in `state->modelPath`.
- Verified: `MambaClapPlugin.cpp:18-26` — `resolveModelPath()` helper extracts directory and appends `"models/mamba_rt.onnx"`.
- Verified: `MambaClapPlugin.cpp:245-247` — `state->modelPath` used in `mamba_activate()` to set `config.modelPath`.
- Verified: `MambaAraController.cpp:9-26` — `resolveModelDirOnce()` uses `dladdr` on itself, `getMambaModelDir()` caches result in `static const std::string`.
- Verified: `MambaAraController.cpp:36` — `getMambaModelDir() + "mamba_rt.onnx"` used in `MambaPlaybackRenderer` constructor.
- Verified: Both plugins fall back to `"./models/"` if `dladdr` returns null.
- Verified: Model files (`mamba_rt.onnx`, `mamba_rt.onnx.data`) are deployed via CMake `file(COPY ...)` and install rules, adjacent to plugin binaries.
- Verified: `$ORIGIN` RPATH is set on both targets for `libonnxruntime.so` resolution.

## Related

- `src/clap/MambaClapPlugin.cpp:18-26` — `resolveModelPath()` helper
- `src/clap/MambaClapPlugin.cpp:422-425` — `dladdr` call in `clap_create_plugin()`
- `src/ara/MambaAraController.cpp:9-26` — `resolveModelDirOnce()` and `getMambaModelDir()`
- `src/ara/MambaAraController.hpp:10` — `getMambaModelDir()` declaration
- `src/core/MambaInferenceEngine.hpp` — `Config::modelPath` field
- `CMakeLists.txt` — `$ORIGIN` RPATH configuration, model deploy rules
- `ADR-001: Shared ONNX Inference Engine` — deployment contract context, model file listing
- `ADR-004: CLAP + ARA Dual-Target Build Architecture` — build system deploy rules, install tree layout
