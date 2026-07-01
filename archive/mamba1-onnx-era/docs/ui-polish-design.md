# CLARA-10: UI Polish Design Document

**Status:** Draft — thinking/planning deliverable
**Date:** 2026-06-02
**Scope:** CLAP plugin GUI (CLAP_EXT_GUI) — no code changes, no source file modifications

---

## Table of Contents

1. [Framework Choice](#1-framework-choice)
2. [Layout & Controls](#2-layout--controls)
3. [Visual Design](#3-visual-design)
4. [Resize Behavior](#4-resize-behavior)
5. [DAW Theme Integration](#5-daw-theme-integration)
6. [Implementation Plan](#6-implementation-plan)
7. [Out of Scope](#7-out-of-scope)
8. [Technical Architecture](#8-technical-architecture)
9. [Open Questions](#9-open-questions)

---

## 1. Framework Choice

### 1.1 Candidates Evaluated

| Criterion | JUCE | pugl | Raw Platform APIs |
|-----------|------|------|-------------------|
| **License** | GPLv3 / commercial ($) | ISC (permissive) | N/A |
| **Codebase size** | ~1.5M LOC | ~15K LOC | ~6K LOC per platform |
| **Compile time** | Slow (precompiled headers needed) | Fast (single .c + headers) | Fast (per-platform) |
| **Widget toolkit** | Rich (sliders, combos, labels, L&F) | None (raw GL/D2D/Cairo) | None (raw GDI/CoreGraphics/Xlib) |
| **CLAP embedding** | Native clap-juce-extensions | Built-in (puglSetParentWindow) | Manual HWND/NSView/X11 Window |
| **ARA editor view** | Native ARA editor host via JUCE | Manual integration | Manual integration |
| **Cross-platform** | Win/Mac/Linux/iOS/Android | Win/Mac/Linux | Per-platform rewrite |
| **Dependency size** | ~100MB git clone | ~2MB git clone | 0 (system APIs) |
| **FetchContent** | Complex (modules, Projucer) | Simple (single CMake target) | N/A |
| **Real-time safety** | message thread, not RT | event loop, not RT | event loop, not RT |
| **DPI awareness** | Built-in | Manual (set_scale callback) | Manual |
| **Rendering backend** | JUCE Graphics (SW), OpenGL, D2D | OpenGL, Vulkan, Cairo | GDI/D2D, CoreGraphics, Xlib/XCB |
| **Plugin industry usage** | Dominant (VST3/AU/AAX) | CLAP ecosystem standard | Rare in modern plugins |

### 1.2 Detailed Analysis

#### JUCE

JUCE is the industry-standard framework for audio plugin UIs. It provides a complete widget toolkit (sliders, buttons, combo boxes, labels), a retained-mode 2D graphics API (`juce::Graphics`), and built-in support for CLAP, VST3, AU, and AAX.

**Arguments for JUCE:**
- Mature LookAndFeel system for custom styling
- Rich set of pre-built widgets (Slider, ComboBox, Label, TextButton)
- Native ARA editor view hosting via `juce::ARADocumentControllerSpecialisation`
- Large community, abundant examples

**Arguments against JUCE:**
- **Licensing**: GPLv3 for open source. CLARA may want a permissive license (the CLAP SDK and ARA SDK are both permissively licensed). JUCE's GPL requires the entire plugin to be GPL-compatible, which conflicts with the proprietary ARA SDK. A commercial JUCE license adds $700+/year cost and licensing complexity.
- **Build complexity**: JUCE requires Projucer or complex CMake integration. Adding it as FetchContent is fragile — JUCE's module system expects a specific directory layout. The Projucer-generated project would need to be committed or regenerated per-platform.
- **Compile time**: JUCE is a large framework. Incremental builds in a plugin project would suffer. CLARA's current build is fast (a few .cpp files).
- **Dependency bloat**: AGENTS.md specifies "Three dependencies via CMake FetchContent" — adding JUCE would be a 4th dependency, but more importantly it's a massive one compared to the others.
- **Overkill**: CLARA's UI is simple — 3 knobs, 1 dropdown, 1 label, optional meters. JUCE's full widget toolkit is disproportionate.

#### pugl

pugl is a minimal, embeddable GUI library designed specifically for audio plugins. It provides a platform window with an OpenGL/Vulkan/Cairo drawing context and keyboard/mouse event dispatch. It does NOT provide widgets — all drawing is done programmatically.

**Arguments for pugl:**
- **ISC license**: Maximally permissive. No viral clauses, no commercial fees. Compatible with all existing CLARA dependencies.
- **CLAP ecosystem standard**: pugl is used by the official CLAP example plugins (`clap-plugins`). U-He uses pugl for their CLAP ports. The CLAP embedding model (parent HWND/NSView/X11 Window) is pugl's primary use case.
- **Tiny codebase**: ~15K lines of C. Easy to audit, fast to compile, trivial to vendor or FetchContent. Keeps CLARA's dependency footprint in line with the "three dependencies" philosophy.
- **Explicit embedding API**: `puglSetParentWindow()` accepts the native handle provided by the CLAP host via `clap_window_t`. This is a 1:1 mapping — no adapter layer needed.
- **Cairo rendering**: pugl's Cairo backend provides a retained-mode 2D vector graphics API (`cairo_t`) that is sufficient for CLARA's simple UI (circles for knobs, rectangles for meters, text for labels). No OpenGL requirement on systems where OpenGL drivers are unreliable (certain Windows configurations with DAW GPU conflicts).
- **DPI scaling**: pugl's `set_scale` callback maps directly to CLAP's `clap_plugin_gui.set_scale`. No impedance mismatch.
- **Resize**: pugl supports resizable windows natively. CLARA's required resize behavior (proportional knob scaling, vertical→horizontal layout breakpoint) is straightforward with a raw draw loop.

**Arguments against pugl:**
- **No built-in widgets**: Knobs, dropdowns, and meters must be drawn from scratch using Cairo/OpenGL primitives. This is more work than JUCE's `juce::Slider`, but CLARA's UI is simple enough that custom drawing is manageable (estimated 300-500 lines of drawing code for all controls).
- **No accessibility**: pugl provides no screen-reader support. JUCE has accessibility features. Given CLARA's current stage (no trained model, no GUI at all), accessibility can be deferred to a later iteration.
- **Manual layout**: Responsive layout (breakpoint at 500px width) must be implemented manually. pugl has no layout manager. However, with 6-8 total controls, a manual layout function is straightforward.
- **Smaller community**: pugl has fewer examples and Stack Overflow answers than JUCE. However, the CLAP example plugins provide working reference implementations.

#### Raw Platform APIs (Win32/Cocoa/X11)

Writing platform-specific UI code using Win32 `CreateWindowEx` + GDI/GDI+, Cocoa `NSView` + CoreGraphics, and X11 `XCreateWindow` + Cairo.

**Arguments against:**
- **3x code**: Every drawing routine, layout calculation, event handler, and DPI scaling logic must be written 3 times. For CLARA's simple UI this is ~900-1500 lines per platform, but the maintenance burden is high — a bug fix or feature addition requires changes in 3 places.
- **CLAP already abstracts this**: The CLAP GUI extension provides the parent window handle. pugl already wraps the per-platform window creation. Writing raw platform code duplicates what pugl already solves.
- **No benefit over pugl**: pugl is essentially a thin wrapper over these same APIs with a consistent C interface. There is no performance or capability advantage to raw APIs for a plugin UI.

### 1.3 Recommendation: **pugl + Cairo**

**Rationale summary:**

1. **License compatibility**: ISC license is fully compatible with CLARA's existing dependencies (CLAP SDK ISC-like, ARA SDK proprietary-but-free, ONNX Runtime MIT). No GPL contamination risk.
2. **Minimal dependency**: ~15K LOC, simple FetchContent integration. Aligns with AGENTS.md philosophy of "Three dependencies via CMake FetchContent." pugl can be fetched as a 4th dependency with negligible build impact.
3. **CLAP native**: pugl was designed for exactly this use case — embedding a plugin UI in a DAW-provided window. The API matches CLAP's `clap_plugin_gui_t` callbacks 1:1.
4. **Sufficient for CLARA's needs**: 3 knobs, 1 dropdown, 1 label, 2 optional meters. This is well within pugl's sweet spot. Custom Cairo drawing for a rotary knob is ~60 lines; for a gain reduction meter ~80 lines.
5. **Future-proof**: If CLARA later needs an ARA editor view, pugl can be embedded in the ARA host's view hierarchy. If CLARA later needs OpenGL-accelerated visuals (e.g., real-time spectrogram), pugl supports an OpenGL backend without changing the window management code.
6. **Industry precedent**: U-He (MFM2, ColourCopy, Presswerk CLAP ports), DISTRHO, and the official CLAP example plugins all use pugl. This is the de facto standard for CLAP plugin UIs outside the JUCE ecosystem.

**Rejected alternatives and why:**

- **JUCE**: GPL license incompatible with proprietary ARA SDK; massive dependency disproportionate to CLARA's simple UI; complex build integration.
- **Raw platform APIs**: Triples implementation and maintenance cost with zero benefit over pugl.
- **SDL2**: Larger than pugl (~500K LOC), designed for games not plugin UIs, embedding model is less clean.
- **Dear ImGui**: Immediate-mode GUI is poor for plugin UIs (no DAW-standard look-and-feel, requires continuous redraw even when idle, GPU-dependent).
- **NanoVG**: OpenGL-only. Cairo fallback desirable for systems where OpenGL in a DAW context is unreliable.

### 1.4 Build Integration Strategy

```cmake
# pugl is header + single .c compilation unit, or FetchContent from GitHub
FetchContent_Declare(
    pugl
    GIT_REPOSITORY https://github.com/lv2/pugl.git
    GIT_TAG v0.4.0   # or latest stable
)
FetchContent_MakeAvailable(pugl)
# pugl's CMake produces a static library target: pugl::pugl_cairo
```

Cairo can be provided by the system on Linux (`libcairo2-dev`) and macOS (built-in CoreGraphics backend via pugl's Quartz Cairo surface). On Windows, cairo can be fetched as a prebuilt binary or built from source via vcpkg/MSYS2. Alternatively, pugl's built-in `pugl_cairo` backend includes a `cairo.h` wrapper — the actual cairo library must be linked separately.

**If Cairo is too heavy**: pugl's `stub` backend provides window creation and event dispatch without any drawing API. The drawing can then be done directly with platform APIs (GDI+ on Windows, CoreGraphics on macOS, Xlib drawing on Linux) through the native window handle exposed by pugl. This eliminates the Cairo dependency but reintroduces per-platform drawing code. Given CLARA's simple UI, the per-platform drawing surface is estimated at ~200 lines per platform.

**Preferred path**: pugl + Cairo on Linux/macOS, pugl + GDI+ on Windows (cairo on Windows can be tricky to link statically). pugl exposes the native window handle, so mixed backends are possible.

---

## 2. Layout & Controls

### 2.1 Control Inventory

| Priority | Control | CLAP Param | Visual Type | Range | Default |
|----------|---------|------------|-------------|-------|---------|
| **P1** | Bypass Toggle | PARAM_BYPASS (0) | Large push button | Off/On (stepped) | Off |
| **P1** | Wet/Dry Knob | PARAM_WET_DRY (1) | Rotary knob w/ readout | 0.0% – 100.0% | 100.0% |
| **P1** | Intensity Knob | PARAM_INTENSITY (2) | Rotary knob w/ readout | 0.0% – 100.0% | 50.0% |
| **P1** | Status Indicator | — | Colored text label | Ready / Loading / Error | "Ready" (green) |
| **P2** | Gain Reduction Meter | — | Vertical bar, dB scale | -∞ to 0 dB | -∞ (idle) |
| **P2** | Cond Vector Viz | — | 2×4 colored bar grid | 0.0 – 1.0 per dim | All zeros |
| **P3** | Model Selector | — | Dropdown | *.onnx in models/ | First model found |
| **P3** | About/Version | — | Popup / tooltip | — | — |

### 2.2 Layout Specifications

#### Minimum Window: 400 × 300 px
#### Maximum Window: 1200 × 800 px
#### Default Window: 600 × 400 px

```
┌────────────────────────────────────────────────────┐
│  ┌──────────┐                    ┌────────────────┐ │
│  │          │                    │  Status: Ready │ │
│  │ BYPASS   │   ┌──────┐ ┌──────┐└────────────────┘ │
│  │  (○/◉)   │   │ WET  │ │ INT  │                  │
│  │          │   │ /DRY │ │ENSITY│                  │
│  │          │   │ 100% │ │ 50%  │                  │
│  └──────────┘   └──────┘ └──────┘                  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │         GAIN REDUCTION METER (Phase 2)       │   │
│  │  dB ┤                                      │   │
│  │   0 ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │   │
│  │  -3 ┤▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │  -6 ┤▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │ -12 ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │ -∞ ┤░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │        CONDITIONING VECTOR (Phase 2)          │   │
│  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐  ┌──┐ ┌──┐ ┌──┐ ┌──┐  │   │
│  │  │▐▐│ │▐▐│ │░░│ │░░│  │▐▐│ │░░│ │░░│ │░░│  │   │
│  │  │▐▐│ │▐▐│ │░░│ │░░│  │▐▐│ │░░│ │░░│ │░░│  │   │
│  │  └──┘ └──┘ └──┘ └──┘  └──┘ └──┘ └──┘ └──┘  │   │
│  │   d0   d1   d2   d3     d4   d5   d6   d7    │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  Model: [mamba_rt.onnx        ▾]   (Phase 3) │   │
│  │  ───────────────────────────────────────────  │   │
│  │  Mamba-SSM Processor v1.1.0        (Phase 3) │   │
│  │  © 2026 Elite DSP                           │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

### 2.3 Control Details

#### 2.3.1 Bypass Toggle

**Behavior:**
- Large circular button (min 48×48 px at default size, scales proportionally)
- Mouse click toggles between Off (0.0) and On (1.0)
- Writes to `std::atomic<float> bypass` via `memory_order_release`
- The existing click-free 50ms linear crossfade in `mamba_process()` handles the audio transition — the GUI just sets the parameter
- The GUI reads `bypass.load(memory_order_acquire)` on each frame update to display current state

**Visual states:**
- **Off (0.0)**: Circle with dim border (#333), dark fill (#1a1a2e), gray "BYPASS" text, gray indicator ring
- **On (1.0)**: Circle with bright border (#e94560), dark fill (#1a1a2e), red indicator ring, bright "BYPASS" text
- **Hover**: Border brightens slightly, subtle outer glow
- **Pressed**: Fill darkens, ring expands slightly (2px → 4px)
- **Transitional**: During the 50ms crossfade, the ring color lerps between off and on colors (driven by `currentBypassValue` read from `MambaPluginState`)

**Accessibility label**: "Bypass — Off" / "Bypass — On"

#### 2.3.2 Wet/Dry Knob

**Behavior:**
- Large rotary knob (min 60×60 px at default size)
- Mouse drag vertical or circular to adjust value
- Writes to `std::atomic<float> wetDry` via `memory_order_release`
- Host automation writes flow through the existing `clap_params_flush` → atomic store path

**Visual design:**
- Circular track: 270° arc (from 7-o'clock to 5-o'clock, bottom gap)
- Filled arc: accent color (#e94560) proportional to value
- Center text: numeric value as percentage (e.g., "100%")
- Label below knob: "WET/DRY" in sans-serif, muted color (#8a8a9a)
- Tick marks: 11 ticks at 0%, 10%, 20%, ... 100%
- Knob indicator: thin white line from center to track position

**Interaction:**
- Click + drag vertical: value moves linearly with mouse Y position
- Click + drag circular: value follows angular position (requires arctan2 calculation)
- Shift + drag: fine adjustment (0.1% increments)
- Scroll wheel: ±1% per notch, ±5% with Shift
- Double-click: reset to default (100.0%)
- Right-click: context menu with "Reset to Default" and "Enter Value..."

**Value display:** Monospace font, centered. Format: `XX%` (integer for values >= 10), `X.X%` (one decimal for values < 10). Show "100%" not "100.0%".

#### 2.3.3 Intensity Knob

**Behavior:**
- Identical visual design to Wet/Dry knob
- Controls `std::atomic<float> intensity` via `memory_order_release`
- Default value: 50.0%
- Label: "INTENSITY"

**Interaction:** Same as Wet/Dry knob. Double-click resets to 50.0%.

#### 2.3.4 Status Indicator

**Behavior:**
- Read-only text label, updated each frame (16ms timer in pugl's event loop)
- Color indicates current plugin state

**States:**

| State | Color | Hex | Text | Condition |
|-------|-------|-----|------|-----------|
| **Ready** | Green | #2ecc71 | "Ready" | Engine initialized, model loaded, processing active |
| **Loading** | Yellow | #f1c40f | "Loading model..." | Engine being initialized in background (model reload) |
| **Error** | Red | #e74c3c | "Error: {message}" | Engine failed to init (model not found, ONNX error, etc.) |
| **Bypassed** | Dim green | #1a6e3a | "Ready (Bypassed)" | Engine ready but bypass is ON |
| **Inactive** | Gray | #666 | "Inactive" | Plugin not activated (between deactivate/activate) |

**Error message source:** The GUI thread reads `engine->initialized_` (via a new public accessor) and if false, checks a new `std::string engineError_` member set during `init()`. If the status is Error, the full error string from the last `init()` failure is displayed.

**Implementation note:** The status label should not query the engine's initialization state on every frame — use a cached `Status` enum updated when the engine state changes (stale reads of cached enum are fine — it's a display-only value).

#### 2.3.5 Model Selector (Phase 3)

**Behavior:**
- Dropdown `<select>` style control (custom-drawn, not native — native dropdowns don't embed well in plugin windows)
- Populated by scanning `models/` directory relative to the plugin binary path for `*.onnx` files
- Directory scan occurs once at plugin activation, result cached in `std::vector<std::string>`
- Selecting a different model pushes a `DeferredTask::ModelReload` message to the SPSC queue
- Audio thread processes the reload at next `mamba_on_main_thread` call (currently stub — will be implemented in Phase 3)
- During reload: status changes to "Loading model..." (yellow)
- On failure: status changes to "Error: {message}" (red), reverts to previous model

**Visual design:**
- Width: 200px min, stretches to available width
- Height: 24px
- Current model name displayed, chevron (▾) on right
- Click: dropdown list appears below, showing all .onnx files in models/
- Selected item: highlighted with accent color
- Hover item: subtle background highlight

**Corner case:** If only one .onnx file exists in models/, the dropdown is disabled (grayed out) with a tooltip: "Only one model available."

#### 2.3.6 About/Version Popup (Phase 3)

**Behavior:**
- Small "ℹ" or "?" button in the bottom-right corner
- Click opens a tooltip or small floating panel above the button
- Click outside dismisses

**Content:**

```
 ┌─────────────────────────────────┐
 │  Mamba-SSM Processor            │
 │  Version: 1.1.0                 │
 │                                 │
 │  Model: mamba_rt.onnx           │
 │  Model dim: d=64, layers=4      │
 │  ONNX Runtime: 1.17.1           │
 │                                 │
 │  Build: Jun 02 2026             │
 │  Platform: linux-x86-64         │
 │                                 │
 │  github.com/elite-dsp/clara      │
 └─────────────────────────────────┘
```

**Data sources:**
- Plugin version: `mamba_plugin_info.version` (currently `"1.1.0"`)
- Model name: `config_.modelPath` filename portion
- Model architecture: `config_.dModel`, `config_.nLayers` from engine config
- ORT version: `ORT_API_VERSION` macro or runtime query via `Ort::GetVersionString()`
- Build date: `__DATE__` compiled into a new `const char* kBuildDate` in `MambaClapPlugin.cpp`
- Platform: `#ifdef` determined string, same as `Platform.hpp` already defines
- Project URL: `MAMBA_PROJECT_URL` macro

### 2.4 Gain Reduction Meter (Phase 2)

**Behavior:**
- Vertical meter bar, 200px tall minimum, scales with window height
- Shows real-time gain reduction applied by the Mamba SSM model
- Dual indicator: peak hold (bright bar) + RMS average (dimmer bar underneath)
- Peak hold: decays at 3 dB/sec after 1.5s hold
- RMS: sliding window of ~100ms

**Data source:** The gain reduction value must be calculated in `processBlock()` and exposed to the GUI. This requires a new `std::atomic<float> gainReductionDb_` in `MambaInferenceEngine`, updated per-block as:
```
maxInputAbs = max(|input[i]|)
maxOutputAbs = max(|output[i]|)
gainReductionDb = 20 * log10(maxOutputAbs / maxInputAbs)  // clamped to [-60, 0]
```

The GUI reads this atomic each frame. The meter visual is updated at the pugl idle timer rate (~30-60 FPS).

**Visual design:**
- Background: dark rectangle (#1a1a2e)
- GR bar: gradient from green (#2ecc71 top, -3 dB) → yellow (#f1c40f, -12 dB) → red (#e94560, -∞)
- Peak hold dot: small white triangle or dot at the peak position
- Scale labels: 0, -3, -6, -12, -24, -60, -∞ on the left side (tiny monospace font)
- Width: 24px fixed, height: full available

**Corner case:** When bypass is ON, meter shows 0 dB GR (no processing). When engine is not initialized, meter is dimmed and shows -∞.

### 2.5 Conditioning Vector Visualization (Phase 2)

**Behavior:**
- 2×4 grid of small colored bars, each representing one dimension of `cond[COND_DIM]` (COND_DIM=8)
- Reads `activeCond_` pointer from the engine via `getActiveConditioningForTest()` (or new public accessor `getActiveCond()`)
- Each bar height proportional to the absolute value of that dimension's scalar
- Color indicates sign: warm (red/orange) for positive, cool (blue) for negative
- All zeros: bars are flat/dimmed (idle state when no ARA context is available)

**Visual design:**
- Each cell: 24×48 px minimum
- Bar fill: gradient from bottom (dark) to top (bright)
- Label below each bar: "d0" through "d7" in tiny monospace
- Hover on a bar: tooltip showing exact value, e.g., "d3: +0.457"
- Grid spacing: 4px gap between cells

**Data source:** `engine->getActiveCond()` returns `const float*` to current active buffer (or nullptr if never set). Since the cond vector is double-buffered with atomic pointer swap, the GUI reads the pointer once per frame — it may get a slightly stale buffer, which is acceptable for a visualization.

**Why 2×4 not 1×8:** A 2×4 grid is more visually compact and fits better in the plugin's horizontal layout at wider sizes. At narrow (<500px) sizes, this grid stacks below the other controls.

### 2.6 Control Placement Map

#### Wide Layout (≥ 500px width) — Side by Side

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│   ┌─────────┐   ┌──────────┐   ┌──────────┐         │
│   │         │   │          │   │          │         │
│   │ BYPASS  │   │ WET/DRY  │   │INTENSITY │         │
│   │  (○)    │   │   100%   │   │   50%    │         │
│   │         │   │          │   │          │         │
│   └─────────┘   └──────────┘   └──────────┘         │
│                                                      │
│   ┌──────────────────────────────────────────────┐   │
│   │  [Status: Ready]                  [Model ▾]  │   │
│   └──────────────────────────────────────────────┘   │
│                                                      │
│   ┌──────────────┐  ┌────────────────────────────┐   │
│   │  Gain Red.   │  │  Conditioning Vector       │   │
│   │  Meter       │  │  [d0][d1][d2][d3]         │   │
│   │  (Phase 2)   │  │  [d4][d5][d6][d7]         │   │
│   └──────────────┘  └────────────────────────────┘   │
│                                                      │
│   ┌────────────────────────────────────────────────┐ │
│   │  v1.1.0 | build Jun 02 2026 | mamba_rt.onnx  │ │
│   └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

#### Narrow Layout (< 500px width) — Stacked Vertically

```
┌─────────────────┐
│                 │
│  ┌───────────┐  │
│  │  BYPASS   │  │
│  │   (○)     │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │ WET/DRY   │  │
│  │   100%    │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │INTENSITY  │  │
│  │   50%     │  │
│  └───────────┘  │
│                 │
│  [Status: Ready]│
│  [Model ▾]      │
│                 │
│  ┌───────────┐  │
│  │GR Meter   │  │
│  │(Phase 2)  │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │Cond Viz   │  │
│  │(Phase 2)  │  │
│  └───────────┘  │
│                 │
│  v1.1.0 | ...  │
└─────────────────┘
```

---

## 3. Visual Design

### 3.1 Color Palette

```
┌─────────────────────────────────────────────────────┐
│ ROLE              │ HEX       │ RGB           │ USE │
├───────────────────┼───────────┼───────────────┼─────┤
│ Background        │ #1a1a2e   │ 26, 26, 46    │ Window background, panel fills │
│ Accent Primary    │ #e94560   │ 233, 69, 96   │ Knob fills, bypass-on ring, active highlights │
│ Secondary         │ #0f3460   │ 15, 52, 96    │ Knob tracks, meter backgrounds, borders │
│ Text Primary      │ #eaeaea   │ 234, 234, 234 │ Labels, knob values, status text │
│ Text Secondary    │ #8a8a9a   │ 138, 138, 154 │ Sub-labels, scale text, dimmed text │
│ Text Disabled     │ #555566   │ 85, 85, 102   │ Inactive controls, disabled text │
│ Success/Ready     │ #2ecc71   │ 46, 204, 113  │ Status Ready, GR meter -3 to 0 dB │
│ Warning/Loading   │ #f1c40f   │ 241, 196, 15  │ Status Loading, GR meter -12 to -6 dB │
│ Error/Danger      │ #e74c3c   │ 231, 76, 60   │ Status Error, GR meter -∞ to -12 dB │
│ Meter Peak Hold   │ #ffffff   │ 255, 255, 255 │ Peak hold indicator dot │
│ Border/Divider    │ #2a2a4a   │ 42, 42, 74    │ Control borders, section dividers │
│ Hover Highlight   │ rgba(233,69,96,0.15)        │ Subtle hover glow on controls │
│ Cond Pos (warm)   │ #e94560   │ 233, 69, 96   │ Positive cond dimension bar │
│ Cond Neg (cool)   │ #3498db   │ 52, 152, 219  │ Negative cond dimension bar │
│ Cond Zero (idle)  │ #2a2a3a   │ 42, 42, 58    │ Zero cond dimension bar (flat) │
└─────────────────────────────────────────────────────┘
```

**Rationale:**
- **#1a1a2e** background: Very dark blue-black. Easier on the eyes in a dim studio than pure #000000. Provides enough color temperature to make the accent and status colors pop without being harsh.
- **#e94560** accent: Warm coral-red. High visibility against dark backgrounds. Used sparingly — only for the active state indicator and knob fill arcs. Avoids the "Christmas decoration" effect of pure red on black.
- **#0f3460** secondary: Deep navy blue. Good for inactive track arcs and meter backgrounds — provides visual structure without competing with the accent color.
- **Status colors**: Standard traffic-light semantics (green=good, yellow=caution, red=bad). High enough saturation to be visible at small sizes but not garish.
- **Gain reduction gradient**: Green (safe) → Yellow (moderate) → Red (heavy). Standard metering convention in pro audio.

### 3.2 Typography

| Role | Font Family | Weight | Size (default) | Example |
|------|------------|--------|----------------|---------|
| Knob values | Monospace (Consolas / SF Mono / DejaVu Sans Mono) | Bold | 18px | `100%` |
| Knob labels | Sans-serif (Segoe UI / SF Pro Text / Roboto) | Regular | 11px | `WET/DRY` |
| Status text | Sans-serif | Semibold | 12px | `Ready` |
| Meter scale | Monospace | Regular | 8px | `-12` |
| Section headers | Sans-serif | Semibold | 10px | `GAIN REDUCTION` |
| Version info | Monospace | Regular | 9px | `v1.1.0` |
| Cond dimension labels | Monospace | Regular | 7px | `d3` |
| Dropdown items | Sans-serif | Regular | 11px | `mamba_rt.onnx` |

**Font fallback chain (per platform):**

```
Windows:  "Segoe UI", "Consolas", "Arial", sans-serif
macOS:    "SF Pro Text", "SF Mono", "Helvetica Neue", sans-serif
Linux:    "Roboto", "DejaVu Sans Mono", "Noto Sans", sans-serif
```

**Implementation:** Font rendering via Cairo's `cairo_show_text()` + `cairo_select_font_face()`. Cairo handles the font fallback chain. Monospace vs sans-serif is selected per text element.

### 3.3 Spacing & Sizing

| Element | Default Size | Scaling Behavior |
|---------|-------------|------------------|
| Window (default) | 600 × 400 | Free resize |
| Window (minimum) | 400 × 300 | Hard constraint, enforced in `adjust_size` |
| Window (maximum) | 1200 × 800 | Soft constraint (can exceed if host forces) |
| Bypass button diameter | 64px | `clamp(48, 0.11 * windowWidth, 120)` |
| Knob diameter | 72px | `clamp(52, 0.12 * windowWidth, 144)` |
| Knob label font size | 11px | `clamp(8, 0.018 * windowWidth, 16)` |
| Knob value font size | 18px | `clamp(12, 0.03 * windowWidth, 24)` |
| Padding (horizontal) | 24px | `clamp(12, 0.04 * windowWidth, 48)` |
| Padding (vertical) | 20px | `clamp(10, 0.05 * windowHeight, 40)` |
| Gap between controls | 16px | `clamp(8, 0.027 * windowWidth, 32)` |
| Status bar height | 24px | Fixed (does not scale) |
| Meter width | 24px | Fixed (does not scale) |
| Cond cell size | 24×48px | `clamp(16, 0.04 * windowWidth, 48) × clamp(32, 0.12 * windowHeight, 96)` |

**Scaling formula:** `clamp(minPx, scaleFactor * windowDim, maxPx)` — linear scaling within the defined range, clamped at both ends. This prevents controls from becoming unusably small at minimum size or absurdly large at maximum size.

### 3.4 Visual States

Every interactive control has 4 states:

1. **Idle**: Default appearance. No mouse interaction.
2. **Hover**: Mouse is over the control. Subtle brightness increase on border/fill. Cursor changes to `hand` (pointer).
3. **Active/Pressed**: Mouse button down on the control. Slight scale-down (98% for buttons, no scale change for knobs) or fill darkening.
4. **Disabled**: Control is non-functional (e.g., model selector when only 1 model, or knobs when engine is not initialized). Appears dimmed (50% opacity overlay). Cursor is `default`.

**Animation/Transitions:**
- Bypass ring color transition: lerp over the 50ms crossfade period (5-6 frames at 60fps)
- Hover glow: 150ms ease-out transition
- Status color change: 300ms ease (loading → ready, ready → error)
- Meter bar update: instant (no smoothing — gain reduction values change fast)
- Peak hold decay: animated (3 dB/sec, updated per frame)
- Cond bar height: 200ms ease-out (smooths jitter from per-block updates)

### 3.5 High-Contrast Mode

Triggered by a new plugin parameter or a right-click context menu option (not by OS setting, since plugin windows don't reliably receive OS accessibility events).

**Changes when enabled:**
- Border thickness: 2px → 4px on all controls
- Font sizes: +2px on all text
- Accent color: #e94560 → #ff6b81 (brighter red for visibility)
- Text color: #eaeaea → #ffffff (pure white)
- Control backgrounds: #1a1a2e → #0a0a1e (darker for contrast)
- Knob track: #0f3460 → #1a4a80 (brighter track)
- Meter scale labels: 8px → 10px
- Minimum window size: 400×300 → 450×340 (to accommodate larger text)

---

## 4. Resize Behavior

### 4.1 Resize Hints

```cpp
// clap_plugin_gui.get_resize_hints
// Returns:
//   can_resize_horizontally = true
//   can_resize_vertically = true
//   aspect_ratio_preserved = false
//   min_width = 400, min_height = 300
//   max_width = 1200, max_height = 800
```

### 4.2 Layout Breakpoints

```
Window Width (px)     Layout Mode         Control Arrangement
─────────────────     ───────────         ───────────────────
400 – 499             Vertical Stack      Bypass → Wet/Dry → Intensity → Status → Meters (stacked vertically)
500 – 799             Horizontal Compact  Bypass | Wet/Dry | Intensity (horizontal row), Status + Meters below
800 – 1200            Horizontal Full     Bypass | Wet/Dry | Intensity (spaced), Meters side-by-side, full footer
```

### 4.3 Proportional Scaling

All control positions and sizes are computed as fractions of the window dimensions, not as absolute pixel positions.

```cpp
struct LayoutParams {
    float bypassX, bypassY, bypassDiam;
    float wetDryX, wetDryY, wetDryDiam;
    float intensityX, intensityY, intensityDiam;
    float statusX, statusY, statusW, statusH;
    float meterX, meterY, meterW, meterH;
    float condGridX, condGridY;
    float footerX, footerY, footerW, footerH;
};

LayoutParams computeLayout(int windowWidth, int windowHeight) {
    // All positions are relative to window dimensions
    // Breakpoints trigger different layout modes
    if (windowWidth < 500) {
        return computeStackedLayout(windowWidth, windowHeight);
    } else if (windowWidth < 800) {
        return computeCompactLayout(windowWidth, windowHeight);
    } else {
        return computeFullLayout(windowWidth, windowHeight);
    }
}
```

### 4.4 Resize Implementation

```
User drags window corner
    │
    ▼
Host calls clap_plugin_gui.adjust_size(&width, &height)
    │  Plugin clamps to [400, 1200] × [300, 800]
    │  Returns true (size accepted)
    ▼
Host calls clap_plugin_gui.set_size(width, height)
    │  Plugin resizes pugl view: puglSetSize(view, width, height)
    │  Plugin invalidates layout cache
    │  Next frame: computeLayout() returns new params, redraw all
    ▼
pugl expose event → redraw with new layout
```

**Performance:** Layout computation is O(1) — just arithmetic on window dimensions. Redraw is triggered by pugl's expose event. At idle (no mouse interaction, no parameter changes), the plugin does not redraw (pugl is damage-based, not continuous).

### 4.5 Minimum Size Enforcement

If the host ignores the resize hints and requests a size below 400×300, the plugin still draws correctly at any size — but controls will be clipped and the layout becomes unusable. The `adjust_size` callback is the primary defense:

```cpp
bool gui_adjust_size(const clap_plugin_gui_t*, uint32_t* width, uint32_t* height) {
    *width  = std::clamp(*width, 400u, 1200u);
    *height = std::clamp(*height, 300u, 800u);
    return true;
}
```

If the host sends sizes below minimum in `set_size` (non-conforming host), the layout computation still works — it just produces tiny/clipped controls. No crash, but visually degraded. Log a warning via `clap_host_log` if available.

---

## 5. DAW Theme Integration

### 5.1 CLAP Theme Extension

**Problem:** CLAP does not currently have a standardized host theme query extension. VST3 has `IContextInfoProvider` which returns `IComponentHandler::restartComponent(kLatencyChanged)` but no theme color query. AU has `AUEventListenerNotify` for property changes but no theme API. The DAW plugin UI landscape lacks a universal theme protocol.

**CLARA's approach:**

1. **Check for CLAP host theme extension** (if defined by the CLAP spec by the time UI is implemented). This would be a `CLAP_EXT_THEME` or similar extension defined by the CLAP working group.
2. **Fallback to dark theme** — the standard for professional DAWs (Ableton Live, Bitwig Studio, REAPER, Cubase, Logic Pro, Studio One all default to dark themes).
3. **Expose theme as a plugin parameter** — a `PARAM_THEME` control (stepped, 3 values: Dark, Light, High-Contrast) that is saved in plugin state. This gives users control regardless of host capabilities.
4. **Future: host-specific detection** — if CLAP standardizes a theme extension, CLARA can query it and override the user-selectable theme with `CLAP_PARAM_IS_HIDDEN` for the theme parameter.

### 5.2 Dark Theme (Default)

As specified in Section 3 (Visual Design). This is the only theme implemented in Phase 1.

### 5.3 Light Theme (Phase 3+)

A light variant for users who prefer it or whose DAW uses a light theme (GarageBand, some Linux hosts):

```
Background:        #f0f0f4
Accent Primary:    #d63031
Secondary:         #b2bec3
Text Primary:      #2d3436
Text Secondary:    #636e72
Border/Divider:    #dfe6e9
Control Track:     #b2bec3
Meter Bg:          #dfe6e9
```

Light theme is lower priority — the dark theme serves 95%+ of users. Implementation is just a color palette swap; all layout and widget code is unchanged.

### 5.4 Host Window Background Detection

If a CLAP theme extension is not available, CLARA can attempt a fallback heuristic:
- On Windows: Read the parent window's background color via `GetPixel()` in a non-client area. If darker than 50% luminance, use dark theme. If lighter, use light theme.
- On macOS: Query `NSAppearance` from the parent `NSView.window.effectiveAppearance`. Map `.darkAqua` / `.vibrantDark` → dark theme.
- On Linux: Check GTK theme via `g_settings_get_value()` or environment variable `GTK_THEME`. Heuristic only; may be unreliable across different desktop environments.

**Decision:** For Phase 1, hardcode dark theme. If heuristic detection is feasible and requested, implement in Phase 3. The host detection heuristic is unreliable across all DAWs — REAPER on Windows uses a custom window manager, Bitwig uses Java/SWT, Ableton Live uses custom UI framework. A user-selectable theme parameter is more reliable than heuristic detection.

---

## 6. Implementation Plan

### 6.1 Phase Breakdown

```
Phase 1: Static UI                 Phase 2: Real-Time Metering       Phase 3: Polish + Extras
┌─────────────────────┐           ┌─────────────────────┐           ┌─────────────────────┐
│ Bypass toggle       │           │ Gain reduction meter │           │ Model selector       │
│ Wet/Dry knob        │           │ Cond vector viz      │           │ About/version popup  │
│ Intensity knob      │           │ Peak hold + RMS      │           │ Light theme option   │
│ Status indicator    │           │ GR atomic in engine  │           │ High-contrast mode   │
│ Resize + layout     │           │ Meter animation      │           │ Error recovery UI    │
│ Dark theme colors   │           │                      │           │ Final testing pass   │
│ CLAP GUI extension  │           │                      │           │                      │
│ pugl window + Cairo │           │                      │           │                      │
│ RT-safe param read  │           │                      │           │                      │
├─────────────────────┤           ├─────────────────────┤           ├─────────────────────┤
│ 3-5 days            │           │ 2-3 days            │           │ 1-2 days            │
└─────────────────────┘           └─────────────────────┘           └─────────────────────┘
```

### 6.2 Phase 1: Static UI (3-5 days)

**Prerequisites:**
- pugl integrated as CMake FetchContent dependency
- Cairo library available on target platforms (or pugl stub + platform drawing code)

**Day 1: Infrastructure**
- Add `CLAP_EXT_GUI` to `mamba_get_extension()`
- Implement all `clap_plugin_gui_t` callbacks:
  - `is_api_supported` → return true for platform's native API
  - `get_preferred_api` → return `CLAP_WINDOW_API_WIN32` / `COCOA` / `X11`
  - `create` → create pugl view, attach to host parent window via `puglSetParentWindow()`
  - `destroy` → destroy pugl view, free resources
  - `set_scale` → forward to pugl `puglSetScaleFactor()`
  - `get_size` → return current pugl view size
  - `can_resize` → return true
  - `get_resize_hints` → return min/max size, no aspect ratio lock
  - `adjust_size` → clamp to [400,1200]×[300,800]
  - `set_size` → `puglSetSize()`, trigger relayout
  - `set_parent` → `puglSetParentWindow()` (for DAW window re-parenting)
  - `show`/`hide` → `puglShow()`/`puglHide()`
- Create new source file `src/clap/MambaClapGUI.cpp` (or integrate into `MambaClapPlugin.cpp`)
- Add pugl event loop integration: pugl runs its own event loop on the main thread. The CLAP host's main thread calls `mamba_on_main_thread` periodically — use this or run pugl's event loop in a dedicated timer callback (`clap_host_timer_register`).

**Day 2: Drawing Primitives**
- Implement Cairo-based drawing helpers:
  - `drawRotaryKnob(cairo_t*, x, y, diameter, value, label, color, trackColor)`
  - `drawBypassButton(cairo_t*, x, y, diameter, isOn, hoverState, pressState)`
  - `drawStatusLabel(cairo_t*, x, y, w, h, status, message)`
  - `drawSectionDivider(cairo_t*, x, y, w)`
- Implement `computeLayout(width, height)` function with the breakpoint logic from Section 4.2

**Day 3: Control Interaction**
- Implement mouse event handling in pugl event callback:
  - `PUGL_BUTTON_PRESS` / `PUGL_BUTTON_RELEASE`: hit-test controls, start/end drag
  - `PUGL_MOTION`: update knob values during drag
  - `PUGL_SCROLL`: wheel adjustment on knobs
- Implement keyboard input (Tab for control focus, arrow keys for value adjustment)
- Wire control changes to parameter atomics:
  - Knob drag → `wetDry.store(newValue, memory_order_release)` + `host->request_param_flush()`
  - Bypass click → `bypass.store(newValue, memory_order_release)` + `host->request_param_flush()`
  - Read atomics on each frame for display values
- Implement the resize + relayout flow

**Day 4: Integration Testing**
- Test on Linux with Carla / clap-host
- Test resize behavior at boundary conditions
- Test parameter automation from host (host drives parameter → GUI reflects it)
- Test bypass click-free transition (GUI sends 0/1, audio thread ramps)
- Verify zero allocations on audio thread (GUI never touches audio thread allocations)
- Verify no crash on rapid open/close of GUI window

**Day 5: Polish**
- Refine visual appearance (colors, font sizes, margins)
- Add hover states and cursor changes
- Test on macOS (if build available)
- Cross-platform path: stub out Cairo on Windows if not ready; use pugl stub backend + GDI+ drawing for Windows Phase 1

### 6.3 Phase 2: Real-Time Metering (2-3 days)

**Prerequisites:**
- Phase 1 complete and stable

**Day 1: Gain Reduction Atomic**
- Add `std::atomic<float> gainReductionDb_{0.0f}` to `MambaInferenceEngine`
- Add public accessor: `float getGainReductionDb() const { return gainReductionDb_.load(std::memory_order_acquire); }`
- Update `processBlock()` to compute GR: find max absolute input and output magnitudes per block, compute dB ratio, store to atomic
- Handle edge cases: silence input (set GR to 0 dB), zero output (set GR to -60 dB min)
- Add `std::atomic<float> gainReductionPeakDb_{0.0f}` for peak hold

**Day 2: GR Meter Drawing**
- Implement `drawGainReductionMeter(cairo_t*, x, y, w, h, currentDb, peakDb)`
- Draw gradient background, filled bar (current GR), peak hold indicator
- Implement peak hold decay timer (per-frame decay of 3 dB/sec from peak value)
- Implement meter scale labels
- Wire into pugl's expose event: read `gainReductionDb_` atomic each frame

**Day 3: Cond Vector Visualization**
- Implement `drawConditioningGrid(cairo_t*, x, y, cellW, cellH, const float* cond, size_t dim)`
- Draw 2×4 colored bar grid
- Read `engine->getActiveCond()` each frame
- Handle nullptr case (before first `setConditioning()` call): draw all bars flat/dim

### 6.4 Phase 3: Polish + Extras (1-2 days)

**Day 1: Model Selector + Error States**
- Scan `models/` directory for `*.onnx` files at activate time
- Implement custom dropdown drawing
- Wire model selection to `DeferredTask::ModelReload` via SPSC queue
- Implement reload logic in `mamba_on_main_thread` (currently stub):
  ```cpp
  case DeferredTask::ModelReload:
      engine->resetState();
      if (!engine->init(newModelPath)) {
          status = Status::Error;
          errorMessage = engine->getLastError();
      } else {
          status = Status::Ready;
      }
      break;
  ```
- Add error message display with auto-dismiss after 5 seconds
- Handle model reload failure gracefully (revert to previous model, show error)

**Day 2: About Popup + Testing**
- Implement about/version tooltip
- Add light theme color palette (swap color constants)
- Add high-contrast mode toggle
- Add theme parameter (`PARAM_THEME`, stepped, 0=Dark, 1=Light, 2=HighContrast)
- Final cross-platform testing pass
- Test on all three platforms (Linux→macOS→Windows, in order of stability)

### 6.5 Estimated Effort Summary

| Phase | Description | Est. Days | Key Risk |
|-------|-------------|-----------|----------|
| **Phase 1** | Static UI (bypass, knobs, status, layout) | 3-5 | Cairo availability on Windows; pugl FetchContent integration |
| **Phase 2** | GR meter + cond vector viz | 2-3 | GR computation accuracy; atomic visibility from GUI thread |
| **Phase 3** | Model selector, about, polish | 1-2 | Model reload thread safety; error recovery edge cases |
| **Total** | | **6-10 days** | |

### 6.6 Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cairo not easily available on Windows | Medium | Medium | Fallback: pugl stub + GDI+ drawing (~200 extra LOC). Cairo can be bundled as static lib. |
| pugl FetchContent fails on MSVC | Low | Medium | Vendor pugl source (public domain/ISC, ~15K LOC). Add as git submodule or copy into `third_party/pugl/`. |
| GR meter shows flickering values | Low | Low | Add smoothing: exponential moving average on GR atomic reads before displaying. |
| Model reload crashes audio thread | Low | High | Make model reload happen entirely on main thread. Audio thread only checks a flag and dequeues. |
| Host doesn't call `mamba_on_main_thread` fast enough | Medium | Low | If host main thread callback is slow, the GUI event loop in pugl can process the queue directly (pugl runs on main thread too). |
| DAW embeds GUI in unexpected way | Medium | Medium | Test with 3 major DAWs on each platform. Fallback: floating window mode if parent embedding fails. |

---

## 7. Out of Scope

The following features are explicitly excluded from CLARA-10 UI polish. They are documented here so their absence is intentional, not an oversight.

### 7.1 Preset Browser

A preset browser requires:
- A preset file format (beyond the current 16-byte binary state)
- Preset metadata (name, author, category, tags, description)
- File system browsing UI
- Preview/hearing functionality
- Preset management (save, delete, rename, organize, import, export)

**Deferred to:** CLARA-20 (post model training). There are no trained models to make meaningful presets for yet.

### 7.2 A/B Comparison

A/B comparison requires:
- Two complete sets of parameter values (A state and B state)
- Quick toggling between states
- Optional: morphing/blending between A and B

**Deferred to:** CLARA-20. Low value for a 3-parameter plugin — users can just move the knobs back.

### 7.3 Waveform Display

A scrolling waveform display of input and output audio requires:
- Ring buffer of audio samples from the audio thread
- Scrolling graph rendering (OpenGL or fast Cairo)
- Significant per-frame rendering cost

**Deferred to:** CLARA-30. Interesting for demonstrating what the model does, but a "nice to have" not a "must have."

### 7.4 EQ Curve Display

An EQ-style frequency response curve requires:
- FFT analysis of the model's impulse response
- Impulse response measurement (run an impulse through the model, capture output)
- Frequency-domain visualization

**Deferred to:** CLARA-30. Requires significant DSP infrastructure that doesn't exist yet.

### 7.5 MIDI Mapping / Learn

MIDI mapping for parameter control requires:
- MIDI input port from the host
- Mapping learn mode UI
- Stored mapping table

**Deferred to:** Indefinite. CLARA is an audio effect, not a MIDI-controlled instrument. MIDI mapping may be added if user feedback demands it.

### 7.6 Parameter Modulation

Host-side parameter modulation (LFO, envelope follower, etc.) requires:
- Modulation source definitions
- Modulation routing matrix UI
- Per-modulation depth controls

**Deferred to:** Indefinite. CLAP supports host-side modulation via `CLAP_EXT_PARAMS` and `CLAP_PARAM_IS_MODULATABLE` flags. The GUI does not need to implement its own modulation system.

### 7.7 Anything Requiring Trained Models

The entire UI is useful and functional even with untrained random weights:
- Controls work (bypass, wet/dry, intensity) — these are DSP-side only
- Status indicator shows engine health
- Meters show meaningful values (gain reduction from random-weight processing is technically correct, just sounds like noise)
- Model selector and about work regardless of training

No UI feature depends on model quality. The UI is decoupled from the model.

---

## 8. Technical Architecture

### 8.1 CLAP GUI Extension API

The `clap_plugin_gui_t` struct must be fully implemented. Below is the complete callback contract:

```cpp
typedef struct clap_plugin_gui {
    // Returns true if the requested API is supported (WIN32/COCOA/X11)
    bool (*is_api_supported)(const clap_plugin_t* plugin, const char* api, bool is_floating);

    // Returns the preferred API for this platform
    bool (*get_preferred_api)(const clap_plugin_t* plugin, const char** api, bool* is_floating);

    // Create the window. Host provides parent via clap_window_t.
    bool (*create)(const clap_plugin_t* plugin, const char* api, bool is_floating);

    // Destroy the window and free all GUI resources.
    void (*destroy)(const clap_plugin_t* plugin);

    // Set the scale factor (DPI scaling).
    void (*set_scale)(const clap_plugin_t* plugin, double scale);

    // Get the current size of the plugin UI.
    bool (*get_size)(const clap_plugin_t* plugin, uint32_t* width, uint32_t* height);

    // Returns true if the window is resizable.
    bool (*can_resize)(const clap_plugin_t* plugin);

    // Get resize hints (min/max size, aspect ratio constraint).
    bool (*get_resize_hints)(const clap_plugin_t* plugin, clap_gui_resize_hints_t* hints);

    // Adjust a proposed size. Plugin can modify width/height to constrain.
    bool (*adjust_size)(const clap_plugin_t* plugin, uint32_t* width, uint32_t* height);

    // Host requests a size change.
    bool (*set_size)(const clap_plugin_t* plugin, uint32_t width, uint32_t height);

    // Host provides a new parent window (re-parenting).
    bool (*set_parent)(const clap_plugin_t* plugin, const clap_window_t* window);

    // Set transient parent for popup windows.
    bool (*set_transient)(const clap_plugin_t* plugin, const clap_window_t* window);

    // Host suggests a window title.
    void (*suggest_title)(const clap_plugin_t* plugin, const char* title);

    // Show/hide the window.
    bool (*show)(const clap_plugin_t* plugin);
    bool (*hide)(const clap_plugin_t* plugin);
} clap_plugin_gui_t;
```

**Note on `is_floating`:** CLARA's default mode is embedded (not floating). Floating window mode is a potential future feature for detached plugin windows but is out of scope for Phase 1-3.

### 8.2 pugl Integration

```
┌─────────────────────────────────────────────────────────┐
│                    CLAP Host (DAW)                       │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  clap_host_t                                       │  │
│  │  ├─ request_param_flush()                          │  │
│  │  ├─ request_restart()                              │  │
│  │  └─ get_extension()                                │  │
│  └───────────────────────────────────────────────────┘  │
│                         ▲                               │
│                         │ clap_plugin_gui_t callbacks    │
│                         │ (host → plugin)               │
│  ┌──────────────────────┴────────────────────────────┐  │
│  │              MambaGUI (pugl-backed)                │  │
│  │                                                    │  │
│  │  ┌─────────────────────────┐  ┌─────────────────┐ │  │
│  │  │ puglView               │  │ Layout Engine   │ │  │
│  │  │ ┌─────────────────────┐ │  │ ┌─────────────┐ │ │  │
│  │  │ │ Parent HWND/NSView  │ │  │ │ computeLayout│ │ │  │
│  │  │ │ (from host)         │ │  │ │ (width,height│ │ │  │
│  │  │ └─────────────────────┘ │  │ │  → LayoutParams│ │  │
│  │  │ ┌─────────────────────┐ │  │ └─────────────┘ │ │  │
│  │  │ │ Cairo drawing ctx   │ │  │ ┌─────────────┐ │ │  │
│  │  │ │ (per-frame expose)  │ │  │ │ Control Hit  │ │ │  │
│  │  │ └─────────────────────┘ │  │ │ Test         │ │ │  │
│  │  │ ┌─────────────────────┐ │  │ └─────────────┘ │ │  │
│  │  │ │ Mouse/Key events    │ │  │                  │ │  │
│  │  │ │ (PUGL_BUTTON_PRESS, │ │  │                  │ │  │
│  │  │ │  PUGL_MOTION, etc.) │ │  │                  │ │  │
│  │  │ └─────────────────────┘ │  │                  │ │  │
│  │  └─────────────────────────┘  └─────────────────┘ │  │
│  │                                                    │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │              Drawing Primitives               │  │  │
│  │  │  drawRotaryKnob()  drawBypassButton()         │  │  │
│  │  │  drawStatusLabel() drawGainReductionMeter()   │  │  │
│  │  │  drawCondGrid()    drawDropdown()             │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────┬──────────────────────────────┘  │
│                       │                                 │
│                       │ reads atomics (no lock)          │
│                       ▼                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │           MambaPluginState                         │  │
│  │  bypass       ─ std::atomic<float>                 │  │
│  │  wetDry       ─ std::atomic<float>                 │  │
│  │  intensity    ─ std::atomic<float>                 │  │
│  │  engine       ─ unique_ptr<MambaInferenceEngine>   │  │
│  │  modelPath    ─ std::string                        │  │
│  │  deferredTask ─ SPSCQueue<DeferredTask, 64>        │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.3 Thread Safety

```
┌──────────────────────┐          ┌──────────────────────┐
│    GUI Thread          │          │    Audio Thread       │
│    (main/host thread)  │          │    (RT, high pri)     │
├──────────────────────┤          ├──────────────────────┤
│                        │          │                        │
│ Reads bypass     ◄────┼──────────┼── writes bypass        │
│ (acquire load)   ◄────┼──────────┼── (release store)      │
│                        │          │                        │
│ Reads wetDry     ◄────┼──────────┼── writes wetDry        │
│ (acquire load)   ◄────┼──────────┼── (release store)      │
│                        │          │                        │
│ Reads intensity  ◄────┼──────────┼── writes intensity     │
│ (acquire load)   ◄────┼──────────┼── (release store)      │
│                        │          │                        │
│ Reads GR meter   ◄────┼──────────┼── writes GR meter      │
│ (acquire load)   ◄────┼──────────┼── (release store)      │
│                        │          │                        │
│ Reads cond ptr   ◄────┼──────────┼── writes cond ptr      │
│ (acquire load)   ◄────┼──────────┼── (release store)      │
│                        │          │                        │
│ Writes bypass ───┼──────►────────┼── (release store)      │
│ (user knob drag)  │          │   │                        │
│                        │          │                        │
│ Pushes deferred   ───┼──────►────┼── pops deferred        │
│ tasks (ModelReload,│          │   │ tasks in on_main_thread│
│ AnalysisQueueDrain) │          │   │                        │
│ via SPSCQueue.push │          │   │ via SPSCQueue.pop      │
│                        │          │                        │
│ Reads engine status  │          │                        │
│ (non-atomic, safe   │          │                        │
│  because engine     │          │                        │
│  lifecycle is       │          │                        │
│  single-threaded)   │          │                        │
│                        │          │                        │
└──────────────────────┘          └──────────────────────┘
```

**Key thread safety rules:**

1. **GUI reads atomics, never writes** (except for parameter changes from user interaction — these go through `clap_host.request_param_flush()` → host → `clap_params_flush` → atomic store on audio thread — but for simplicity, direct atomic store from GUI is also safe since the audio thread only reads atomics via acquire ordering).

2. **Parameter updates from GUI:** When the user drags a knob, the GUI writes the new value directly to the atomic (`wetDry.store(val, memory_order_release)`) and calls `host->request_param_flush()`. This is safe because:
   - The audio thread reads with `memory_order_acquire`, establishing a happens-before relationship.
   - The CLAP spec allows `request_param_flush()` to be called from any thread.
   - There is no possibility of a write-write conflict (only GUI writes, audio thread reads).

3. **Parameter updates from host automation:** The host calls `mamba_params_flush()` → writes atomics directly. The GUI reads atomics on each frame. Since both use release/acquire ordering, the GUI always sees consistent values.

4. **Deferred tasks:** The audio thread pushes tasks (ModelReload, AnalysisQueueDrain) to the SPSC queue. The main thread (or pugl event loop) pops them. This is the existing `SPSCQueue` mechanism — no changes needed.

5. **Engine lifecycle:** `activate()`/`deactivate()`/`init()` are called from the main thread (not the audio thread). The GUI can safely read engine status (initialized, error message) without atomics because these are only modified on the main thread.

6. **pugl event loop**: pugl's event loop runs on the main thread. The CLAP host calls `mamba_on_main_thread` on the main thread as well. These can be integrated: the pugl event loop processes GUI events, and `mamba_on_main_thread` processes deferred tasks during idle.

### 8.4 File Organization

```
src/
  clap/
    MambaClapPlugin.cpp      # Existing — add GUI extension to get_extension()
    MambaClapPlugin.hpp      # Existing — add GUI state fields
    MambaClapGUI.hpp          # NEW — GUI class declaration
    MambaClapGUI.cpp          # NEW — GUI implementation:
                              #   - CLAP GUI extension callbacks
                              #   - pugl event handlers
                              #   - Cairo drawing primitives
                              #   - Layout computation
  core/
    MambaInferenceEngine.hpp  # Modify — add:
                              #   std::atomic<float> gainReductionDb_
                              #   float getGainReductionDb() const
                              #   const float* getActiveCond() const (public)
                              #   std::string engineError_
                              #   std::string getLastError() const
    MambaInferenceEngine.cpp  # Modify — update processBlock() to compute GR
CMakeLists.txt                # Modify — add pugl FetchContent,
                              #   link MambaSuiteClap against pugl + cairo
```

### 8.5 GUI State Storage

The `MambaPluginState` struct gains a new member:

```cpp
struct MambaPluginState {
    // ... existing members ...

    // GUI state (nullptr when GUI is not created)
    std::unique_ptr<MambaGUI> gui;
};
```

`MambaGUI` is a new class that owns:
- `puglView* view_` — the pugl window
- `LayoutParams currentLayout_` — cached layout for current window size
- `std::vector<std::string> availableModels_` — scanned from models/ dir
- `int selectedModelIndex_` — index into availableModels_
- `Status currentStatus_` — cached status enum { Ready, Loading, Error, Bypassed, Inactive }
- `std::string statusMessage_` — error message when status is Error
- `bool highContrastMode_` — high contrast toggle state
- `float currentGainReductionDb_` — cached GR value for drawing
- `float peakHoldDb_` — peak hold value with decay
- `std::array<float, COND_DIM> cachedCond_` — last-read cond vector values

---

## 9. Open Questions

### Q1: Framework Decision — pugl vs. JUCE?

**Status:** Recommended pugl + Cairo. Needs final decision before starting.

**Decision criteria to resolve:**
- Confirm Cairo availability on Windows (MSYS2 prebuilt or vcpkg cairo package)
- If Cairo on Windows is too problematic: decide between pugl + GDI+ drawing vs. switching to JUCE
- JUCE licensing: if the project decides to go GPL, JUCE becomes viable. If permissive is preferred, JUCE is out.

**Recommendation:** Start with pugl + Cairo on Linux (confirmed workable). Port to macOS (Cairo via Homebrew or built-in CoreGraphics backend). If Windows Cairo is difficult, use pugl + GDI+ direct drawing (no Cairo on Windows). This keeps the decision reversible — JUCE can be adopted later if pugl proves insufficient.

### Q2: Dark-Only vs. Theme-Aware?

**Status:** Dark-only for Phase 1. Theme parameter for Phase 3.

**Pros of dark-only:**
- Less code, less testing
- 95%+ of professional DAW users use dark themes
- No heuristic detection reliability issues

**Pros of theme-aware:**
- Accessibility (some users need light themes)
- Professional polish
- Future-proof if CLAP standardizes theme extension

**Recommendation:** Dark-only for Phase 1-2. Add theme parameter in Phase 3. Do NOT implement heuristic host theme detection — it is unreliable across platforms/DAWs and adds complexity with minimal benefit.

### Q3: How to Visually Represent 8 Cond Dimensions Meaningfully?

**Status:** Open design question. Proposed solution: 2×4 colored bar grid (Section 2.5).

**Alternative ideas:**
- **Radar chart**: 8-axis spider/radar plot. Visually appealing but takes significant space and is harder to read at a glance.
- **Color-mapped circle**: A single circle whose color is a weighted blend of 8 colors (one per dimension). Very compact but loses per-dimension information.
- **Text-only**: Display numeric values as a 2×4 grid of text. Simple but not glanceable.
- **Bar grid (proposed)**: 2×4 colored bars. Compact, glanceable, shows sign+ magnitude per dimension.

**Recommendation:** Start with the 2×4 bar grid. If user feedback indicates a different visualization would be more useful (e.g., once an ARA host bridge populates the cond vector with real musical data), redesign in a future iteration.

### Q4: UI in CLAP Only, or Also ARA Editor Views?

**Status:** CLAP-only for Phase 1-3. ARA editor deferred.

**Rationale:**
- ARA 2.0 `PlaybackRendererInterface` has no `renderSamples()` method — audio processing is companion CLAP plugin only.
- The ARA `EditorView` interface (`ARA_EDITOR_VIEW`) provides an editor window but requires the host to support it.
- The ARA `DocumentController` can provide a custom view, but the ARA SDK's editor API is complex and poorly documented.
- Most ARA hosts (Melodyne, Cubase, Studio One) don't display plugin UIs for ARA processors — they use ARA's own UI framework.
- If an ARA editor view is needed later, pugl can be embedded in the ARA `EditorView` by providing the native view handle from the ARA host. The same `MambaGUI` class can be reused — only the embedding context changes.

**Recommendation:** CLAP-only for now. Revisit ARA editor in CLARA-30 after the CLAP GUI is stable and ARA host integration is more mature.

### Q5: Floating Window Support?

**Status:** Deferred. Embedded-only for Phase 1.

CLAP supports floating plugin windows (`is_floating = true` in `clap_plugin_gui.create()`). Some hosts (Carla, clap-host) prefer floating windows because they don't provide a parent window for embedding.

**Implementation difference:**
- Embedded: `puglSetParentWindow(parentHandle)` — pugl becomes a child of the DAW window.
- Floating: No parent window set — pugl creates a top-level window. Requires `suggest_title` callback to set window title.

**Recommendation:** Support both embedded and floating in `create()`. Use `is_floating` parameter to decide. Falling back to floating when parent embedding fails is a good robustness pattern.

### Q6: GPU Acceleration for Cairo Rendering?

**Status:** Not needed. Software rendering is sufficient.

CLARA's UI is simple enough that Cairo software rendering at 60fps is trivial:
- 3 knobs × ~50 draw calls each = 150 draw calls
- 1 bypass button = ~20 draw calls
- 1 status label = ~5 draw calls
- 1 GR meter = ~30 draw calls
- 1 cond grid = ~40 draw calls
- Total: ~250 Cairo draw calls per frame

Modern CPUs handle this in <1ms. GPU acceleration (via pugl's OpenGL backend) would be over-engineering. If later phases add a real-time waveform display or spectrogram, OpenGL may be warranted — but for Phase 1-3, Cairo software rendering is the right choice.

### Q7: Multi-Instance GUI Considerations?

**Status:** Each plugin instance creates its own `MambaGUI`.

CLAP supports multiple plugin instances in a single process. Each `clap_create_plugin` call creates a new `MambaPluginState` with its own atomics, engine, and GUI. The GUI code must be reentrant — each instance's pugl view runs in its own event loop context.

This is handled naturally: pugl creates one `puglView*` per GUI instance, and each instance's event callbacks receive the `plugin` pointer via pugl's user data (`puglSetHandle(view, plugin)`). No global state in the GUI code.

### Q8: Parameter Readout Precision?

**Status:** Integer for values ≥10, one decimal for values <10.

- Wet/Dry at 100% → "100%" (not "100.0%")
- Wet/Dry at 7.5% → "7.5%" (not "8%")
- Intensity at 50% → "50%" (not "50.0%")
- Intensity at 0.3% → "0.3%" (not "0%")

This matches the existing `mamba_params_value_to_display` logic in `MambaClapPlugin.cpp` for consistency. The GUI display should match the host's parameter display string.

---

## Appendix A: Reference Implementations

| Project | URL | Notes |
|---------|-----|-------|
| clap-plugins (official) | https://github.com/free-audio/clap-plugins | Uses pugl for GUI. Good reference for CLAP GUI extension implementation. |
| DISTRHO | https://github.com/DISTRHO/DPF | DPF framework supports pugl. Reference for pugl + Cairo drawing. |
| U-He CLAP ports | https://u-he.com/ | Commercial plugins using pugl for CLAP GUIs. Demonstrates pugl at production quality. |
| pugl examples | https://github.com/lv2/pugl/tree/main/examples | Official pugl examples showing window creation, Cairo drawing, and event handling. |

## Appendix B: CLAP GUI Extension ID

```cpp
#define CLAP_EXT_GUI "clap.gui"
```

The extension ID is `"clap.gui"` (not `"clap.gui.dnd"` or `"clap.gui.free-standing"` — those are separate extensions for drag-and-drop and detached window support, respectively, and are out of scope for Phase 1).

## Appendix C: Testing Checklist (Phase 1)

- [ ] Plugin appears in host's plugin list
- [ ] GUI window opens when host requests it
- [ ] Bypass button toggles correctly, visual state matches audio state
- [ ] Wet/Dry knob drag updates parameter, host automation reflected in GUI
- [ ] Intensity knob drag updates parameter, host automation reflected in GUI
- [ ] Status shows "Ready" (green) when engine is active
- [ ] Status shows "Error: {msg}" (red) when model fails to load
- [ ] Window resizes and controls reposition correctly
- [ ] Minimum size (400×300) is enforced
- [ ] Maximum size (1200×800) is enforced
- [ ] Vertical stack layout at <500px width
- [ ] Horizontal layout at ≥500px width
- [ ] GUI closes without crash, can reopen
- [ ] Multiple plugin instances each have independent GUI
- [ ] Zero memory leaks on GUI open/close (Valgrind/ASan)
- [ ] Audio thread continues processing while GUI is open
- [ ] Audio thread has zero heap allocations (verified with `malloc` hook or heaptrack)
- [ ] Rapid GUI open/close (stress test) doesn't crash
- [ ] Rapid window resize doesn't crash
- [ ] Hover states appear and disappear correctly
- [ ] Knob double-click resets to default value
- [ ] Scroll wheel on knobs adjusts values
