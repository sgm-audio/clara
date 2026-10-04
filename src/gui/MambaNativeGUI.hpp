#pragma once

// CLARA Native GUI — CLAP_EXT_GUI framebuffer-based param editor.
//
// Shared RGBA8 pixel buffer with software 2D drawing primitives.
// Per-platform blit: X11 on Linux, Win32 GDI on Windows.
//
// Renders 3 sliders (bypass, wet/dry, intensity) and a level meter.
// Mouse input: click+drag on slider tracks to change param values.
// Timer-based redraw for level meter updates.

// CLARA_NO_NATIVE_GUI: build without the framebuffer editor. Linux needs the X11
// development headers for it; Windows uses GDI and needs no extra dependency.
// Defined automatically by CMake when X11 is unavailable (see CMakeLists.txt),
// which is what lets headless containers and MinGW cross-compiles build the CLAP
// plugin. The plugin then advertises no GUI extension and hosts show their
// generic editor.
#if !defined(CLARA_NO_NATIVE_GUI)

#include "clap/MambaClapPlugin.hpp"

#include <clap/ext/gui.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Platform-specific includes
#if defined(_WIN32)
#    define NOMINMAX
#    include <windows.h>
#elif defined(__APPLE__)
#    include <TargetConditionals.h>
#else
#    include <X11/Xlib.h>
#    include <X11/Xutil.h>
#endif

namespace clara::mamba3 {

// ── Constants ──────────────────────────────────────────────────────────────
static constexpr int kGUIWidth = 400;
static constexpr int kGUIHeight = 300;
static constexpr int kSliderHeight = 40;
static constexpr int kSliderY[] = {50, 100, 150}; // bypass, wet/dry, intensity
static constexpr int kMeterY = 210;
static constexpr int kMeterHeight = 30;
static constexpr int kLabelX = 20;
static constexpr int kSliderX = 120;
static constexpr int kSliderW = 250;

// ── Colors (RGBA8) ─────────────────────────────────────────────────────────
static constexpr uint32_t kColorBg = 0xFF1A1A2E;
static constexpr uint32_t kColorTrack = 0xFF2A2A4A;
static constexpr uint32_t kColorKnob = 0xFF00D4AA;
static constexpr uint32_t kColorText = 0xFF888888;
static constexpr uint32_t kColorLabel = 0xFF00D4AA;
static constexpr uint32_t kColorMeterBg = 0xFF2A2A4A;
static constexpr uint32_t kColorMeterIn = 0xFF00D4AA;
static constexpr uint32_t kColorMeterOut = 0xFF00AAFF;
static constexpr uint32_t kColorBypassOn = 0xFF00D4AA;
static constexpr uint32_t kColorBypassOff = 0xFF444444;

// ── Pixel Buffer ───────────────────────────────────────────────────────────
struct MambaPixelBuffer {
    uint32_t pixels[kGUIWidth * kGUIHeight] = {};

    void clear() {
        std::memset(pixels, 0, sizeof(pixels));
    }

    void fillRect(int x, int y, int w, int h, uint32_t color) {
        int x0 = std::max(0, x);
        int y0 = std::max(0, y);
        int x1 = std::min(kGUIWidth, x + w);
        int y1 = std::min(kGUIHeight, y + h);
        for (int row = y0; row < y1; ++row) {
            for (int col = x0; col < x1; ++col) {
                pixels[row * kGUIWidth + col] = color;
            }
        }
    }

    void fillGradient(int x, int y, int w, int h, uint32_t c0, uint32_t c1) {
        int x0 = std::max(0, x);
        int y0 = std::max(0, y);
        int x1 = std::min(kGUIWidth, x + w);
        int y1 = std::min(kGUIHeight, y + h);
        for (int row = y0; row < y1; ++row) {
            float t = (float) (row - y0) / (float) (y1 - y0);
            uint32_t r = (uint32_t) (((c0 >> 16) & 0xFF) * (1 - t) + ((c1 >> 16) & 0xFF) * t);
            uint32_t g = (uint32_t) (((c0 >> 8) & 0xFF) * (1 - t) + ((c1 >> 8) & 0xFF) * t);
            uint32_t b = (uint32_t) ((c0 & 0xFF) * (1 - t) + (c1 & 0xFF) * t);
            uint32_t color = 0xFF000000 | (r << 16) | (g << 8) | b;
            for (int col = x0; col < x1; ++col) {
                pixels[row * kGUIWidth + col] = color;
            }
        }
    }

    void drawSlider(int x, int y, int w, int h, float value, const char *label, bool isToggle) {
        // Background
        fillRect(x, y, w, h, kColorTrack);
        // Knob position
        int knobW = isToggle ? w : (int) (w * value);
        uint32_t knobColor = isToggle ? (value > 0.5f ? kColorBypassOn : kColorBypassOff) : kColorKnob;
        fillRect(x, y, knobW, h, knobColor);
    }

    void drawLevelMeter(int x, int y, int w, int h, float level, uint32_t color) {
        fillRect(x, y, w, h, kColorMeterBg);
        int fillW = (int) (w * std::min(1.0f, level));
        if (fillW > 0) {
            fillRect(x, y, fillW, h, color);
        }
    }
};

// ── Native GUI State ───────────────────────────────────────────────────────
struct MambaNativeGUIState {
    MambaPixelBuffer buffer;
    MambaPluginState *pluginState{nullptr};
    const clap_host_gui_t *hostGUI{nullptr};

    // Platform window handles
#if defined(_WIN32)
    HWND hwnd{nullptr};
    HDC hdc{nullptr};
    HBITMAP hbitmap{nullptr};
    void *bitmapBits{nullptr};
    WNDPROC originalWndProc{nullptr};
#elif defined(__APPLE__)
    void *nsview{nullptr};
#else
    Display *display{nullptr};
    Window window{0};
    GC gc{nullptr};
    XImage *ximage{nullptr};
#endif

    bool visible{false};
    bool floating{false};
    int width{kGUIWidth};
    int height{kGUIHeight};
    int mouseX{-1}, mouseY{-1};
    bool mouseDown{false};
    int dragParam{-1}; // -1 = none, 0=bypass, 1=wetdry, 2=intensity

    // ── Drawing ──
    void render() {
        buffer.clear();

        // Title
        buffer.fillRect(0, 0, kGUIWidth, 30, 0xFF1A1A2E);
        // Title text is rendered as a colored bar — no font rendering
        buffer.fillRect(10, 8, 200, 14, kColorLabel);

        // Sliders
        const char *labels[] = {"Bypass", "Wet/Dry", "Intensity"};
        float values[] = {
            pluginState ? pluginState->bypass.load(std::memory_order_relaxed) : 0.0f,
            pluginState ? pluginState->wetDry.load(std::memory_order_relaxed) : 1.0f,
            pluginState ? pluginState->intensity.load(std::memory_order_relaxed) : 0.5f,
        };
        bool isToggle[] = {true, false, false};

        for (int i = 0; i < 3; ++i) {
            // Label area (colored rect)
            buffer.fillRect(kLabelX, kSliderY[i] + 8, 80, 24, kColorTrack);
            // Slider track
            buffer.drawSlider(kSliderX, kSliderY[i], kSliderW, kSliderHeight, values[i], labels[i], isToggle[i]);
        }

        // Level meter
        float peakIn = pluginState ? pluginState->peakIn.load(std::memory_order_relaxed) : 0.0f;
        float peakOut = pluginState ? pluginState->peakOut.load(std::memory_order_relaxed) : 0.0f;
        buffer.drawLevelMeter(kLabelX, kMeterY, kGUIWidth - 2 * kLabelX, kMeterHeight, peakIn, kColorMeterIn);
        buffer.drawLevelMeter(kLabelX, kMeterY + kMeterHeight + 4, kGUIWidth - 2 * kLabelX, kMeterHeight, peakOut,
                              kColorMeterOut);

        // Info bar
        buffer.fillRect(0, kGUIHeight - 20, kGUIWidth, 20, 0xFF111122);
    }

    // ── Hit testing ──
    int hitTest(int mx, int my) {
        for (int i = 0; i < 3; ++i) {
            if (mx >= kSliderX && mx < kSliderX + kSliderW && my >= kSliderY[i] && my < kSliderY[i] + kSliderHeight) {
                return i;
            }
        }
        return -1;
    }

    float sliderValueFromMouse(int mx) {
        float v = (float) (mx - kSliderX) / (float) kSliderW;
        return std::clamp(v, 0.0f, 1.0f);
    }
};

// ── CLAP_EXT_GUI Implementation ────────────────────────────────────────────

inline bool mamba_gui_is_api_supported(const clap_plugin_t *, const char *api, bool is_floating) {
#if defined(_WIN32)
    return std::strcmp(api, CLAP_WINDOW_API_WIN32) == 0 && !is_floating;
#elif defined(__APPLE__)
    return std::strcmp(api, CLAP_WINDOW_API_COCOA) == 0 && !is_floating;
#else
    return std::strcmp(api, CLAP_WINDOW_API_X11) == 0 && !is_floating;
#endif
}

inline bool mamba_gui_get_preferred_api(const clap_plugin_t *, const char **api, bool *is_floating) {
#if defined(_WIN32)
    *api = CLAP_WINDOW_API_WIN32;
#elif defined(__APPLE__)
    *api = CLAP_WINDOW_API_COCOA;
#else
    *api = CLAP_WINDOW_API_X11;
#endif
    *is_floating = false;
    return true;
}

inline bool mamba_gui_create(const clap_plugin_t *plugin, const char *api, bool is_floating) {
    (void) api;
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (!state->nativeGUI) {
        state->nativeGUI = new MambaNativeGUIState();
        state->nativeGUI->pluginState = state;
    }
    state->nativeGUI->floating = is_floating;
    state->nativeGUI->visible = false;

    // Store host GUI extension
    if (state->host && state->host->get_extension) {
        state->nativeGUI->hostGUI = (const clap_host_gui_t *) state->host->get_extension(state->host, CLAP_EXT_GUI);
    }
    return true;
}

inline void mamba_gui_destroy(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state->nativeGUI) {
#if defined(_WIN32)
        if (state->nativeGUI->hbitmap) {
            DeleteObject(state->nativeGUI->hbitmap);
            state->nativeGUI->hbitmap = nullptr;
        }
        if (state->nativeGUI->hdc) {
            DeleteDC(state->nativeGUI->hdc);
            state->nativeGUI->hdc = nullptr;
        }
#elif !defined(__APPLE__)
        if (state->nativeGUI->ximage) {
            XDestroyImage(state->nativeGUI->ximage);
            state->nativeGUI->ximage = nullptr;
        }
        if (state->nativeGUI->gc && state->nativeGUI->display) {
            XFreeGC(state->nativeGUI->display, state->nativeGUI->gc);
            state->nativeGUI->gc = nullptr;
        }
#endif
        delete state->nativeGUI;
        state->nativeGUI = nullptr;
    }
}

inline bool mamba_gui_set_scale(const clap_plugin_t *, double) {
    return false;
}

inline bool mamba_gui_get_size(const clap_plugin_t *, uint32_t *w, uint32_t *h) {
    *w = kGUIWidth;
    *h = kGUIHeight;
    return true;
}

inline bool mamba_gui_can_resize(const clap_plugin_t *) {
    return false;
}

inline bool mamba_gui_get_resize_hints(const clap_plugin_t *, clap_gui_resize_hints_t *) {
    return false;
}

inline bool mamba_gui_adjust_size(const clap_plugin_t *, uint32_t *, uint32_t *) {
    return false;
}

inline bool mamba_gui_set_size(const clap_plugin_t *, uint32_t, uint32_t) {
    return false;
}

// ── Platform-specific blit ─────────────────────────────────────────────────
#if defined(_WIN32)

inline bool mamba_gui_set_parent(const clap_plugin_t *plugin, const clap_window_t *window) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    auto *gui = state->nativeGUI;
    if (!gui) return false;

    gui->hwnd = (HWND) window->win32;

    // Create a memory DC with a DIB section for the pixel buffer
    HDC hdcScreen = GetDC(nullptr);
    gui->hdc = CreateCompatibleDC(hdcScreen);
    ReleaseDC(nullptr, hdcScreen);

    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = kGUIWidth;
    bmi.bmiHeader.biHeight = -kGUIHeight; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    gui->hbitmap = CreateDIBSection(gui->hdc, &bmi, DIB_RGB_COLORS, &gui->bitmapBits, nullptr, 0);
    SelectObject(gui->hdc, gui->hbitmap);

    return true;
}

inline bool mamba_gui_show(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    auto *gui = state->nativeGUI;
    if (!gui || !gui->hwnd) return false;

    gui->render();
    std::memcpy(gui->bitmapBits, gui->buffer.pixels, kGUIWidth * kGUIHeight * 4);

    HDC hdcWin = GetDC(gui->hwnd);
    BitBlt(hdcWin, 0, 0, kGUIWidth, kGUIHeight, gui->hdc, 0, 0, SRCCOPY);
    ReleaseDC(gui->hwnd, hdcWin);

    gui->visible = true;
    return true;
}

inline bool mamba_gui_hide(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state->nativeGUI) state->nativeGUI->visible = false;
    return true;
}

#elif defined(__APPLE__)

inline bool mamba_gui_set_parent(const clap_plugin_t *plugin, const clap_window_t *window) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state->nativeGUI) state->nativeGUI->nsview = window->cocoa;
    return true;
}

inline bool mamba_gui_show(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state->nativeGUI) state->nativeGUI->visible = true;
    return true;
}

inline bool mamba_gui_hide(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state->nativeGUI) state->nativeGUI->visible = false;
    return true;
}

#else // Linux X11

inline bool mamba_gui_set_parent(const clap_plugin_t *plugin, const clap_window_t *window) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    auto *gui = state->nativeGUI;
    if (!gui) return false;

    gui->window = window->x11;
    gui->display = XOpenDisplay(nullptr);
    if (!gui->display) return false;

    gui->gc = XCreateGC(gui->display, gui->window, 0, nullptr);

    // Create XImage for the pixel buffer
    char *imgData = (char *) std::calloc(kGUIWidth * kGUIHeight * 4, 1);
    gui->ximage = XCreateImage(gui->display, DefaultVisual(gui->display, 0), 24, ZPixmap, 0, imgData, kGUIWidth,
                               kGUIHeight, 32, 0);

    // Select input events for mouse interaction
    XSelectInput(gui->display, gui->window, ButtonPressMask | ButtonReleaseMask | PointerMotionMask);

    return true;
}

inline bool mamba_gui_show(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    auto *gui = state->nativeGUI;
    if (!gui || !gui->display || !gui->window) return false;

    gui->render();

    // Copy pixel buffer to XImage
    std::memcpy(gui->ximage->data, gui->buffer.pixels, kGUIWidth * kGUIHeight * 4);

    // Blit to window
    XPutImage(gui->display, gui->window, gui->gc, gui->ximage, 0, 0, 0, 0, kGUIWidth, kGUIHeight);
    XFlush(gui->display);

    gui->visible = true;
    return true;
}

inline bool mamba_gui_hide(const clap_plugin_t *plugin) {
    auto *state = (MambaPluginState *) plugin->plugin_data;
    if (state->nativeGUI) state->nativeGUI->visible = false;
    return true;
}

#endif

// ── Timer callback (for level meter updates) ───────────────────────────────
// Called from the host's timer. We re-render and blit the pixel buffer.
inline void mamba_gui_on_timer(MambaNativeGUIState *gui) {
    if (!gui || !gui->visible) return;

    gui->render();

#if defined(_WIN32)
    if (gui->hdc && gui->hwnd) {
        std::memcpy(gui->bitmapBits, gui->buffer.pixels, kGUIWidth * kGUIHeight * 4);
        HDC hdcWin = GetDC(gui->hwnd);
        BitBlt(hdcWin, 0, 0, kGUIWidth, kGUIHeight, gui->hdc, 0, 0, SRCCOPY);
        ReleaseDC(gui->hwnd, hdcWin);
    }
#elif !defined(__APPLE__)
    if (gui->display && gui->window && gui->ximage) {
        std::memcpy(gui->ximage->data, gui->buffer.pixels, kGUIWidth * kGUIHeight * 4);
        XPutImage(gui->display, gui->window, gui->gc, gui->ximage, 0, 0, 0, 0, kGUIWidth, kGUIHeight);
        XFlush(gui->display);
    }
#endif
}

// ── Mouse event handler ────────────────────────────────────────────────────
// Called from the host when mouse events occur on the plugin window.
inline void mamba_gui_on_mouse(MambaNativeGUIState *gui, int mx, int my, bool pressed, bool released) {
    if (!gui) return;

    if (pressed) {
        gui->dragParam = gui->hitTest(mx, my);
        gui->mouseDown = true;
        if (gui->dragParam >= 0) {
            float val = gui->sliderValueFromMouse(mx);
            auto *ps = gui->pluginState;
            if (gui->dragParam == 0) { // bypass toggle
                float cur = ps->bypass.load(std::memory_order_relaxed);
                ps->bypass.store(cur > 0.5f ? 0.0f : 1.0f, std::memory_order_release);
            } else if (gui->dragParam == 1) {
                ps->wetDry.store(val, std::memory_order_release);
            } else if (gui->dragParam == 2) {
                ps->intensity.store(val, std::memory_order_release);
            }
            // Notify host
            if (gui->hostGUI && gui->hostGUI->resize_hints_changed)
                gui->hostGUI->resize_hints_changed(gui->pluginState->host);
        }
    } else if (released) {
        gui->dragParam = -1;
        gui->mouseDown = false;
    } else if (gui->mouseDown && gui->dragParam > 0) {
        // Drag for wet/dry or intensity
        float val = gui->sliderValueFromMouse(mx);
        auto *ps = gui->pluginState;
        if (gui->dragParam == 1) {
            ps->wetDry.store(val, std::memory_order_release);
        } else if (gui->dragParam == 2) {
            ps->intensity.store(val, std::memory_order_release);
        }
    }
}

// ── Extension struct ───────────────────────────────────────────────────────
static const clap_plugin_gui_t mamba_native_gui_extension = {
    mamba_gui_is_api_supported,
    mamba_gui_get_preferred_api,
    mamba_gui_create,
    mamba_gui_destroy,
    mamba_gui_set_scale,
    mamba_gui_get_size,
    mamba_gui_can_resize,
    mamba_gui_get_resize_hints,
    mamba_gui_adjust_size,
    mamba_gui_set_size,
    mamba_gui_set_parent,
    nullptr, // set_transient (floating not supported)
    nullptr, // suggest_title (floating not supported)
    mamba_gui_show,
    mamba_gui_hide,
};

} // namespace clara::mamba3

#endif // !CLARA_NO_NATIVE_GUI
