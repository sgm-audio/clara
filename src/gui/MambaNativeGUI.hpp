#pragma once

// CLAP native GUI extension — framebuffer-based param editor.
// Platform-specific rendering: X11 on Linux, Win32 GDI on Windows.
// Shared pixel buffer with software 2D drawing primitives.
//
// This file is a placeholder. The full implementation will be added in Phase 3.
// For now, the CLAP_EXT_GUI extension is not registered — the VST3 wrapper
// falls back to the host's generic parameter editor.

#include <clap/ext/gui.h>
