#pragma once

#include <cmath>
#include <algorithm>
#include <array>

class ConditioningGenerator {
public:
    static constexpr size_t COND_DIM = 128;

    // Musical context struct — all fields optional (use -1 or equivalent sentinel for "unknown")
    struct MusicalContext {
        float tempoBpm = -1.0f;           // beats per minute, -1 if unknown
        int   keyPitchClass = -1;          // 0=C, 1=C#, ..., 11=B, -1 if unknown
        int   mode = -1;                   // 0=major, 1=minor, -1 if unknown
        int   timeSigNumerator = -1;       // 3, 4, 6, etc., -1 if unknown
        int   sectionType = -1;            // 0=verse, 1=chorus, 2=bridge, 3=intro, 4=outro, -1 if unknown
    };

    // Encode musical context into cond[COND_DIM].
    // Unknown fields are encoded as zero (no modulation for that dimension).
    static void encode(const MusicalContext& ctx, float* cond, size_t dim) {
        if (!cond || dim == 0) return;
        const size_t n = std::min(dim, COND_DIM);
        std::fill(cond, cond + n, 0.0f);

        // Slot layout for COND_DIM=128:
        // [0-7]   — DAW context (tempo, key, mode, time sig, section, 3 reserved)
        // [8-39]  — CLAP semantic embedding (32 floats, from text + audio encoders)
        // [40-71] — User knobs (32 floats: intensity, character, envelope, etc.)
        // [72-127] — Reserved (future: arrangement position, density, genre, etc.)

        if (n > 0 && ctx.tempoBpm > 0.0f) {
            // Log-scale: center at 120bpm, range roughly 30-480bpm
            cond[0] = std::clamp(std::log2(ctx.tempoBpm / 120.0f) * 0.25f + 0.5f, 0.0f, 1.0f);
        }

        if (n > 1 && ctx.keyPitchClass >= 0 && ctx.keyPitchClass <= 11) {
            cond[1] = static_cast<float>(ctx.keyPitchClass) / 12.0f;
        }

        if (n > 2) {
            if (ctx.mode == 0)      cond[2] = 0.0f;  // major
            else if (ctx.mode == 1) cond[2] = 1.0f;  // minor
            else                    cond[2] = 0.5f;  // unknown
        }

        if (n > 3 && ctx.timeSigNumerator > 0) {
            if (ctx.timeSigNumerator == 3)      cond[3] = 0.0f;
            else if (ctx.timeSigNumerator == 4) cond[3] = 0.5f;
            else                                cond[3] = 1.0f;
        }

        if (n > 4 && ctx.sectionType >= 0 && ctx.sectionType <= 4) {
            cond[4] = static_cast<float>(ctx.sectionType) / 4.0f;
        }

        // Slots 5-7: reserved, remain zero
    }

    // Generate a default "neutral" conditioning (all zeros → no modulation).
    // This is safe to call when no musical context is available.
    static void neutral(float* cond, size_t dim) {
        if (!cond || dim == 0) return;
        const size_t n = std::min(dim, COND_DIM);
        std::fill(cond, cond + n, 0.0f);
    }

    // Generate a test/default conditioning for basic verification.
    // Tempo=120bpm, C major, 4/4, verse — produces non-zero cond for testing.
    static MusicalContext defaultContext() {
        MusicalContext ctx;
        ctx.tempoBpm = 120.0f;
        ctx.keyPitchClass = 0;   // C
        ctx.mode = 0;            // major
        ctx.timeSigNumerator = 4;
        ctx.sectionType = 0;     // verse
        return ctx;
    }
};
