// CLARA Mamba-3 Configuration — Compile-time dimension constants
//
// Part of the Mamba-1 → Mamba-3 migration (MAMBA3_MIGRATION_PLAN.md).
// All constants are template-ready (MAX_ prefix) to support eventual
// model-size variability without runtime overhead.
//
// Architecture: Multi-head MIMO R=4, trapezoidal SSM, RoPE, FiLM
// Reference:  MAMBA3_MIGRATION_PLAN.md §B, §C

#pragma once

#include <cstddef>
#include <cstdint>

namespace clara::mamba3 {

// ── Model Dimensions ──────────────────────────────────────────────

constexpr size_t MAX_D_STATE       = 32;    // SSM state dimension
constexpr size_t MAX_D_MODEL       = 128;   // Embedding dimension
constexpr size_t MAX_D_INNER       = 256;   // 2× d_model expansion
constexpr size_t MAX_NHEADS        = 4;     // Parallel MIMO heads
constexpr size_t MAX_MIMO_RANK     = 4;     // Frequency slots per head (sub/bass/mids/highs)
constexpr size_t MAX_NUM_BC_HEADS  = 1;     // B/C groups per head
constexpr size_t MAX_NUM_ROPE_ANGLES = 16;  // RoPE rotation angles (= d_state/2)
constexpr size_t MAX_LAYERS        = 6;     // SSM layers in the stack

// ── Derived Dimensions ────────────────────────────────────────────

// ponytail: headdim = d_inner / nheads. The migration plan lists MAX_HEADDIM=64
// which equals 256/4. This differs from d_model/nheads (=32) — the MIMO architecture
// uses d_inner as the channel count for head partitioning.
constexpr size_t MAX_HEADDIM       = MAX_D_INNER / MAX_NHEADS;  // 64

// in_proj dimensionality: z(d_inner=256) + x(nheads*headdim=256)
// + B(nheads*mimo_rank*d_state=512) + C(same=512)
// + dd_dt(nheads=4) + dd_A(4) + trap(4) + angles(nheads*rope_angles=64)
// Total: 256 + 256 + 512 + 512 + 4 + 4 + 4 + 64 = 1612
constexpr size_t MAX_IN_PROJ_DIM   = MAX_D_INNER                          // z
                                   + MAX_NHEADS * MAX_HEADDIM             // x per head
                                   + MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE  // B
                                   + MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE  // C
                                   + MAX_NHEADS                           // dd_dt
                                   + MAX_NHEADS                           // dd_A
                                   + MAX_NHEADS                           // trap
                                   + MAX_NHEADS * MAX_NUM_ROPE_ANGLES;   // angles

// dt_film_proj: γ,β per head → 2 × nheads rows, COND_DIM columns
constexpr size_t MAX_DT_FILM_ROWS  = 2 * MAX_NHEADS;   // 8

// ── Conditioning ──────────────────────────────────────────────────

constexpr size_t COND_DIM          = 128;   // DAW(8) + CLAP(32) + knobs(32) + reserved(56)
constexpr size_t COND_DAW_OFFSET   = 0;
constexpr size_t COND_DAW_SIZE     = 8;
constexpr size_t COND_CLAP_OFFSET  = 8;
constexpr size_t COND_CLAP_SIZE    = 32;
constexpr size_t COND_KNOB_OFFSET  = 40;
constexpr size_t COND_KNOB_SIZE    = 32;
constexpr size_t COND_RESERVED_OFFSET = 72;
constexpr size_t COND_RESERVED_SIZE   = 56;

static_assert(COND_DAW_SIZE + COND_CLAP_SIZE + COND_KNOB_SIZE + COND_RESERVED_SIZE == COND_DIM,
              "COND_DIM must equal sum of all conditioning slot groups");

// ── Block Processing ──────────────────────────────────────────────

constexpr size_t MAX_CHUNK_SIZE    = 64;    // Samples per SSM sub-block
constexpr size_t MAX_BLOCK_SIZE    = 512;   // Max samples per processBlock() call
constexpr size_t SSM_STATE_NORM_INTERVAL = 4096;  // Periodic state re-normalization

// ── Weight Format (Version 2) ─────────────────────────────────────

constexpr uint32_t WEIGHT_MAGIC    = 0x4D424D41;  // "MBMA"
constexpr uint32_t WEIGHT_VERSION  = 2;
constexpr size_t   WEIGHTS_PER_LAYER  = 13;  // Arrays per layer in flat binary

// File header size (magic + version + 7 dims + 4 reserved + weight_data_size)
constexpr size_t WEIGHT_HEADER_SIZE = 64;  // bytes

// ── State Memory ──────────────────────────────────────────────────

// Per-layer state: ssm_real + ssm_imag + k_state + angle_state
// ssm_real/imag  = nheads × headdim × d_state = 4 × 64 × 32 = 8192 each
// k_state        = nheads × mimo_rank × d_state = 4 × 4 × 32 = 512
// angle_state    = nheads × rope_angles = 4 × 16 = 64
// Total per layer: 8192 + 8192 + 512 + 64 = 16960
constexpr size_t SSM_REAL_FLOATS   = MAX_NHEADS * MAX_HEADDIM * MAX_D_STATE;     // 8192
constexpr size_t SSM_IMAG_FLOATS   = MAX_NHEADS * MAX_HEADDIM * MAX_D_STATE;     // 8192
constexpr size_t K_STATE_FLOATS    = MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE;   // 512
constexpr size_t ANGLE_STATE_FLOATS = MAX_NHEADS * MAX_NUM_ROPE_ANGLES;          // 64
constexpr size_t STATE_FLOATS_PER_LAYER = SSM_REAL_FLOATS + SSM_IMAG_FLOATS
                                        + K_STATE_FLOATS + ANGLE_STATE_FLOATS;   // 16960

constexpr size_t TOTAL_STATE_FLOATS = STATE_FLOATS_PER_LAYER * MAX_LAYERS;       // 101760

// ── Asset Verification ────────────────────────────────────────────

static_assert(MAX_D_INNER == 2 * MAX_D_MODEL, "MAX_D_INNER must be 2× MAX_D_MODEL");
static_assert(MAX_NHEADS * MAX_HEADDIM == MAX_D_INNER,
              "MAX_NHEADS × MAX_HEADDIM must equal MAX_D_INNER");
static_assert(MAX_NUM_ROPE_ANGLES == MAX_D_STATE / 2,
              "MAX_NUM_ROPE_ANGLES must equal MAX_D_STATE / 2");
static_assert(MAX_CHUNK_SIZE <= MAX_BLOCK_SIZE,
              "MAX_CHUNK_SIZE must be ≤ MAX_BLOCK_SIZE");

}  // namespace clara::mamba3
