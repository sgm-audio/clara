// CLARA Mamba-3 SSM Block — Per-sample 8-step pipeline
//
// Part of Phase 3 of the Mamba-1 → Mamba-3 migration.
// Bit-exact match target: reference_forward.py ssm_block_forward()
//
// Architecture: Multi-head MIMO R=4, trapezoidal SSM, RoPE, FiLM
// Reference: MAMBA3_MIGRATION_PLAN.md §C

#pragma once

#include "core/MambaConfig.h"

#include <cstddef>
#include <cstdint>

namespace clara::mamba3 {

// ── SSM State (per layer, stack-allocated) ────────────────────────

struct alignas(64) SSMState {
    // SSM hidden state: h_real[h][d][s] — real component
    float ssm_real[MAX_NHEADS][MAX_HEADDIM][MAX_D_STATE] = {};

    // SSM hidden state: h_imag[h][d][s] — imaginary component (reserved)
    float ssm_imag[MAX_NHEADS][MAX_HEADDIM][MAX_D_STATE] = {};

    // Kalman-like K state: k[h][r][s]
    float k_state[MAX_NHEADS][MAX_MIMO_RANK][MAX_D_STATE] = {};

    // RoPE accumulated angles: angle[h][a]
    float angle_state[MAX_NHEADS][MAX_NUM_ROPE_ANGLES] = {};

    // Sample counter for periodic state normalization
    uint32_t sample_count = 0;

    void zero();
    void normalize_if_needed();
};

// ── SSM Weights (pointers into mmap'd buffer, no copies) ──────────

struct SSMWeights {
    // Array 0: in_proj_w [IN_PROJ_DIM][D_MODEL] — fused input projection
    const float *in_proj_w;

    // Array 1: dt_bias [NHEADS]
    const float *dt_bias;

    // Array 2: B_bias [NHEADS][MIMO_RANK][D_STATE]
    const float *B_bias;

    // Array 3: C_bias [NHEADS][MIMO_RANK][D_STATE]
    const float *C_bias;

    // Array 4: B_norm_weight [D_STATE]
    const float *B_norm_weight;

    // Array 5: C_norm_weight [D_STATE]
    const float *C_norm_weight;

    // Array 6: mimo_x [NHEADS][MIMO_RANK][HEADDIM]
    const float *mimo_x;

    // Array 7: mimo_z [NHEADS][MIMO_RANK][HEADDIM]
    const float *mimo_z;

    // Array 8: mimo_o [NHEADS][MIMO_RANK][HEADDIM]
    const float *mimo_o;

    // Array 9: D_proj [NHEADS]
    const float *D_proj;

    // Array 10: out_proj_w [D_MODEL][D_INNER]
    const float *out_proj_w;

    // Array 11: film_proj_w [2 * D_MODEL][COND_DIM]
    const float *film_proj_w;

    // Array 12: dt_film_proj_w [2 * NHEADS][COND_DIM]
    const float *dt_film_proj_w;
};

// ── SSM Block — Per-sample forward pass ───────────────────────────

class SSMBlock {
public:
    SSMBlock(const SSMWeights &w, SSMState &s) : weights_(w), state_(s) {
    }

    // Process one sample through the SSM layer.
    // input:  [D_MODEL] — embedding from previous layer (or input)
    // output: [D_MODEL] — transformed embedding
    // cond:   [COND_DIM] — conditioning vector (nullptr = skip FiLM)
    void process_sample(const float *input, float *output, const float *cond = nullptr);

    // Process a block of samples (up to MAX_CHUNK_SIZE).
    void process_block(const float *input, float *output, size_t n_samples, const float *cond = nullptr);

private:
    const SSMWeights &weights_;
    SSMState &state_;

    // Scratch buffers (stack-allocated inside process_sample)
    // — declared here for alignment, used in process_sample

    // Step 0: FiLM conditioning
    void step_film(const float *u, float *u_film, const float *cond);

    // Step 1: Fused in_proj — single matvec, then split
    void step_in_proj(const float *u_film, float *proj);

    // Step 2: A and Δ preprocessing (softplus, trapezoidal)
    void step_dt_A(const float *dd_dt, const float *dd_A, const float *cond, float *DT_t, float *A_t);

    // Steps 3+4: RMSNorm + RoPE on B and C
    void step_norm_rope(const float *B, const float *C, float *B_rot, float *C_rot);

    // Step 5: Multi-head MIMO SSM recurrence (trapezoidal)
    void step_ssm_recurrence(const float *x_h, const float *B_rot, const float *C_rot, const float *DT_t,
                             const float *A_t, float *y_head);

    // Step 6: MIMO z-gating (SiLU)
    void step_z_gating(const float *z, const float *y_head, float *y_gated);

    // Step 7: Output projection + D skip + residual
    void step_output(const float *y_gated, const float *x_h, const float *u_film, float *output);
};

} // namespace clara::mamba3
