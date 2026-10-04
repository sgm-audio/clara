// CLARA Mamba-3 SSM Block — 8-step per-sample pipeline + batched entry points
//
// Reference implementation: python/reference_forward.py ssm_block_forward()
// (R=4 time-domain, multi-head MIMO, trapezoidal SSM, RoPE, FiLM).
//
// All scratch is either stack-allocated or owned by the caller. No heap
// allocations on the audio thread.
//
// Performance note (2026-10-04): the three projections (FiLM, in_proj,
// out_proj) are matrix-vector products per sample in the naive formulation,
// which re-reads the 6.6 MB weight file through the cache once per sample.
// The *_batch() entry points run them as matrix × chunk GEMMs instead, and
// prepare_conditioning() hoists the conditioning projections out of the sample
// loop. The recurrence (steps 2-6) stays per-sample because it is causal.

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

// ── SSM Block ─────────────────────────────────────────────────────
//
// Bound to one layer's weights and state via bind(). A block is a thin
// view over caller-owned memory — it owns nothing but its conditioning cache.

class SSMBlock {
public:
    SSMBlock() = default;
    SSMBlock(const SSMWeights &w, SSMState &s) {
        bind(w, s);
    }

    void bind(const SSMWeights &w, SSMState &s) {
        weights_ = &w;
        state_ = &s;
    }

    bool bound() const {
        return weights_ != nullptr && state_ != nullptr;
    }

    // ── Conditioning (computed once per chunk, not per sample) ────

    // Evaluate the FiLM and Δ-conditioning projections for `cond`.
    // cond == nullptr clears the cache (plain passthrough conditioning).
    void prepare_conditioning(const float *cond);

    // ── Batched entry points (chunk-major) ────────────────────────
    //
    // Batch buffers use channel-major layout: element (row, i) is at
    // row * n_samples + i. Every sample in the chunk shares one conditioning
    // vector, so FiLM coefficients are identical across the chunk.

    // x[d][i] = x[d][i] * (1 + gamma[d]) + beta[d]   (in place)
    void film_batch(float *x, size_t n_samples) const;

    // proj[j][i] = sum_d in_proj_w[j][d] * x[d][i]
    void in_proj_batch(const float *x, size_t n_samples, float *proj) const;

    // Per-sample steps 2-6 for every sample in the chunk.
    // For sample i: reads proj column i and the FiLM'd embedding u_film column i,
    // writes y_gated column i plus the two scalar output terms used by step 7.
    void sample_core_batch(const float *proj, const float *u_film, size_t n_samples, float *y_gated,
                           float *d_skip_scale, float *mimo_o_sum);

    // out[d][i] = sum_j out_proj_w[d][j] * y_gated[j][i]
    void out_proj_batch(const float *y_gated, size_t n_samples, float *out) const;

    // ── Single-sample path (kept for tests and reference comparison) ──

    // Process one sample through the SSM layer.
    // input:  [D_MODEL] — embedding from previous layer (or input)
    // output: [D_MODEL] — transformed embedding
    // cond:   [COND_DIM] — conditioning vector (nullptr = skip FiLM)
    void process_sample(const float *input, float *output, const float *cond = nullptr);

    // Process a block of samples (the count is not bounded by MAX_CHUNK_SIZE).
    // Not used by the engine's hot path — SsmInferenceCore drives the batched
    // entry points.
    void process_block(const float *input, float *output, size_t n_samples, const float *cond = nullptr);

private:
    const SSMWeights *weights_ = nullptr;
    SSMState *state_ = nullptr;

    // Conditioning cache — evaluated once per chunk in prepare_conditioning()
    float film_gamma_[MAX_D_MODEL] = {};
    float film_beta_[MAX_D_MODEL] = {};
    float dt_cond_[2 * MAX_NHEADS] = {};
    bool has_cond_ = false;

    // Step 0: FiLM conditioning using the cached coefficients
    void step_film(const float *u, float *u_film) const;

    // Step 1: Fused in_proj — single matvec, then split
    void step_in_proj(const float *u_film, float *proj) const;

    // Step 2: A and Δ preprocessing (softplus, trapezoidal)
    void step_dt_A(const float *dd_dt, const float *dd_A, float *DT_t, float *A_t) const;

    // Steps 3+4: bias + norm scaling + RoPE on B and C
    void step_norm_rope(const float *B, const float *C, const float *angles_raw, float *B_rot, float *C_rot);

    // Step 5: Multi-head MIMO SSM recurrence (trapezoidal)
    void step_ssm_recurrence(const float *x_h, const float *B_rot, const float *C_rot, const float *DT_t,
                             const float *A_t, float *y_head);

    // Step 6: MIMO z-gating (SiLU)
    void step_z_gating(const float *z, const float *y_head, float *y_gated) const;

    // Step 7 (per-sample part): the two scalar terms added to every output dim
    void step_output_scalars(const float *y_gated, const float *x_h, float *d_skip_scale, float *mimo_o_sum) const;

    // Steps 2-6 + step 7 scalars for one sample. proj is [IN_PROJ_DIM] contiguous.
    void sample_core(const float *u_film, const float *proj, float *y_gated, float *d_skip_scale, float *mimo_o_sum);
};

} // namespace clara::mamba3
