// CLARA Mamba-3 SSM Block — Per-sample 8-step pipeline implementation
//
// Bit-exact target: reference_forward.py ssm_block_forward()
// All arrays stack-allocated. Zero heap allocations on audio thread.

#include "core/SsmBlock.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace clara::mamba3 {

// ── Math helpers (inline for RT-safety) ───────────────────────────

namespace {

inline float softplus(float x) {
    // log(1 + exp(x)), numerically stable
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1pf(std::expf(x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::expf(-x));
}

inline float rsqrt(float x) {
    return 1.0f / std::sqrt(x);
}

}  // namespace

// ── SSMState ──────────────────────────────────────────────────────

void SSMState::zero() {
    std::memset(ssm_real, 0, sizeof(ssm_real));
    std::memset(ssm_imag, 0, sizeof(ssm_imag));
    std::memset(k_state, 0, sizeof(k_state));
    std::memset(angle_state, 0, sizeof(angle_state));
    sample_count = 0;
}

void SSMState::normalize_if_needed() {
    if (sample_count > 0 && (sample_count % SSM_STATE_NORM_INTERVAL) == 0) {
        float max_abs = 0.0f;
        for (size_t h = 0; h < MAX_NHEADS; ++h) {
            for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                for (size_t s = 0; s < MAX_D_STATE; ++s) {
                    float v = std::fabs(ssm_real[h][d][s]);
                    if (v > max_abs) max_abs = v;
                }
            }
        }
        if (max_abs > 1.0f) {
            float scale = 1.0f / max_abs;
            for (size_t h = 0; h < MAX_NHEADS; ++h) {
                for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                    for (size_t s = 0; s < MAX_D_STATE; ++s) {
                        ssm_real[h][d][s] *= scale;
                    }
                }
            }
        }
    }
}

// ── SSMBlock Step 0: FiLM conditioning ────────────────────────────

void SSMBlock::step_film(const float* u, float* u_film, const float* cond) {
    if (cond == nullptr) {
        // No conditioning — pass through
        std::memcpy(u_film, u, MAX_D_MODEL * sizeof(float));
        return;
    }

    // film_proj_w @ cond → [2 * D_MODEL]
    // gamma = first D_MODEL, beta = second D_MODEL
    float gamma_beta[2 * MAX_D_MODEL] = {};
    for (size_t i = 0; i < 2 * MAX_D_MODEL; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < COND_DIM; ++j) {
            sum += weights_.film_proj_w[i * COND_DIM + j] * cond[j];
        }
        gamma_beta[i] = sum;
    }

    // u_film[d] = u[d] * (1 + gamma[d]) + beta[d]
    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        u_film[d] = u[d] * (1.0f + gamma_beta[d]) + gamma_beta[MAX_D_MODEL + d];
    }
}

// ── SSMBlock Step 1: Fused in_proj ────────────────────────────────

void SSMBlock::step_in_proj(const float* u_film, float* proj) {
    // proj = in_proj_w @ u_film  where in_proj_w is [IN_PROJ_DIM, D_MODEL]
    for (size_t i = 0; i < MAX_IN_PROJ_DIM; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < MAX_D_MODEL; ++j) {
            sum += weights_.in_proj_w[i * MAX_D_MODEL + j] * u_film[j];
        }
        proj[i] = sum;
    }
}

// ── SSMBlock Step 2: A and Δ preprocessing ────────────────────────

void SSMBlock::step_dt_A(const float* dd_dt, const float* dd_A,
                         const float* cond,
                         float* DT_t, float* A_t) {
    // dt_film_cond = dt_film_proj_w @ cond → [2 * NHEADS]
    float dt_cond_val[2 * MAX_NHEADS] = {};
    if (cond != nullptr) {
        for (size_t i = 0; i < 2 * MAX_NHEADS; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < COND_DIM; ++j) {
                sum += weights_.dt_film_proj_w[i * COND_DIM + j] * cond[j];
            }
            dt_cond_val[i] = sum;
        }
    }

    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        float dt_h = softplus(dd_dt[h] + weights_.dt_bias[h]);
        if (cond != nullptr) {
            dt_h = softplus(dt_h + dt_cond_val[h] + dt_cond_val[MAX_NHEADS + h]);
        }
        DT_t[h] = dt_h;
        A_t[h] = -softplus(dd_A[h]);
    }
}

// ── SSMBlock Steps 3+4: RMSNorm + RoPE on B and C ────────────────

void SSMBlock::step_norm_rope(const float* B, const float* C,
                               float* B_rot, float* C_rot) {
    // B_rot, C_rot output shapes: [NHEADS][MIMO_RANK][D_STATE]
    // B, C input shapes: same (from in_proj split)

    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        // Update angle state for this head
        for (size_t a = 0; a < MAX_NUM_ROPE_ANGLES; ++a) {
            // angles_raw for this head are at offset
            // (in the proj split, angles_raw comes after everything else)
            // We access angle_state directly here — it was updated in the caller
            // by adding angles_raw to it.
            // For now: state_.angle_state was updated before calling this.
            // We use current angle_state values for RoPE.
        }

        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            // Apply B_norm_weight elementwise
            // Then RoPE rotation
            const float* B_in = B + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;
            const float* C_in = C + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;
            float* B_out = B_rot + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;
            float* C_out = C_rot + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;

            // RoPE: split into even/odd pairs along D_STATE dimension
            // x_even = x[0::2], x_odd = x[1::2]
            // rot_even = x_even * cos - x_odd * sin
            // rot_odd  = x_even * sin + x_odd * cos
            for (size_t a = 0; a < MAX_NUM_ROPE_ANGLES; ++a) {
                float angle = state_.angle_state[h][a];
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);

                size_t even = a * 2;
                size_t odd  = a * 2 + 1;

                // B rotation (with B_norm_weight)
                float b_e = B_in[even] * weights_.B_norm_weight[even];
                float b_o = B_in[odd]  * weights_.B_norm_weight[odd];
                B_out[even] = b_e * cos_a - b_o * sin_a;
                B_out[odd]  = b_e * sin_a + b_o * cos_a;

                // C rotation (with C_norm_weight)
                float c_e = C_in[even] * weights_.C_norm_weight[even];
                float c_o = C_in[odd]  * weights_.C_norm_weight[odd];
                C_out[even] = c_e * cos_a - c_o * sin_a;
                C_out[odd]  = c_e * sin_a + c_o * cos_a;
            }
        }
    }
}

// ── SSMBlock Step 5: Multi-head MIMO SSM recurrence ──────────────

void SSMBlock::step_ssm_recurrence(const float* x_h,
                                    const float* B_rot, const float* C_rot,
                                    const float* DT_t, const float* A_t,
                                    float* y_head) {
    // y_head output: [NHEADS][HEADDIM]
    std::memset(y_head, 0, MAX_NHEADS * MAX_HEADDIM * sizeof(float));

    // Trapezoidal discretization
    float A_bar[MAX_NHEADS];
    float B_bar_scale[MAX_NHEADS];
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        float half_a = A_t[h] * DT_t[h] * 0.5f;
        float denom = 1.0f - half_a;
        A_bar[h] = (1.0f + half_a) / denom;
        B_bar_scale[h] = DT_t[h] / denom;
    }

    // MIMO x-coupling: x_slots[h][r] = sum_d x_h[h][d] * mimo_x[h][r][d]
    float x_slots[MAX_NHEADS][MAX_MIMO_RANK] = {};
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            float sum = 0.0f;
            for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                sum += x_h[h * MAX_HEADDIM + d]
                     * weights_.mimo_x[(h * MAX_MIMO_RANK + r) * MAX_HEADDIM + d];
            }
            x_slots[h][r] = sum;
        }
    }

    // SSM recurrence for each head and slot
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            const float* B_r = B_rot + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;
            float x_s = x_slots[h][r];

            // Update h_state for all headdim channels
            for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                for (size_t s = 0; s < MAX_D_STATE; ++s) {
                    float update = B_bar_scale[h] * B_r[s] * x_s;
                    state_.ssm_real[h][d][s] =
                        A_bar[h] * state_.ssm_real[h][d][s] + update;
                }
            }
        }
    }

    // Readout: y_head[h][d] = sum_r sum_s C_rot[h][r][s] * h_state[h][d][s]
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t d = 0; d < MAX_HEADDIM; ++d) {
            float sum = 0.0f;
            for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
                for (size_t s = 0; s < MAX_D_STATE; ++s) {
                    const float* C_r = C_rot + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;
                    sum += C_r[s] * state_.ssm_real[h][d][s];
                }
            }
            y_head[h * MAX_HEADDIM + d] = sum;
        }
    }
}

// ── SSMBlock Step 6: MIMO z-gating (SiLU) ─────────────────────────

void SSMBlock::step_z_gating(const float* z, const float* y_head,
                              float* y_gated) {
    // y_gated output: [NHEADS][HEADDIM]

    // SiLU on z: z_gated[i] = z[i] * sigmoid(z[i])
    float z_gated[MAX_D_INNER];
    for (size_t i = 0; i < MAX_D_INNER; ++i) {
        z_gated[i] = z[i] * sigmoid(z[i]);
    }

    // MIMO z-coupling: z_slots[h][r] = sum_d z_gated[h][d] * mimo_z[h][r][d]
    float z_slots[MAX_NHEADS][MAX_MIMO_RANK] = {};
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            float sum = 0.0f;
            for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                sum += z_gated[h * MAX_HEADDIM + d]
                     * weights_.mimo_z[(h * MAX_MIMO_RANK + r) * MAX_HEADDIM + d];
            }
            z_slots[h][r] = sum;
        }
    }

    // Per-head gate: sum over slots
    float z_gate_per_head[MAX_NHEADS] = {};
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            z_gate_per_head[h] += z_slots[h][r];
        }
    }

    // Apply gating: y_gated[h][d] = z_gate_per_head[h] * y_head[h][d]
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t d = 0; d < MAX_HEADDIM; ++d) {
            y_gated[h * MAX_HEADDIM + d] =
                z_gate_per_head[h] * y_head[h * MAX_HEADDIM + d];
        }
    }
}

// ── SSMBlock Step 7: Output projection + D skip + residual ────────

void SSMBlock::step_output(const float* y_gated, const float* x_h,
                            const float* u_film, float* output) {
    // out_proj_w @ y_gated → [D_MODEL]
    // out_proj_w is [D_MODEL, D_INNER], y_gated is [D_INNER]
    float out_proj[MAX_D_MODEL] = {};
    for (size_t i = 0; i < MAX_D_MODEL; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < MAX_D_INNER; ++j) {
            sum += weights_.out_proj_w[i * MAX_D_INNER + j] * y_gated[j];
        }
        out_proj[i] = sum;
    }

    // D skip (simplified): D_proj[h] * x_h[h] sum
    float d_skip_scale = 0.0f;
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        float x_sum = 0.0f;
        for (size_t d = 0; d < MAX_HEADDIM; ++d) {
            x_sum += x_h[h * MAX_HEADDIM + d];
        }
        d_skip_scale += weights_.D_proj[h] * x_sum;
    }
    float d_skip_contrib = d_skip_scale * 0.01f;  // ponytail: simplified

    // MIMO output coupling via mimo_o
    float mimo_o_sum = 0.0f;
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            float sum = 0.0f;
            for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                sum += y_gated[h * MAX_HEADDIM + d]
                     * weights_.mimo_o[(h * MAX_MIMO_RANK + r) * MAX_HEADDIM + d];
            }
            mimo_o_sum += sum;
        }
    }
    float mimo_o_contrib = mimo_o_sum * 0.01f;  // ponytail: simplified

    // Combine: output = out_proj + D skip + MIMO + residual
    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        output[d] = out_proj[d] + d_skip_contrib + mimo_o_contrib + u_film[d];
    }
}

// ── SSMBlock: Main per-sample forward ─────────────────────────────

void SSMBlock::process_sample(const float* input, float* output,
                               const float* cond) {
    // ── Step 0: FiLM ──
    float u_film[MAX_D_MODEL];
    step_film(input, u_film, cond);

    // ── Step 1: in_proj ──
    float proj[MAX_IN_PROJ_DIM];
    step_in_proj(u_film, proj);

    // Split proj into components
    size_t idx = 0;
    const float* z     = proj + idx; idx += MAX_D_INNER;
    const float* x_h   = proj + idx; idx += MAX_NHEADS * MAX_HEADDIM;
    const float* B     = proj + idx; idx += MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE;
    const float* C     = proj + idx; idx += MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE;
    const float* dd_dt = proj + idx; idx += MAX_NHEADS;
    const float* dd_A  = proj + idx; idx += MAX_NHEADS;
    const float* trap  = proj + idx; idx += MAX_NHEADS;
    (void)trap;  // reserved — not yet used in this implementation
    const float* angles_raw = proj + idx;

    // Update angle state
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t a = 0; a < MAX_NUM_ROPE_ANGLES; ++a) {
            state_.angle_state[h][a] += angles_raw[h * MAX_NUM_ROPE_ANGLES + a];
        }
    }

    // ── Step 2: A and Δ ──
    float DT_t[MAX_NHEADS], A_t[MAX_NHEADS];
    step_dt_A(dd_dt, dd_A, cond, DT_t, A_t);

    // ── Steps 3+4: RMSNorm + RoPE ──
    float B_rot[MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE];
    float C_rot[MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE];
    step_norm_rope(B, C, B_rot, C_rot);

    // ── Step 5: SSM recurrence ──
    float y_head[MAX_NHEADS * MAX_HEADDIM];
    step_ssm_recurrence(x_h, B_rot, C_rot, DT_t, A_t, y_head);

    // ── Step 6: MIMO z-gating ──
    float y_gated[MAX_NHEADS * MAX_HEADDIM];
    step_z_gating(z, y_head, y_gated);

    // ── Step 7: Output projection + residual ──
    step_output(y_gated, x_h, u_film, output);

    // ── Periodic state normalization ──
    ++state_.sample_count;
    state_.normalize_if_needed();
}

void SSMBlock::process_block(const float* input, float* output,
                              size_t n_samples, const float* cond) {
    for (size_t i = 0; i < n_samples; ++i) {
        process_sample(&input[i * MAX_D_MODEL], &output[i * MAX_D_MODEL], cond);
    }
}

}  // namespace clara::mamba3
