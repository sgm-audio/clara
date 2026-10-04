// CLARA Mamba-3 SSM Block — 8-step pipeline implementation
//
// Reference target: python/reference_forward.py ssm_block_forward()
// All arrays are stack-allocated or caller-owned. Zero heap allocations.
//
// Layout convention for the batched entry points: channel-major, element
// (row, i) at row * n_samples + i. This keeps the inner loop contiguous over
// samples, which is what lets the three projections vectorize.

#include "core/SsmBlock.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace clara::mamba3 {

// ── Math helpers (inline for RT-safety) ───────────────────────────

namespace {

inline float softplus(float x) {
    // log(1 + exp(x)), numerically stable
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float rsqrt(float x) {
    return 1.0f / std::sqrt(x);
}

// ── Blocked GEMM for the projections ──────────────────────────────
//
// out[M][n] = W[M][K] * x[K][n], all channel-major (element (r, i) at r * n + i).
//
// Why blocked: the row-outer formulation issues one load of W and one load of x
// per FMA (~0.5 FMA/load), which is load-port bound. This kernel keeps a BJ × BI
// tile of the output in registers while streaming K, so each x load feeds BJ
// FMAs and each W load feeds BI FMAs.
//
// Measured on the reference machine (Xeon 2.6 GHz, generic SSE2 build), per
// sample: row-outer in_proj 0.0343 ms -> blocked 0.0246 ms (1.4x);
// out_proj 0.0053 ms -> 0.0037 ms (1.4x). Sweeping the tile shape from 2x16 to
// 8x16 moved the total by less than 5%, so 4x16 is kept as a middle setting:
// on this CPU the kernel is at its floating-point issue limit, not load bound,
// once blocked. Wider hardware (AVX-512/FMA builds) is where larger tiles pay
// off, which is why the tile size is a compile-time knob (CLARA_GEMM_BJ/BI).
//
// M and n are runtime values (n varies with the host block size); K is a
// template parameter so the inner loop fully unrolls.
#ifndef CLARA_GEMM_BJ
#    define CLARA_GEMM_BJ 4
#endif
#ifndef CLARA_GEMM_BI
#    define CLARA_GEMM_BI 16
#endif

template <size_t BJ, size_t BI, size_t K>
void gemm_blocked(const float *__restrict w, const float *__restrict x, size_t m, size_t n, float *__restrict out) {
    if (m == 0 || n == 0) return;

    // Contract: m and n index fixed-size engine scratch (m is a projection
    // dimension, n ≤ PROCESS_CHUNK_SIZE). Bounding them lets the compiler prove
    // the index math cannot overflow — without it GCC emits a spurious
    // -Waggressive-loop-optimizations warning on the remainder loops below.
    if (m > MAX_IN_PROJ_DIM || n > MAX_BLOCK_SIZE) return;

    size_t j0 = 0;
    for (; j0 + BJ <= m; j0 += BJ) {
        size_t i0 = 0;
        for (; i0 + BI <= n; i0 += BI) {
            float acc[BJ][BI] = {};

            for (size_t k = 0; k < K; ++k) {
                float xv[BI];
                const float *xrow = x + k * n + i0;
                for (size_t t = 0; t < BI; ++t) {
                    xv[t] = xrow[t];
                }
                for (size_t jj = 0; jj < BJ; ++jj) {
                    const float wv = w[(j0 + jj) * K + k];
                    for (size_t t = 0; t < BI; ++t) {
                        acc[jj][t] += wv * xv[t];
                    }
                }
            }

            for (size_t jj = 0; jj < BJ; ++jj) {
                float *orow = out + (j0 + jj) * n + i0;
                for (size_t t = 0; t < BI; ++t) {
                    orow[t] = acc[jj][t];
                }
            }
        }

        // Tail columns (n is the host block size, so the last chunk can be short)
        for (; i0 < n; ++i0) {
            for (size_t jj = 0; jj < BJ; ++jj) {
                const float *wrow = w + (j0 + jj) * K;
                float s = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    s += wrow[k] * x[k * n + i0];
                }
                out[(j0 + jj) * n + i0] = s;
            }
        }
    }

    // Tail rows (fewer than BJ of them), walked with pointer increments rather
    // than j * K indexing.
    const float *wrow = w + j0 * K;
    float *orow = out + j0 * n;
    for (size_t j = j0; j < m; ++j, wrow += K, orow += n) {
        for (size_t i = 0; i < n; ++i) {
            float s = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                s += wrow[k] * x[k * n + i];
            }
            orow[i] = s;
        }
    }
}

} // namespace

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

// ── Conditioning (once per chunk) ─────────────────────────────────

void SSMBlock::prepare_conditioning(const float *cond) {
    has_cond_ = (cond != nullptr);
    if (!has_cond_) return;

    // film_proj_w @ cond → [2 * D_MODEL]; gamma = first half, beta = second
    for (size_t i = 0; i < MAX_D_MODEL; ++i) {
        float g = 0.0f;
        float b = 0.0f;
        const float *w_g = weights_->film_proj_w + i * COND_DIM;
        const float *w_b = weights_->film_proj_w + (MAX_D_MODEL + i) * COND_DIM;
        for (size_t j = 0; j < COND_DIM; ++j) {
            const float c = cond[j];
            g += w_g[j] * c;
            b += w_b[j] * c;
        }
        film_gamma_[i] = g;
        film_beta_[i] = b;
    }

    // dt_film_proj_w @ cond → [2 * NHEADS]
    for (size_t i = 0; i < 2 * MAX_NHEADS; ++i) {
        float sum = 0.0f;
        const float *w = weights_->dt_film_proj_w + i * COND_DIM;
        for (size_t j = 0; j < COND_DIM; ++j) {
            sum += w[j] * cond[j];
        }
        dt_cond_[i] = sum;
    }
}

// ── Step 0: FiLM conditioning ─────────────────────────────────────

void SSMBlock::step_film(const float *u, float *u_film) const {
    if (!has_cond_) {
        std::memcpy(u_film, u, MAX_D_MODEL * sizeof(float));
        return;
    }
    // u_film[d] = u[d] * (1 + gamma[d]) + beta[d]
    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        u_film[d] = u[d] * (1.0f + film_gamma_[d]) + film_beta_[d];
    }
}

void SSMBlock::film_batch(float *x, size_t n_samples) const {
    if (!has_cond_) return; // passthrough
    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        const float scale = 1.0f + film_gamma_[d];
        const float beta = film_beta_[d];
        float *row = x + d * n_samples;
        for (size_t i = 0; i < n_samples; ++i) {
            row[i] = row[i] * scale + beta;
        }
    }
}

// ── Step 1: Fused in_proj ─────────────────────────────────────────

void SSMBlock::step_in_proj(const float *u_film, float *proj) const {
    for (size_t i = 0; i < MAX_IN_PROJ_DIM; ++i) {
        float sum = 0.0f;
        const float *row = weights_->in_proj_w + i * MAX_D_MODEL;
        for (size_t j = 0; j < MAX_D_MODEL; ++j) {
            sum += row[j] * u_film[j];
        }
        proj[i] = sum;
    }
}

// proj[j][i] = sum_d in_proj_w[j][d] * x[d][i]
// 1612 × 128 × n: two thirds of the engine's FLOPs, so it uses the blocked
// kernel (see gemm_blocked above).
void SSMBlock::in_proj_batch(const float *x, size_t n_samples, float *proj) const {
    gemm_blocked<CLARA_GEMM_BJ, CLARA_GEMM_BI, MAX_D_MODEL>(weights_->in_proj_w, x, MAX_IN_PROJ_DIM, n_samples, proj);
}

// ── Step 2: A and Δ preprocessing ────────────────────────────────

void SSMBlock::step_dt_A(const float *dd_dt, const float *dd_A, float *DT_t, float *A_t) const {
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        float dt_h = softplus(dd_dt[h] + weights_->dt_bias[h]);
        if (has_cond_) {
            dt_h = softplus(dt_h + dt_cond_[h] + dt_cond_[MAX_NHEADS + h]);
        }
        DT_t[h] = dt_h;
        A_t[h] = -softplus(dd_A[h]);
    }
}

// ── Steps 3+4: bias, norm scaling, RoPE on B and C ───────────────

void SSMBlock::step_norm_rope(const float *B, const float *C, const float *angles_raw, float *B_rot, float *C_rot) {
    // Order matters: the angle state is advanced first and the rotation uses
    // the advanced values (matching process_sample() and reference_forward.py).
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t a = 0; a < MAX_NUM_ROPE_ANGLES; ++a) {
            state_->angle_state[h][a] += angles_raw[h * MAX_NUM_ROPE_ANGLES + a];
        }
    }

    // The rotation angle for head h is state_.angle_state[h][a] — it does not
    // depend on the MIMO rank, so cos/sin are computed once per (head, angle)
    // and reused across ranks. The naive version evaluated 4× as many
    // transcendentals (256 per layer per sample instead of 64).
    float cos_t[MAX_NUM_ROPE_ANGLES];
    float sin_t[MAX_NUM_ROPE_ANGLES];

    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t a = 0; a < MAX_NUM_ROPE_ANGLES; ++a) {
            const float angle = state_->angle_state[h][a];
            cos_t[a] = std::cos(angle);
            sin_t[a] = std::sin(angle);
        }

        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            const size_t slot = h * MAX_MIMO_RANK + r;
            const float *B_in = B + slot * MAX_D_STATE;
            const float *C_in = C + slot * MAX_D_STATE;
            float *B_out = B_rot + slot * MAX_D_STATE;
            float *C_out = C_rot + slot * MAX_D_STATE;

            const float *B_bias_entry = weights_->B_bias + slot * MAX_D_STATE;
            const float *C_bias_entry = weights_->C_bias + slot * MAX_D_STATE;

            for (size_t a = 0; a < MAX_NUM_ROPE_ANGLES; ++a) {
                const float cos_a = cos_t[a];
                const float sin_a = sin_t[a];
                const size_t even = a * 2;
                const size_t odd = a * 2 + 1;

                // B rotation (with B_bias + B_norm_weight)
                const float b_e = (B_in[even] + B_bias_entry[even]) * weights_->B_norm_weight[even];
                const float b_o = (B_in[odd] + B_bias_entry[odd]) * weights_->B_norm_weight[odd];
                B_out[even] = b_e * cos_a - b_o * sin_a;
                B_out[odd] = b_e * sin_a + b_o * cos_a;

                // C rotation (with C_bias + C_norm_weight)
                const float c_e = (C_in[even] + C_bias_entry[even]) * weights_->C_norm_weight[even];
                const float c_o = (C_in[odd] + C_bias_entry[odd]) * weights_->C_norm_weight[odd];
                C_out[even] = c_e * cos_a - c_o * sin_a;
                C_out[odd] = c_e * sin_a + c_o * cos_a;
            }
        }
    }
}

// ── Step 5: Multi-head MIMO SSM recurrence ───────────────────────

void SSMBlock::step_ssm_recurrence(const float *x_h, const float *B_rot, const float *C_rot, const float *DT_t,
                                   const float *A_t, float *y_head) {
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
                sum += x_h[h * MAX_HEADDIM + d] * weights_->mimo_x[(h * MAX_MIMO_RANK + r) * MAX_HEADDIM + d];
            }
            x_slots[h][r] = sum;
        }
    }

    // SSM recurrence for each head and slot
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            const float *B_r = B_rot + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;
            const float x_s = x_slots[h][r];

            for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                for (size_t s = 0; s < MAX_D_STATE; ++s) {
                    float update = B_bar_scale[h] * B_r[s] * x_s;
                    state_->ssm_real[h][d][s] = A_bar[h] * state_->ssm_real[h][d][s] + update;
                }
            }
        }
    }

    // Readout: y_head[h][d] = sum_r sum_s C_rot[h][r][s] * h_state[h][d][s]
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t d = 0; d < MAX_HEADDIM; ++d) {
            float sum = 0.0f;
            for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
                const float *C_r = C_rot + (h * MAX_MIMO_RANK + r) * MAX_D_STATE;
                for (size_t s = 0; s < MAX_D_STATE; ++s) {
                    sum += C_r[s] * state_->ssm_real[h][d][s];
                }
            }
            y_head[h * MAX_HEADDIM + d] = sum;
        }
    }
}

// ── Step 6: MIMO z-gating (SiLU) ─────────────────────────────────

void SSMBlock::step_z_gating(const float *z, const float *y_head, float *y_gated) const {
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
                sum += z_gated[h * MAX_HEADDIM + d] * weights_->mimo_z[(h * MAX_MIMO_RANK + r) * MAX_HEADDIM + d];
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
            y_gated[h * MAX_HEADDIM + d] = z_gate_per_head[h] * y_head[h * MAX_HEADDIM + d];
        }
    }
}

// ── Step 7: output terms ─────────────────────────────────────────

void SSMBlock::step_output_scalars(const float *y_gated, const float *x_h, float *d_skip_scale,
                                   float *mimo_o_sum) const {
    // D skip: sum_h D_proj[h] * sum_d x_h[h][d]
    float d_skip = 0.0f;
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        float x_sum = 0.0f;
        for (size_t d = 0; d < MAX_HEADDIM; ++d) {
            x_sum += x_h[h * MAX_HEADDIM + d];
        }
        d_skip += weights_->D_proj[h] * x_sum;
    }
    *d_skip_scale = d_skip * 0.01f;

    // MIMO output coupling: sum_h sum_r dot(y_gated[h], mimo_o[h][r])
    float mimo_o = 0.0f;
    for (size_t h = 0; h < MAX_NHEADS; ++h) {
        for (size_t r = 0; r < MAX_MIMO_RANK; ++r) {
            const float *mrow = weights_->mimo_o + (h * MAX_MIMO_RANK + r) * MAX_HEADDIM;
            float sum = 0.0f;
            for (size_t d = 0; d < MAX_HEADDIM; ++d) {
                sum += y_gated[h * MAX_HEADDIM + d] * mrow[d];
            }
            mimo_o += sum;
        }
    }
    *mimo_o_sum = mimo_o * 0.01f;
}

void SSMBlock::out_proj_batch(const float *y_gated, size_t n_samples, float *out) const {
    gemm_blocked<CLARA_GEMM_BJ, CLARA_GEMM_BI, MAX_D_INNER>(weights_->out_proj_w, y_gated, MAX_D_MODEL, n_samples, out);
}

// ── Steps 2-7 for one sample ─────────────────────────────────────

void SSMBlock::sample_core(const float *u_film, const float *proj, float *y_gated, float *d_skip_scale,
                           float *mimo_o_sum) {
    (void) u_film; // residual is applied by the caller (it owns the batch buffer)

    size_t idx = 0;
    const float *z = proj + idx;
    idx += MAX_D_INNER;
    const float *x_h = proj + idx;
    idx += MAX_NHEADS * MAX_HEADDIM;
    const float *B = proj + idx;
    idx += MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE;
    const float *C = proj + idx;
    idx += MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE;
    const float *dd_dt = proj + idx;
    idx += MAX_NHEADS;
    const float *dd_A = proj + idx;
    idx += MAX_NHEADS;
    idx += MAX_NHEADS; // trap — reserved
    const float *angles_raw = proj + idx;

    // ── Step 2: A and Δ ──
    float DT_t[MAX_NHEADS], A_t[MAX_NHEADS];
    step_dt_A(dd_dt, dd_A, DT_t, A_t);

    // ── Steps 3+4: bias, norm scaling, RoPE (advances angle state) ──
    float B_rot[MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE];
    float C_rot[MAX_NHEADS * MAX_MIMO_RANK * MAX_D_STATE];
    step_norm_rope(B, C, angles_raw, B_rot, C_rot);

    // ── Step 5: SSM recurrence ──
    float y_head[MAX_NHEADS * MAX_HEADDIM];
    step_ssm_recurrence(x_h, B_rot, C_rot, DT_t, A_t, y_head);

    // ── Step 6: MIMO z-gating ──
    step_z_gating(z, y_head, y_gated);

    // ── Step 7 (scalars; the projection runs batched in the caller) ──
    step_output_scalars(y_gated, x_h, d_skip_scale, mimo_o_sum);

    // ── Periodic state normalization ──
    ++state_->sample_count;
    state_->normalize_if_needed();
}

void SSMBlock::sample_core_batch(const float *proj, const float *u_film, size_t n_samples, float *y_gated,
                                 float *d_skip_scale, float *mimo_o_sum) {
    float proj_col[MAX_IN_PROJ_DIM];
    float film_col[MAX_D_MODEL];
    float y_col[MAX_D_INNER];

    for (size_t i = 0; i < n_samples; ++i) {
        // Gather this sample's columns out of the channel-major batch buffers
        for (size_t j = 0; j < MAX_IN_PROJ_DIM; ++j) {
            proj_col[j] = proj[j * n_samples + i];
        }
        for (size_t d = 0; d < MAX_D_MODEL; ++d) {
            film_col[d] = u_film[d * n_samples + i];
        }

        sample_core(film_col, proj_col, y_col, &d_skip_scale[i], &mimo_o_sum[i]);

        for (size_t j = 0; j < MAX_D_INNER; ++j) {
            y_gated[j * n_samples + i] = y_col[j];
        }
    }
}

// ── Single-sample path ───────────────────────────────────────────

void SSMBlock::process_sample(const float *input, float *output, const float *cond) {
    prepare_conditioning(cond);

    float u_film[MAX_D_MODEL];
    step_film(input, u_film);

    float proj[MAX_IN_PROJ_DIM];
    step_in_proj(u_film, proj);

    float y_gated[MAX_D_INNER];
    float d_skip_contrib = 0.0f;
    float mimo_o_contrib = 0.0f;
    sample_core(u_film, proj, y_gated, &d_skip_contrib, &mimo_o_contrib);

    float out_proj[MAX_D_MODEL];
    for (size_t i = 0; i < MAX_D_MODEL; ++i) {
        const float *wrow = weights_->out_proj_w + i * MAX_D_INNER;
        float sum = 0.0f;
        for (size_t j = 0; j < MAX_D_INNER; ++j) {
            sum += wrow[j] * y_gated[j];
        }
        out_proj[i] = sum;
    }

    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        output[d] = out_proj[d] + d_skip_contrib + mimo_o_contrib + u_film[d];
    }
}

void SSMBlock::process_block(const float *input, float *output, size_t n_samples, const float *cond) {
    for (size_t i = 0; i < n_samples; ++i) {
        process_sample(&input[i * MAX_D_MODEL], &output[i * MAX_D_MODEL], cond);
    }
}

} // namespace clara::mamba3
