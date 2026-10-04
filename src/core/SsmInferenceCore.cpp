// CLARA Mamba-3 Inference Core — Multi-layer engine implementation
//
// Pipeline per chunk (layer-major):
//   1. x_batch[d][i] = input[i]                    (embed scalar input)
//   2. for each layer:
//        a. prepare_conditioning(cond)            (once per chunk, not per sample)
//        b. film_batch(x_batch)                   (in place: x -> u_film)
//        c. in_proj_batch(x_batch) -> proj_batch  (GEMM)
//        d. sample_core_batch(...) -> y_batch      (per-sample recurrence)
//        e. out_proj_batch(y_batch) -> out_batch   (GEMM)
//        f. x_batch = out_batch + d_skip + mimo_o + u_film; LayerNorm
//   3. output[i] = x_batch[0][i]
//
// Numerically this is the same arithmetic as the per-sample formulation; only
// the summation order inside the projections differs (float32 rounding).

#include "core/SsmInferenceCore.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace clara::mamba3 {

// ── Lifecycle ──────────────────────────────────────────────────────

bool SsmInferenceCore::init(const char *weight_filepath) {
    if (!loader_.load(weight_filepath)) {
        return false;
    }

    for (size_t l = 0; l < MAX_LAYERS; ++l) {
        layer_weights_[l] = loader_.get_layer_weights(l);
        if (layer_weights_[l] == nullptr) return false;
        blocks_[l].bind(*layer_weights_[l], layer_states_[l]);

        // Initialize LayerNorm: gamma = 1, beta = 0
        for (size_t d = 0; d < MAX_D_MODEL; ++d) {
            ln_gamma_[l][d] = 1.0f;
            ln_beta_[l][d] = 0.0f;
        }
    }

    reset_states();
    ready_ = true;
    return true;
}

// ── Audio Processing ───────────────────────────────────────────────

bool SsmInferenceCore::process_ready_reset() {
    // Check-and-clear the deferred reset flag
    bool expected = true;
    if (reset_requested_.compare_exchange_strong(expected, false, std::memory_order_acq_rel)) {
        for (size_t l = 0; l < MAX_LAYERS; ++l) {
            layer_states_[l].zero();
        }
        sample_count_ = 0;
        return true;
    }
    return false;
}

// LayerNorm across d_model, per sample. x is channel-major [D_MODEL][n].
void SsmInferenceCore::apply_layer_norm(float *x, size_t n_samples, size_t layer_idx) const {
    const float *gamma = ln_gamma_[layer_idx];
    const float *beta = ln_beta_[layer_idx];

    for (size_t i = 0; i < n_samples; ++i) {
        float mean = 0.0f;
        for (size_t d = 0; d < MAX_D_MODEL; ++d) {
            mean += x[d * n_samples + i];
        }
        mean /= static_cast<float>(MAX_D_MODEL);

        float var = 0.0f;
        for (size_t d = 0; d < MAX_D_MODEL; ++d) {
            const float diff = x[d * n_samples + i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(MAX_D_MODEL);

        const float inv_std = 1.0f / std::sqrt(var + 1e-5f);

        for (size_t d = 0; d < MAX_D_MODEL; ++d) {
            float *v = &x[d * n_samples + i];
            *v = gamma[d] * (*v - mean) * inv_std + beta[d];
        }
    }
}

void SsmInferenceCore::process_block(const float *input, float *output, size_t n_samples) {
    if (!ready_ || n_samples == 0) return;

    // Handle deferred reset
    process_ready_reset();

    const float *cond = cond_proc_.get_conditioning();

    size_t offset = 0;
    while (offset < n_samples) {
        const size_t n = std::min(PROCESS_CHUNK_SIZE, n_samples - offset);
        process_chunk(input + offset, output + offset, n, cond);
        offset += n;
    }
}

void SsmInferenceCore::process_chunk(const float *input, float *output, size_t n_samples, const float *cond) {
    const size_t n = n_samples;

    // ── Embed scalar input into the first channel ──
    std::memset(x_batch_, 0, MAX_D_MODEL * n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        x_batch_[i] = input[i];
    }

    // ── Layer stack ──
    for (size_t l = 0; l < MAX_LAYERS; ++l) {
        SSMBlock &block = blocks_[l];

        block.prepare_conditioning(cond); // FiLM + Δ coefficients, once per chunk
        block.film_batch(x_batch_, n);    // x_batch becomes u_film in place
        block.in_proj_batch(x_batch_, n, proj_batch_);
        block.sample_core_batch(proj_batch_, x_batch_, n, y_batch_, d_skip_scratch_, mimo_o_scratch_);
        block.out_proj_batch(y_batch_, n, out_batch_);

        // output[d] = out_proj[d] + d_skip + mimo_o + u_film[d]
        for (size_t i = 0; i < n; ++i) {
            const float extra = d_skip_scratch_[i] + mimo_o_scratch_[i];
            for (size_t d = 0; d < MAX_D_MODEL; ++d) {
                const size_t idx = d * n + i;
                x_batch_[idx] = out_batch_[idx] + extra + x_batch_[idx];
            }
        }

        apply_layer_norm(x_batch_, n, l);
    }

    // ── Read the output sample from channel 0 ──
    for (size_t i = 0; i < n; ++i) {
        output[i] = x_batch_[i];
    }

    sample_count_ += n;
}

void SsmInferenceCore::reset_states() {
    reset_requested_.store(true, std::memory_order_release);
}

void SsmInferenceCore::set_conditioning(const float *cond) {
    cond_proc_.set_conditioning(cond);
}

} // namespace clara::mamba3
