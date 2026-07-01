// CLARA Mamba-3 Inference Core — Multi-layer engine implementation

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

void SsmInferenceCore::apply_layer_norm(float *x, size_t layer_idx) {
    // Compute mean and variance
    float mean = 0.0f;
    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        mean += x[d];
    }
    mean /= static_cast<float>(MAX_D_MODEL);

    float var = 0.0f;
    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        float diff = x[d] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(MAX_D_MODEL);

    float inv_std = 1.0f / std::sqrt(var + 1e-5f);

    for (size_t d = 0; d < MAX_D_MODEL; ++d) {
        x[d] = ln_gamma_[layer_idx][d] * (x[d] - mean) * inv_std + ln_beta_[layer_idx][d];
    }
}

void SsmInferenceCore::process_block(const float *input, float *output, size_t n_samples) {
    if (!ready_ || n_samples == 0) return;

    // Handle deferred reset
    process_ready_reset();

    const float *cond = cond_proc_.get_conditioning();

    // Process samples one at a time through all layers
    for (size_t i = 0; i < n_samples; ++i) {
        // Embed scalar input → D_MODEL
        float x[MAX_D_MODEL] = {};
        x[0] = input[i]; // ponytail: first channel gets the mono sample

        // Forward through 6 layers
        for (size_t l = 0; l < MAX_LAYERS; ++l) {
            SSMBlock block(*layer_weights_[l], layer_states_[l]);
            float out[MAX_D_MODEL];
            block.process_sample(x, out, cond);

            // LayerNorm
            std::memcpy(x, out, MAX_D_MODEL * sizeof(float));
            apply_layer_norm(x, l);
        }

        // Extract scalar output from first D_MODEL channel
        output[i] = x[0];
        ++sample_count_;
    }
}

void SsmInferenceCore::reset_states() {
    reset_requested_.store(true, std::memory_order_release);
}

void SsmInferenceCore::set_conditioning(const float *cond) {
    cond_proc_.set_conditioning(cond);
}

} // namespace clara::mamba3
