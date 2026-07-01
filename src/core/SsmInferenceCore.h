// CLARA Mamba-3 Inference Core — Multi-layer engine wrapper
//
// Phase 4 of the Mamba-1 → Mamba-3 migration.
// Replaces MambaInferenceEngine (ONNX Runtime wrapper).
//
// Manages 6 SSMBlock layers, per-layer LayerNorm, conditioning,
// chunked 64-sample block processing, and deferred state reset.

#pragma once

#include "core/ConditionProcessor.h"
#include "core/MambaConfig.h"
#include "core/SsmBlock.h"
#include "core/WeightLoader.h"

#include <atomic>
#include <cstddef>

namespace clara::mamba3 {

class SsmInferenceCore {
public:
    SsmInferenceCore() = default;
    ~SsmInferenceCore() = default;

    // ── Lifecycle ──────────────────────────────────────────────────

    // Load weights from a flat binary .bin file and initialize all layers.
    // Returns true on success, false on any error (file not found, dim mismatch).
    // Must be called from the main thread (file I/O). Not RT-safe.
    bool init(const char *weight_filepath);

    // True after successful init().
    bool is_ready() const {
        return ready_;
    }

    // ── Audio Processing (RT-safe) ─────────────────────────────────

    // Process a block of mono audio samples.
    // input:  [n_samples] float audio samples
    // output: [n_samples] float processed audio
    // n_samples: must be ≤ MAX_BLOCK_SIZE (512)
    //
    // Splits into 64-sample sub-blocks internally. State carries across
    // calls. Zero heap allocations.
    void process_block(const float *input, float *output, size_t n_samples);

    // Reset all SSM states to zero. Deferred — takes effect on the next
    // process_block() call. Safe from any thread.
    void reset_states();

    // ── Conditioning (RT-safe reads, non-RT writes) ───────────────

    // Set the conditioning vector. Called from non-RT threads (ARA, UI).
    // Takes effect on the next process_block() call.
    void set_conditioning(const float *cond);

    // Get access to the condition processor for direct manipulation.
    ConditionProcessor &conditioning() {
        return cond_proc_;
    }

private:
    bool ready_ = false;

    // Weights (loaded once)
    WeightLoader loader_;
    const SSMWeights *layer_weights_[MAX_LAYERS] = {};

    // Per-layer SSM blocks (stack-allocated, 6 layers)
    SSMState layer_states_[MAX_LAYERS];
    // ponytail: SSMBlock instances created on-stack in process_block
    // to avoid heap — each is a thin wrapper around weights+state pointers.

    // Per-layer LayerNorm parameters
    float ln_gamma_[MAX_LAYERS][MAX_D_MODEL] = {};
    float ln_beta_[MAX_LAYERS][MAX_D_MODEL] = {};

    // Conditioning (RT-safe atomic double-buffer)
    ConditionProcessor cond_proc_;

    // Deferred reset flag
    std::atomic<bool> reset_requested_{false};

    // Sub-block counter for state continuity
    size_t sample_count_ = 0;

    // Internal helpers
    void apply_layer_norm(float *x, size_t layer_idx);
    bool process_ready_reset();
};

} // namespace clara::mamba3
