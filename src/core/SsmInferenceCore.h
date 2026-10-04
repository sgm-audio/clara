// CLARA Mamba-3 Inference Core — Multi-layer engine wrapper
//
// Runs the R=4 time-domain Mamba-3 model (SsmBlock) over 6 layers.
// Weights are mmap'd from a v2 flat binary file; all scratch is owned by the
// engine and allocated once in init(), so the audio path never touches the heap.
//
// Execution model (2026-10-04): the host's block is processed in chunks of
// PROCESS_CHUNK_SIZE samples, layer-major. Within a chunk the three projections
// run as matrix × chunk GEMMs and the conditioning projections are evaluated
// once; the SSM recurrence stays per-sample because it is causal. See
// MambaConfig.h for why that is a scheduling decision, not a model change.

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
    // Returns true on success, false on any error (file not found, dim mismatch,
    // truncated payload). Must be called from the main thread (file I/O).
    // Not RT-safe — it may allocate, but only here.
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
    // Splits into PROCESS_CHUNK_SIZE chunks internally. State carries across
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

    // ── Introspection (tests, diagnostics) ────────────────────────

    // Scratch this engine owns, in bytes (state + batch buffers). Useful for
    // memory budgeting in hosts.
    static size_t scratch_bytes() {
        return sizeof(SsmInferenceCore);
    }

private:
    bool ready_ = false;

    // Weights (loaded once)
    WeightLoader loader_;
    const SSMWeights *layer_weights_[MAX_LAYERS] = {};

    // Per-layer SSM blocks and states. Blocks are thin views over the weights
    // and state below — no per-chunk construction, and nothing allocated on the
    // audio thread.
    SSMBlock blocks_[MAX_LAYERS];
    SSMState layer_states_[MAX_LAYERS];

    // Per-layer LayerNorm parameters
    float ln_gamma_[MAX_LAYERS][MAX_D_MODEL] = {};
    float ln_beta_[MAX_LAYERS][MAX_D_MODEL] = {};

    // Conditioning (RT-safe atomic double-buffer)
    ConditionProcessor cond_proc_;

    // Deferred reset flag
    std::atomic<bool> reset_requested_{false};

    // Sub-block counter for state continuity
    size_t sample_count_ = 0;

    // ── Batch scratch (channel-major: element (row, i) at row * n + i) ──
    //
    // x_batch holds the layer input and, after film_batch(), the FiLM'd
    // embedding; the final combine writes the next layer's input back into it.
    alignas(64) float x_batch_[MAX_D_MODEL * PROCESS_CHUNK_SIZE] = {};
    alignas(64) float proj_batch_[MAX_IN_PROJ_DIM * PROCESS_CHUNK_SIZE] = {};
    alignas(64) float y_batch_[MAX_D_INNER * PROCESS_CHUNK_SIZE] = {};
    alignas(64) float out_batch_[MAX_D_MODEL * PROCESS_CHUNK_SIZE] = {};
    alignas(64) float d_skip_scratch_[PROCESS_CHUNK_SIZE] = {};
    alignas(64) float mimo_o_scratch_[PROCESS_CHUNK_SIZE] = {};

    // Internal helpers
    void process_chunk(const float *input, float *output, size_t n_samples, const float *cond);
    void apply_layer_norm(float *x, size_t n_samples, size_t layer_idx) const;
    bool process_ready_reset();
};

} // namespace clara::mamba3
