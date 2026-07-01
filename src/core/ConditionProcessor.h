// CLARA Mamba-3 Condition Processor — RT-safe conditioning double-buffer
//
// The conditioning vector (128 floats) is updated from non-RT threads
// (ARA, CLAP text encoder, UI knobs) and read lock-free from the audio thread.
//
// Pattern: producer writes to inactive buffer, then atomic-swaps the active
// pointer. Consumer always reads through the atomic pointer — never blocks.

#pragma once

#include "core/MambaConfig.h"

#include <atomic>
#include <cstring>

namespace clara::mamba3 {

class ConditionProcessor {
public:
    ConditionProcessor() {
        active_.store(buffer_a_, std::memory_order_release);
    }

    // Called from audio thread — returns pointer to current conditioning vector.
    // Pointer is valid until the next call to this method (or until the
    // ConditionProcessor is destroyed). Never blocks.
    const float *get_conditioning() const {
        return active_.load(std::memory_order_acquire);
    }

    // Called from any non-RT thread — writes a new conditioning vector.
    // The update becomes visible to the audio thread on the next call to
    // get_conditioning().
    void set_conditioning(const float *cond) {
        // Write to the buffer NOT currently being read
        float *target = (active_.load(std::memory_order_relaxed) == buffer_a_) ? buffer_b_ : buffer_a_;
        std::memcpy(target, cond, COND_DIM * sizeof(float));
        // Atomic swap — makes the new buffer visible
        active_.store(target, std::memory_order_release);
    }

    // Zero out both buffers (e.g., on reset)
    void clear() {
        std::memset(buffer_a_, 0, sizeof(buffer_a_));
        std::memset(buffer_b_, 0, sizeof(buffer_b_));
    }

private:
    float buffer_a_[COND_DIM] = {};
    float buffer_b_[COND_DIM] = {};
    std::atomic<float *> active_{};
};

} // namespace clara::mamba3
