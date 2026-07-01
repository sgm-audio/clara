// CLARA Mamba-3 Weight Loader — Flat binary version 2 reader
//
// Reads model.bin via mmap (or fallback read for platforms without mmap).
// Validates header magic, version, dimensions. Sets up SSMWeights pointers.

#pragma once

#include "core/MambaConfig.h"
#include "core/SsmBlock.h"

#include <cstddef>
#include <cstdint>

namespace clara::mamba3 {

struct WeightHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t d_state;
    uint32_t d_model;
    uint32_t d_inner;
    uint32_t nheads;
    uint32_t headdim;
    uint32_t mimo_rank;
    uint32_t n_layers;
    uint32_t cond_dim;
    uint32_t in_proj_dim;
    uint32_t weight_data_size;
};

class WeightLoader {
public:
    WeightLoader() = default;
    ~WeightLoader();

    // Load a .bin file and set up weight pointers.
    // Returns true on success, false on any error (bad magic, dim mismatch, etc.)
    bool load(const char *filepath);

    // Get weights for a specific layer (0-indexed).
    // The returned pointer is valid until the WeightLoader is destroyed.
    const SSMWeights *get_layer_weights(size_t layer) const;

    // Number of layers loaded.
    size_t num_layers() const {
        return header_.n_layers;
    }

    // Release the loaded data (called automatically by destructor).
    void unload();

private:
    WeightHeader header_ = {};
    uint8_t *mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
    bool loaded_ = false;

    // Per-layer SSMWeights structs (pointers into mapped_data_)
    SSMWeights layer_weights_[MAX_LAYERS] = {};

    bool validate_header() const;
    bool setup_layer_pointers();
};

} // namespace clara::mamba3
