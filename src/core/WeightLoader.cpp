// CLARA Mamba-3 Weight Loader — Flat binary version 2 implementation

#include "core/WeightLoader.h"

#ifndef _WIN32
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace clara::mamba3 {

WeightLoader::~WeightLoader() {
    unload();
}

bool WeightLoader::load(const char *filepath) {
    unload(); // Release any previous data

#ifdef _WIN32
    // ponytail: read-into-heap on Windows — model is ~6MB loaded once at init;
    // switch to CreateFileMapping only if load time ever matters.
    FILE *f = std::fopen(filepath, "rb");
    if (f == nullptr) return false;
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    long file_size = std::ftell(f);
    if (file_size <= 0 || std::fseek(f, 0, SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }
    mapped_size_ = static_cast<size_t>(file_size);
    void *data = std::malloc(mapped_size_);
    if (data == nullptr || std::fread(data, 1, mapped_size_, f) != mapped_size_) {
        std::free(data);
        std::fclose(f);
        mapped_size_ = 0;
        return false;
    }
    std::fclose(f);
#else
    // Open file
    int fd = ::open(filepath, O_RDONLY);
    if (fd < 0) return false;

    // Get file size
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return false;
    }
    mapped_size_ = static_cast<size_t>(st.st_size);

    // mmap the entire file
    void *data = ::mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (data == MAP_FAILED) return false;
#endif

    mapped_data_ = static_cast<uint8_t *>(data);

    // Read header (first 64 bytes)
    static_assert(sizeof(WeightHeader) <= 64, "Header must fit in 64 bytes");
    std::memcpy(&header_, mapped_data_, sizeof(WeightHeader));

    // Validate header
    if (!validate_header()) {
        unload();
        return false;
    }

    // Setup per-layer weight pointers
    if (!setup_layer_pointers()) {
        unload();
        return false;
    }

    loaded_ = true;
    return true;
}

void WeightLoader::unload() {
    if (mapped_data_ != nullptr) {
#ifdef _WIN32
        std::free(mapped_data_);
#else
        ::munmap(mapped_data_, mapped_size_);
#endif
        mapped_data_ = nullptr;
        mapped_size_ = 0;
    }
    std::memset(&header_, 0, sizeof(header_));
    std::memset(layer_weights_, 0, sizeof(layer_weights_));
    loaded_ = false;
}

const SSMWeights *WeightLoader::get_layer_weights(size_t layer) const {
    if (!loaded_ || layer >= header_.n_layers) return nullptr;
    return &layer_weights_[layer];
}

bool WeightLoader::validate_header() const {
    if (header_.magic != WEIGHT_MAGIC) return false;
    if (header_.version != WEIGHT_VERSION) return false;

    // Dimension checks must match compile-time constants
    if (header_.d_state != MAX_D_STATE) return false;
    if (header_.d_model != MAX_D_MODEL) return false;
    if (header_.d_inner != MAX_D_INNER) return false;
    if (header_.nheads != MAX_NHEADS) return false;
    if (header_.headdim != MAX_HEADDIM) return false;
    if (header_.mimo_rank != MAX_MIMO_RANK) return false;
    if (header_.n_layers != MAX_LAYERS) return false;
    if (header_.cond_dim != COND_DIM) return false;
    if (header_.in_proj_dim != MAX_IN_PROJ_DIM) return false;

    // Check that weight data fits in the file
    size_t expected_size = 64 + header_.weight_data_size;
    if (mapped_size_ < expected_size) return false;

    return true;
}

bool WeightLoader::setup_layer_pointers() {
    // Weight data starts after 64-byte header
    const float *data = reinterpret_cast<const float *>(mapped_data_ + 64);
    size_t offset = 0;

    for (size_t l = 0; l < header_.n_layers; ++l) {
        SSMWeights &w = layer_weights_[l];

        // Array 0: in_proj_w [IN_PROJ_DIM * D_MODEL]
        w.in_proj_w = data + offset;
        offset += W_IN_PROJ_ELEMS;

        // Array 1: dt_bias [NHEADS]
        w.dt_bias = data + offset;
        offset += W_DT_BIAS_ELEMS;

        // Array 2: B_bias [NHEADS * MIMO_RANK * D_STATE]
        w.B_bias = data + offset;
        offset += W_B_BIAS_ELEMS;

        // Array 3: C_bias
        w.C_bias = data + offset;
        offset += W_C_BIAS_ELEMS;

        // Array 4: B_norm_weight [D_STATE]
        w.B_norm_weight = data + offset;
        offset += W_B_NORM_ELEMS;

        // Array 5: C_norm_weight
        w.C_norm_weight = data + offset;
        offset += W_C_NORM_ELEMS;

        // Array 6: mimo_x [NHEADS * MIMO_RANK * HEADDIM]
        w.mimo_x = data + offset;
        offset += W_MIMO_X_ELEMS;

        // Array 7: mimo_z
        w.mimo_z = data + offset;
        offset += W_MIMO_Z_ELEMS;

        // Array 8: mimo_o
        w.mimo_o = data + offset;
        offset += W_MIMO_O_ELEMS;

        // Array 9: D_proj [NHEADS]
        w.D_proj = data + offset;
        offset += W_D_PROJ_ELEMS;

        // Array 10: out_proj_w [D_MODEL * D_INNER]
        w.out_proj_w = data + offset;
        offset += W_OUT_PROJ_ELEMS;

        // Array 11: film_proj_w [2 * D_MODEL * COND_DIM]
        w.film_proj_w = data + offset;
        offset += W_FILM_PROJ_ELEMS;

        // Array 12: dt_film_proj_w [2 * NHEADS * COND_DIM]
        w.dt_film_proj_w = data + offset;
        offset += W_DT_FILM_PROJ_ELEMS;
    }

    // Verify total offset matches header
    if (offset != header_.weight_data_size / sizeof(float)) {
        return false;
    }

    return true;
}

} // namespace clara::mamba3
