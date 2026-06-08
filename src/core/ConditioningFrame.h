#pragma once

#include <cstddef>
#include <array>
#include <atomic>
#include <cstring>
#include <algorithm>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

struct alignas(16) ConditioningFrame {
    static constexpr size_t kCondDim = 8;

    std::array<float, kCondDim> data{};
    std::atomic<uint64_t> seq{0};

    ConditioningFrame() = default;

    ConditioningFrame(const ConditioningFrame& other)
        : data(other.data)
        , seq(other.seq.load(std::memory_order_relaxed))
    {}

    ConditioningFrame& operator=(const ConditioningFrame& other) {
        data = other.data;
        seq.store(other.seq.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    void fill(const float* src, size_t dim) {
        size_t n = std::min(dim, kCondDim);
        for (size_t i = 0; i < n; ++i) {
            data[i] = src[i];
        }
        for (size_t i = n; i < kCondDim; ++i) {
            data[i] = 0.0f;
        }
    }

    bool operator==(const ConditioningFrame& other) const {
        return data == other.data;
    }

    bool operator!=(const ConditioningFrame& other) const {
        return !(*this == other);
    }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif

static_assert(sizeof(ConditioningFrame) <= 64,
    "ConditioningFrame must fit in one cache line (64 bytes on x86)");
static_assert(alignof(ConditioningFrame) >= 16,
    "ConditioningFrame must be at least 16-byte aligned");
