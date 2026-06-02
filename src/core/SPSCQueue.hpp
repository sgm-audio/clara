#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <type_traits>

template<typename T, size_t Capacity = 64>
class SPSCQueue {
    static_assert(Capacity > 0 && Capacity <= 65536, "Capacity must be 1..65536");
    static_assert(std::is_nothrow_move_assignable_v<T> || std::is_copy_assignable_v<T>,
                  "T must be nothrow move-assignable or copy-assignable");

    // Power of 2 for efficient mask
    static constexpr size_t kMask = Capacity - 1;
    static_assert((Capacity & kMask) == 0, "Capacity must be power of 2");

    std::array<T, Capacity> buffer_;
    // Align head and tail on separate cache lines to avoid false sharing
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

public:
    SPSCQueue() = default;
    ~SPSCQueue() = default;

    // Non-copyable, non-movable
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    // Push from producer thread. Returns true if slot was available.
    bool push(const T& item) {
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_acquire);
        if (((tail + 1) & kMask) == (head & kMask)) {
            return false;  // full
        }
        buffer_[tail & kMask] = item;
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    bool push(T&& item) {
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_acquire);
        if (((tail + 1) & kMask) == (head & kMask)) {
            return false;  // full
        }
        buffer_[tail & kMask] = std::move(item);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Pop from consumer thread. Returns false if empty.
    bool pop(T& item) {
        auto head = head_.load(std::memory_order_relaxed);
        auto tail = tail_.load(std::memory_order_acquire);
        if (head == tail) {
            return false;  // empty
        }
        item = std::move(buffer_[head & kMask]);
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    size_t size() const {
        auto head = head_.load(std::memory_order_acquire);
        auto tail = tail_.load(std::memory_order_acquire);
        return tail - head;
    }

    bool empty() const {
        return size() == 0;
    }

    static constexpr size_t capacity() { return Capacity; }
};
