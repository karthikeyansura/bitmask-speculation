#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <cstdint>

namespace bitmask {

constexpr unsigned kFullWarp = 0xffffffffu;

// Candidate for a masked argmax. index == INT_MAX means "no allowed token yet".
// bad carries NaN observations, which are errors even at disallowed positions.
struct Best {
    float value;
    int index;
    int bad;
};

__device__ __forceinline__ Best empty_best() { return {-CUDART_INF_F, INT_MAX, 0}; }

// Larger value wins; equal values resolve to the lower token ID, so the result
// is independent of reduction order.
__device__ __forceinline__ Best combine(Best a, Best b) {
    const int bad = a.bad | b.bad;
    const bool take = b.index != INT_MAX &&
        (a.index == INT_MAX || b.value > a.value || (b.value == a.value && b.index < a.index));
    Best out = take ? b : a;
    out.bad = bad;
    return out;
}

__device__ __forceinline__ Best warp_best(Best local) {
    for (int offset = 16; offset > 0; offset /= 2) {
        const Best other{__shfl_down_sync(kFullWarp, local.value, offset),
                         __shfl_down_sync(kFullWarp, local.index, offset),
                         __shfl_down_sync(kFullWarp, local.bad, offset)};
        local = combine(local, other);
    }
    return local;
}

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(kFullWarp, value, offset);
    return value;
}

// Packed mask word with bits at or beyond the vocabulary cleared. Callers never
// see tail bits as candidates, whatever the producer left there.
__device__ __forceinline__ uint32_t mask_word(const int32_t* mask, int64_t word, int64_t words,
                                              int64_t vocab) {
    if (word >= words) return 0u;
    uint32_t bits = static_cast<uint32_t>(mask[word]);
    const int tail = static_cast<int>(vocab % 32);
    if (word == words - 1 && tail != 0) bits &= (1u << tail) - 1u;
    return bits;
}

__device__ __forceinline__ bool allowed(const int32_t* mask, int64_t token) {
    return (static_cast<uint32_t>(mask[token / 32]) >> (token % 32)) & 1u;
}

}  // namespace bitmask
