#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>

namespace bitmask {
struct Best { double value; int index; int bad; };
__device__ inline Best combine(Best a, Best b) {
    const int bad = a.bad | b.bad;
    if (b.index != INT_MAX && (a.index == INT_MAX || b.value > a.value ||
        (b.value == a.value && b.index < a.index))) a = b;
    a.bad = bad;
    return a;
}
// Exactly 256 participating threads. Only thread zero consumes the result.
__device__ inline Best block_best(Best local) {
    for (int offset = 16; offset > 0; offset /= 2) {
        Best other{__shfl_down_sync(0xffffffff, local.value, offset),
                   __shfl_down_sync(0xffffffff, local.index, offset),
                   __shfl_down_sync(0xffffffff, local.bad, offset)};
        local = combine(local, other);
    }
    __shared__ Best warps[8];
    if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = local;
    __syncthreads();
    Best result{-CUDART_INF, INT_MAX, 0};
    if (threadIdx.x == 0)
        for (int warp = 0; warp < 8; ++warp) result = combine(result, warps[warp]);
    return result;
}
__device__ inline bool allowed(const int32_t* mask, int64_t token) {
    return (static_cast<uint32_t>(mask[token / 32]) >> (token % 32)) & 1u;
}
}
