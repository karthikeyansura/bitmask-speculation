#include "checks.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {
__global__ void count_bits(const int32_t* mask, int64_t* output, int64_t words, int tail) {
    int local = 0;
    for (int64_t i = threadIdx.x; i < words; i += blockDim.x) {
        uint32_t word = static_cast<uint32_t>(mask[i]);
        if (i == words - 1 && tail != 0) word &= (uint32_t(1) << tail) - 1;
        local += __popc(word);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        local += __shfl_down_sync(0xffffffff, local, offset);
    __shared__ int sums[8];
    if (threadIdx.x % 32 == 0) sums[threadIdx.x / 32] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        int64_t total = 0;
        for (int warp = 0; warp < 8; ++warp) total += sums[warp];
        *output = total;  // Always written; no caller zeroing or atomic accumulation.
    }
}
}

torch::Tensor mask_count_cuda(const torch::Tensor& mask, int64_t vocab) {
    bitmask::check_mask(mask, vocab);
    const c10::cuda::CUDAGuard guard(mask.device());
    auto output = torch::empty({}, mask.options().dtype(torch::kInt64));
    auto stream = c10::cuda::getCurrentCUDAStream(mask.get_device());
    count_bits<<<1, 256, 0, stream.stream()>>>(
        mask.data_ptr<int32_t>(), output.data_ptr<int64_t>(), mask.numel(), vocab % 32);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
