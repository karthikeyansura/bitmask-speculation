#include "checks.h"
#include "reduce.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

namespace {
__global__ void masked_max(const float* logits, const int32_t* mask,
                           int64_t* output, int32_t* status, int64_t vocab) {
    bitmask::Best local{-CUDART_INF, INT_MAX, 0};
    for (int64_t token = threadIdx.x; token < vocab; token += blockDim.x) {
        const double value = logits[token];
        if (isnan(value)) local.bad |= 1;  // Includes invalid positions.
        else if (bitmask::allowed(mask, token))
            local = bitmask::combine(local, {value, static_cast<int>(token), 0});
    }
    auto best = bitmask::block_best(local);
    if (threadIdx.x == 0) {
        *status = best.bad ? 2 : (best.index == INT_MAX ? 1 : 0);
        *output = *status == 0 ? best.index : -1;
    }
}
}

std::vector<torch::Tensor> masked_argmax_cuda(const torch::Tensor& logits,
                                             const torch::Tensor& mask) {
    const int64_t vocab = bitmask::check_selection(logits, mask);
    const c10::cuda::CUDAGuard guard(logits.device());
    auto output = torch::empty({}, logits.options().dtype(torch::kInt64));
    auto status = torch::empty({}, logits.options().dtype(torch::kInt32));
    auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
    masked_max<<<1, 256, 0, stream.stream()>>>(logits.data_ptr<float>(), mask.data_ptr<int32_t>(),
        output.data_ptr<int64_t>(), status.data_ptr<int32_t>(), vocab);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, status};
}
