#include "checks.h"
#include "reduce.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <vector>

std::vector<torch::Tensor> masked_argmax_cuda(const torch::Tensor&, const torch::Tensor&);

namespace {
__global__ void sample_cdf(const float* logits, const int32_t* mask, const double* uniform,
                           int64_t* output, int32_t* status, int64_t vocab, double temperature) {
    bitmask::Best local{-CUDART_INF, INT_MAX, 0};
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const double value = logits[i];
        if (isnan(value)) local.bad |= 1;
        else if (bitmask::allowed(mask, i)) {
            if (isinf(value) && value > 0) local.bad |= 2;
            local = bitmask::combine(local, {value, static_cast<int>(i), 0});
        }
    }
    auto best = bitmask::block_best(local);
    __shared__ double maximum;
    __shared__ int error;
    if (threadIdx.x == 0) {
        maximum = best.value;
        error = (best.bad & 1) ? 2 : (best.bad & 2) ? 5 :
                best.index == INT_MAX ? 1 : !isfinite(best.value) ? 4 : 0;
        if (!error && !(*uniform >= 0 && *uniform < 1)) error = 3;
        *status = error;
        *output = -1;
    }
    __syncthreads();
    if (error) return;

    double sum = 0;
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x)
        if (bitmask::allowed(mask, i)) sum += exp((double(logits[i]) - maximum) / temperature);
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ double sums[8];
    if (threadIdx.x % 32 == 0) sums[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0;
        for (int warp = 0; warp < 8; ++warp) total += sums[warp];
        const double cutoff = *uniform * total;
        double cumulative = 0;
        int64_t last_positive = -1;
        // This final CDF pass is serial. No softmax tensor is materialized.
        for (int64_t i = 0; i < vocab; ++i) {
            if (!bitmask::allowed(mask, i)) continue;
            const double weight = exp((double(logits[i]) - maximum) / temperature);
            if (weight == 0) continue;
            last_positive = i;
            cumulative += weight;
            if (cutoff < cumulative) { *output = i; return; }
        }
        // Parallel and serial sums can differ in their last rounding bits.
        *output = last_positive;
    }
}
}

std::vector<torch::Tensor> masked_sample_cuda(const torch::Tensor& logits,
                                             const torch::Tensor& mask,
                                             double temperature,
                                             const torch::Tensor& uniform) {
    const int64_t vocab = bitmask::check_selection(logits, mask);
    TORCH_CHECK(std::isfinite(temperature) && temperature >= 0, "invalid temperature");
    if (temperature == 0) return masked_argmax_cuda(logits, mask);
    TORCH_CHECK(uniform.is_cuda() && uniform.device() == logits.device(), "uniform device differs");
    TORCH_CHECK(uniform.scalar_type() == torch::kFloat64 && uniform.layout() == torch::kStrided &&
                uniform.dim() == 0 && uniform.is_contiguous(), "uniform must be a float64 scalar");
    const c10::cuda::CUDAGuard guard(logits.device());
    auto output = torch::empty({}, logits.options().dtype(torch::kInt64));
    auto status = torch::empty({}, logits.options().dtype(torch::kInt32));
    auto stream = c10::cuda::getCurrentCUDAStream(logits.get_device());
    sample_cdf<<<1, 256, 0, stream.stream()>>>(logits.data_ptr<float>(), mask.data_ptr<int32_t>(),
        uniform.data_ptr<double>(), output.data_ptr<int64_t>(), status.data_ptr<int32_t>(), vocab, temperature);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, status};
}
