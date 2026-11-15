// Masked softmax sampling by inverse CDF in one read of the logits.
//
// Each thread owns a contiguous token range and folds it with the online
// softmax recurrence (Milakov & Gimelshein, 2018): a running maximum m and a
// sum s rescaled whenever m grows. Rescaling every thread's sum to the block
// maximum gives per-range masses in token-ID order; a block scan locates the
// range holding u * total, and only that thread walks its range. No probability
// vector is materialized and the CDF order matches the reference sampler.

#include "checks.h"
#include "reduce.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <vector>

std::vector<torch::Tensor> masked_argmax_cuda(const torch::Tensor&, const torch::Tensor&);

namespace {

constexpr int kThreads = 256;

// Status: 0 ok, 1 no valid token, 2 NaN logits, 3 invalid uniform,
// 4 no finite probability mass, 5 positive infinity.
__global__ void __launch_bounds__(kThreads)
sample_online(const float* __restrict__ logits, const int32_t* __restrict__ mask,
              const double* __restrict__ uniform, int64_t* __restrict__ output,
              int32_t* __restrict__ status, int64_t vocab, double temperature) {
    const int t = threadIdx.x;
    const int64_t span = (vocab + kThreads - 1) / kThreads;
    const int64_t begin = t * span < vocab ? t * span : vocab;
    const int64_t end = begin + span < vocab ? begin + span : vocab;

    double m = -CUDART_INF, s = 0;
    int flags = 0, found = 0;
    for (int64_t i = begin; i < end; ++i) {
        const float x = logits[i];
        if (isnan(x)) { flags |= 1; continue; }
        if (!bitmask::allowed(mask, i)) continue;
        found = 1;
        if (isinf(x)) { if (x > 0) flags |= 2; continue; }  // -inf carries zero mass.
        const double v = x;
        if (v > m) {
            s = s * exp((m - v) / temperature) + 1.0;
            m = v;
        } else {
            s += exp((v - m) / temperature);
        }
    }

    __shared__ double maxima[kThreads];
    __shared__ double masses[kThreads];
    __shared__ int shared_flags, error, owner, fallback;
    __shared__ double maximum, cutoff, prefix;
    if (t == 0) shared_flags = 0;
    __syncthreads();
    if (flags) atomicOr(&shared_flags, flags);
    maxima[t] = m;
    const int any_valid = __syncthreads_or(found);

    if (t == 0) {
        double block_max = -CUDART_INF;
        for (int k = 0; k < kThreads; ++k) block_max = fmax(block_max, maxima[k]);
        int code = (shared_flags & 1) ? 2 : (shared_flags & 2) ? 5 :
                   !any_valid ? 1 : !isfinite(block_max) ? 4 : 0;
        if (!code && !(*uniform >= 0 && *uniform < 1)) code = 3;
        *status = code;
        *output = -1;
        error = code;
        maximum = block_max;
    }
    __syncthreads();
    if (error) return;

    masses[t] = s > 0 ? s * exp((m - maximum) / temperature) : 0.0;
    __syncthreads();
    if (t == 0) {
        double total = 0;
        for (int k = 0; k < kThreads; ++k) total += masses[k];
        const double target = *uniform * total;
        double running = 0;
        int chosen = -1, last = -1;
        for (int k = 0; k < kThreads; ++k) {
            if (masses[k] <= 0) continue;
            last = k;
            if (target < running + masses[k]) { chosen = k; break; }
            running += masses[k];
        }
        // Rounding can leave the target past the final range; take its last token.
        fallback = chosen < 0;
        owner = fallback ? last : chosen;
        prefix = running;
        cutoff = target;
    }
    __syncthreads();
    if (t != owner) return;

    double cumulative = prefix;
    int64_t last_positive = -1;
    for (int64_t i = begin; i < end; ++i) {
        if (!bitmask::allowed(mask, i)) continue;
        const float x = logits[i];
        if (isinf(x)) continue;
        const double weight = exp((double(x) - maximum) / temperature);
        if (weight == 0) continue;
        last_positive = i;
        cumulative += weight;
        if (!fallback && cutoff < cumulative) { *output = i; return; }
    }
    *output = last_positive;
}

}  // namespace

std::vector<torch::Tensor> masked_sample_cuda(const torch::Tensor& logits, const torch::Tensor& mask,
                                             double temperature, const torch::Tensor& uniform) {
    const int64_t vocab = bitmask::check_selection(logits, mask);
    TORCH_CHECK(std::isfinite(temperature) && temperature >= 0, "invalid temperature");
    if (temperature == 0) return masked_argmax_cuda(logits, mask);
    TORCH_CHECK(uniform.is_cuda() && uniform.device() == logits.device(), "uniform device differs");
    TORCH_CHECK(uniform.scalar_type() == torch::kFloat64 && uniform.layout() == torch::kStrided &&
                uniform.dim() == 0 && uniform.is_contiguous(), "uniform must be a float64 scalar");
    const c10::cuda::CUDAGuard guard(logits.device());
    auto output = torch::empty({}, logits.options().dtype(torch::kInt64));
    auto status = torch::empty({}, logits.options().dtype(torch::kInt32));
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
    sample_online<<<1, kThreads, 0, stream>>>(logits.data_ptr<float>(), mask.data_ptr<int32_t>(),
        uniform.data_ptr<double>(), output.data_ptr<int64_t>(), status.data_ptr<int32_t>(), vocab, temperature);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, status};
}
