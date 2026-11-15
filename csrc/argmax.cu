// Masked argmax with a fused valid-token count, for one row or a [B, V] batch.
//
// Pass 1 tiles each row across blocks. A warp owns 1,024 consecutive tokens:
// every lane loads one 32-bit mask word, counts it with __popc, and the warp
// broadcasts word j with __shfl_sync while lanes read logits j*32 + lane. The
// mask is read once and logit loads stay coalesced. Pass 2 reduces the
// per-block partials of each row in a single warp. Ties resolve to the lowest
// token ID, so results do not depend on block scheduling.

#include "checks.h"
#include "reduce.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

namespace {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int64_t kTokensPerWarp = 32 * 32;
constexpr int64_t kTokensPerBlock = kWarps * kTokensPerWarp;

struct Partial {
    float value;
    int index;
    int count;
    int bad;
};

__global__ void __launch_bounds__(kThreads)
partial_reduce(const float* __restrict__ logits, const int32_t* __restrict__ masks,
               Partial* __restrict__ partials, int64_t vocab, int64_t words, int blocks) {
    const int64_t row = blockIdx.y;
    logits += row * vocab;
    masks += row * words;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t first = blockIdx.x * kTokensPerBlock + warp * kTokensPerWarp;

    const uint32_t bits = bitmask::mask_word(masks, first / 32 + lane, words, vocab);
    int count = __popc(bits);
    bitmask::Best local = bitmask::empty_best();
#pragma unroll 4
    for (int j = 0; j < 32; ++j) {
        const uint32_t word = __shfl_sync(bitmask::kFullWarp, bits, j);
        const int64_t token = first + j * 32 + lane;
        if (token >= vocab) continue;
        const float value = logits[token];
        if (isnan(value)) {
            local.bad = 1;  // NaN fails the row even at a disallowed position.
        } else if ((word >> lane) & 1u) {
            local = bitmask::combine(local, {value, static_cast<int>(token), 0});
        }
    }
    local = bitmask::warp_best(local);
    count = bitmask::warp_sum(count);

    __shared__ bitmask::Best best[kWarps];
    __shared__ int counts[kWarps];
    if (lane == 0) {
        best[warp] = local;
        counts[warp] = count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bitmask::Best total = bitmask::empty_best();
        int valid = 0;
        for (int w = 0; w < kWarps; ++w) {
            total = bitmask::combine(total, best[w]);
            valid += counts[w];
        }
        partials[row * blocks + blockIdx.x] = {total.value, total.index, valid, total.bad};
    }
}

// One warp per row. Status: 0 ok, 1 no valid token, 2 NaN logits.
__global__ void finish_reduce(const Partial* __restrict__ partials, int blocks,
                              int64_t* __restrict__ tokens, int64_t* __restrict__ counts,
                              int32_t* __restrict__ status) {
    const int64_t row = blockIdx.x;
    partials += row * blocks;
    bitmask::Best local = bitmask::empty_best();
    long long valid = 0;
    for (int b = threadIdx.x; b < blocks; b += 32) {
        const Partial p = partials[b];
        local = bitmask::combine(local, {p.value, p.index, p.bad});
        valid += p.count;
    }
    local = bitmask::warp_best(local);
    valid = bitmask::warp_sum(valid);
    if (threadIdx.x == 0) {
        const int code = local.bad ? 2 : (local.index == INT_MAX ? 1 : 0);
        status[row] = code;
        tokens[row] = code == 0 ? local.index : -1;
        if (counts != nullptr) counts[row] = valid;
    }
}

void launch(const float* logits, const int32_t* masks, int64_t rows, int64_t vocab,
            int64_t* tokens, int64_t* counts, int32_t* status, const torch::Tensor& like) {
    const int64_t words = (vocab + 31) / 32;
    const int blocks = static_cast<int>((vocab + kTokensPerBlock - 1) / kTokensPerBlock);
    auto scratch = torch::empty({rows * blocks * static_cast<int64_t>(sizeof(Partial))},
                                like.options().dtype(torch::kUInt8));
    auto* partials = reinterpret_cast<Partial*>(scratch.data_ptr<uint8_t>());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(like.get_device()).stream();
    partial_reduce<<<dim3(blocks, static_cast<unsigned>(rows)), kThreads, 0, stream>>>(
        logits, masks, partials, vocab, words, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finish_reduce<<<static_cast<unsigned>(rows), 32, 0, stream>>>(partials, blocks, tokens, counts, status);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::vector<torch::Tensor> masked_argmax_cuda(const torch::Tensor& logits, const torch::Tensor& mask) {
    const int64_t vocab = bitmask::check_selection(logits, mask);
    const c10::cuda::CUDAGuard guard(logits.device());
    auto output = torch::empty({}, logits.options().dtype(torch::kInt64));
    auto status = torch::empty({}, logits.options().dtype(torch::kInt32));
    launch(logits.data_ptr<float>(), mask.data_ptr<int32_t>(), 1, vocab,
           output.data_ptr<int64_t>(), nullptr, status.data_ptr<int32_t>(), logits);
    return {output, status};
}

std::vector<torch::Tensor> masked_reduce_batch_cuda(const torch::Tensor& logits, const torch::Tensor& masks) {
    const auto [rows, vocab] = bitmask::check_batch(logits, masks);
    const c10::cuda::CUDAGuard guard(logits.device());
    auto tokens = torch::empty({rows}, logits.options().dtype(torch::kInt64));
    auto counts = torch::empty({rows}, logits.options().dtype(torch::kInt64));
    auto status = torch::empty({rows}, logits.options().dtype(torch::kInt32));
    if (rows > 0)
        launch(logits.data_ptr<float>(), masks.data_ptr<int32_t>(), rows, vocab,
               tokens.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), status.data_ptr<int32_t>(), logits);
    return {tokens, counts, status};
}
