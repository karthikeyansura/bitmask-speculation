#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <limits>

namespace bitmask {
inline int64_t check_mask(const torch::Tensor& mask, int64_t vocab) {
    TORCH_CHECK(vocab > 0 && vocab <= std::numeric_limits<int>::max(), "invalid vocabulary size");
    TORCH_CHECK(mask.is_cuda(), "mask must be CUDA");
    TORCH_CHECK(mask.layout() == torch::kStrided && mask.dim() == 1 && mask.is_contiguous(),
                "mask must be a contiguous strided row");
    TORCH_CHECK(mask.scalar_type() == torch::kInt32, "mask must be int32");
    TORCH_CHECK(mask.numel() == (vocab + 31) / 32, "incorrect mask word count");
    return vocab;
}
inline int64_t check_selection(const torch::Tensor& logits, const torch::Tensor& mask) {
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(logits.layout() == torch::kStrided && logits.dim() == 1 && logits.is_contiguous(),
                "logits must be a contiguous strided row");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    check_mask(mask, logits.numel());
    TORCH_CHECK(logits.device() == mask.device(), "logits and mask devices differ");
    return logits.numel();
}
}  // namespace bitmask
