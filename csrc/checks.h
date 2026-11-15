#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <limits>
#include <utility>

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
inline std::pair<int64_t, int64_t> check_batch(const torch::Tensor& logits, const torch::Tensor& masks) {
    TORCH_CHECK(logits.is_cuda() && masks.is_cuda(), "batch inputs must be CUDA");
    TORCH_CHECK(logits.device() == masks.device(), "logits and mask devices differ");
    TORCH_CHECK(logits.layout() == torch::kStrided && logits.dim() == 2 && logits.is_contiguous(),
                "logits must be a contiguous [B, V] tensor");
    TORCH_CHECK(masks.layout() == torch::kStrided && masks.dim() == 2 && masks.is_contiguous(),
                "masks must be a contiguous [B, ceil(V / 32)] tensor");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(masks.scalar_type() == torch::kInt32, "masks must be int32");
    const int64_t rows = logits.size(0), vocab = logits.size(1);
    TORCH_CHECK(rows <= 65535, "batch exceeds the CUDA grid-y limit");
    TORCH_CHECK(vocab > 0 && vocab <= std::numeric_limits<int>::max(), "invalid vocabulary size");
    TORCH_CHECK(masks.size(0) == rows && masks.size(1) == (vocab + 31) / 32, "incorrect mask shape");
    return {rows, vocab};
}
}  // namespace bitmask
