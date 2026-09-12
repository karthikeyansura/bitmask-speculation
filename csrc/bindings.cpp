#include <torch/extension.h>
#include <vector>

int cuda_runtime_version();
std::vector<torch::Tensor> masked_sample_cuda(const torch::Tensor&, const torch::Tensor&, double, const torch::Tensor&);
std::vector<torch::Tensor> masked_argmax_cuda(const torch::Tensor&, const torch::Tensor&);
torch::Tensor mask_count_cuda(const torch::Tensor&, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Optional packed-mask CUDA primitives";
    module.def("runtime_version", &cuda_runtime_version);
    module.def("mask_count", &mask_count_cuda);
    module.def("masked_argmax", &masked_argmax_cuda);
    module.def("masked_sample", &masked_sample_cuda);
}
