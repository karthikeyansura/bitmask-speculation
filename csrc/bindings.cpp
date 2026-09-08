#include <torch/extension.h>

int cuda_runtime_version();
torch::Tensor mask_count_cuda(const torch::Tensor&, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Optional packed-mask CUDA primitives";
    module.def("runtime_version", &cuda_runtime_version);
    module.def("mask_count", &mask_count_cuda);
}
