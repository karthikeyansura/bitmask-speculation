#include <torch/extension.h>

int cuda_runtime_version();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Optional packed-mask CUDA primitives";
    module.def("runtime_version", &cuda_runtime_version);
}
