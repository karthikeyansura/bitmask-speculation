#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

int cuda_runtime_version() {
    int version = 0;
    C10_CUDA_CHECK(cudaRuntimeGetVersion(&version));
    return version;
}
