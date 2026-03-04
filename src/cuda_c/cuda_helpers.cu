// CUDA runtime helpers callable from Fortran via iso_c_binding.
// Extracted from mom6_coriolis_kernels.cu to avoid duplicate symbols
// when multiple .cu files are linked.

#include <cuda_runtime.h>
#include "cuda_helpers.h"

extern "C" int cuda_device_synchronize_c(void)
{
    return (int)cudaDeviceSynchronize();
}

extern "C" void* cuda_malloc_c(size_t bytes)
{
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    return (err == cudaSuccess) ? ptr : NULL;
}

extern "C" void cuda_free_c(void* ptr)
{
    if (ptr) cudaFree(ptr);
}

extern "C" int cuda_memcpy_h2d_c(void* dst, const void* src, size_t bytes)
{
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

extern "C" int cuda_memcpy_d2h_c(void* dst, const void* src, size_t bytes)
{
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

extern "C" int cuda_memset_c(void* ptr, int value, size_t bytes)
{
    return (int)cudaMemset(ptr, value, bytes);
}
