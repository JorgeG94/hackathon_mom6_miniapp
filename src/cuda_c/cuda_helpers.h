#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int    cuda_set_device_c(int device);
int    cuda_device_synchronize_c(void);
void*  cuda_malloc_c(size_t bytes);
void   cuda_free_c(void* ptr);
int    cuda_memcpy_h2d_c(void* dst, const void* src, size_t bytes);
int    cuda_memcpy_d2h_c(void* dst, const void* src, size_t bytes);
int    cuda_memcpy_d2d_c(void* dst, const void* src, size_t bytes);
int    cuda_memset_c(void* ptr, int value, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // CUDA_HELPERS_H
