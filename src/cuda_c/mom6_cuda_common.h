#ifndef MOM6_CUDA_COMMON_H
#define MOM6_CUDA_COMMON_H

// 1-based column-major indexing (matching Fortran memory layout)
// For a 2D array of dimensions (n1, n2):  arr(i,j) -> arr[IDX2(i,j,n1)]
// For a 3D array of dimensions (n1, n2, n3): arr(i,j,k) -> arr[IDX3(i,j,k,n1,n2)]
#define IDX2(i, j, n1)         (((j)-1)*(n1) + ((i)-1))
#define IDX3(i, j, k, n1, n2)  (((k)-1)*(n1)*(n2) + ((j)-1)*(n1) + ((i)-1))

// Standard thread indexing (1-based, matching Fortran convention)
// blockIdx and threadIdx are 0-based in CUDA C, so add 1
#define THREAD_I  ((int)(blockIdx.x * blockDim.x + threadIdx.x + 1))
#define THREAD_J  ((int)(blockIdx.y * blockDim.y + threadIdx.y + 1))
#define THREAD_K  ((int)(blockIdx.z + 1))

// Constants matching Fortran parameters
#define C1_12  (1.0 / 12.0)
#define C1_24  (1.0 / 24.0)

#endif // MOM6_CUDA_COMMON_H
