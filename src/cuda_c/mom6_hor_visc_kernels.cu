// MOM6 Horizontal Viscosity — CUDA C kernels
//
// 1:1 port of the two attributes(global) kernels from
// mom6_hor_visc_cuda.F90.  Uses 1-based column-major indexing macros
// so that array accesses are bit-identical to the Fortran CUDA version.
//
// Two fused 3D kernels:
//   1. stress_kernel_c    — vel_grad + strain + stress (str_xx, str_xy)
//   2. divergence_kernel_c — viscous acceleration (diffu, diffv)
//
// Intermediate values (dudx, dvdy, dvdx, dudy, sh_xx, sh_xy) remain
// in registers and never touch global memory.
//
// 1-based local indexing, k in blockIdx.z.

#include "mom6_cuda_common.h"
#include <cuda_runtime.h>

// =========================================================================
// Kernel 1: stress_kernel_c — fused vel_grad + strain + stress (3D)
//
// Computes str_xx at h-points and str_xy at q-points directly from
// u, v, h.
//
// str_xx range: i in [Isq_l : Ieq_l+1], j in [Jsq_l : Jeq_l+1]
// str_xy range: i in [Isq_l : Ieq_l],   j in [Jsq_l : Jeq_l]
//
// where Isq_l = is_l - 1, Ieq_l = ie_l, Jsq_l = js_l - 1, Jeq_l = je_l
// =========================================================================
__global__ void stress_kernel_c(
    double* __restrict__ str_xx,
    double* __restrict__ str_xy,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ h,
    const double* __restrict__ DY_dxT,
    const double* __restrict__ DX_dyT,
    const double* __restrict__ DY_dxBu,
    const double* __restrict__ DX_dyBu,
    const double* __restrict__ IdyCu,
    const double* __restrict__ IdxCu,
    const double* __restrict__ IdyCv,
    const double* __restrict__ IdxCv,
    const double* __restrict__ mask2dBu,
    const double* __restrict__ reduction_xx,
    const double* __restrict__ reduction_xy,
    double Kh_bg,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i > n1 || j > n2 || k < 1 || k > n3) return;

    int Isq_l = is_l - 1;
    int Ieq_l = ie_l;
    int Jsq_l = js_l - 1;
    int Jeq_l = je_l;

    // --- h-point: vel_grad -> strain -> str_xx (all in registers) ---
    if (i >= Isq_l && i <= Ieq_l + 1 &&
        j >= Jsq_l && j <= Jeq_l + 1) {
        double dudx_l = DY_dxT[IDX2(i, j, n1)] * (IdyCu[IDX2(i, j, n1)] * u[IDX3(i, j, k, n1, n2)] -
                                                     IdyCu[IDX2(i - 1, j, n1)] * u[IDX3(i - 1, j, k, n1, n2)]);
        double dvdy_l = DX_dyT[IDX2(i, j, n1)] * (IdxCv[IDX2(i, j, n1)] * v[IDX3(i, j, k, n1, n2)] -
                                                     IdxCv[IDX2(i, j - 1, n1)] * v[IDX3(i, j - 1, k, n1, n2)]);
        double sh_xx_l = dudx_l - dvdy_l;
        str_xx[IDX3(i, j, k, n1, n2)] = -Kh_bg * sh_xx_l * h[IDX3(i, j, k, n1, n2)] * reduction_xx[IDX2(i, j, n1)];
    }

    // --- q-point: vel_grad -> strain -> str_xy (all in registers) ---
    if (i >= Isq_l && i <= Ieq_l &&
        j >= Jsq_l && j <= Jeq_l) {
        double dvdx_l = DY_dxBu[IDX2(i, j, n1)] * (v[IDX3(i + 1, j, k, n1, n2)] * IdyCv[IDX2(i + 1, j, n1)] -
                                                      v[IDX3(i, j, k, n1, n2)] * IdyCv[IDX2(i, j, n1)]);
        double dudy_l = DX_dyBu[IDX2(i, j, n1)] * (u[IDX3(i, j + 1, k, n1, n2)] * IdxCu[IDX2(i, j + 1, n1)] -
                                                      u[IDX3(i, j, k, n1, n2)] * IdxCu[IDX2(i, j, n1)]);
        double sh_xy_l = mask2dBu[IDX2(i, j, n1)] * (dvdx_l + dudy_l);
        double hq_val = 0.25 * ((h[IDX3(i, j, k, n1, n2)] + h[IDX3(i + 1, j + 1, k, n1, n2)]) +
                                 (h[IDX3(i + 1, j, k, n1, n2)] + h[IDX3(i, j + 1, k, n1, n2)]));
        str_xy[IDX3(i, j, k, n1, n2)] = -Kh_bg * sh_xy_l * hq_val *
                                          mask2dBu[IDX2(i, j, n1)] * reduction_xy[IDX2(i, j, n1)];
    }
}

// =========================================================================
// Kernel 2: divergence_kernel_c — stress divergence -> viscous acceleration (3D)
//
// diffu at u-points: i in [Isq_l : Ieq_l], j in [js_l : je_l]
// diffv at v-points: i in [is_l : ie_l],    j in [Jsq_l : Jeq_l]
//
// where Isq_l = is_l - 1, Ieq_l = ie_l, Jsq_l = js_l - 1, Jeq_l = je_l
// =========================================================================
__global__ void divergence_kernel_c(
    double* __restrict__ diffu,
    double* __restrict__ diffv,
    const double* __restrict__ str_xx,
    const double* __restrict__ str_xy,
    const double* __restrict__ h,
    const double* __restrict__ mask2dT,
    const double* __restrict__ IdyCu,
    const double* __restrict__ IdxCu,
    const double* __restrict__ IdyCv,
    const double* __restrict__ IdxCv,
    const double* __restrict__ IareaCu,
    const double* __restrict__ IareaCv,
    const double* __restrict__ dy2h,
    const double* __restrict__ dx2h,
    const double* __restrict__ dy2q,
    const double* __restrict__ dx2q,
    double h_neglect,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i > n1 || j > n2 || k < 1 || k > n3) return;

    int Isq_l = is_l - 1;
    int Ieq_l = ie_l;
    int Jsq_l = js_l - 1;
    int Jeq_l = je_l;

    // diffu at u-points: i in [Isq_l : Ieq_l], j in [js_l : je_l]
    if (i >= Isq_l && i <= Ieq_l &&
        j >= js_l && j <= je_l) {
        double h_u = 0.5 * (mask2dT[IDX2(i, j, n1)] * h[IDX3(i, j, k, n1, n2)] +
                             mask2dT[IDX2(i + 1, j, n1)] * h[IDX3(i + 1, j, k, n1, n2)]);
        diffu[IDX3(i, j, k, n1, n2)] = ((IdxCu[IDX2(i, j, n1)] * (dx2q[IDX2(i, j - 1, n1)] * str_xy[IDX3(i, j - 1, k, n1, n2)] -
                                                                      dx2q[IDX2(i, j, n1)] * str_xy[IDX3(i, j, k, n1, n2)]) +
                                           IdyCu[IDX2(i, j, n1)] * (dy2h[IDX2(i, j, n1)] * str_xx[IDX3(i, j, k, n1, n2)] -
                                                                      dy2h[IDX2(i + 1, j, n1)] * str_xx[IDX3(i + 1, j, k, n1, n2)])) *
                                          IareaCu[IDX2(i, j, n1)]) / (h_u + h_neglect);
    }

    // diffv at v-points: i in [is_l : ie_l], j in [Jsq_l : Jeq_l]
    if (i >= is_l && i <= ie_l &&
        j >= Jsq_l && j <= Jeq_l) {
        double h_v = 0.5 * (mask2dT[IDX2(i, j, n1)] * h[IDX3(i, j, k, n1, n2)] +
                             mask2dT[IDX2(i, j + 1, n1)] * h[IDX3(i, j + 1, k, n1, n2)]);
        diffv[IDX3(i, j, k, n1, n2)] = ((IdyCv[IDX2(i, j, n1)] * (dy2q[IDX2(i - 1, j, n1)] * str_xy[IDX3(i - 1, j, k, n1, n2)] -
                                                                      dy2q[IDX2(i, j, n1)] * str_xy[IDX3(i, j, k, n1, n2)]) -
                                           IdxCv[IDX2(i, j, n1)] * (dx2h[IDX2(i, j, n1)] * str_xx[IDX3(i, j, k, n1, n2)] -
                                                                      dx2h[IDX2(i, j + 1, n1)] * str_xx[IDX3(i, j + 1, k, n1, n2)])) *
                                          IareaCv[IDX2(i, j, n1)]) / (h_v + h_neglect);
    }
}

// =========================================================================
// Extern "C" launch wrappers (called from Fortran via iso_c_binding)
// =========================================================================

extern "C" void launch_stress_kernel(
    double* str_xx, double* str_xy,
    double* u, double* v, double* h,
    double* DY_dxT, double* DX_dyT, double* DY_dxBu, double* DX_dyBu,
    double* IdyCu, double* IdxCu, double* IdyCv, double* IdxCv,
    double* mask2dBu, double* reduction_xx, double* reduction_xy,
    double Kh_bg,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    stress_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        str_xx, str_xy, u, v, h,
        DY_dxT, DX_dyT, DY_dxBu, DX_dyBu,
        IdyCu, IdxCu, IdyCv, IdxCv,
        mask2dBu, reduction_xx, reduction_xy,
        Kh_bg,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_divergence_kernel(
    double* diffu, double* diffv,
    double* str_xx, double* str_xy, double* h,
    double* mask2dT,
    double* IdyCu, double* IdxCu, double* IdyCv, double* IdxCv,
    double* IareaCu, double* IareaCv,
    double* dy2h, double* dx2h, double* dy2q, double* dx2q,
    double h_neglect,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    divergence_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        diffu, diffv,
        str_xx, str_xy, h,
        mask2dT,
        IdyCu, IdxCu, IdyCv, IdxCv,
        IareaCu, IareaCv,
        dy2h, dx2h, dy2q, dx2q,
        h_neglect,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}
