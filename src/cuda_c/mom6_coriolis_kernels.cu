// MOM6 Coriolis and Momentum Advection — CUDA C kernels
//
// 1:1 port of every attributes(global) and attributes(device) from
// mom6_coriolis_cuda.F90.  Uses 1-based column-major indexing macros
// so that array accesses are bit-identical to the Fortran CUDA version.

#include "mom6_cuda_common.h"
#include <cuda_runtime.h>

// =========================================================================
// Device helper functions
// =========================================================================

/// Compute potential vorticity q at vorticity point (ii,jj,k).
static __device__ double compute_pv_inline_c(
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ h,
    const double* __restrict__ dyCv,
    const double* __restrict__ dxCu,
    const double* __restrict__ areaT,
    const double* __restrict__ IareaBu,
    const double* __restrict__ CoriolisBu,
    const double* __restrict__ mask2dBu,
    int n1, int n2, int n3,
    int ii, int jj, int k)
{
    double dvdx = v[IDX3(ii+1, jj, k, n1, n2)] * dyCv[IDX2(ii+1, jj, n1)]
                - v[IDX3(ii,   jj, k, n1, n2)] * dyCv[IDX2(ii,   jj, n1)];
    double dudy = u[IDX3(ii, jj+1, k, n1, n2)] * dxCu[IDX2(ii, jj+1, n1)]
                - u[IDX3(ii, jj,   k, n1, n2)] * dxCu[IDX2(ii, jj,   n1)];
    double rel_vort = mask2dBu[IDX2(ii, jj, n1)] * (dvdx - dudy) * IareaBu[IDX2(ii, jj, n1)];
    double abs_vort = CoriolisBu[IDX2(ii, jj, n1)] + rel_vort;

    double hau     = 0.5 * (areaT[IDX2(ii, jj, n1)]     * h[IDX3(ii, jj, k, n1, n2)]
                           + areaT[IDX2(ii+1, jj, n1)]   * h[IDX3(ii+1, jj, k, n1, n2)]);
    double hau_jp1 = 0.5 * (areaT[IDX2(ii, jj+1, n1)]   * h[IDX3(ii, jj+1, k, n1, n2)]
                           + areaT[IDX2(ii+1, jj+1, n1)] * h[IDX3(ii+1, jj+1, k, n1, n2)]);
    double hav     = 0.5 * (areaT[IDX2(ii, jj, n1)]      * h[IDX3(ii, jj, k, n1, n2)]
                           + areaT[IDX2(ii, jj+1, n1)]    * h[IDX3(ii, jj+1, k, n1, n2)]);
    double hav_ip1 = 0.5 * (areaT[IDX2(ii+1, jj, n1)]    * h[IDX3(ii+1, jj, k, n1, n2)]
                           + areaT[IDX2(ii+1, jj+1, n1)]  * h[IDX3(ii+1, jj+1, k, n1, n2)]);
    double haq = hau + hau_jp1 + hav + hav_ip1;
    double aq  = areaT[IDX2(ii, jj, n1)] + areaT[IDX2(ii+1, jj+1, n1)]
               + areaT[IDX2(ii+1, jj, n1)] + areaT[IDX2(ii, jj+1, n1)];

    return abs_vort * aq / (haq + 1.0e-20);
}

/// Compute kinetic energy at tracer point (ii,jj,k).
static __device__ double compute_ke_inline_c(
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ dyCu,
    const double* __restrict__ dxCv,
    const double* __restrict__ areaT,
    int n1, int n2, int n3,
    int ii, int jj, int k)
{
    return 0.25 * (
        (dyCu[IDX2(ii, jj, n1)] * u[IDX3(ii, jj, k, n1, n2)] * u[IDX3(ii, jj, k, n1, n2)]
       + dyCu[IDX2(ii-1, jj, n1)] * u[IDX3(ii-1, jj, k, n1, n2)] * u[IDX3(ii-1, jj, k, n1, n2)])
      + (dxCv[IDX2(ii, jj, n1)] * v[IDX3(ii, jj, k, n1, n2)] * v[IDX3(ii, jj, k, n1, n2)]
       + dxCv[IDX2(ii, jj-1, n1)] * v[IDX3(ii, jj-1, k, n1, n2)] * v[IDX3(ii, jj-1, k, n1, n2)])
    ) / areaT[IDX2(ii, jj, n1)];
}

// =========================================================================
// Kernel 1: phase1_kernel — circulation + area-weighted thickness + KE
// =========================================================================
__global__ void phase1_kernel_c(
    double* __restrict__ dvdx,
    double* __restrict__ dudy,
    double* __restrict__ hArea_u,
    double* __restrict__ hArea_v,
    double* __restrict__ KE,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ h,
    const double* __restrict__ dyCv,
    const double* __restrict__ dxCu,
    const double* __restrict__ dyCu,
    const double* __restrict__ dxCv,
    const double* __restrict__ areaT,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i > n1 || j > n2 || k < 1 || k > n3) return;

    // dvdx, dudy: i in [is_l-1 : ie_l], j in [js_l-1 : je_l]
    if (i >= is_l - 1 && i <= ie_l && j >= js_l - 1 && j <= je_l) {
        dvdx[IDX3(i, j, k, n1, n2)] = v[IDX3(i+1, j, k, n1, n2)] * dyCv[IDX2(i+1, j, n1)]
                                     - v[IDX3(i,   j, k, n1, n2)] * dyCv[IDX2(i,   j, n1)];
        dudy[IDX3(i, j, k, n1, n2)] = u[IDX3(i, j+1, k, n1, n2)] * dxCu[IDX2(i, j+1, n1)]
                                     - u[IDX3(i, j,   k, n1, n2)] * dxCu[IDX2(i, j,   n1)];
    }

    // hArea_v: i in [is_l : ie_l+1], j in [js_l-1 : je_l]
    if (i >= is_l && i <= ie_l + 1 && j >= js_l - 1 && j <= je_l) {
        hArea_v[IDX3(i, j, k, n1, n2)] = 0.5 * (areaT[IDX2(i, j, n1)]   * h[IDX3(i, j,   k, n1, n2)]
                                                + areaT[IDX2(i, j+1, n1)] * h[IDX3(i, j+1, k, n1, n2)]);
    }

    // hArea_u: i in [is_l-1 : ie_l], j in [js_l : je_l+1]
    if (i >= is_l - 1 && i <= ie_l && j >= js_l && j <= je_l + 1) {
        hArea_u[IDX3(i, j, k, n1, n2)] = 0.5 * (areaT[IDX2(i, j, n1)]   * h[IDX3(i,   j, k, n1, n2)]
                                                + areaT[IDX2(i+1, j, n1)] * h[IDX3(i+1, j, k, n1, n2)]);
    }

    // KE: i in [is_l : ie_l], j in [js_l : je_l]
    if (i >= is_l && i <= ie_l && j >= js_l && j <= je_l) {
        KE[IDX3(i, j, k, n1, n2)] = 0.25 * (
            (dyCu[IDX2(i, j, n1)]   * u[IDX3(i, j, k, n1, n2)]   * u[IDX3(i, j, k, n1, n2)]
           + dyCu[IDX2(i-1, j, n1)] * u[IDX3(i-1, j, k, n1, n2)] * u[IDX3(i-1, j, k, n1, n2)])
          + (dxCv[IDX2(i, j, n1)]   * v[IDX3(i, j, k, n1, n2)]   * v[IDX3(i, j, k, n1, n2)]
           + dxCv[IDX2(i, j-1, n1)] * v[IDX3(i, j-1, k, n1, n2)] * v[IDX3(i, j-1, k, n1, n2)])
        ) / areaT[IDX2(i, j, n1)];
    }
}

// =========================================================================
// Kernel 2: phase1_kernel_smem — shared-memory variant (32x4 block)
// =========================================================================
__global__ void phase1_kernel_smem_c(
    double* __restrict__ dvdx,
    double* __restrict__ dudy,
    double* __restrict__ hArea_u,
    double* __restrict__ hArea_v,
    double* __restrict__ KE,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ h,
    const double* __restrict__ dyCv,
    const double* __restrict__ dxCu,
    const double* __restrict__ dyCu,
    const double* __restrict__ dxCv,
    const double* __restrict__ areaT,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    const int BLK_X = 32, BLK_Y = 4;
    const int SX = BLK_X + 2, SY = BLK_Y + 2;  // 34 x 6

    __shared__ double s_u[SY * SX];
    __shared__ double s_v[SY * SX];
    __shared__ double s_h[SY * SX];

    int tx = (int)threadIdx.x + 1;  // 1-based within block
    int ty = (int)threadIdx.y + 1;
    int i = (int)blockIdx.x * BLK_X + tx;
    int j = (int)blockIdx.y * BLK_Y + ty;
    int k = (int)blockIdx.z + 1;

    // Cooperatively load tile + halo into shared memory
    // s_*(1,1) = global(i_base-1, j_base-1, k)
    // s_*(tx+1, ty+1) = global(i, j, k)
    int tid = ((int)threadIdx.y) * BLK_X + (int)threadIdx.x + 1;  // 1-based [1..128]

    for (int idx = tid; idx <= SX * SY; idx += BLK_X * BLK_Y) {
        int si = ((idx - 1) % SX) + 1;
        int sj = ((idx - 1) / SX) + 1;
        int gi = (int)blockIdx.x * BLK_X + si - 1;
        int gj = (int)blockIdx.y * BLK_Y + sj - 1;

        if (gi >= 1 && gi <= n1 && gj >= 1 && gj <= n2 && k >= 1 && k <= n3) {
            s_u[IDX2(si, sj, SX)] = u[IDX3(gi, gj, k, n1, n2)];
            s_v[IDX2(si, sj, SX)] = v[IDX3(gi, gj, k, n1, n2)];
            s_h[IDX2(si, sj, SX)] = h[IDX3(gi, gj, k, n1, n2)];
        }
    }

    __syncthreads();

    if (i > n1 || j > n2 || k < 1 || k > n3) return;

    // Shared indexing: s_*(tx+1, ty+1) = (i,j)
    //   (tx, ty+1) = (i-1,j),  (tx+2, ty+1) = (i+1,j)
    //   (tx+1, ty) = (i,j-1),  (tx+1, ty+2) = (i,j+1)

    // dvdx, dudy
    if (i >= is_l - 1 && i <= ie_l && j >= js_l - 1 && j <= je_l) {
        dvdx[IDX3(i, j, k, n1, n2)] = s_v[IDX2(tx+2, ty+1, SX)] * dyCv[IDX2(i+1, j, n1)]
                                     - s_v[IDX2(tx+1, ty+1, SX)] * dyCv[IDX2(i,   j, n1)];
        dudy[IDX3(i, j, k, n1, n2)] = s_u[IDX2(tx+1, ty+2, SX)] * dxCu[IDX2(i, j+1, n1)]
                                     - s_u[IDX2(tx+1, ty+1, SX)] * dxCu[IDX2(i, j,   n1)];
    }

    // hArea_v
    if (i >= is_l && i <= ie_l + 1 && j >= js_l - 1 && j <= je_l) {
        hArea_v[IDX3(i, j, k, n1, n2)] = 0.5 * (areaT[IDX2(i, j, n1)]   * s_h[IDX2(tx+1, ty+1, SX)]
                                                + areaT[IDX2(i, j+1, n1)] * s_h[IDX2(tx+1, ty+2, SX)]);
    }

    // hArea_u
    if (i >= is_l - 1 && i <= ie_l && j >= js_l && j <= je_l + 1) {
        hArea_u[IDX3(i, j, k, n1, n2)] = 0.5 * (areaT[IDX2(i, j, n1)]   * s_h[IDX2(tx+1, ty+1, SX)]
                                                + areaT[IDX2(i+1, j, n1)] * s_h[IDX2(tx+2, ty+1, SX)]);
    }

    // KE
    if (i >= is_l && i <= ie_l && j >= js_l && j <= je_l) {
        KE[IDX3(i, j, k, n1, n2)] = 0.25 * (
            (dyCu[IDX2(i, j, n1)]   * s_u[IDX2(tx+1, ty+1, SX)] * s_u[IDX2(tx+1, ty+1, SX)]
           + dyCu[IDX2(i-1, j, n1)] * s_u[IDX2(tx,   ty+1, SX)] * s_u[IDX2(tx,   ty+1, SX)])
          + (dxCv[IDX2(i, j, n1)]   * s_v[IDX2(tx+1, ty+1, SX)] * s_v[IDX2(tx+1, ty+1, SX)]
           + dxCv[IDX2(i, j-1, n1)] * s_v[IDX2(tx+1, ty,   SX)] * s_v[IDX2(tx+1, ty,   SX)])
        ) / areaT[IDX2(i, j, n1)];
    }
}

// =========================================================================
// Kernel 3: phase1a_circ_kernel — circulation + area-weighted thickness only
// =========================================================================
__global__ void phase1a_circ_kernel_c(
    double* __restrict__ dvdx,
    double* __restrict__ dudy,
    double* __restrict__ hArea_u,
    double* __restrict__ hArea_v,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ h,
    const double* __restrict__ dyCv,
    const double* __restrict__ dxCu,
    const double* __restrict__ areaT,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i > n1 || j > n2 || k < 1 || k > n3) return;

    if (i >= is_l - 1 && i <= ie_l && j >= js_l - 1 && j <= je_l) {
        dvdx[IDX3(i, j, k, n1, n2)] = v[IDX3(i+1, j, k, n1, n2)] * dyCv[IDX2(i+1, j, n1)]
                                     - v[IDX3(i,   j, k, n1, n2)] * dyCv[IDX2(i,   j, n1)];
        dudy[IDX3(i, j, k, n1, n2)] = u[IDX3(i, j+1, k, n1, n2)] * dxCu[IDX2(i, j+1, n1)]
                                     - u[IDX3(i, j,   k, n1, n2)] * dxCu[IDX2(i, j,   n1)];
    }

    if (i >= is_l && i <= ie_l + 1 && j >= js_l - 1 && j <= je_l) {
        hArea_v[IDX3(i, j, k, n1, n2)] = 0.5 * (areaT[IDX2(i, j, n1)]   * h[IDX3(i, j,   k, n1, n2)]
                                                + areaT[IDX2(i, j+1, n1)] * h[IDX3(i, j+1, k, n1, n2)]);
    }

    if (i >= is_l - 1 && i <= ie_l && j >= js_l && j <= je_l + 1) {
        hArea_u[IDX3(i, j, k, n1, n2)] = 0.5 * (areaT[IDX2(i, j, n1)]   * h[IDX3(i,   j, k, n1, n2)]
                                                + areaT[IDX2(i+1, j, n1)] * h[IDX3(i+1, j, k, n1, n2)]);
    }
}

// =========================================================================
// Kernel 4: phase1b_ke_kernel — KE only
// =========================================================================
__global__ void phase1b_ke_kernel_c(
    double* __restrict__ KE,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ dyCu,
    const double* __restrict__ dxCv,
    const double* __restrict__ areaT,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i > n1 || j > n2 || k < 1 || k > n3) return;

    if (i >= is_l && i <= ie_l && j >= js_l && j <= je_l) {
        KE[IDX3(i, j, k, n1, n2)] = 0.25 * (
            (dyCu[IDX2(i, j, n1)]   * u[IDX3(i, j, k, n1, n2)]   * u[IDX3(i, j, k, n1, n2)]
           + dyCu[IDX2(i-1, j, n1)] * u[IDX3(i-1, j, k, n1, n2)] * u[IDX3(i-1, j, k, n1, n2)])
          + (dxCv[IDX2(i, j, n1)]   * v[IDX3(i, j, k, n1, n2)]   * v[IDX3(i, j, k, n1, n2)]
           + dxCv[IDX2(i, j-1, n1)] * v[IDX3(i, j-1, k, n1, n2)] * v[IDX3(i, j-1, k, n1, n2)])
        ) / areaT[IDX2(i, j, n1)];
    }
}

// =========================================================================
// Kernel 5: vorticity_pv_kernel — vorticity and potential vorticity
// =========================================================================
__global__ void vorticity_pv_kernel_c(
    double* __restrict__ rel_vort,
    double* __restrict__ abs_vort,
    double* __restrict__ q,
    double* __restrict__ Ih_q,
    const double* __restrict__ dvdx,
    const double* __restrict__ dudy,
    const double* __restrict__ hArea_u,
    const double* __restrict__ hArea_v,
    const double* __restrict__ Area_q,
    const double* __restrict__ IareaBu,
    const double* __restrict__ CoriolisBu,
    const double* __restrict__ mask2dBu,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i < is_l - 1 || i > ie_l || j < js_l - 1 || j > je_l
        || k < 1 || k > n3) return;

    rel_vort[IDX3(i, j, k, n1, n2)] = mask2dBu[IDX2(i, j, n1)]
        * (dvdx[IDX3(i, j, k, n1, n2)] - dudy[IDX3(i, j, k, n1, n2)])
        * IareaBu[IDX2(i, j, n1)];
    abs_vort[IDX3(i, j, k, n1, n2)] = CoriolisBu[IDX2(i, j, n1)]
        + rel_vort[IDX3(i, j, k, n1, n2)];

    double hArea_q_val = (hArea_u[IDX3(i, j,   k, n1, n2)] + hArea_u[IDX3(i, j+1, k, n1, n2)])
                       + (hArea_v[IDX3(i, j,   k, n1, n2)] + hArea_v[IDX3(i+1, j, k, n1, n2)]);
    Ih_q[IDX3(i, j, k, n1, n2)] = Area_q[IDX3(i, j, k, n1, n2)] / (hArea_q_val + 1.0e-20);
    q[IDX3(i, j, k, n1, n2)] = abs_vort[IDX3(i, j, k, n1, n2)] * Ih_q[IDX3(i, j, k, n1, n2)];
}

// =========================================================================
// Kernel 6: arakawa_hsu90_coef_kernel — Arakawa HSU90 coefficients
// =========================================================================
__global__ void arakawa_hsu90_coef_kernel_c(
    double* __restrict__ a,
    double* __restrict__ b,
    double* __restrict__ c,
    double* __restrict__ d,
    const double* __restrict__ q,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (j < js_l || j > je_l || k < 1 || k > n3) return;

    // a, d: i in [is_l-1 : ie_l]
    if (i >= is_l - 1 && i <= ie_l) {
        a[IDX3(i, j, k, n1, n2)] = (q[IDX3(i, j, k, n1, n2)]
            + (q[IDX3(i+1, j, k, n1, n2)] + q[IDX3(i, j-1, k, n1, n2)])) * C1_12;
        d[IDX3(i, j, k, n1, n2)] = ((q[IDX3(i, j, k, n1, n2)]
            + q[IDX3(i+1, j-1, k, n1, n2)]) + q[IDX3(i, j-1, k, n1, n2)]) * C1_12;
    }

    // b, c: i in [is_l : ie_l]
    if (i >= is_l && i <= ie_l) {
        b[IDX3(i, j, k, n1, n2)] = (q[IDX3(i, j, k, n1, n2)]
            + (q[IDX3(i-1, j, k, n1, n2)] + q[IDX3(i, j-1, k, n1, n2)])) * C1_12;
        c[IDX3(i, j, k, n1, n2)] = ((q[IDX3(i, j, k, n1, n2)]
            + q[IDX3(i-1, j-1, k, n1, n2)]) + q[IDX3(i, j-1, k, n1, n2)]) * C1_12;
    }
}

// =========================================================================
// Kernel 7: arakawa_lamb81_coef_kernel — Arakawa LAMB81 coefficients
// =========================================================================
__global__ void arakawa_lamb81_coef_kernel_c(
    double* __restrict__ a,
    double* __restrict__ b,
    double* __restrict__ c,
    double* __restrict__ d,
    const double* __restrict__ q,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i < is_l || i > ie_l || j < js_l || j > je_l || k < 1 || k > n3) return;

    a[IDX3(i-1, j, k, n1, n2)] = (2.0 * (q[IDX3(i, j, k, n1, n2)] + q[IDX3(i-1, j-1, k, n1, n2)])
        + (q[IDX3(i-1, j, k, n1, n2)] + q[IDX3(i, j-1, k, n1, n2)])) * C1_24;
    d[IDX3(i-1, j, k, n1, n2)] = ((q[IDX3(i, j, k, n1, n2)] + q[IDX3(i-1, j-1, k, n1, n2)])
        + 2.0 * (q[IDX3(i-1, j, k, n1, n2)] + q[IDX3(i, j-1, k, n1, n2)])) * C1_24;
    b[IDX3(i, j, k, n1, n2)] = ((q[IDX3(i, j, k, n1, n2)] + q[IDX3(i-1, j-1, k, n1, n2)])
        + 2.0 * (q[IDX3(i-1, j, k, n1, n2)] + q[IDX3(i, j-1, k, n1, n2)])) * C1_24;
    c[IDX3(i, j, k, n1, n2)] = (2.0 * (q[IDX3(i, j, k, n1, n2)] + q[IDX3(i-1, j-1, k, n1, n2)])
        + (q[IDX3(i-1, j, k, n1, n2)] + q[IDX3(i, j-1, k, n1, n2)])) * C1_24;
}

// =========================================================================
// Kernel 8: coriolis_sadourny_kernel — Sadourny energy-conserving scheme
// =========================================================================
__global__ void coriolis_sadourny_kernel_c(
    double* __restrict__ CAu,
    double* __restrict__ CAv,
    const double* __restrict__ q,
    const double* __restrict__ KE,
    const double* __restrict__ uh,
    const double* __restrict__ vh,
    const double* __restrict__ IdxCu,
    const double* __restrict__ IdyCv,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (k < 1 || k > n3) return;

    // CAu: i in [is_l : ie_l-1], j in [js_l : je_l]
    if (i >= is_l && i <= ie_l - 1 && j >= js_l && j <= je_l) {
        double KEx = (KE[IDX3(i+1, j, k, n1, n2)] - KE[IDX3(i, j, k, n1, n2)]) * IdxCu[IDX2(i, j, n1)];
        CAu[IDX3(i, j, k, n1, n2)] = 0.25 * (
            (q[IDX3(i, j, k, n1, n2)]   * (vh[IDX3(i+1, j, k, n1, n2)] + vh[IDX3(i, j, k, n1, n2)]))
          + (q[IDX3(i, j-1, k, n1, n2)] * (vh[IDX3(i, j-1, k, n1, n2)] + vh[IDX3(i+1, j-1, k, n1, n2)]))
        ) * IdxCu[IDX2(i, j, n1)] - KEx;
    }

    // CAv: i in [is_l : ie_l], j in [js_l : je_l-1]
    if (i >= is_l && i <= ie_l && j >= js_l && j <= je_l - 1) {
        double KEy = (KE[IDX3(i, j+1, k, n1, n2)] - KE[IDX3(i, j, k, n1, n2)]) * IdyCv[IDX2(i, j, n1)];
        CAv[IDX3(i, j, k, n1, n2)] = -0.25 * (
            (q[IDX3(i-1, j, k, n1, n2)] * (uh[IDX3(i-1, j, k, n1, n2)] + uh[IDX3(i-1, j+1, k, n1, n2)]))
          + (q[IDX3(i,   j, k, n1, n2)] * (uh[IDX3(i,   j, k, n1, n2)] + uh[IDX3(i,   j+1, k, n1, n2)]))
        ) * IdyCv[IDX2(i, j, n1)] - KEy;
    }
}

// =========================================================================
// Kernel 9: coriolis_arakawa_kernel — Arakawa schemes (HSU90 or LAMB81)
// =========================================================================
__global__ void coriolis_arakawa_kernel_c(
    double* __restrict__ CAu,
    double* __restrict__ CAv,
    const double* __restrict__ a,
    const double* __restrict__ b,
    const double* __restrict__ c,
    const double* __restrict__ d,
    const double* __restrict__ KE,
    const double* __restrict__ uh,
    const double* __restrict__ vh,
    const double* __restrict__ IdxCu,
    const double* __restrict__ IdyCv,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (k < 1 || k > n3) return;

    // CAu: i in [is_l : ie_l-1], j in [js_l : je_l]
    if (i >= is_l && i <= ie_l - 1 && j >= js_l && j <= je_l) {
        double KEx = (KE[IDX3(i+1, j, k, n1, n2)] - KE[IDX3(i, j, k, n1, n2)]) * IdxCu[IDX2(i, j, n1)];
        CAu[IDX3(i, j, k, n1, n2)] = (
            ((a[IDX3(i, j, k, n1, n2)] * vh[IDX3(i+1, j,   k, n1, n2)])
           + (c[IDX3(i, j, k, n1, n2)] * vh[IDX3(i,   j-1, k, n1, n2)]))
          + ((b[IDX3(i, j, k, n1, n2)] * vh[IDX3(i,   j,   k, n1, n2)])
           + (d[IDX3(i, j, k, n1, n2)] * vh[IDX3(i+1, j-1, k, n1, n2)]))
        ) * IdxCu[IDX2(i, j, n1)] - KEx;
    }

    // CAv: i in [is_l : ie_l], j in [js_l : je_l-1]
    if (i >= is_l && i <= ie_l && j >= js_l && j <= je_l - 1) {
        double KEy = (KE[IDX3(i, j+1, k, n1, n2)] - KE[IDX3(i, j, k, n1, n2)]) * IdyCv[IDX2(i, j, n1)];
        CAv[IDX3(i, j, k, n1, n2)] = -(
            ((a[IDX3(i-1, j,   k, n1, n2)] * uh[IDX3(i-1, j,   k, n1, n2)])
           + (c[IDX3(i,   j+1, k, n1, n2)] * uh[IDX3(i,   j+1, k, n1, n2)]))
          + ((b[IDX3(i,   j,   k, n1, n2)] * uh[IDX3(i,   j,   k, n1, n2)])
           + (d[IDX3(i-1, j+1, k, n1, n2)] * uh[IDX3(i-1, j+1, k, n1, n2)]))
        ) * IdyCv[IDX2(i, j, n1)] - KEy;
    }
}

// =========================================================================
// Kernel 10: coriolis_fused_sadourny_kernel — Fused single-kernel Sadourny
// =========================================================================
__global__ void coriolis_fused_sadourny_kernel_c(
    double* __restrict__ CAu,
    double* __restrict__ CAv,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ h,
    const double* __restrict__ uh,
    const double* __restrict__ vh,
    const double* __restrict__ dyCv,
    const double* __restrict__ dxCu,
    const double* __restrict__ dyCu,
    const double* __restrict__ dxCv,
    const double* __restrict__ areaT,
    const double* __restrict__ IareaBu,
    const double* __restrict__ CoriolisBu,
    const double* __restrict__ mask2dBu,
    const double* __restrict__ IdxCu,
    const double* __restrict__ IdyCv,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (k < 1 || k > n3) return;

    int do_cau = (i >= is_l && i <= ie_l - 1 && j >= js_l && j <= je_l);
    int do_cav = (i >= is_l && i <= ie_l && j >= js_l && j <= je_l - 1);

    if (!do_cau && !do_cav) return;

    // q(i,j,k) — needed by both CAu and CAv
    double q_ij = compute_pv_inline_c(u, v, h, dyCv, dxCu, areaT,
        IareaBu, CoriolisBu, mask2dBu, n1, n2, n3, i, j, k);

    // KE(i,j,k) — needed by both CAu and CAv
    double KE_ij = compute_ke_inline_c(u, v, dyCu, dxCv, areaT, n1, n2, n3, i, j, k);

    // --- CAu(i,j,k) ---
    if (do_cau) {
        double q_jm1 = compute_pv_inline_c(u, v, h, dyCv, dxCu, areaT,
            IareaBu, CoriolisBu, mask2dBu, n1, n2, n3, i, j-1, k);
        double KE_ip1 = compute_ke_inline_c(u, v, dyCu, dxCv, areaT, n1, n2, n3, i+1, j, k);
        double KEx = (KE_ip1 - KE_ij) * IdxCu[IDX2(i, j, n1)];
        CAu[IDX3(i, j, k, n1, n2)] = 0.25 * (
            (q_ij  * (vh[IDX3(i+1, j, k, n1, n2)] + vh[IDX3(i, j, k, n1, n2)]))
          + (q_jm1 * (vh[IDX3(i, j-1, k, n1, n2)] + vh[IDX3(i+1, j-1, k, n1, n2)]))
        ) * IdxCu[IDX2(i, j, n1)] - KEx;
    }

    // --- CAv(i,j,k) ---
    if (do_cav) {
        double q_im1 = compute_pv_inline_c(u, v, h, dyCv, dxCu, areaT,
            IareaBu, CoriolisBu, mask2dBu, n1, n2, n3, i-1, j, k);
        double KE_jp1 = compute_ke_inline_c(u, v, dyCu, dxCv, areaT, n1, n2, n3, i, j+1, k);
        double KEy = (KE_jp1 - KE_ij) * IdyCv[IDX2(i, j, n1)];
        CAv[IDX3(i, j, k, n1, n2)] = -0.25 * (
            (q_im1 * (uh[IDX3(i-1, j, k, n1, n2)] + uh[IDX3(i-1, j+1, k, n1, n2)]))
          + (q_ij  * (uh[IDX3(i,   j, k, n1, n2)] + uh[IDX3(i,   j+1, k, n1, n2)]))
        ) * IdyCv[IDX2(i, j, n1)] - KEy;
    }
}


// =========================================================================
// Extern "C" launch wrappers (called from Fortran via iso_c_binding)
// =========================================================================

extern "C" void launch_phase1_kernel(
    double* dvdx, double* dudy, double* hArea_u, double* hArea_v, double* KE,
    double* u, double* v, double* h,
    double* dyCv, double* dxCu, double* dyCu, double* dxCv, double* areaT,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    phase1_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        dvdx, dudy, hArea_u, hArea_v, KE,
        u, v, h, dyCv, dxCu, dyCu, dxCv, areaT,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_phase1_kernel_smem(
    double* dvdx, double* dudy, double* hArea_u, double* hArea_v, double* KE,
    double* u, double* v, double* h,
    double* dyCv, double* dxCu, double* dyCu, double* dxCv, double* areaT,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    phase1_kernel_smem_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        dvdx, dudy, hArea_u, hArea_v, KE,
        u, v, h, dyCv, dxCu, dyCu, dxCv, areaT,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_phase1a_circ_kernel(
    double* dvdx, double* dudy, double* hArea_u, double* hArea_v,
    double* u, double* v, double* h,
    double* dyCv, double* dxCu, double* areaT,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    phase1a_circ_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        dvdx, dudy, hArea_u, hArea_v,
        u, v, h, dyCv, dxCu, areaT,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_phase1b_ke_kernel(
    double* KE, double* u, double* v,
    double* dyCu, double* dxCv, double* areaT,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    phase1b_ke_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        KE, u, v, dyCu, dxCv, areaT,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_vorticity_pv_kernel(
    double* rel_vort, double* abs_vort, double* q, double* Ih_q,
    double* dvdx, double* dudy, double* hArea_u, double* hArea_v, double* Area_q,
    double* IareaBu, double* CoriolisBu, double* mask2dBu,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    vorticity_pv_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        rel_vort, abs_vort, q, Ih_q,
        dvdx, dudy, hArea_u, hArea_v, Area_q,
        IareaBu, CoriolisBu, mask2dBu,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_arakawa_hsu90_coef_kernel(
    double* a, double* b, double* c, double* d, double* q,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    arakawa_hsu90_coef_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        a, b, c, d, q,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_arakawa_lamb81_coef_kernel(
    double* a, double* b, double* c, double* d, double* q,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    arakawa_lamb81_coef_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        a, b, c, d, q,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_coriolis_sadourny_kernel(
    double* CAu, double* CAv,
    double* q, double* KE, double* uh, double* vh,
    double* IdxCu, double* IdyCv,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    coriolis_sadourny_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        CAu, CAv, q, KE, uh, vh, IdxCu, IdyCv,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_coriolis_arakawa_kernel(
    double* CAu, double* CAv,
    double* a, double* b, double* c, double* d,
    double* KE, double* uh, double* vh,
    double* IdxCu, double* IdyCv,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    coriolis_arakawa_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        CAu, CAv, a, b, c, d, KE, uh, vh, IdxCu, IdyCv,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_coriolis_fused_sadourny_kernel(
    double* CAu, double* CAv,
    double* u, double* v, double* h, double* uh, double* vh,
    double* dyCv, double* dxCu, double* dyCu, double* dxCv, double* areaT,
    double* IareaBu, double* CoriolisBu, double* mask2dBu,
    double* IdxCu, double* IdyCv,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    coriolis_fused_sadourny_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        CAu, CAv, u, v, h, uh, vh,
        dyCv, dxCu, dyCu, dxCv, areaT,
        IareaBu, CoriolisBu, mask2dBu, IdxCu, IdyCv,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

// CUDA runtime helpers moved to cuda_helpers.cu
