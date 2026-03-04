// MOM6 Continuity PPM Solver — CUDA C kernels
//
// 1:1 port of every attributes(global) kernel from mom6_continuity_cuda.F90.
// Uses 1-based column-major indexing macros so that array accesses are
// bit-identical to the Fortran CUDA version.
//
// Fortran->C translations:
//   sign(a,b) -> copysign(a,b)
//   max()     -> fmax()
//   min()     -> fmin()
//   abs()     -> fabs()

#include "mom6_cuda_common.h"
#include <cuda_runtime.h>
#include <math.h>

static const double oneSixth = 1.0 / 6.0;

// =========================================================================
// Kernel 1: ppm_reconstruction_3d_kernel_c
//   PPM edge values h_W, h_E with limiter.
//   3D kernel: i in [is_l-1 : ie_l+1], j in [js_l : je_l], k in [1:n3]
// =========================================================================
__global__ void ppm_reconstruction_3d_kernel_c(
    double* __restrict__ h_W,
    double* __restrict__ h_E,
    const double* __restrict__ h_in,
    const double* __restrict__ mask2dT,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    double h_min,
    int monotonic,
    int upwind_1st_flag,
    int simple_2nd_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i < is_l - 1 || i > ie_l + 1 ||
        j < js_l || j > je_l ||
        k < 1 || k > n3) return;

    // Upwind first-order: flat reconstruction
    if (upwind_1st_flag == 1) {
        h_W[IDX3(i, j, k, n1, n2)] = h_in[IDX3(i, j, k, n1, n2)];
        h_E[IDX3(i, j, k, n1, n2)] = h_in[IDX3(i, j, k, n1, n2)];
        return;
    }

    double h_im1_val, h_ip1_val;
    double slp_im1, slp_i, slp_ip1;
    double dMx, dMn;

    int im1 = (i - 1 >= 1) ? (i - 1) : 1;
    int ip1 = (i + 1 <= n1) ? (i + 1) : n1;

    if (simple_2nd_flag == 1) {
        // Simple second-order: arithmetic mean edges
        h_im1_val = mask2dT[IDX2(im1, j, n1)] * h_in[IDX3(im1, j, k, n1, n2)]
                  + (1.0 - mask2dT[IDX2(im1, j, n1)]) * h_in[IDX3(i, j, k, n1, n2)];
        h_ip1_val = mask2dT[IDX2(ip1, j, n1)] * h_in[IDX3(ip1, j, k, n1, n2)]
                  + (1.0 - mask2dT[IDX2(ip1, j, n1)]) * h_in[IDX3(i, j, k, n1, n2)];
        h_W[IDX3(i, j, k, n1, n2)] = 0.5 * (h_im1_val + h_in[IDX3(i, j, k, n1, n2)]);
        h_E[IDX3(i, j, k, n1, n2)] = 0.5 * (h_ip1_val + h_in[IDX3(i, j, k, n1, n2)]);
    } else {
        // Full PPM with slopes

        // Slope at i-1
        if (i - 2 >= 1 && i >= 1 &&
            (mask2dT[IDX2(i-2, j, n1)] * mask2dT[IDX2(i-1, j, n1)] * mask2dT[IDX2(i, j, n1)]) != 0.0) {
            slp_im1 = 0.5 * (h_in[IDX3(i, j, k, n1, n2)] - h_in[IDX3(i-2, j, k, n1, n2)]);
            dMx = fmax(fmax(h_in[IDX3(i, j, k, n1, n2)], h_in[IDX3(i-2, j, k, n1, n2)]),
                       h_in[IDX3(i-1, j, k, n1, n2)]) - h_in[IDX3(i-1, j, k, n1, n2)];
            dMn = h_in[IDX3(i-1, j, k, n1, n2)] -
                  fmin(fmin(h_in[IDX3(i, j, k, n1, n2)], h_in[IDX3(i-2, j, k, n1, n2)]),
                       h_in[IDX3(i-1, j, k, n1, n2)]);
            slp_im1 = copysign(1.0, slp_im1) * fmin(fabs(slp_im1), 2.0 * fmin(dMx, dMn));
        } else {
            slp_im1 = 0.0;
        }

        // Slope at i
        if (i - 1 >= 1 && i + 1 <= n1 &&
            (mask2dT[IDX2(i-1, j, n1)] * mask2dT[IDX2(i, j, n1)] * mask2dT[IDX2(i+1, j, n1)]) != 0.0) {
            slp_i = 0.5 * (h_in[IDX3(i+1, j, k, n1, n2)] - h_in[IDX3(i-1, j, k, n1, n2)]);
            dMx = fmax(fmax(h_in[IDX3(i+1, j, k, n1, n2)], h_in[IDX3(i-1, j, k, n1, n2)]),
                       h_in[IDX3(i, j, k, n1, n2)]) - h_in[IDX3(i, j, k, n1, n2)];
            dMn = h_in[IDX3(i, j, k, n1, n2)] -
                  fmin(fmin(h_in[IDX3(i+1, j, k, n1, n2)], h_in[IDX3(i-1, j, k, n1, n2)]),
                       h_in[IDX3(i, j, k, n1, n2)]);
            slp_i = copysign(1.0, slp_i) * fmin(fabs(slp_i), 2.0 * fmin(dMx, dMn));
        } else {
            slp_i = 0.0;
        }

        // Slope at i+1
        if (i + 2 <= n1 && i >= 1 &&
            (mask2dT[IDX2(i, j, n1)] * mask2dT[IDX2(i+1, j, n1)] * mask2dT[IDX2(i+2, j, n1)]) != 0.0) {
            slp_ip1 = 0.5 * (h_in[IDX3(i+2, j, k, n1, n2)] - h_in[IDX3(i, j, k, n1, n2)]);
            dMx = fmax(fmax(h_in[IDX3(i+2, j, k, n1, n2)], h_in[IDX3(i, j, k, n1, n2)]),
                       h_in[IDX3(i+1, j, k, n1, n2)]) - h_in[IDX3(i+1, j, k, n1, n2)];
            dMn = h_in[IDX3(i+1, j, k, n1, n2)] -
                  fmin(fmin(h_in[IDX3(i+2, j, k, n1, n2)], h_in[IDX3(i, j, k, n1, n2)]),
                       h_in[IDX3(i+1, j, k, n1, n2)]);
            slp_ip1 = copysign(1.0, slp_ip1) * fmin(fabs(slp_ip1), 2.0 * fmin(dMx, dMn));
        } else {
            slp_ip1 = 0.0;
        }

        // Edge values from slopes
        h_im1_val = mask2dT[IDX2(im1, j, n1)] * h_in[IDX3(im1, j, k, n1, n2)]
                  + (1.0 - mask2dT[IDX2(im1, j, n1)]) * h_in[IDX3(i, j, k, n1, n2)];
        h_ip1_val = mask2dT[IDX2(ip1, j, n1)] * h_in[IDX3(ip1, j, k, n1, n2)]
                  + (1.0 - mask2dT[IDX2(ip1, j, n1)]) * h_in[IDX3(i, j, k, n1, n2)];

        h_W[IDX3(i, j, k, n1, n2)] = 0.5 * (h_im1_val + h_in[IDX3(i, j, k, n1, n2)])
                                    + oneSixth * (slp_im1 - slp_i);
        h_E[IDX3(i, j, k, n1, n2)] = 0.5 * (h_ip1_val + h_in[IDX3(i, j, k, n1, n2)])
                                    + oneSixth * (slp_i - slp_ip1);
    }

    // Limiter (applies to both simple_2nd and full PPM)
    double h_i = h_in[IDX3(i, j, k, n1, n2)];
    double hW = h_W[IDX3(i, j, k, n1, n2)];
    double hE = h_E[IDX3(i, j, k, n1, n2)];

    if (monotonic == 1) {
        if ((hE - h_i) * (h_i - hW) <= 0.0) {
            hW = h_i;
            hE = h_i;
        } else {
            double RLdiff = hE - hW;
            double RLmean = 0.5 * (hE + hW);
            double FunFac = 6.0 * RLdiff * (h_i - RLmean);
            double RLdiff2 = RLdiff * RLdiff;
            if (FunFac > RLdiff2) hW = 3.0 * h_i - 2.0 * hE;
            if (FunFac < -RLdiff2) hE = 3.0 * h_i - 2.0 * hW;
        }
    } else {
        double curv = 3.0 * ((hW + hE) - 2.0 * h_i);
        if (curv > 0.0) {
            double dh = hE - hW;
            if (fabs(dh) < curv) {
                if (h_i <= h_min) {
                    hW = h_i;
                    hE = h_i;
                } else if (12.0 * curv * (h_i - h_min) < (curv * curv + 3.0 * dh * dh)) {
                    double scale_val = 12.0 * curv * (h_i - h_min) / (curv * curv + 3.0 * dh * dh);
                    hW = h_i + scale_val * (hW - h_i);
                    hE = h_i + scale_val * (hE - h_i);
                }
            }
        }
    }

    h_W[IDX3(i, j, k, n1, n2)] = hW;
    h_E[IDX3(i, j, k, n1, n2)] = hE;
}

// =========================================================================
// Kernel 2: zonal_flux_layer_3d_kernel_c
//   Flux uh + duhdu with CFL check.
//   3D kernel: i in [is_l-1 : ie_l], j in [js_l : je_l], k in [1:n3]
// =========================================================================
__global__ void zonal_flux_layer_3d_kernel_c(
    double* __restrict__ uh,
    double* __restrict__ duhdu,
    const double* __restrict__ u,
    const double* __restrict__ h_in,
    const double* __restrict__ h_W,
    const double* __restrict__ h_E,
    const double* __restrict__ dy_Cu,
    const double* __restrict__ IdxT,
    const double* __restrict__ IareaT,
    const double* __restrict__ por_face_areaU,
    const double* __restrict__ visc_rem_u,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    double dt,
    int vol_CFL_flag,
    int use_visc_rem_flag,
    int use_por_face_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l ||
        k < 1 || k > n3) return;

    double visc_rem_val;
    if (use_visc_rem_flag == 1) {
        visc_rem_val = visc_rem_u[IDX3(i, j, k, n1, n2)];
    } else {
        visc_rem_val = 1.0;
    }

    double dy_pfa;
    if (use_por_face_flag == 1) {
        dy_pfa = dy_Cu[IDX2(i, j, n1)] * por_face_areaU[IDX3(i, j, k, n1, n2)];
    } else {
        dy_pfa = dy_Cu[IDX2(i, j, n1)];
    }

    double u_ijk = u[IDX3(i, j, k, n1, n2)];
    double CFL, curv_3, h_marg;

    if (u_ijk > 0.0) {
        if (vol_CFL_flag == 1) {
            CFL = (u_ijk * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
        } else {
            CFL = u_ijk * dt * IdxT[IDX2(i, j, n1)];
        }
        curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
               - 2.0 * h_in[IDX3(i, j, k, n1, n2)];
        uh[IDX3(i, j, k, n1, n2)] = dy_pfa * u_ijk *
            (h_E[IDX3(i, j, k, n1, n2)] + CFL * (0.5 * (h_W[IDX3(i, j, k, n1, n2)]
            - h_E[IDX3(i, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
        h_marg = h_E[IDX3(i, j, k, n1, n2)] + CFL * ((h_W[IDX3(i, j, k, n1, n2)]
                - h_E[IDX3(i, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
    } else if (u_ijk < 0.0) {
        if (vol_CFL_flag == 1) {
            CFL = (-u_ijk * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
        } else {
            CFL = -u_ijk * dt * IdxT[IDX2(i+1, j, n1)];
        }
        curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
               - 2.0 * h_in[IDX3(i+1, j, k, n1, n2)];
        uh[IDX3(i, j, k, n1, n2)] = dy_pfa * u_ijk *
            (h_W[IDX3(i+1, j, k, n1, n2)] + CFL * (0.5 * (h_E[IDX3(i+1, j, k, n1, n2)]
            - h_W[IDX3(i+1, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
        h_marg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * ((h_E[IDX3(i+1, j, k, n1, n2)]
                - h_W[IDX3(i+1, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
    } else {
        uh[IDX3(i, j, k, n1, n2)] = 0.0;
        h_marg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
    }

    duhdu[IDX3(i, j, k, n1, n2)] = dy_pfa * h_marg * visc_rem_val;
}

// =========================================================================
// Kernel 3: zonal_convergence_kernel_c
//   3D h update from divergence.
//   Range: i in [is_l : ie_l], j in [js_l : je_l], k in [1:n3]
// =========================================================================
__global__ void zonal_convergence_kernel_c(
    double* __restrict__ h,
    const double* __restrict__ hin,
    const double* __restrict__ uh,
    const double* __restrict__ IareaT,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    double dt)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i < is_l || i > ie_l ||
        j < js_l || j > je_l ||
        k < 1 || k > n3) return;

    h[IDX3(i, j, k, n1, n2)] = fmax(hin[IDX3(i, j, k, n1, n2)]
        - dt * IareaT[IDX2(i, j, n1)]
        * (uh[IDX3(i, j, k, n1, n2)] - uh[IDX3(i-1, j, k, n1, n2)]), 0.0);
}

// =========================================================================
// Kernel 4: visc_rem_max_kernel_c
//   2D column max of visc_rem_u.
//   Range: i in [is_l-1 : ie_l], j in [js_l : je_l]
// =========================================================================
__global__ void visc_rem_max_kernel_c(
    double* __restrict__ visc_rem_max_out,
    const double* __restrict__ visc_rem_u,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    int use_vrm_max_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l) return;

    if (use_vrm_max_flag == 1) {
        double vmax = 0.0;
        for (int k = 1; k <= n3; k++) {
            vmax = fmax(vmax, visc_rem_u[IDX3(i, j, k, n1, n2)]);
        }
        visc_rem_max_out[IDX2(i, j, n1)] = vmax;
    } else {
        visc_rem_max_out[IDX2(i, j, n1)] = 1.0;
    }
}

// =========================================================================
// Kernel 5: uh_duhdu_tot_kernel_c
//   2D column sums of uh and duhdu.
//   Range: i in [is_l-1 : ie_l], j in [js_l : je_l]
// =========================================================================
__global__ void uh_duhdu_tot_kernel_c(
    double* __restrict__ uh_tot_0,
    double* __restrict__ duhdu_tot_0,
    const double* __restrict__ uh,
    const double* __restrict__ duhdu,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l) return;

    double uh_sum = 0.0;
    double duhdu_sum = 0.0;
    for (int k = 1; k <= n3; k++) {
        uh_sum += uh[IDX3(i, j, k, n1, n2)];
        duhdu_sum += duhdu[IDX3(i, j, k, n1, n2)];
    }
    uh_tot_0[IDX2(i, j, n1)] = uh_sum;
    duhdu_tot_0[IDX2(i, j, n1)] = duhdu_sum;
}

// =========================================================================
// Kernel 6: zonal_CFL_limits_kernel_c
//   2D CFL bounds on du correction.
//   Range: i in [is_l-1 : ie_l], j in [js_l : je_l]
// =========================================================================
__global__ void zonal_CFL_limits_kernel_c(
    double* __restrict__ du_max_CFL_out,
    double* __restrict__ du_min_CFL_out,
    const double* __restrict__ u,
    const double* __restrict__ visc_rem_u,
    const double* __restrict__ visc_rem_max_in,
    const double* __restrict__ dxT,
    const double* __restrict__ areaT,
    const double* __restrict__ dy_Cu,
    const double* __restrict__ mask2dCu,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    double CFL_dt, double I_dt,
    int vol_CFL_flag, int use_visc_rem_flag, int aggress_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l) return;

    // Initial CFL limits from visc_rem_max
    double I_vrm = 0.0;
    if (visc_rem_max_in[IDX2(i, j, n1)] > 0.0)
        I_vrm = 1.0 / visc_rem_max_in[IDX2(i, j, n1)];

    double dx_W, dx_E;
    if (vol_CFL_flag == 1) {
        dx_W = areaT[IDX2(i, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
        if (fabs(dx_W) > 1000.0 * dxT[IDX2(i, j, n1)]) dx_W = 1000.0 * dxT[IDX2(i, j, n1)];
        dx_E = areaT[IDX2(i+1, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
        if (fabs(dx_E) > 1000.0 * dxT[IDX2(i+1, j, n1)]) dx_E = 1000.0 * dxT[IDX2(i+1, j, n1)];
    } else {
        dx_W = dxT[IDX2(i, j, n1)];
        dx_E = dxT[IDX2(i+1, j, n1)];
    }
    double du_max_val = 2.0 * (CFL_dt * dx_W) * I_vrm;
    double du_min_val = -2.0 * (CFL_dt * dx_E) * I_vrm;

    // Tighten CFL limits over k
    if (use_visc_rem_flag == 1) {
        if (aggress_flag == 1) {
            for (int k = 1; k <= n3; k++) {
                double vrm_k = visc_rem_u[IDX3(i, j, k, n1, n2)];
                if (vol_CFL_flag == 1) {
                    dx_W = areaT[IDX2(i, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_W) > 1000.0 * dxT[IDX2(i, j, n1)]) dx_W = 1000.0 * dxT[IDX2(i, j, n1)];
                    dx_E = areaT[IDX2(i+1, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_E) > 1000.0 * dxT[IDX2(i+1, j, n1)]) dx_E = 1000.0 * dxT[IDX2(i+1, j, n1)];
                } else {
                    dx_W = dxT[IDX2(i, j, n1)];
                    dx_E = dxT[IDX2(i+1, j, n1)];
                }
                double du_lim = 0.499 * ((dx_W * I_dt - u[IDX3(i, j, k, n1, n2)])
                              + fmin(0.0, u[IDX3(i-1, j, k, n1, n2)]));
                if (du_max_val * vrm_k > du_lim)
                    du_max_val = du_lim / vrm_k;
                du_lim = 0.499 * ((-dx_E * I_dt - u[IDX3(i, j, k, n1, n2)])
                       + fmax(0.0, u[IDX3(i+1, j, k, n1, n2)]));
                if (du_min_val * vrm_k < du_lim)
                    du_min_val = du_lim / vrm_k;
            }
        } else {
            for (int k = 1; k <= n3; k++) {
                double vrm_k = visc_rem_u[IDX3(i, j, k, n1, n2)];
                if (vol_CFL_flag == 1) {
                    dx_W = areaT[IDX2(i, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_W) > 1000.0 * dxT[IDX2(i, j, n1)]) dx_W = 1000.0 * dxT[IDX2(i, j, n1)];
                    dx_E = areaT[IDX2(i+1, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_E) > 1000.0 * dxT[IDX2(i+1, j, n1)]) dx_E = 1000.0 * dxT[IDX2(i+1, j, n1)];
                } else {
                    dx_W = dxT[IDX2(i, j, n1)];
                    dx_E = dxT[IDX2(i+1, j, n1)];
                }
                if (du_max_val * vrm_k > dx_W * CFL_dt - u[IDX3(i, j, k, n1, n2)] * mask2dCu[IDX2(i, j, n1)])
                    du_max_val = (dx_W * CFL_dt - u[IDX3(i, j, k, n1, n2)]) / vrm_k;
                if (du_min_val * vrm_k < -dx_E * CFL_dt - u[IDX3(i, j, k, n1, n2)] * mask2dCu[IDX2(i, j, n1)])
                    du_min_val = -(dx_E * CFL_dt + u[IDX3(i, j, k, n1, n2)]) / vrm_k;
            }
        }
    } else {
        if (aggress_flag == 1) {
            for (int k = 1; k <= n3; k++) {
                if (vol_CFL_flag == 1) {
                    dx_W = areaT[IDX2(i, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_W) > 1000.0 * dxT[IDX2(i, j, n1)]) dx_W = 1000.0 * dxT[IDX2(i, j, n1)];
                    dx_E = areaT[IDX2(i+1, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_E) > 1000.0 * dxT[IDX2(i+1, j, n1)]) dx_E = 1000.0 * dxT[IDX2(i+1, j, n1)];
                } else {
                    dx_W = dxT[IDX2(i, j, n1)];
                    dx_E = dxT[IDX2(i+1, j, n1)];
                }
                du_max_val = fmin(du_max_val, 0.499 * ((dx_W * I_dt - u[IDX3(i, j, k, n1, n2)])
                           + fmin(0.0, u[IDX3(i-1, j, k, n1, n2)])));
                du_min_val = fmax(du_min_val, 0.499 * ((-dx_E * I_dt - u[IDX3(i, j, k, n1, n2)])
                           + fmax(0.0, u[IDX3(i+1, j, k, n1, n2)])));
            }
        } else {
            for (int k = 1; k <= n3; k++) {
                if (vol_CFL_flag == 1) {
                    dx_W = areaT[IDX2(i, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_W) > 1000.0 * dxT[IDX2(i, j, n1)]) dx_W = 1000.0 * dxT[IDX2(i, j, n1)];
                    dx_E = areaT[IDX2(i+1, j, n1)] / (dy_Cu[IDX2(i, j, n1)] + 1.0e-30);
                    if (fabs(dx_E) > 1000.0 * dxT[IDX2(i+1, j, n1)]) dx_E = 1000.0 * dxT[IDX2(i+1, j, n1)];
                } else {
                    dx_W = dxT[IDX2(i, j, n1)];
                    dx_E = dxT[IDX2(i+1, j, n1)];
                }
                du_max_val = fmin(du_max_val, dx_W * CFL_dt - u[IDX3(i, j, k, n1, n2)]);
                du_min_val = fmax(du_min_val, -(dx_E * CFL_dt + u[IDX3(i, j, k, n1, n2)]));
            }
        }
    }

    // Ensure bounds include 0
    du_max_CFL_out[IDX2(i, j, n1)] = fmax(du_max_val, 0.0);
    du_min_CFL_out[IDX2(i, j, n1)] = fmin(du_min_val, 0.0);
}

// =========================================================================
// Kernel 7: zonal_flux_adjust_kernel_c
//   Newton iteration to adjust zonal fluxes to match barotropic transport.
//   2D thread mapping: one thread per (I,j), serial k-loops.
//   Range: i in [is_l-1 : ie_l], j in [js_l : je_l]
// =========================================================================
__global__ void zonal_flux_adjust_kernel_c(
    double* __restrict__ uh,
    double* __restrict__ du_out,
    const double* __restrict__ u,
    const double* __restrict__ h_in,
    const double* __restrict__ h_W,
    const double* __restrict__ h_E,
    const double* __restrict__ uhbt,
    const double* __restrict__ visc_rem_u,
    const double* __restrict__ por_face_areaU,
    const double* __restrict__ IareaT,
    const double* __restrict__ dy_Cu,
    const double* __restrict__ IdxT,
    const double* __restrict__ du_max_CFL_in,
    const double* __restrict__ du_min_CFL_in,
    const double* __restrict__ uh_tot_0_in,
    const double* __restrict__ duhdu_tot_0_in,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    double dt,
    double tol_eta_base, double tol_vel_val,
    int vol_CFL_flag, int use_visc_rem_flag, int better_iter_flag,
    int use_por_face_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l) return;

    const int max_itts = 20;

    double du_val = 0.0;
    double du_max_val = du_max_CFL_in[IDX2(i, j, n1)];
    double du_min_val = du_min_CFL_in[IDX2(i, j, n1)];
    double uh_err_val = uh_tot_0_in[IDX2(i, j, n1)] - uhbt[IDX2(i, j, n1)];
    double duhdu_tot_val = duhdu_tot_0_in[IDX2(i, j, n1)];
    double uh_err_best_val = fabs(uh_err_val);
    int do_more = 1;

    for (int itt = 1; itt <= max_itts; itt++) {
        double tol_eta;
        if (itt <= 1) {
            tol_eta = 1.0e-6 * tol_eta_base;
        } else if (itt == 2) {
            tol_eta = 1.0e-4 * tol_eta_base;
        } else if (itt == 3) {
            tol_eta = 1.0e-2 * tol_eta_base;
        } else {
            tol_eta = tol_eta_base;
        }
        double tol_vel = tol_vel_val;

        if (uh_err_val > 0.0) {
            du_max_val = du_val;
        } else if (uh_err_val < 0.0) {
            du_min_val = du_val;
        } else {
            do_more = 0;
        }

        if (do_more) {
            if ((dt * fmin(IareaT[IDX2(i, j, n1)], IareaT[IDX2(i+1, j, n1)]) * fabs(uh_err_val) > tol_eta) ||
                (better_iter_flag == 1 && ((fabs(uh_err_val) > tol_vel * duhdu_tot_val) ||
                                            (fabs(uh_err_val) > uh_err_best_val)))) {
                double ddu = -uh_err_val / duhdu_tot_val;
                double du_prev = du_val;
                du_val = du_val + ddu;
                if (fabs(ddu) < 1.0e-15 * fabs(du_val)) {
                    do_more = 0;
                } else if (ddu > 0.0) {
                    if (du_val >= du_max_val) {
                        du_val = 0.5 * (du_prev + du_max_val);
                        if (du_max_val - du_prev < 1.0e-15 * fabs(du_val)) do_more = 0;
                    }
                } else {
                    if (du_val <= du_min_val) {
                        du_val = 0.5 * (du_prev + du_min_val);
                        if (du_prev - du_min_val < 1.0e-15 * fabs(du_val)) do_more = 0;
                    }
                }
            } else {
                do_more = 0;
            }
        }

        if (!do_more) break;

        // Recompute flux with adjusted velocity
        uh_err_val = -uhbt[IDX2(i, j, n1)];
        duhdu_tot_val = 0.0;
        for (int k = 1; k <= n3; k++) {
            double visc_rem_val;
            if (use_visc_rem_flag == 1) {
                visc_rem_val = visc_rem_u[IDX3(i, j, k, n1, n2)];
            } else {
                visc_rem_val = 1.0;
            }
            double dy_pfa;
            if (use_por_face_flag == 1) {
                dy_pfa = dy_Cu[IDX2(i, j, n1)] * por_face_areaU[IDX3(i, j, k, n1, n2)];
            } else {
                dy_pfa = dy_Cu[IDX2(i, j, n1)];
            }
            double u_adj = u[IDX3(i, j, k, n1, n2)] + du_val * visc_rem_val;

            double CFL, curv_3, h_marg, uh_k;
            if (u_adj > 0.0) {
                if (vol_CFL_flag == 1) {
                    CFL = (u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
                } else {
                    CFL = u_adj * dt * IdxT[IDX2(i, j, n1)];
                }
                curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
                       - 2.0 * h_in[IDX3(i, j, k, n1, n2)];
                uh_k = dy_pfa * u_adj *
                    (h_E[IDX3(i, j, k, n1, n2)] + CFL * (0.5 * (h_W[IDX3(i, j, k, n1, n2)]
                    - h_E[IDX3(i, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
                h_marg = h_E[IDX3(i, j, k, n1, n2)] + CFL * ((h_W[IDX3(i, j, k, n1, n2)]
                        - h_E[IDX3(i, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
            } else if (u_adj < 0.0) {
                if (vol_CFL_flag == 1) {
                    CFL = (-u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
                } else {
                    CFL = -u_adj * dt * IdxT[IDX2(i+1, j, n1)];
                }
                curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
                       - 2.0 * h_in[IDX3(i+1, j, k, n1, n2)];
                uh_k = dy_pfa * u_adj *
                    (h_W[IDX3(i+1, j, k, n1, n2)] + CFL * (0.5 * (h_E[IDX3(i+1, j, k, n1, n2)]
                    - h_W[IDX3(i+1, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
                h_marg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * ((h_E[IDX3(i+1, j, k, n1, n2)]
                        - h_W[IDX3(i+1, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
            } else {
                uh_k = 0.0;
                h_marg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
            }
            double duhdu_k = dy_pfa * h_marg * visc_rem_val;
            uh_err_val += uh_k;
            duhdu_tot_val += duhdu_k;
        }
        uh_err_best_val = fmin(uh_err_best_val, fabs(uh_err_val));
    } // Newton iterations

    // Final pass: write converged uh values
    for (int k = 1; k <= n3; k++) {
        double visc_rem_val;
        if (use_visc_rem_flag == 1) {
            visc_rem_val = visc_rem_u[IDX3(i, j, k, n1, n2)];
        } else {
            visc_rem_val = 1.0;
        }
        double dy_pfa;
        if (use_por_face_flag == 1) {
            dy_pfa = dy_Cu[IDX2(i, j, n1)] * por_face_areaU[IDX3(i, j, k, n1, n2)];
        } else {
            dy_pfa = dy_Cu[IDX2(i, j, n1)];
        }
        double u_adj = u[IDX3(i, j, k, n1, n2)] + du_val * visc_rem_val;

        if (u_adj > 0.0) {
            double CFL;
            if (vol_CFL_flag == 1) {
                CFL = (u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
            } else {
                CFL = u_adj * dt * IdxT[IDX2(i, j, n1)];
            }
            double curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
                          - 2.0 * h_in[IDX3(i, j, k, n1, n2)];
            uh[IDX3(i, j, k, n1, n2)] = dy_pfa * u_adj *
                (h_E[IDX3(i, j, k, n1, n2)] + CFL * (0.5 * (h_W[IDX3(i, j, k, n1, n2)]
                - h_E[IDX3(i, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
        } else if (u_adj < 0.0) {
            double CFL;
            if (vol_CFL_flag == 1) {
                CFL = (-u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
            } else {
                CFL = -u_adj * dt * IdxT[IDX2(i+1, j, n1)];
            }
            double curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
                          - 2.0 * h_in[IDX3(i+1, j, k, n1, n2)];
            uh[IDX3(i, j, k, n1, n2)] = dy_pfa * u_adj *
                (h_W[IDX3(i+1, j, k, n1, n2)] + CFL * (0.5 * (h_E[IDX3(i+1, j, k, n1, n2)]
                - h_W[IDX3(i+1, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
        } else {
            uh[IDX3(i, j, k, n1, n2)] = 0.0;
        }
    }

    du_out[IDX2(i, j, n1)] = du_val;
}

// =========================================================================
// Kernel 8: set_zonal_BT_cont_kernel_c
//   BT_cont face areas via Newton iteration + 3 test velocities.
//   2D thread mapping: one thread per (I,j).
//   Range: i in [is_l-1 : ie_l], j in [js_l : je_l]
// =========================================================================
__global__ void set_zonal_BT_cont_kernel_c(
    double* __restrict__ FA_u_W0,
    double* __restrict__ FA_u_WW,
    double* __restrict__ uBT_WW,
    double* __restrict__ FA_u_E0,
    double* __restrict__ FA_u_EE,
    double* __restrict__ uBT_EE,
    const double* __restrict__ u,
    const double* __restrict__ h_in,
    const double* __restrict__ h_W,
    const double* __restrict__ h_E,
    const double* __restrict__ visc_rem_u,
    const double* __restrict__ por_face_areaU,
    const double* __restrict__ IareaT,
    const double* __restrict__ dy_Cu,
    const double* __restrict__ IdxT,
    const double* __restrict__ dxCu,
    const double* __restrict__ visc_rem_max_in,
    const double* __restrict__ du_max_CFL_in,
    const double* __restrict__ du_min_CFL_in,
    const double* __restrict__ uh_tot_0_in,
    const double* __restrict__ duhdu_tot_0_in,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    double dt,
    double tol_eta_base, double tol_vel_val,
    int vol_CFL_flag, int use_visc_rem_flag, int better_iter_flag,
    int use_por_face_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l) return;

    const int max_itts = 20;
    const double min_visc_rem = 0.1;
    const double CFL_min_val = 1.0e-6;

    double Idt = 1.0 / dt;

    // --- Phase 1: Newton iteration to find du0 (zero-transport correction) ---
    double du_val = 0.0;
    double du_max_val = du_max_CFL_in[IDX2(i, j, n1)];
    double du_min_val = du_min_CFL_in[IDX2(i, j, n1)];
    double uh_err_val = uh_tot_0_in[IDX2(i, j, n1)];  // target is 0
    double duhdu_tot_val = duhdu_tot_0_in[IDX2(i, j, n1)];
    double uh_err_best_val = fabs(uh_err_val);
    int do_more = 1;

    for (int itt = 1; itt <= max_itts; itt++) {
        double tol_eta;
        if (itt <= 1) {
            tol_eta = 1.0e-6 * tol_eta_base;
        } else if (itt == 2) {
            tol_eta = 1.0e-4 * tol_eta_base;
        } else if (itt == 3) {
            tol_eta = 1.0e-2 * tol_eta_base;
        } else {
            tol_eta = tol_eta_base;
        }
        double tol_vel = tol_vel_val;

        if (uh_err_val > 0.0) {
            du_max_val = du_val;
        } else if (uh_err_val < 0.0) {
            du_min_val = du_val;
        } else {
            do_more = 0;
        }

        if (do_more) {
            if ((dt * fmin(IareaT[IDX2(i, j, n1)], IareaT[IDX2(i+1, j, n1)]) * fabs(uh_err_val) > tol_eta) ||
                (better_iter_flag == 1 && ((fabs(uh_err_val) > tol_vel * duhdu_tot_val) ||
                                            (fabs(uh_err_val) > uh_err_best_val)))) {
                double ddu = -uh_err_val / duhdu_tot_val;
                double du_prev = du_val;
                du_val = du_val + ddu;
                if (fabs(ddu) < 1.0e-15 * fabs(du_val)) {
                    do_more = 0;
                } else if (ddu > 0.0) {
                    if (du_val >= du_max_val) {
                        du_val = 0.5 * (du_prev + du_max_val);
                        if (du_max_val - du_prev < 1.0e-15 * fabs(du_val)) do_more = 0;
                    }
                } else {
                    if (du_val <= du_min_val) {
                        du_val = 0.5 * (du_prev + du_min_val);
                        if (du_prev - du_min_val < 1.0e-15 * fabs(du_val)) do_more = 0;
                    }
                }
            } else {
                do_more = 0;
            }
        }

        if (!do_more) break;

        uh_err_val = 0.0;  // target is zero transport
        duhdu_tot_val = 0.0;
        for (int k = 1; k <= n3; k++) {
            double visc_rem_val;
            if (use_visc_rem_flag == 1) {
                visc_rem_val = visc_rem_u[IDX3(i, j, k, n1, n2)];
            } else {
                visc_rem_val = 1.0;
            }
            double dy_pfa;
            if (use_por_face_flag == 1) {
                dy_pfa = dy_Cu[IDX2(i, j, n1)] * por_face_areaU[IDX3(i, j, k, n1, n2)];
            } else {
                dy_pfa = dy_Cu[IDX2(i, j, n1)];
            }
            double u_adj = u[IDX3(i, j, k, n1, n2)] + du_val * visc_rem_val;
            double CFL, curv_3, h_marg, uh_k;
            if (u_adj > 0.0) {
                if (vol_CFL_flag == 1) {
                    CFL = (u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
                } else {
                    CFL = u_adj * dt * IdxT[IDX2(i, j, n1)];
                }
                curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
                       - 2.0 * h_in[IDX3(i, j, k, n1, n2)];
                uh_k = dy_pfa * u_adj *
                    (h_E[IDX3(i, j, k, n1, n2)] + CFL * (0.5 * (h_W[IDX3(i, j, k, n1, n2)]
                    - h_E[IDX3(i, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
                h_marg = h_E[IDX3(i, j, k, n1, n2)] + CFL * ((h_W[IDX3(i, j, k, n1, n2)]
                        - h_E[IDX3(i, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
            } else if (u_adj < 0.0) {
                if (vol_CFL_flag == 1) {
                    CFL = (-u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
                } else {
                    CFL = -u_adj * dt * IdxT[IDX2(i+1, j, n1)];
                }
                curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
                       - 2.0 * h_in[IDX3(i+1, j, k, n1, n2)];
                uh_k = dy_pfa * u_adj *
                    (h_W[IDX3(i+1, j, k, n1, n2)] + CFL * (0.5 * (h_E[IDX3(i+1, j, k, n1, n2)]
                    - h_W[IDX3(i+1, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
                h_marg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * ((h_E[IDX3(i+1, j, k, n1, n2)]
                        - h_W[IDX3(i+1, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
            } else {
                uh_k = 0.0;
                h_marg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
            }
            double duhdu_k = dy_pfa * h_marg * visc_rem_val;
            uh_err_val += uh_k;
            duhdu_tot_val += duhdu_k;
        }
        uh_err_best_val = fmin(uh_err_best_val, fabs(uh_err_val));
    } // Newton for du0

    double du0_val = du_val;

    // --- Phase 2: Determine test velocities duL, duR ---
    double du_CFL_val = (CFL_min_val * Idt) * dxCu[IDX2(i, j, n1)];
    double duR_val = fmin(0.0, du0_val - du_CFL_val);
    double duL_val = fmax(0.0, du0_val + du_CFL_val);

    // Adjust duR, duL so test velocities are truly upwind
    for (int k = 1; k <= n3; k++) {
        double visc_rem_val;
        if (use_visc_rem_flag == 1) {
            visc_rem_val = visc_rem_u[IDX3(i, j, k, n1, n2)];
        } else {
            visc_rem_val = 1.0;
        }
        double visc_rem_lim = fmax(visc_rem_val, min_visc_rem * visc_rem_max_in[IDX2(i, j, n1)]);
        if (visc_rem_lim > 0.0) {
            if (u[IDX3(i, j, k, n1, n2)] + duR_val * visc_rem_lim > -du_CFL_val * visc_rem_val)
                duR_val = -(u[IDX3(i, j, k, n1, n2)] + du_CFL_val * visc_rem_val) / visc_rem_lim;
            if (u[IDX3(i, j, k, n1, n2)] + duL_val * visc_rem_lim < du_CFL_val * visc_rem_val)
                duL_val = -(u[IDX3(i, j, k, n1, n2)] - du_CFL_val * visc_rem_val) / visc_rem_lim;
        }
    }

    // --- Phase 3: Evaluate fluxes at 3 test velocities ---
    double FAmt_0_val = 0.0, FAmt_L_val = 0.0, FAmt_R_val = 0.0;
    double uhtot_L_val = 0.0, uhtot_R_val = 0.0;

    for (int k = 1; k <= n3; k++) {
        double visc_rem_val;
        if (use_visc_rem_flag == 1) {
            visc_rem_val = visc_rem_u[IDX3(i, j, k, n1, n2)];
        } else {
            visc_rem_val = 1.0;
        }
        double dy_pfa;
        if (use_por_face_flag == 1) {
            dy_pfa = dy_Cu[IDX2(i, j, n1)] * por_face_areaU[IDX3(i, j, k, n1, n2)];
        } else {
            dy_pfa = dy_Cu[IDX2(i, j, n1)];
        }

        double CFL, curv_3, h_marg, uh_k, u_adj;

        // u_0 test velocity
        u_adj = u[IDX3(i, j, k, n1, n2)] + du0_val * visc_rem_val;
        if (u_adj > 0.0) {
            if (vol_CFL_flag == 1) {
                CFL = (u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
            } else {
                CFL = u_adj * dt * IdxT[IDX2(i, j, n1)];
            }
            curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
                   - 2.0 * h_in[IDX3(i, j, k, n1, n2)];
            h_marg = h_E[IDX3(i, j, k, n1, n2)] + CFL * ((h_W[IDX3(i, j, k, n1, n2)]
                    - h_E[IDX3(i, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
        } else if (u_adj < 0.0) {
            if (vol_CFL_flag == 1) {
                CFL = (-u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
            } else {
                CFL = -u_adj * dt * IdxT[IDX2(i+1, j, n1)];
            }
            curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
                   - 2.0 * h_in[IDX3(i+1, j, k, n1, n2)];
            h_marg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * ((h_E[IDX3(i+1, j, k, n1, n2)]
                    - h_W[IDX3(i+1, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
        } else {
            h_marg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
        }
        FAmt_0_val += dy_pfa * h_marg * visc_rem_val;

        // u_L test velocity (westerly, positive)
        u_adj = u[IDX3(i, j, k, n1, n2)] + duL_val * visc_rem_val;
        if (u_adj > 0.0) {
            if (vol_CFL_flag == 1) {
                CFL = (u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
            } else {
                CFL = u_adj * dt * IdxT[IDX2(i, j, n1)];
            }
            curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
                   - 2.0 * h_in[IDX3(i, j, k, n1, n2)];
            uh_k = dy_pfa * u_adj *
                (h_E[IDX3(i, j, k, n1, n2)] + CFL * (0.5 * (h_W[IDX3(i, j, k, n1, n2)]
                - h_E[IDX3(i, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
            h_marg = h_E[IDX3(i, j, k, n1, n2)] + CFL * ((h_W[IDX3(i, j, k, n1, n2)]
                    - h_E[IDX3(i, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
        } else if (u_adj < 0.0) {
            if (vol_CFL_flag == 1) {
                CFL = (-u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
            } else {
                CFL = -u_adj * dt * IdxT[IDX2(i+1, j, n1)];
            }
            curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
                   - 2.0 * h_in[IDX3(i+1, j, k, n1, n2)];
            uh_k = dy_pfa * u_adj *
                (h_W[IDX3(i+1, j, k, n1, n2)] + CFL * (0.5 * (h_E[IDX3(i+1, j, k, n1, n2)]
                - h_W[IDX3(i+1, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
            h_marg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * ((h_E[IDX3(i+1, j, k, n1, n2)]
                    - h_W[IDX3(i+1, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
        } else {
            uh_k = 0.0;
            h_marg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
        }
        FAmt_L_val += dy_pfa * h_marg * visc_rem_val;
        uhtot_L_val += uh_k;

        // u_R test velocity (easterly, negative)
        u_adj = u[IDX3(i, j, k, n1, n2)] + duR_val * visc_rem_val;
        if (u_adj > 0.0) {
            if (vol_CFL_flag == 1) {
                CFL = (u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
            } else {
                CFL = u_adj * dt * IdxT[IDX2(i, j, n1)];
            }
            curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
                   - 2.0 * h_in[IDX3(i, j, k, n1, n2)];
            uh_k = dy_pfa * u_adj *
                (h_E[IDX3(i, j, k, n1, n2)] + CFL * (0.5 * (h_W[IDX3(i, j, k, n1, n2)]
                - h_E[IDX3(i, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
            h_marg = h_E[IDX3(i, j, k, n1, n2)] + CFL * ((h_W[IDX3(i, j, k, n1, n2)]
                    - h_E[IDX3(i, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
        } else if (u_adj < 0.0) {
            if (vol_CFL_flag == 1) {
                CFL = (-u_adj * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
            } else {
                CFL = -u_adj * dt * IdxT[IDX2(i+1, j, n1)];
            }
            curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
                   - 2.0 * h_in[IDX3(i+1, j, k, n1, n2)];
            uh_k = dy_pfa * u_adj *
                (h_W[IDX3(i+1, j, k, n1, n2)] + CFL * (0.5 * (h_E[IDX3(i+1, j, k, n1, n2)]
                - h_W[IDX3(i+1, j, k, n1, n2)]) + curv_3 * (CFL - 1.5)));
            h_marg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * ((h_E[IDX3(i+1, j, k, n1, n2)]
                    - h_W[IDX3(i+1, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
        } else {
            uh_k = 0.0;
            h_marg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
        }
        FAmt_R_val += dy_pfa * h_marg * visc_rem_val;
        uhtot_R_val += uh_k;
    } // k

    // --- Phase 4: Compute BT_cont fields ---
    double FA_0, FA_avg;

    // Westerly (W0, WW)
    FA_0 = FAmt_0_val;
    FA_avg = FAmt_0_val;
    if ((duL_val - du0_val) != 0.0)
        FA_avg = uhtot_L_val / (duL_val - du0_val);
    if (FA_avg > fmax(FA_0, FAmt_L_val)) {
        FA_avg = fmax(FA_0, FAmt_L_val);
    } else if (FA_avg < fmin(FA_0, FAmt_L_val)) {
        FA_0 = FA_avg;
    }
    FA_u_W0[IDX2(i, j, n1)] = FA_0;
    FA_u_WW[IDX2(i, j, n1)] = FAmt_L_val;
    if (fabs(FA_0 - FAmt_L_val) <= 1.0e-12 * FA_0) {
        uBT_WW[IDX2(i, j, n1)] = 0.0;
    } else {
        uBT_WW[IDX2(i, j, n1)] = (1.5 * (duL_val - du0_val))
            * ((FAmt_L_val - FA_avg) / (FAmt_L_val - FA_0));
    }

    // Easterly (E0, EE)
    FA_0 = FAmt_0_val;
    FA_avg = FAmt_0_val;
    if ((duR_val - du0_val) != 0.0)
        FA_avg = uhtot_R_val / (duR_val - du0_val);
    if (FA_avg > fmax(FA_0, FAmt_R_val)) {
        FA_avg = fmax(FA_0, FAmt_R_val);
    } else if (FA_avg < fmin(FA_0, FAmt_R_val)) {
        FA_0 = FA_avg;
    }
    FA_u_E0[IDX2(i, j, n1)] = FA_0;
    FA_u_EE[IDX2(i, j, n1)] = FAmt_R_val;
    if (fabs(FAmt_R_val - FA_0) <= 1.0e-12 * FA_0) {
        uBT_EE[IDX2(i, j, n1)] = 0.0;
    } else {
        uBT_EE[IDX2(i, j, n1)] = (1.5 * (duR_val - du0_val))
            * ((FAmt_R_val - FA_avg) / (FAmt_R_val - FA_0));
    }
}

// =========================================================================
// Kernel 9: zonal_flux_thickness_kernel_c
//   3D flux thickness h_u.
//   Range: i in [is_l-1 : ie_l], j in [js_l : je_l], k in [1:n3]
// =========================================================================
__global__ void zonal_flux_thickness_kernel_c(
    double* __restrict__ h_u,
    const double* __restrict__ u,
    const double* __restrict__ h,
    const double* __restrict__ h_W,
    const double* __restrict__ h_E,
    const double* __restrict__ por_face_areaU,
    const double* __restrict__ visc_rem_u,
    const double* __restrict__ IareaT,
    const double* __restrict__ IdxT,
    const double* __restrict__ dy_Cu,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    double dt,
    int vol_CFL_flag, int marginal_flag, int has_visc_rem_flag, int use_por_face_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l ||
        k < 1 || k > n3) return;

    double u_ijk = u[IDX3(i, j, k, n1, n2)];
    double CFL, curv_3, h_avg, h_marg;

    if (u_ijk > 0.0) {
        if (vol_CFL_flag == 1) {
            CFL = (u_ijk * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i, j, n1)]);
        } else {
            CFL = u_ijk * dt * IdxT[IDX2(i, j, n1)];
        }
        curv_3 = (h_W[IDX3(i, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)])
               - 2.0 * h[IDX3(i, j, k, n1, n2)];
        h_avg = h_E[IDX3(i, j, k, n1, n2)] + CFL * (0.5 * (h_W[IDX3(i, j, k, n1, n2)]
              - h_E[IDX3(i, j, k, n1, n2)]) + curv_3 * (CFL - 1.5));
        h_marg = h_E[IDX3(i, j, k, n1, n2)] + CFL * ((h_W[IDX3(i, j, k, n1, n2)]
                - h_E[IDX3(i, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
    } else if (u_ijk < 0.0) {
        if (vol_CFL_flag == 1) {
            CFL = (-u_ijk * dt) * (dy_Cu[IDX2(i, j, n1)] * IareaT[IDX2(i+1, j, n1)]);
        } else {
            CFL = -u_ijk * dt * IdxT[IDX2(i+1, j, n1)];
        }
        curv_3 = (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i+1, j, k, n1, n2)])
               - 2.0 * h[IDX3(i+1, j, k, n1, n2)];
        h_avg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * (0.5 * (h_E[IDX3(i+1, j, k, n1, n2)]
              - h_W[IDX3(i+1, j, k, n1, n2)]) + curv_3 * (CFL - 1.5));
        h_marg = h_W[IDX3(i+1, j, k, n1, n2)] + CFL * ((h_E[IDX3(i+1, j, k, n1, n2)]
                - h_W[IDX3(i+1, j, k, n1, n2)]) + 3.0 * curv_3 * (CFL - 1.0));
    } else {
        h_avg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
        h_marg = 0.5 * (h_W[IDX3(i+1, j, k, n1, n2)] + h_E[IDX3(i, j, k, n1, n2)]);
    }

    double result;
    if (marginal_flag == 1) {
        result = h_marg;
    } else {
        result = h_avg;
    }

    // Scale by visc_rem and por_face_areaU
    if (has_visc_rem_flag == 1 && use_por_face_flag == 1) {
        result *= visc_rem_u[IDX3(i, j, k, n1, n2)] * por_face_areaU[IDX3(i, j, k, n1, n2)];
    } else if (has_visc_rem_flag == 1) {
        result *= visc_rem_u[IDX3(i, j, k, n1, n2)];
    } else if (use_por_face_flag == 1) {
        result *= por_face_areaU[IDX3(i, j, k, n1, n2)];
    }

    h_u[IDX3(i, j, k, n1, n2)] = result;
}

// =========================================================================
// Kernel 10: u_cor_kernel_c
//   Corrected velocity: u_cor = u + du * visc_rem_u.
//   3D: i in [is_l-1 : ie_l], j in [js_l : je_l], k in [1:n3]
// =========================================================================
__global__ void u_cor_kernel_c(
    double* __restrict__ u_cor,
    const double* __restrict__ u,
    const double* __restrict__ du,
    const double* __restrict__ visc_rem_u,
    int n1, int n2, int n3,
    int is_l, int ie_l, int js_l, int je_l,
    int use_visc_rem_flag)
{
    int i = THREAD_I;
    int j = THREAD_J;
    int k = THREAD_K;

    if (i < is_l - 1 || i > ie_l ||
        j < js_l || j > je_l ||
        k < 1 || k > n3) return;

    if (use_visc_rem_flag == 1) {
        u_cor[IDX3(i, j, k, n1, n2)] = u[IDX3(i, j, k, n1, n2)]
            + du[IDX2(i, j, n1)] * visc_rem_u[IDX3(i, j, k, n1, n2)];
    } else {
        u_cor[IDX3(i, j, k, n1, n2)] = u[IDX3(i, j, k, n1, n2)] + du[IDX2(i, j, n1)];
    }
}


// =========================================================================
// Extern "C" launch wrappers (called from Fortran via iso_c_binding)
// =========================================================================

extern "C" void launch_ppm_reconstruction_3d_kernel(
    double* h_W, double* h_E, double* h_in, double* mask2dT,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    double h_min, int monotonic, int upwind_1st_flag, int simple_2nd_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    ppm_reconstruction_3d_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        h_W, h_E, h_in, mask2dT,
        n1, n2, n3, is_l, ie_l, js_l, je_l,
        h_min, monotonic, upwind_1st_flag, simple_2nd_flag);
}

extern "C" void launch_zonal_flux_layer_3d_kernel(
    double* uh, double* duhdu, double* u, double* h_in,
    double* h_W, double* h_E,
    double* dy_Cu, double* IdxT, double* IareaT,
    double* por_face_areaU, double* visc_rem_u,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    double dt, int vol_CFL_flag, int use_visc_rem_flag, int use_por_face_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    zonal_flux_layer_3d_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        uh, duhdu, u, h_in, h_W, h_E,
        dy_Cu, IdxT, IareaT, por_face_areaU, visc_rem_u,
        n1, n2, n3, is_l, ie_l, js_l, je_l,
        dt, vol_CFL_flag, use_visc_rem_flag, use_por_face_flag);
}

extern "C" void launch_zonal_convergence_kernel(
    double* h, double* hin, double* uh, double* IareaT,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    double dt,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    zonal_convergence_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        h, hin, uh, IareaT,
        n1, n2, n3, is_l, ie_l, js_l, je_l, dt);
}

extern "C" void launch_visc_rem_max_kernel(
    double* visc_rem_max_out, double* visc_rem_u,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int use_vrm_max_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    visc_rem_max_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        visc_rem_max_out, visc_rem_u,
        n1, n2, n3, is_l, ie_l, js_l, je_l, use_vrm_max_flag);
}

extern "C" void launch_uh_duhdu_tot_kernel(
    double* uh_tot_0, double* duhdu_tot_0,
    double* uh, double* duhdu,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    uh_duhdu_tot_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        uh_tot_0, duhdu_tot_0, uh, duhdu,
        n1, n2, n3, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_zonal_CFL_limits_kernel(
    double* du_max_CFL_out, double* du_min_CFL_out,
    double* u, double* visc_rem_u, double* visc_rem_max_in,
    double* dxT, double* areaT, double* dy_Cu, double* mask2dCu,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    double CFL_dt, double I_dt,
    int vol_CFL_flag, int use_visc_rem_flag, int aggress_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    zonal_CFL_limits_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        du_max_CFL_out, du_min_CFL_out,
        u, visc_rem_u, visc_rem_max_in,
        dxT, areaT, dy_Cu, mask2dCu,
        n1, n2, n3, is_l, ie_l, js_l, je_l,
        CFL_dt, I_dt, vol_CFL_flag, use_visc_rem_flag, aggress_flag);
}

extern "C" void launch_zonal_flux_adjust_kernel(
    double* uh, double* du_out,
    double* u, double* h_in, double* h_W, double* h_E, double* uhbt,
    double* visc_rem_u, double* por_face_areaU,
    double* IareaT, double* dy_Cu, double* IdxT,
    double* du_max_CFL_in, double* du_min_CFL_in,
    double* uh_tot_0_in, double* duhdu_tot_0_in,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    double dt, double tol_eta_base, double tol_vel_val,
    int vol_CFL_flag, int use_visc_rem_flag, int better_iter_flag,
    int use_por_face_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    zonal_flux_adjust_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        uh, du_out, u, h_in, h_W, h_E, uhbt,
        visc_rem_u, por_face_areaU,
        IareaT, dy_Cu, IdxT,
        du_max_CFL_in, du_min_CFL_in,
        uh_tot_0_in, duhdu_tot_0_in,
        n1, n2, n3, is_l, ie_l, js_l, je_l,
        dt, tol_eta_base, tol_vel_val,
        vol_CFL_flag, use_visc_rem_flag, better_iter_flag, use_por_face_flag);
}

extern "C" void launch_set_zonal_BT_cont_kernel(
    double* FA_u_W0, double* FA_u_WW, double* uBT_WW,
    double* FA_u_E0, double* FA_u_EE, double* uBT_EE,
    double* u, double* h_in, double* h_W, double* h_E,
    double* visc_rem_u, double* por_face_areaU,
    double* IareaT, double* dy_Cu, double* IdxT, double* dxCu,
    double* visc_rem_max_in,
    double* du_max_CFL_in, double* du_min_CFL_in,
    double* uh_tot_0_in, double* duhdu_tot_0_in,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    double dt, double tol_eta_base, double tol_vel_val,
    int vol_CFL_flag, int use_visc_rem_flag, int better_iter_flag,
    int use_por_face_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    set_zonal_BT_cont_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        FA_u_W0, FA_u_WW, uBT_WW,
        FA_u_E0, FA_u_EE, uBT_EE,
        u, h_in, h_W, h_E,
        visc_rem_u, por_face_areaU,
        IareaT, dy_Cu, IdxT, dxCu,
        visc_rem_max_in,
        du_max_CFL_in, du_min_CFL_in,
        uh_tot_0_in, duhdu_tot_0_in,
        n1, n2, n3, is_l, ie_l, js_l, je_l,
        dt, tol_eta_base, tol_vel_val,
        vol_CFL_flag, use_visc_rem_flag, better_iter_flag, use_por_face_flag);
}

extern "C" void launch_zonal_flux_thickness_kernel(
    double* h_u, double* u, double* h, double* h_W, double* h_E,
    double* por_face_areaU, double* visc_rem_u,
    double* IareaT, double* IdxT, double* dy_Cu,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    double dt,
    int vol_CFL_flag, int marginal_flag, int has_visc_rem_flag, int use_por_face_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    zonal_flux_thickness_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        h_u, u, h, h_W, h_E,
        por_face_areaU, visc_rem_u,
        IareaT, IdxT, dy_Cu,
        n1, n2, n3, is_l, ie_l, js_l, je_l,
        dt, vol_CFL_flag, marginal_flag, has_visc_rem_flag, use_por_face_flag);
}

extern "C" void launch_u_cor_kernel(
    double* u_cor, double* u, double* du, double* visc_rem_u,
    int n1, int n2, int n3, int is_l, int ie_l, int js_l, int je_l,
    int use_visc_rem_flag,
    int grid_x, int grid_y, int grid_z,
    int block_x, int block_y, int block_z,
    void* stream)
{
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);
    u_cor_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        u_cor, u, du, visc_rem_u,
        n1, n2, n3, is_l, ie_l, js_l, je_l, use_visc_rem_flag);
}
