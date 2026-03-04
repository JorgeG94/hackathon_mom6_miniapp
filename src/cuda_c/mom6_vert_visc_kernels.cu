// MOM6 Vertical Viscosity — CUDA C kernels
//
// 1:1 port of the CUDA Fortran kernels from mom6_vert_visc_cuda.F90.
// Uses 1-based column-major indexing macros so that array accesses are
// bit-identical to the Fortran CUDA version.
//
// Two 2D-threaded kernels (no blockIdx.z): each thread handles one
// full water column.  Fused coef+remnant+apply eliminates all intermediate
// global-memory traffic for coupling coefficients.

#include "mom6_cuda_common.h"
#include <cuda_runtime.h>

static const double RHO_0 = 1035.0;

// =========================================================================
// Kernel 1: vert_visc_cra_u_kernel_c — fused coef+remnant+apply, u-points
// =========================================================================
__global__ void vert_visc_cra_u_kernel_c(
    double* __restrict__ u,
    const double* __restrict__ h,
    const double* __restrict__ mask2dCu,
    double* __restrict__ visc_rem_u,
    double* __restrict__ taux_bot,
    const double* __restrict__ taux,
    double Kv, double Kv_ml, double Kv_extra_bbl,
    double Hmix, double Hbbl,
    double dt, double dt_Rho0, double h_neglect, double I_Hbbl,
    int n1, int n2, int nz,
    int is_l, int ie_l, int js_l, int je_l,
    double* __restrict__ hvel,
    double* __restrict__ z_i)
{
    int i = THREAD_I;
    int j = THREAD_J;

    // U-points range: i in [is_l-1 : ie_l], j in [js_l : je_l]
    if (i < is_l - 1 || i > ie_l || j < js_l || j > je_l) return;

    if (mask2dCu[IDX2(i, j, n1)] > 0.0) {

        // ============================================================
        // Coefficient phase: bottom-up hvel and z_i computation
        // ============================================================
        // z_i has nz+1 layers in dim3: index nz+1 in 1-based
        z_i[IDX3(i, j, nz + 1, n1, n2)] = 0.0;

        for (int k = nz; k >= 1; k--) {
            double h_ij  = h[IDX3(i, j, k, n1, n2)];
            double h_ip1 = h[IDX3(i + 1, j, k, n1, n2)];

            double h_harm  = 2.0 * h_ij * h_ip1 / (h_ij + h_ip1 + h_neglect);
            double h_arith = 0.5 * (h_ip1 + h_ij);
            double h_delta = h_ip1 - h_ij;

            hvel[IDX3(i, j, k, n1, n2)] = h_harm;

            if (u[IDX3(i, j, k, n1, n2)] * h_delta < 0.0) {
                double z2 = z_i[IDX3(i, j, k + 1, n1, n2)];
                double z2_6 = z2 * z2 * z2 * z2 * z2 * z2;
                double botfn = 1.0 / (1.0 + 0.09 * z2_6);
                hvel[IDX3(i, j, k, n1, n2)] = (1.0 - botfn) * h_harm + botfn * h_arith;
            }

            z_i[IDX3(i, j, k, n1, n2)] = z_i[IDX3(i, j, k + 1, n1, n2)] + h_harm * I_Hbbl;
        }

        // ============================================================
        // Combined remnant + apply: single tridiagonal solve
        // ============================================================
        double sfc_stress = dt_Rho0 * taux[IDX2(i, j, n1)] * mask2dCu[IDX2(i, j, n1)];

        double a_k = 0.0;  // a_col(1) = 0 (surface)
        double a_k1;
        double z_top;

        // Compute a_col(2) on the fly
        if (nz > 1) {
            z_top = hvel[IDX3(i, j, 1, n1, n2)];
            double Kv_tot = Kv;
            if (z_top < Hmix) {
                double topfn = 1.0 - z_top / Hmix;
                Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn;
            }
            double z2 = z_i[IDX3(i, j, 2, n1, n2)];
            double z2_6 = z2 * z2 * z2 * z2 * z2 * z2;
            double botfn = 1.0 / (1.0 + 0.09 * z2_6);
            Kv_tot = Kv_tot + Kv_extra_bbl * botfn;
            double h_shear = 0.5 * (hvel[IDX3(i, j, 2, n1, n2)] + hvel[IDX3(i, j, 1, n1, n2)] + h_neglect);
            a_k1 = Kv_tot / h_shear;
        } else {
            a_k1 = (Kv + Kv_extra_bbl) / (0.5 * hvel[IDX3(i, j, nz, n1, n2)] + h_neglect);
        }

        // Layer 1
        double h_eff = hvel[IDX3(i, j, 1, n1, n2)] + h_neglect;
        double b_denom_1 = h_eff + dt * a_k;
        double b1 = 1.0 / (b_denom_1 + dt * a_k1);
        double d1 = b_denom_1 * b1;
        visc_rem_u[IDX3(i, j, 1, n1, n2)] = b1 * h_eff;
        u[IDX3(i, j, 1, n1, n2)] = b1 * (h_eff * u[IDX3(i, j, 1, n1, n2)] + sfc_stress);

        // Interior layers -- a_col computed on the fly, c1 stored in z_i
        for (int k = 2; k <= nz; k++) {
            // Store c1(k) in z_i(i,j,k) (z_i dead after coef phase for k-1)
            z_i[IDX3(i, j, k, n1, n2)] = dt * a_k1 * b1;

            a_k = a_k1;  // shift: a_col(k) = previous a_col(k+1)

            // Compute a_col(k+1) on the fly
            if (k < nz) {
                z_top = z_top + hvel[IDX3(i, j, k, n1, n2)];
                double Kv_tot = Kv;
                if (z_top < Hmix) {
                    double topfn = 1.0 - z_top / Hmix;
                    Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn;
                }
                double z2 = z_i[IDX3(i, j, k + 1, n1, n2)];
                double z2_6 = z2 * z2 * z2 * z2 * z2 * z2;
                double botfn = 1.0 / (1.0 + 0.09 * z2_6);
                Kv_tot = Kv_tot + Kv_extra_bbl * botfn;
                double h_shear = 0.5 * (hvel[IDX3(i, j, k + 1, n1, n2)] + hvel[IDX3(i, j, k, n1, n2)] + h_neglect);
                a_k1 = Kv_tot / h_shear;
            } else {
                a_k1 = (Kv + Kv_extra_bbl) / (0.5 * hvel[IDX3(i, j, nz, n1, n2)] + h_neglect);
            }

            h_eff = hvel[IDX3(i, j, k, n1, n2)] + h_neglect;
            b_denom_1 = h_eff + dt * (a_k * d1);
            b1 = 1.0 / (b_denom_1 + dt * a_k1);
            d1 = b_denom_1 * b1;
            visc_rem_u[IDX3(i, j, k, n1, n2)] = (h_eff +
                dt * a_k * visc_rem_u[IDX3(i, j, k - 1, n1, n2)]) * b1;
            u[IDX3(i, j, k, n1, n2)] = (h_eff * u[IDX3(i, j, k, n1, n2)] +
                dt * a_k * u[IDX3(i, j, k - 1, n1, n2)]) * b1;
        }

        // Combined back substitution (c1 stored in z_i)
        for (int k = nz - 1; k >= 1; k--) {
            visc_rem_u[IDX3(i, j, k, n1, n2)] = visc_rem_u[IDX3(i, j, k, n1, n2)] +
                z_i[IDX3(i, j, k + 1, n1, n2)] * visc_rem_u[IDX3(i, j, k + 1, n1, n2)];
            u[IDX3(i, j, k, n1, n2)] = u[IDX3(i, j, k, n1, n2)] +
                z_i[IDX3(i, j, k + 1, n1, n2)] * u[IDX3(i, j, k + 1, n1, n2)];
        }

        // Bottom stress (a_k1 is a_col(nz+1) after the loop)
        taux_bot[IDX2(i, j, n1)] = RHO_0 * u[IDX3(i, j, nz, n1, n2)] * a_k1;

    } else {
        // Masked points: remnant = 1, stress = 0
        for (int k = 1; k <= nz; k++) {
            visc_rem_u[IDX3(i, j, k, n1, n2)] = 1.0;
        }
        taux_bot[IDX2(i, j, n1)] = 0.0;
    }
}

// =========================================================================
// Kernel 2: vert_visc_cra_v_kernel_c — fused coef+remnant+apply, v-points
// =========================================================================
__global__ void vert_visc_cra_v_kernel_c(
    double* __restrict__ v,
    const double* __restrict__ h,
    const double* __restrict__ mask2dCv,
    double* __restrict__ visc_rem_v,
    double* __restrict__ tauy_bot,
    const double* __restrict__ tauy,
    double Kv, double Kv_ml, double Kv_extra_bbl,
    double Hmix, double Hbbl,
    double dt, double dt_Rho0, double h_neglect, double I_Hbbl,
    int n1, int n2, int nz,
    int is_l, int ie_l, int js_l, int je_l,
    double* __restrict__ hvel,
    double* __restrict__ z_i)
{
    int i = THREAD_I;
    int j = THREAD_J;

    // V-points range: i in [is_l : ie_l], j in [js_l-1 : je_l]
    if (i < is_l || i > ie_l || j < js_l - 1 || j > je_l) return;

    if (mask2dCv[IDX2(i, j, n1)] > 0.0) {

        // ============================================================
        // Coefficient phase: bottom-up hvel and z_i computation
        // ============================================================
        z_i[IDX3(i, j, nz + 1, n1, n2)] = 0.0;

        for (int k = nz; k >= 1; k--) {
            double h_ij  = h[IDX3(i, j, k, n1, n2)];
            double h_jp1 = h[IDX3(i, j + 1, k, n1, n2)];

            double h_harm  = 2.0 * h_ij * h_jp1 / (h_ij + h_jp1 + h_neglect);
            double h_arith = 0.5 * (h_jp1 + h_ij);
            double h_delta = h_jp1 - h_ij;

            hvel[IDX3(i, j, k, n1, n2)] = h_harm;

            if (v[IDX3(i, j, k, n1, n2)] * h_delta < 0.0) {
                double z2 = z_i[IDX3(i, j, k + 1, n1, n2)];
                double z2_6 = z2 * z2 * z2 * z2 * z2 * z2;
                double botfn = 1.0 / (1.0 + 0.09 * z2_6);
                hvel[IDX3(i, j, k, n1, n2)] = (1.0 - botfn) * h_harm + botfn * h_arith;
            }

            z_i[IDX3(i, j, k, n1, n2)] = z_i[IDX3(i, j, k + 1, n1, n2)] + h_harm * I_Hbbl;
        }

        // ============================================================
        // Combined remnant + apply: single tridiagonal solve
        // ============================================================
        double sfc_stress = dt_Rho0 * tauy[IDX2(i, j, n1)] * mask2dCv[IDX2(i, j, n1)];

        double a_k = 0.0;  // a_col(1) = 0 (surface)
        double a_k1;
        double z_top;

        // Compute a_col(2) on the fly
        if (nz > 1) {
            z_top = hvel[IDX3(i, j, 1, n1, n2)];
            double Kv_tot = Kv;
            if (z_top < Hmix) {
                double topfn = 1.0 - z_top / Hmix;
                Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn;
            }
            double z2 = z_i[IDX3(i, j, 2, n1, n2)];
            double z2_6 = z2 * z2 * z2 * z2 * z2 * z2;
            double botfn = 1.0 / (1.0 + 0.09 * z2_6);
            Kv_tot = Kv_tot + Kv_extra_bbl * botfn;
            double h_shear = 0.5 * (hvel[IDX3(i, j, 2, n1, n2)] + hvel[IDX3(i, j, 1, n1, n2)] + h_neglect);
            a_k1 = Kv_tot / h_shear;
        } else {
            a_k1 = (Kv + Kv_extra_bbl) / (0.5 * hvel[IDX3(i, j, nz, n1, n2)] + h_neglect);
        }

        // Layer 1
        double h_eff = hvel[IDX3(i, j, 1, n1, n2)] + h_neglect;
        double b_denom_1 = h_eff + dt * a_k;
        double b1 = 1.0 / (b_denom_1 + dt * a_k1);
        double d1 = b_denom_1 * b1;
        visc_rem_v[IDX3(i, j, 1, n1, n2)] = b1 * h_eff;
        v[IDX3(i, j, 1, n1, n2)] = b1 * (h_eff * v[IDX3(i, j, 1, n1, n2)] + sfc_stress);

        // Interior layers
        for (int k = 2; k <= nz; k++) {
            z_i[IDX3(i, j, k, n1, n2)] = dt * a_k1 * b1;  // c1(k) stored in z_i(i,j,k)

            a_k = a_k1;

            if (k < nz) {
                z_top = z_top + hvel[IDX3(i, j, k, n1, n2)];
                double Kv_tot = Kv;
                if (z_top < Hmix) {
                    double topfn = 1.0 - z_top / Hmix;
                    Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn;
                }
                double z2 = z_i[IDX3(i, j, k + 1, n1, n2)];
                double z2_6 = z2 * z2 * z2 * z2 * z2 * z2;
                double botfn = 1.0 / (1.0 + 0.09 * z2_6);
                Kv_tot = Kv_tot + Kv_extra_bbl * botfn;
                double h_shear = 0.5 * (hvel[IDX3(i, j, k + 1, n1, n2)] + hvel[IDX3(i, j, k, n1, n2)] + h_neglect);
                a_k1 = Kv_tot / h_shear;
            } else {
                a_k1 = (Kv + Kv_extra_bbl) / (0.5 * hvel[IDX3(i, j, nz, n1, n2)] + h_neglect);
            }

            h_eff = hvel[IDX3(i, j, k, n1, n2)] + h_neglect;
            b_denom_1 = h_eff + dt * (a_k * d1);
            b1 = 1.0 / (b_denom_1 + dt * a_k1);
            d1 = b_denom_1 * b1;
            visc_rem_v[IDX3(i, j, k, n1, n2)] = (h_eff +
                dt * a_k * visc_rem_v[IDX3(i, j, k - 1, n1, n2)]) * b1;
            v[IDX3(i, j, k, n1, n2)] = (h_eff * v[IDX3(i, j, k, n1, n2)] +
                dt * a_k * v[IDX3(i, j, k - 1, n1, n2)]) * b1;
        }

        // Combined back substitution (c1 stored in z_i)
        for (int k = nz - 1; k >= 1; k--) {
            visc_rem_v[IDX3(i, j, k, n1, n2)] = visc_rem_v[IDX3(i, j, k, n1, n2)] +
                z_i[IDX3(i, j, k + 1, n1, n2)] * visc_rem_v[IDX3(i, j, k + 1, n1, n2)];
            v[IDX3(i, j, k, n1, n2)] = v[IDX3(i, j, k, n1, n2)] +
                z_i[IDX3(i, j, k + 1, n1, n2)] * v[IDX3(i, j, k + 1, n1, n2)];
        }

        // Bottom stress (a_k1 is a_col(nz+1) after the loop)
        tauy_bot[IDX2(i, j, n1)] = RHO_0 * v[IDX3(i, j, nz, n1, n2)] * a_k1;

    } else {
        // Masked points: remnant = 1, stress = 0
        for (int k = 1; k <= nz; k++) {
            visc_rem_v[IDX3(i, j, k, n1, n2)] = 1.0;
        }
        tauy_bot[IDX2(i, j, n1)] = 0.0;
    }
}

// =========================================================================
// Extern "C" launch wrappers (called from Fortran via iso_c_binding)
// =========================================================================

extern "C" void launch_vert_visc_cra_u_kernel(
    double* u, double* h, double* mask2dCu,
    double* visc_rem_u, double* taux_bot, double* taux,
    double Kv, double Kv_ml, double Kv_extra_bbl,
    double Hmix, double Hbbl,
    double dt, double dt_Rho0, double h_neglect, double I_Hbbl,
    int n1, int n2, int nz,
    int is_l, int ie_l, int js_l, int je_l,
    double* hvel, double* z_i,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    vert_visc_cra_u_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        u, h, mask2dCu, visc_rem_u, taux_bot, taux,
        Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl,
        dt, dt_Rho0, h_neglect, I_Hbbl,
        n1, n2, nz, is_l, ie_l, js_l, je_l,
        hvel, z_i);
}

extern "C" void launch_vert_visc_cra_v_kernel(
    double* v, double* h, double* mask2dCv,
    double* visc_rem_v, double* tauy_bot, double* tauy,
    double Kv, double Kv_ml, double Kv_extra_bbl,
    double Hmix, double Hbbl,
    double dt, double dt_Rho0, double h_neglect, double I_Hbbl,
    int n1, int n2, int nz,
    int is_l, int ie_l, int js_l, int je_l,
    double* hvel, double* z_i,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    vert_visc_cra_v_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        v, h, mask2dCv, visc_rem_v, tauy_bot, tauy,
        Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl,
        dt, dt_Rho0, h_neglect, I_Hbbl,
        n1, n2, nz, is_l, ie_l, js_l, je_l,
        hvel, z_i);
}
