// MOM6 Barotropic Solver — CUDA C kernels
//
// 1:1 port of every attributes(global) kernel from
// mom6_barotropic_cuda.F90.  Uses 1-based column-major indexing macros
// so that array accesses are bit-identical to the Fortran CUDA version.
//
// All kernels are 2D (no k-dimension).

#include "mom6_cuda_common.h"
#include <cuda_runtime.h>

// =========================================================================
// Kernel 1: bt_init_kernel_c
// Copies eta_in, ubt_in, vbt_in into state arrays over the full domain
// and zeros the accumulation arrays.
// Range: full domain (1:n1, 1:n2)
// =========================================================================
__global__ void bt_init_kernel_c(
    double* __restrict__ eta,
    double* __restrict__ ubt,
    double* __restrict__ vbt,
    double* __restrict__ ubt_av,
    double* __restrict__ vbt_av,
    double* __restrict__ uhbt_av,
    double* __restrict__ vhbt_av,
    const double* __restrict__ eta_in,
    const double* __restrict__ ubt_in,
    const double* __restrict__ vbt_in,
    int n1, int n2)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;

    eta[IDX2(i, j, n1)]     = eta_in[IDX2(i, j, n1)];
    ubt[IDX2(i, j, n1)]     = ubt_in[IDX2(i, j, n1)];
    vbt[IDX2(i, j, n1)]     = vbt_in[IDX2(i, j, n1)];
    ubt_av[IDX2(i, j, n1)]  = 0.0;
    vbt_av[IDX2(i, j, n1)]  = 0.0;
    uhbt_av[IDX2(i, j, n1)] = 0.0;
    vhbt_av[IDX2(i, j, n1)] = 0.0;
}

// =========================================================================
// Kernel 2: bt_store_prev_u_kernel_c
// Stores ubt to ubt_prev for transport weighting.
// Range: i in [is_l-1 : ie_l+1], j in [js_l : je_l]
// =========================================================================
__global__ void bt_store_prev_u_kernel_c(
    double* __restrict__ ubt_prev,
    const double* __restrict__ ubt,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l - 1 || i > ie_l + 1 || j < js_l || j > je_l) return;

    ubt_prev[IDX2(i, j, n1)] = ubt[IDX2(i, j, n1)];
}

// =========================================================================
// Kernel 3: bt_store_prev_v_kernel_c
// Stores vbt to vbt_prev for transport weighting.
// Range: i in [is_l : ie_l], j in [js_l-1 : je_l+1]
// =========================================================================
__global__ void bt_store_prev_v_kernel_c(
    double* __restrict__ vbt_prev,
    const double* __restrict__ vbt,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l || i > ie_l || j < js_l - 1 || j > je_l + 1) return;

    vbt_prev[IDX2(i, j, n1)] = vbt[IDX2(i, j, n1)];
}

// =========================================================================
// Kernel 4: bt_eta_pred_kernel_c
// Eta predictor from velocity divergence.
// Range: i in [is_l : ie_l], j in [js_l : je_l]
// =========================================================================
__global__ void bt_eta_pred_kernel_c(
    double* __restrict__ eta_pred,
    const double* __restrict__ eta,
    const double* __restrict__ ubt,
    const double* __restrict__ vbt,
    const double* __restrict__ Datu,
    const double* __restrict__ Datv,
    const double* __restrict__ IareaT,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l,
    double dtbt)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l || i > ie_l || j < js_l || j > je_l) return;

    eta_pred[IDX2(i, j, n1)] = eta[IDX2(i, j, n1)] + (dtbt * IareaT[IDX2(i, j, n1)]) *
        ((Datu[IDX2(i - 1, j, n1)] * ubt[IDX2(i - 1, j, n1)] - Datu[IDX2(i, j, n1)] * ubt[IDX2(i, j, n1)]) +
         (Datv[IDX2(i, j - 1, n1)] * vbt[IDX2(i, j - 1, n1)] - Datv[IDX2(i, j, n1)] * vbt[IDX2(i, j, n1)]));
}

// =========================================================================
// Kernel 5: bt_pressure_force_u_kernel_c
// Pressure force at u-points.
// Range: i in [is_l : ie_l-1], j in [js_l : je_l]
// =========================================================================
__global__ void bt_pressure_force_u_kernel_c(
    double* __restrict__ PFu,
    const double* __restrict__ eta_pred,
    const double* __restrict__ gtot_E,
    const double* __restrict__ gtot_W,
    const double* __restrict__ IdxCu,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l,
    double dgeo_de)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l || i > ie_l - 1 || j < js_l || j > je_l) return;

    PFu[IDX2(i, j, n1)] = (eta_pred[IDX2(i, j, n1)] * gtot_E[IDX2(i, j, n1)] -
                            eta_pred[IDX2(i + 1, j, n1)] * gtot_W[IDX2(i + 1, j, n1)]) *
                           dgeo_de * IdxCu[IDX2(i, j, n1)];
}

// =========================================================================
// Kernel 6: bt_pressure_force_v_kernel_c
// Pressure force at v-points.
// Range: i in [is_l : ie_l], j in [js_l : je_l-1]
// =========================================================================
__global__ void bt_pressure_force_v_kernel_c(
    double* __restrict__ PFv,
    const double* __restrict__ eta_pred,
    const double* __restrict__ gtot_N,
    const double* __restrict__ gtot_S,
    const double* __restrict__ IdyCv,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l,
    double dgeo_de)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l || i > ie_l || j < js_l || j > je_l - 1) return;

    PFv[IDX2(i, j, n1)] = (eta_pred[IDX2(i, j, n1)] * gtot_N[IDX2(i, j, n1)] -
                            eta_pred[IDX2(i, j + 1, n1)] * gtot_S[IDX2(i, j + 1, n1)]) *
                           dgeo_de * IdyCv[IDX2(i, j, n1)];
}

// =========================================================================
// Kernel 7: bt_coriolis_update_u_kernel_c
// Coriolis update for u-points: computes Cor_u from vbt and f_4_u,
// then updates ubt.
// Range: i in [is_l : ie_l-1], j in [js_l : je_l]
//
// CRITICAL: f_4_u has shape (4, n1, n2) in Fortran, i.e. leading dim = 4.
// Access: f_4_u(c, i, j) -> f_4_u[IDX3(c, i, j, 4, n1)]
// =========================================================================
__global__ void bt_coriolis_update_u_kernel_c(
    double* __restrict__ ubt,
    double* __restrict__ Cor_u,
    const double* __restrict__ vbt,
    const double* __restrict__ PFu,
    const double* __restrict__ f_4_u,
    const double* __restrict__ bt_rem_u,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l,
    double dtbt)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l || i > ie_l - 1 || j < js_l || j > je_l) return;

    Cor_u[IDX2(i, j, n1)] = (f_4_u[IDX3(4, i, j, 4, n1)] * vbt[IDX2(i + 1, j, n1)]     +
                              f_4_u[IDX3(1, i, j, 4, n1)] * vbt[IDX2(i, j - 1, n1)])    +
                             (f_4_u[IDX3(3, i, j, 4, n1)] * vbt[IDX2(i, j, n1)]         +
                              f_4_u[IDX3(2, i, j, 4, n1)] * vbt[IDX2(i + 1, j - 1, n1)]);

    ubt[IDX2(i, j, n1)] = bt_rem_u[IDX2(i, j, n1)] * (ubt[IDX2(i, j, n1)] + dtbt * (Cor_u[IDX2(i, j, n1)] + PFu[IDX2(i, j, n1)]));
}

// =========================================================================
// Kernel 8: bt_coriolis_update_v_kernel_c
// Coriolis update for v-points: computes Cor_v from ubt and f_4_v,
// then updates vbt.
// Range: i in [is_l : ie_l], j in [js_l : je_l-1]
//
// CRITICAL: f_4_v has shape (4, n1, n2) in Fortran, i.e. leading dim = 4.
// Access: f_4_v(c, i, j) -> f_4_v[IDX3(c, i, j, 4, n1)]
// =========================================================================
__global__ void bt_coriolis_update_v_kernel_c(
    double* __restrict__ vbt,
    double* __restrict__ Cor_v,
    const double* __restrict__ ubt,
    const double* __restrict__ PFv,
    const double* __restrict__ f_4_v,
    const double* __restrict__ bt_rem_v,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l,
    double dtbt)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l || i > ie_l || j < js_l || j > je_l - 1) return;

    Cor_v[IDX2(i, j, n1)] = -1.0 *
                             ((f_4_v[IDX3(1, i, j, 4, n1)] * ubt[IDX2(i - 1, j, n1)]     +
                               f_4_v[IDX3(4, i, j, 4, n1)] * ubt[IDX2(i, j + 1, n1)])    +
                              (f_4_v[IDX3(2, i, j, 4, n1)] * ubt[IDX2(i, j, n1)]         +
                               f_4_v[IDX3(3, i, j, 4, n1)] * ubt[IDX2(i - 1, j + 1, n1)]));

    vbt[IDX2(i, j, n1)] = bt_rem_v[IDX2(i, j, n1)] * (vbt[IDX2(i, j, n1)] + dtbt * (Cor_v[IDX2(i, j, n1)] + PFv[IDX2(i, j, n1)]));
}

// =========================================================================
// Kernel 9: bt_transport_eta_accum_kernel_c
// Compute transports (uhbt, vhbt) and accumulate time averages.
// Extended ranges for uhbt/vhbt for MPI boundary correctness.
//
// uhbt range: i in [is_l-1 : ie_l], j in [js_l : je_l]
// vhbt range: i in [is_l : ie_l], j in [js_l-1 : je_l]
// ubt_av accumulation: same range as uhbt
// vbt_av accumulation: same range as vhbt
// =========================================================================
__global__ void bt_transport_eta_accum_kernel_c(
    double* __restrict__ uhbt,
    double* __restrict__ vhbt,
    double* __restrict__ eta,
    double* __restrict__ ubt_av,
    double* __restrict__ vbt_av,
    double* __restrict__ uhbt_av,
    double* __restrict__ vhbt_av,
    const double* __restrict__ ubt,
    const double* __restrict__ vbt,
    const double* __restrict__ ubt_prev,
    const double* __restrict__ vbt_prev,
    const double* __restrict__ Datu,
    const double* __restrict__ Datv,
    const double* __restrict__ IareaT,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l,
    double dtbt, double trans_wt1, double trans_wt2, double inv_nstep)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;

    // Compute u-transport: i in [is_l-1 : ie_l], j in [js_l : je_l]
    if (i >= is_l - 1 && i <= ie_l && j >= js_l && j <= je_l) {
        uhbt[IDX2(i, j, n1)] = Datu[IDX2(i, j, n1)] * (trans_wt1 * ubt[IDX2(i, j, n1)] + trans_wt2 * ubt_prev[IDX2(i, j, n1)]);
    }

    // Compute v-transport: i in [is_l : ie_l], j in [js_l-1 : je_l]
    if (i >= is_l && i <= ie_l && j >= js_l - 1 && j <= je_l) {
        vhbt[IDX2(i, j, n1)] = Datv[IDX2(i, j, n1)] * (trans_wt1 * vbt[IDX2(i, j, n1)] + trans_wt2 * vbt_prev[IDX2(i, j, n1)]);
    }

    // Accumulate u time averages: same range as uhbt
    if (i >= is_l - 1 && i <= ie_l && j >= js_l && j <= je_l) {
        ubt_av[IDX2(i, j, n1)]  = ubt_av[IDX2(i, j, n1)]  + ubt[IDX2(i, j, n1)] * inv_nstep;
        uhbt_av[IDX2(i, j, n1)] = uhbt_av[IDX2(i, j, n1)] + uhbt[IDX2(i, j, n1)] * inv_nstep;
    }

    // Accumulate v time averages: same range as vhbt
    if (i >= is_l && i <= ie_l && j >= js_l - 1 && j <= je_l) {
        vbt_av[IDX2(i, j, n1)]  = vbt_av[IDX2(i, j, n1)]  + vbt[IDX2(i, j, n1)] * inv_nstep;
        vhbt_av[IDX2(i, j, n1)] = vhbt_av[IDX2(i, j, n1)] + vhbt[IDX2(i, j, n1)] * inv_nstep;
    }
}

// =========================================================================
// Kernel 10: bt_eta_update_kernel_c
// Update eta from divergence of transports (uhbt, vhbt).
// Must be launched after bt_transport_eta_accum_kernel_c has completed
// (implicit sync between kernel launches on the same stream).
// Range: i in [is_l : ie_l], j in [js_l : je_l]
// =========================================================================
__global__ void bt_eta_update_kernel_c(
    double* __restrict__ eta,
    const double* __restrict__ uhbt,
    const double* __restrict__ vhbt,
    const double* __restrict__ IareaT,
    int n1, int n2,
    int is_l, int ie_l, int js_l, int je_l,
    double dtbt)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;
    if (i < is_l || i > ie_l || j < js_l || j > je_l) return;

    eta[IDX2(i, j, n1)] = eta[IDX2(i, j, n1)] - dtbt * IareaT[IDX2(i, j, n1)] *
        ((uhbt[IDX2(i, j, n1)] - uhbt[IDX2(i - 1, j, n1)]) + (vhbt[IDX2(i, j, n1)] - vhbt[IDX2(i, j - 1, n1)]));
}

// =========================================================================
// Kernel 11: bt_copy_output_kernel_c
// Copy final output arrays over the full domain.
// Range: full domain (1:n1, 1:n2)
// =========================================================================
__global__ void bt_copy_output_kernel_c(
    double* __restrict__ u_av_out,
    double* __restrict__ v_av_out,
    double* __restrict__ eta_out,
    const double* __restrict__ ubt_av,
    const double* __restrict__ vbt_av,
    const double* __restrict__ eta,
    int n1, int n2)
{
    int i = THREAD_I;
    int j = THREAD_J;

    if (i > n1 || j > n2) return;

    u_av_out[IDX2(i, j, n1)] = ubt_av[IDX2(i, j, n1)];
    v_av_out[IDX2(i, j, n1)] = vbt_av[IDX2(i, j, n1)];
    eta_out[IDX2(i, j, n1)]  = eta[IDX2(i, j, n1)];
}


// =========================================================================
// Extern "C" launch wrappers (called from Fortran via iso_c_binding)
// All use 2D grid (no z dimension).
// =========================================================================

extern "C" void launch_bt_init_kernel(
    double* eta, double* ubt, double* vbt,
    double* ubt_av, double* vbt_av, double* uhbt_av, double* vhbt_av,
    double* eta_in, double* ubt_in, double* vbt_in,
    int n1, int n2,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_init_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        eta, ubt, vbt, ubt_av, vbt_av, uhbt_av, vhbt_av,
        eta_in, ubt_in, vbt_in,
        n1, n2);
}

extern "C" void launch_bt_store_prev_u_kernel(
    double* ubt_prev, double* ubt,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_store_prev_u_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        ubt_prev, ubt,
        n1, n2, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_bt_store_prev_v_kernel(
    double* vbt_prev, double* vbt,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_store_prev_v_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        vbt_prev, vbt,
        n1, n2, is_l, ie_l, js_l, je_l);
}

extern "C" void launch_bt_eta_pred_kernel(
    double* eta_pred, double* eta, double* ubt, double* vbt,
    double* Datu, double* Datv, double* IareaT,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    double dtbt,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_eta_pred_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        eta_pred, eta, ubt, vbt,
        Datu, Datv, IareaT,
        n1, n2, is_l, ie_l, js_l, je_l, dtbt);
}

extern "C" void launch_bt_pressure_force_u_kernel(
    double* PFu, double* eta_pred,
    double* gtot_E, double* gtot_W, double* IdxCu,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    double dgeo_de,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_pressure_force_u_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        PFu, eta_pred, gtot_E, gtot_W, IdxCu,
        n1, n2, is_l, ie_l, js_l, je_l, dgeo_de);
}

extern "C" void launch_bt_pressure_force_v_kernel(
    double* PFv, double* eta_pred,
    double* gtot_N, double* gtot_S, double* IdyCv,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    double dgeo_de,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_pressure_force_v_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        PFv, eta_pred, gtot_N, gtot_S, IdyCv,
        n1, n2, is_l, ie_l, js_l, je_l, dgeo_de);
}

extern "C" void launch_bt_coriolis_update_u_kernel(
    double* ubt, double* Cor_u, double* vbt,
    double* PFu, double* f_4_u, double* bt_rem_u,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    double dtbt,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_coriolis_update_u_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        ubt, Cor_u, vbt, PFu, f_4_u, bt_rem_u,
        n1, n2, is_l, ie_l, js_l, je_l, dtbt);
}

extern "C" void launch_bt_coriolis_update_v_kernel(
    double* vbt, double* Cor_v, double* ubt,
    double* PFv, double* f_4_v, double* bt_rem_v,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    double dtbt,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_coriolis_update_v_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        vbt, Cor_v, ubt, PFv, f_4_v, bt_rem_v,
        n1, n2, is_l, ie_l, js_l, je_l, dtbt);
}

extern "C" void launch_bt_transport_eta_accum_kernel(
    double* uhbt, double* vhbt, double* eta,
    double* ubt_av, double* vbt_av, double* uhbt_av, double* vhbt_av,
    double* ubt, double* vbt, double* ubt_prev, double* vbt_prev,
    double* Datu, double* Datv, double* IareaT,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    double dtbt, double trans_wt1, double trans_wt2, double inv_nstep,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_transport_eta_accum_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        uhbt, vhbt, eta, ubt_av, vbt_av, uhbt_av, vhbt_av,
        ubt, vbt, ubt_prev, vbt_prev,
        Datu, Datv, IareaT,
        n1, n2, is_l, ie_l, js_l, je_l,
        dtbt, trans_wt1, trans_wt2, inv_nstep);
}

extern "C" void launch_bt_eta_update_kernel(
    double* eta, double* uhbt, double* vhbt, double* IareaT,
    int n1, int n2, int is_l, int ie_l, int js_l, int je_l,
    double dtbt,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_eta_update_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        eta, uhbt, vhbt, IareaT,
        n1, n2, is_l, ie_l, js_l, je_l, dtbt);
}

extern "C" void launch_bt_copy_output_kernel(
    double* u_av_out, double* v_av_out, double* eta_out,
    double* ubt_av, double* vbt_av, double* eta,
    int n1, int n2,
    int grid_x, int grid_y,
    int block_x, int block_y,
    void* stream)
{
    dim3 grid(grid_x, grid_y, 1);
    dim3 block(block_x, block_y, 1);
    bt_copy_output_kernel_c<<<grid, block, 0, (cudaStream_t)stream>>>(
        u_av_out, v_av_out, eta_out,
        ubt_av, vbt_av, eta,
        n1, n2);
}
