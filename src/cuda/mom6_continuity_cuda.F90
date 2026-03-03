!> MOM6 Continuity PPM Solver Module (CUDA Fortran variant)
!!
!! Full-parity CUDA implementation matching OpenACC/OpenMP functionality.
!! Includes: PPM reconstruction (upwind_1st, simple_2nd, full PPM),
!! zonal flux with duhdu/por_face/visc_rem/vol_CFL support,
!! Newton iteration for barotropic transport matching,
!! CFL-bounded velocity correction, BT_cont face-area computation,
!! and flux thickness.
!!
module mom6_continuity_cuda
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: continuity_init_cuda, continuity_PPM_cuda, continuity_end_cuda
    public :: continuity_CS_cuda
    public :: BT_cont_type_cuda, alloc_BT_cont_type_cuda, dealloc_BT_cont_type_cuda

    real(dp), parameter :: oneSixth = 1.0_dp / 6.0_dp

    !> Control structure for CUDA Fortran continuity solver
    type :: continuity_CS_cuda
        logical :: initialized = .false.
        integer :: is, ie, js, je, nz
        integer :: isd, ied, jsd, jed

        !> PPM parameters
        logical :: monotonic
        logical :: upwind_1st
        logical :: simple_2nd

        !> Solver flags (match OpenACC CS)
        logical :: vol_CFL
        logical :: aggress_adjust
        logical :: better_iter
        logical :: use_visc_rem_max
        logical :: marginal_faces
        real(dp) :: tol_eta
        real(dp) :: tol_vel
        real(dp) :: CFL_limit_adjust

        !> Device 2D grid metrics (existing)
        real(dp), device, allocatable :: IareaT_d(:,:)
        real(dp), device, allocatable :: IdxT_d(:,:)
        real(dp), device, allocatable :: dy_Cu_d(:,:)
        real(dp), device, allocatable :: mask2dT_d(:,:)

        !> Device 2D grid metrics (new — for vol_CFL and BT_cont)
        real(dp), device, allocatable :: dxT_d(:,:)
        real(dp), device, allocatable :: areaT_d(:,:)
        real(dp), device, allocatable :: dxCu_d(:,:)
        real(dp), device, allocatable :: mask2dCu_d(:,:)

        !> Device 3D work arrays for PPM edge values
        real(dp), device, allocatable :: h_W(:,:,:)
        real(dp), device, allocatable :: h_E(:,:,:)

        !> Device work arrays for full solver
        real(dp), device, allocatable :: duhdu(:,:,:)
        real(dp), device, allocatable :: du(:,:)
        real(dp), device, allocatable :: du_max_CFL(:,:)
        real(dp), device, allocatable :: du_min_CFL(:,:)
        real(dp), device, allocatable :: duhdu_tot_0(:,:)
        real(dp), device, allocatable :: uh_tot_0(:,:)
        real(dp), device, allocatable :: visc_rem_max_arr(:,:)

        !> Default device arrays (for when optional args are absent)
        real(dp), device, allocatable :: por_face_areaU_def(:,:,:)
        real(dp), device, allocatable :: visc_rem_u_def(:,:,:)
    end type continuity_CS_cuda

    !> CUDA-specific BT_cont type with device arrays
    type :: BT_cont_type_cuda
        real(dp), device, allocatable :: FA_u_EE(:,:), FA_u_E0(:,:)
        real(dp), device, allocatable :: FA_u_W0(:,:), FA_u_WW(:,:)
        real(dp), device, allocatable :: uBT_WW(:,:), uBT_EE(:,:)
        real(dp), device, allocatable :: h_u(:,:,:)
    end type BT_cont_type_cuda

contains

    !=========================================================================
    ! CUDA Kernels (attributes(global))
    !=========================================================================

    !> PPM reconstruction kernel with upwind_1st, simple_2nd, and full PPM modes.
    attributes(global) subroutine ppm_reconstruction_3d_kernel( &
            h_W, h_E, h_in, mask2dT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, h_min, monotonic, &
            upwind_1st_flag, simple_2nd_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: h_W(n1, n2, n3), h_E(n1, n2, n3)
        real(dp), intent(in)  :: h_in(n1, n2, n3)
        real(dp), intent(in)  :: mask2dT(n1, n2)
        real(dp), value, intent(in) :: h_min
        integer, value, intent(in) :: monotonic
        integer, value, intent(in) :: upwind_1st_flag, simple_2nd_flag

        integer :: i, j, k
        real(dp) :: h_im1, h_ip1, h_i
        real(dp) :: dMx, dMn, slp_im1, slp_i, slp_ip1
        real(dp) :: RLdiff, RLdiff2, RLmean, FunFac
        real(dp) :: curv, dh, scale_val

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i < is_l - 1 .or. i > ie_l + 1 .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        ! ----- Upwind first-order: flat reconstruction, return early -----
        if (upwind_1st_flag == 1) then
            h_W(i, j, k) = h_in(i, j, k)
            h_E(i, j, k) = h_in(i, j, k)
            return
        end if

        ! ----- Simple second-order: arithmetic mean edges -----
        if (simple_2nd_flag == 1) then
            h_im1 = mask2dT(max(i - 1, 1), j) * h_in(max(i - 1, 1), j, k) + &
                     (1.0_dp - mask2dT(max(i - 1, 1), j)) * h_in(i, j, k)
            h_ip1 = mask2dT(min(i + 1, n1), j) * h_in(min(i + 1, n1), j, k) + &
                     (1.0_dp - mask2dT(min(i + 1, n1), j)) * h_in(i, j, k)
            h_W(i, j, k) = 0.5_dp * (h_im1 + h_in(i, j, k))
            h_E(i, j, k) = 0.5_dp * (h_ip1 + h_in(i, j, k))
        else
            ! ----- Full PPM with slopes (existing logic) -----
            ! Slope at i-1
            if (i - 2 >= 1 .and. i >= 1 .and. &
                (mask2dT(i - 2, j) * mask2dT(i - 1, j) * mask2dT(i, j)) /= 0.0_dp) then
                slp_im1 = 0.5_dp * (h_in(i, j, k) - h_in(i - 2, j, k))
                dMx = max(h_in(i, j, k), h_in(i - 2, j, k), h_in(i - 1, j, k)) - h_in(i - 1, j, k)
                dMn = h_in(i - 1, j, k) - min(h_in(i, j, k), h_in(i - 2, j, k), h_in(i - 1, j, k))
                slp_im1 = sign(1.0_dp, slp_im1) * min(abs(slp_im1), 2.0_dp * min(dMx, dMn))
            else
                slp_im1 = 0.0_dp
            end if

            ! Slope at i
            if (i - 1 >= 1 .and. i + 1 <= n1 .and. &
                (mask2dT(i - 1, j) * mask2dT(i, j) * mask2dT(i + 1, j)) /= 0.0_dp) then
                slp_i = 0.5_dp * (h_in(i + 1, j, k) - h_in(i - 1, j, k))
                dMx = max(h_in(i + 1, j, k), h_in(i - 1, j, k), h_in(i, j, k)) - h_in(i, j, k)
                dMn = h_in(i, j, k) - min(h_in(i + 1, j, k), h_in(i - 1, j, k), h_in(i, j, k))
                slp_i = sign(1.0_dp, slp_i) * min(abs(slp_i), 2.0_dp * min(dMx, dMn))
            else
                slp_i = 0.0_dp
            end if

            ! Slope at i+1
            if (i + 2 <= n1 .and. i >= 1 .and. &
                (mask2dT(i, j) * mask2dT(i + 1, j) * mask2dT(i + 2, j)) /= 0.0_dp) then
                slp_ip1 = 0.5_dp * (h_in(i + 2, j, k) - h_in(i, j, k))
                dMx = max(h_in(i + 2, j, k), h_in(i, j, k), h_in(i + 1, j, k)) - h_in(i + 1, j, k)
                dMn = h_in(i + 1, j, k) - min(h_in(i + 2, j, k), h_in(i, j, k), h_in(i + 1, j, k))
                slp_ip1 = sign(1.0_dp, slp_ip1) * min(abs(slp_ip1), 2.0_dp * min(dMx, dMn))
            else
                slp_ip1 = 0.0_dp
            end if

            ! Edge values from slopes
            h_im1 = mask2dT(max(i - 1, 1), j) * h_in(max(i - 1, 1), j, k) + &
                     (1.0_dp - mask2dT(max(i - 1, 1), j)) * h_in(i, j, k)
            h_ip1 = mask2dT(min(i + 1, n1), j) * h_in(min(i + 1, n1), j, k) + &
                     (1.0_dp - mask2dT(min(i + 1, n1), j)) * h_in(i, j, k)

            h_W(i, j, k) = 0.5_dp * (h_im1 + h_in(i, j, k)) + oneSixth * (slp_im1 - slp_i)
            h_E(i, j, k) = 0.5_dp * (h_ip1 + h_in(i, j, k)) + oneSixth * (slp_i - slp_ip1)
        end if

        ! ----- Limiter (applies to both simple_2nd and full PPM) -----
        h_i = h_in(i, j, k)

        if (monotonic == 1) then
            if ((h_E(i, j, k) - h_i) * (h_i - h_W(i, j, k)) <= 0.0_dp) then
                h_W(i, j, k) = h_i
                h_E(i, j, k) = h_i
            else
                RLdiff = h_E(i, j, k) - h_W(i, j, k)
                RLmean = 0.5_dp * (h_E(i, j, k) + h_W(i, j, k))
                FunFac = 6.0_dp * RLdiff * (h_i - RLmean)
                RLdiff2 = RLdiff * RLdiff
                if (FunFac > RLdiff2) h_W(i, j, k) = 3.0_dp * h_i - 2.0_dp * h_E(i, j, k)
                if (FunFac < -RLdiff2) h_E(i, j, k) = 3.0_dp * h_i - 2.0_dp * h_W(i, j, k)
            end if
        else
            curv = 3.0_dp * ((h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_i)
            if (curv > 0.0_dp) then
                dh = h_E(i, j, k) - h_W(i, j, k)
                if (abs(dh) < curv) then
                    if (h_i <= h_min) then
                        h_W(i, j, k) = h_i
                        h_E(i, j, k) = h_i
                    elseif (12.0_dp * curv * (h_i - h_min) < (curv**2 + 3.0_dp * dh**2)) then
                        scale_val = 12.0_dp * curv * (h_i - h_min) / (curv**2 + 3.0_dp * dh**2)
                        h_W(i, j, k) = h_i + scale_val * (h_W(i, j, k) - h_i)
                        h_E(i, j, k) = h_i + scale_val * (h_E(i, j, k) - h_i)
                    end if
                end if
            end if
        end if

    end subroutine ppm_reconstruction_3d_kernel

    !> Expanded zonal flux kernel: computes uh and duhdu with por_face_areaU, visc_rem, vol_CFL.
    attributes(global) subroutine zonal_flux_layer_3d_kernel( &
            uh, duhdu, u, h_in, h_W, h_E, dy_Cu, IdxT, IareaT, &
            por_face_areaU, visc_rem_u, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
            vol_CFL_flag, use_visc_rem_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: uh(n1, n2, n3), duhdu(n1, n2, n3)
        real(dp), intent(in)  :: u(n1, n2, n3), h_in(n1, n2, n3)
        real(dp), intent(in)  :: h_W(n1, n2, n3), h_E(n1, n2, n3)
        real(dp), intent(in)  :: por_face_areaU(n1, n2, n3), visc_rem_u(n1, n2, n3)
        real(dp), intent(in)  :: dy_Cu(n1, n2), IdxT(n1, n2), IareaT(n1, n2)
        real(dp), value, intent(in) :: dt
        integer, value, intent(in) :: vol_CFL_flag, use_visc_rem_flag

        integer :: i, j, k
        real(dp) :: CFL, curv_3, h_marg, visc_rem_val

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        if (use_visc_rem_flag == 1) then
            visc_rem_val = visc_rem_u(i, j, k)
        else
            visc_rem_val = 1.0_dp
        end if

        if (u(i, j, k) > 0.0_dp) then
            if (vol_CFL_flag == 1) then
                CFL = (u(i, j, k) * dt) * (dy_Cu(i, j) * IareaT(i, j))
            else
                CFL = u(i, j, k) * dt * IdxT(i, j)
            end if
            curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
            uh(i, j, k) = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u(i, j, k) * &
                (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                 curv_3 * (CFL - 1.5_dp)))
            h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + &
                     3.0_dp * curv_3 * (CFL - 1.0_dp))
        elseif (u(i, j, k) < 0.0_dp) then
            if (vol_CFL_flag == 1) then
                CFL = (-u(i, j, k) * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
            else
                CFL = -u(i, j, k) * dt * IdxT(i + 1, j)
            end if
            curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
            uh(i, j, k) = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u(i, j, k) * &
                (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                 curv_3 * (CFL - 1.5_dp)))
            h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                     3.0_dp * curv_3 * (CFL - 1.0_dp))
        else
            uh(i, j, k) = 0.0_dp
            h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
        end if

        duhdu(i, j, k) = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * h_marg * visc_rem_val

    end subroutine zonal_flux_layer_3d_kernel

    !> Zonal convergence kernel (unchanged).
    attributes(global) subroutine zonal_convergence_kernel( &
            h, hin, uh, IareaT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: h(n1, n2, n3)
        real(dp), intent(in)  :: hin(n1, n2, n3)
        real(dp), intent(in)  :: uh(n1, n2, n3)
        real(dp), intent(in)  :: IareaT(n1, n2)
        real(dp), value, intent(in) :: dt

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i < is_l .or. i > ie_l .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        h(i, j, k) = max(hin(i, j, k) - dt * IareaT(i, j) * &
                     (uh(i, j, k) - uh(i - 1, j, k)), 0.0_dp)

    end subroutine zonal_convergence_kernel

    !> Compute column-max of visc_rem_u. Collapse(2): one thread per (I,j).
    attributes(global) subroutine visc_rem_max_kernel( &
            visc_rem_max_out, visc_rem_u, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, use_vrm_max_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: visc_rem_max_out(n1, n2)
        real(dp), intent(in)  :: visc_rem_u(n1, n2, n3)
        integer, value, intent(in) :: use_vrm_max_flag

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l) return

        if (use_vrm_max_flag == 1) then
            visc_rem_max_out(i, j) = 0.0_dp
            do k = 1, n3
                visc_rem_max_out(i, j) = max(visc_rem_max_out(i, j), visc_rem_u(i, j, k))
            end do
        else
            visc_rem_max_out(i, j) = 1.0_dp
        end if

    end subroutine visc_rem_max_kernel

    !> Compute column sums of uh and duhdu. Collapse(2): one thread per (I,j).
    attributes(global) subroutine uh_duhdu_tot_kernel( &
            uh_tot_0, duhdu_tot_0, uh, duhdu, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: uh_tot_0(n1, n2), duhdu_tot_0(n1, n2)
        real(dp), intent(in)  :: uh(n1, n2, n3), duhdu(n1, n2, n3)

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l) return

        uh_tot_0(i, j) = 0.0_dp
        duhdu_tot_0(i, j) = 0.0_dp
        do k = 1, n3
            uh_tot_0(i, j) = uh_tot_0(i, j) + uh(i, j, k)
            duhdu_tot_0(i, j) = duhdu_tot_0(i, j) + duhdu(i, j, k)
        end do

    end subroutine uh_duhdu_tot_kernel

    !> Compute CFL-based limits on du correction. Collapse(2): one thread per (I,j).
    attributes(global) subroutine zonal_CFL_limits_kernel( &
            du_max_CFL_out, du_min_CFL_out, u, visc_rem_u, visc_rem_max_in, &
            dxT, areaT, dy_Cu, mask2dCu, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            CFL_dt, I_dt, vol_CFL_flag, use_visc_rem_flag, aggress_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: du_max_CFL_out(n1, n2), du_min_CFL_out(n1, n2)
        real(dp), intent(in)  :: u(n1, n2, n3), visc_rem_u(n1, n2, n3)
        real(dp), intent(in)  :: visc_rem_max_in(n1, n2)
        real(dp), intent(in)  :: dxT(n1, n2), areaT(n1, n2), dy_Cu(n1, n2), mask2dCu(n1, n2)
        real(dp), value, intent(in) :: CFL_dt, I_dt
        integer, value, intent(in) :: vol_CFL_flag, use_visc_rem_flag, aggress_flag

        integer :: i, j, k
        real(dp) :: I_vrm, dx_W, dx_E, du_lim, vrm_k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l) return

        ! Initial CFL limits from visc_rem_max
        I_vrm = 0.0_dp
        if (visc_rem_max_in(i, j) > 0.0_dp) I_vrm = 1.0_dp / visc_rem_max_in(i, j)
        if (vol_CFL_flag == 1) then
            dx_W = areaT(i, j) / (dy_Cu(i, j) + 1.0e-30_dp)
            if (abs(dx_W) > 1000.0_dp * dxT(i, j)) dx_W = 1000.0_dp * dxT(i, j)
            dx_E = areaT(i + 1, j) / (dy_Cu(i, j) + 1.0e-30_dp)
            if (abs(dx_E) > 1000.0_dp * dxT(i + 1, j)) dx_E = 1000.0_dp * dxT(i + 1, j)
        else
            dx_W = dxT(i, j)
            dx_E = dxT(i + 1, j)
        end if
        du_max_CFL_out(i, j) = 2.0_dp * (CFL_dt * dx_W) * I_vrm
        du_min_CFL_out(i, j) = -2.0_dp * (CFL_dt * dx_E) * I_vrm

        ! Tighten CFL limits over k
        if (use_visc_rem_flag == 1) then
            if (aggress_flag == 1) then
                do k = 1, n3
                    vrm_k = visc_rem_u(i, j, k)
                    if (vol_CFL_flag == 1) then
                        dx_W = areaT(i, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_W) > 1000.0_dp * dxT(i, j)) dx_W = 1000.0_dp * dxT(i, j)
                        dx_E = areaT(i + 1, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_E) > 1000.0_dp * dxT(i + 1, j)) dx_E = 1000.0_dp * dxT(i + 1, j)
                    else
                        dx_W = dxT(i, j)
                        dx_E = dxT(i + 1, j)
                    end if
                    du_lim = 0.499_dp * ((dx_W * I_dt - u(i, j, k)) + min(0.0_dp, u(i - 1, j, k)))
                    if (du_max_CFL_out(i, j) * vrm_k > du_lim) &
                        du_max_CFL_out(i, j) = du_lim / vrm_k
                    du_lim = 0.499_dp * ((-dx_E * I_dt - u(i, j, k)) + max(0.0_dp, u(i + 1, j, k)))
                    if (du_min_CFL_out(i, j) * vrm_k < du_lim) &
                        du_min_CFL_out(i, j) = du_lim / vrm_k
                end do
            else
                do k = 1, n3
                    vrm_k = visc_rem_u(i, j, k)
                    if (vol_CFL_flag == 1) then
                        dx_W = areaT(i, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_W) > 1000.0_dp * dxT(i, j)) dx_W = 1000.0_dp * dxT(i, j)
                        dx_E = areaT(i + 1, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_E) > 1000.0_dp * dxT(i + 1, j)) dx_E = 1000.0_dp * dxT(i + 1, j)
                    else
                        dx_W = dxT(i, j)
                        dx_E = dxT(i + 1, j)
                    end if
                    if (du_max_CFL_out(i, j) * vrm_k > dx_W * CFL_dt - u(i, j, k) * mask2dCu(i, j)) &
                        du_max_CFL_out(i, j) = (dx_W * CFL_dt - u(i, j, k)) / vrm_k
                    if (du_min_CFL_out(i, j) * vrm_k < -dx_E * CFL_dt - u(i, j, k) * mask2dCu(i, j)) &
                        du_min_CFL_out(i, j) = -(dx_E * CFL_dt + u(i, j, k)) / vrm_k
                end do
            end if
        else
            if (aggress_flag == 1) then
                do k = 1, n3
                    if (vol_CFL_flag == 1) then
                        dx_W = areaT(i, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_W) > 1000.0_dp * dxT(i, j)) dx_W = 1000.0_dp * dxT(i, j)
                        dx_E = areaT(i + 1, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_E) > 1000.0_dp * dxT(i + 1, j)) dx_E = 1000.0_dp * dxT(i + 1, j)
                    else
                        dx_W = dxT(i, j)
                        dx_E = dxT(i + 1, j)
                    end if
                    du_max_CFL_out(i, j) = min(du_max_CFL_out(i, j), 0.499_dp * &
                        ((dx_W * I_dt - u(i, j, k)) + min(0.0_dp, u(i - 1, j, k))))
                    du_min_CFL_out(i, j) = max(du_min_CFL_out(i, j), 0.499_dp * &
                        ((-dx_E * I_dt - u(i, j, k)) + max(0.0_dp, u(i + 1, j, k))))
                end do
            else
                do k = 1, n3
                    if (vol_CFL_flag == 1) then
                        dx_W = areaT(i, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_W) > 1000.0_dp * dxT(i, j)) dx_W = 1000.0_dp * dxT(i, j)
                        dx_E = areaT(i + 1, j) / (dy_Cu(i, j) + 1.0e-30_dp)
                        if (abs(dx_E) > 1000.0_dp * dxT(i + 1, j)) dx_E = 1000.0_dp * dxT(i + 1, j)
                    else
                        dx_W = dxT(i, j)
                        dx_E = dxT(i + 1, j)
                    end if
                    du_max_CFL_out(i, j) = min(du_max_CFL_out(i, j), dx_W * CFL_dt - u(i, j, k))
                    du_min_CFL_out(i, j) = max(du_min_CFL_out(i, j), -(dx_E * CFL_dt + u(i, j, k)))
                end do
            end if
        end if

        ! Ensure bounds include 0
        du_max_CFL_out(i, j) = max(du_max_CFL_out(i, j), 0.0_dp)
        du_min_CFL_out(i, j) = min(du_min_CFL_out(i, j), 0.0_dp)

    end subroutine zonal_CFL_limits_kernel

    !> Newton iteration to adjust zonal fluxes to match barotropic transport.
    !! Collapse(2): one thread per (I,j), independent Newton iteration with serial k-loops.
    attributes(global) subroutine zonal_flux_adjust_kernel( &
            uh, du_out, u, h_in, h_W, h_E, uhbt, &
            visc_rem_u, por_face_areaU, &
            IareaT, dy_Cu, IdxT, &
            du_max_CFL_in, du_min_CFL_in, uh_tot_0_in, duhdu_tot_0_in, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
            tol_eta_base, tol_vel_val, vol_CFL_flag, use_visc_rem_flag, better_iter_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(inout) :: uh(n1, n2, n3)
        real(dp), intent(out)   :: du_out(n1, n2)
        real(dp), intent(in)    :: u(n1, n2, n3), h_in(n1, n2, n3)
        real(dp), intent(in)    :: h_W(n1, n2, n3), h_E(n1, n2, n3)
        real(dp), intent(in)    :: uhbt(n1, n2)
        real(dp), intent(in)    :: visc_rem_u(n1, n2, n3), por_face_areaU(n1, n2, n3)
        real(dp), intent(in)    :: IareaT(n1, n2), dy_Cu(n1, n2), IdxT(n1, n2)
        real(dp), intent(in)    :: du_max_CFL_in(n1, n2), du_min_CFL_in(n1, n2)
        real(dp), intent(in)    :: uh_tot_0_in(n1, n2), duhdu_tot_0_in(n1, n2)
        real(dp), value, intent(in) :: dt, tol_eta_base, tol_vel_val
        integer, value, intent(in) :: vol_CFL_flag, use_visc_rem_flag, better_iter_flag

        integer :: i, j, k, itt
        integer, parameter :: max_itts = 20
        real(dp) :: du_val, du_prev, ddu, uh_err_val, uh_err_best_val
        real(dp) :: duhdu_tot_val, du_max_val, du_min_val
        real(dp) :: tol_eta, tol_vel
        real(dp) :: CFL, curv_3, h_marg, u_adj, uh_k, duhdu_k, visc_rem_val
        logical :: do_more

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l) return

        du_val = 0.0_dp
        du_max_val = du_max_CFL_in(i, j)
        du_min_val = du_min_CFL_in(i, j)
        uh_err_val = uh_tot_0_in(i, j) - uhbt(i, j)
        duhdu_tot_val = duhdu_tot_0_in(i, j)
        uh_err_best_val = abs(uh_err_val)
        do_more = .true.

        do itt = 1, max_itts
            if (itt <= 1) then
                tol_eta = 1.0e-6_dp * tol_eta_base
            elseif (itt == 2) then
                tol_eta = 1.0e-4_dp * tol_eta_base
            elseif (itt == 3) then
                tol_eta = 1.0e-2_dp * tol_eta_base
            else
                tol_eta = tol_eta_base
            end if
            tol_vel = tol_vel_val

            if (uh_err_val > 0.0_dp) then
                du_max_val = du_val
            elseif (uh_err_val < 0.0_dp) then
                du_min_val = du_val
            else
                do_more = .false.
            end if

            if (do_more) then
                if ((dt * min(IareaT(i, j), IareaT(i + 1, j)) * abs(uh_err_val) > tol_eta) .or. &
                    (better_iter_flag == 1 .and. ((abs(uh_err_val) > tol_vel * duhdu_tot_val) .or. &
                                                   (abs(uh_err_val) > uh_err_best_val)))) then
                    ddu = -uh_err_val / duhdu_tot_val
                    du_prev = du_val
                    du_val = du_val + ddu
                    if (abs(ddu) < 1.0e-15_dp * abs(du_val)) then
                        do_more = .false.
                    elseif (ddu > 0.0_dp) then
                        if (du_val >= du_max_val) then
                            du_val = 0.5_dp * (du_prev + du_max_val)
                            if (du_max_val - du_prev < 1.0e-15_dp * abs(du_val)) do_more = .false.
                        end if
                    else
                        if (du_val <= du_min_val) then
                            du_val = 0.5_dp * (du_prev + du_min_val)
                            if (du_prev - du_min_val < 1.0e-15_dp * abs(du_val)) do_more = .false.
                        end if
                    end if
                else
                    do_more = .false.
                end if
            end if

            if (.not. do_more) exit

            ! Recompute flux with adjusted velocity
            uh_err_val = -uhbt(i, j)
            duhdu_tot_val = 0.0_dp
            do k = 1, n3
                if (use_visc_rem_flag == 1) then
                    visc_rem_val = visc_rem_u(i, j, k)
                else
                    visc_rem_val = 1.0_dp
                end if
                u_adj = u(i, j, k) + du_val * visc_rem_val

                if (u_adj > 0.0_dp) then
                    if (vol_CFL_flag == 1) then
                        CFL = (u_adj * dt) * (dy_Cu(i, j) * IareaT(i, j))
                    else
                        CFL = u_adj * dt * IdxT(i, j)
                    end if
                    curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                    uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                        (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                         curv_3 * (CFL - 1.5_dp)))
                    h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + &
                             3.0_dp * curv_3 * (CFL - 1.0_dp))
                elseif (u_adj < 0.0_dp) then
                    if (vol_CFL_flag == 1) then
                        CFL = (-u_adj * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
                    else
                        CFL = -u_adj * dt * IdxT(i + 1, j)
                    end if
                    curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                    uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                        (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                         curv_3 * (CFL - 1.5_dp)))
                    h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                             3.0_dp * curv_3 * (CFL - 1.0_dp))
                else
                    uh_k = 0.0_dp
                    h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                end if
                duhdu_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * h_marg * visc_rem_val
                uh_err_val = uh_err_val + uh_k
                duhdu_tot_val = duhdu_tot_val + duhdu_k
            end do
            uh_err_best_val = min(uh_err_best_val, abs(uh_err_val))
        end do ! Newton iterations

        ! Final pass: write converged uh values
        do k = 1, n3
            if (use_visc_rem_flag == 1) then
                visc_rem_val = visc_rem_u(i, j, k)
            else
                visc_rem_val = 1.0_dp
            end if
            u_adj = u(i, j, k) + du_val * visc_rem_val

            if (u_adj > 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (u_adj * dt) * (dy_Cu(i, j) * IareaT(i, j))
                else
                    CFL = u_adj * dt * IdxT(i, j)
                end if
                curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                uh(i, j, k) = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                    (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                     curv_3 * (CFL - 1.5_dp)))
            elseif (u_adj < 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (-u_adj * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
                else
                    CFL = -u_adj * dt * IdxT(i + 1, j)
                end if
                curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                uh(i, j, k) = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                    (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                     curv_3 * (CFL - 1.5_dp)))
            else
                uh(i, j, k) = 0.0_dp
            end if
        end do

        du_out(i, j) = du_val

    end subroutine zonal_flux_adjust_kernel

    !> Compute BT_cont face areas via Newton iteration + 3 test velocities.
    !! Collapse(2): one thread per (I,j).
    attributes(global) subroutine set_zonal_BT_cont_kernel( &
            FA_u_W0, FA_u_WW, uBT_WW, FA_u_E0, FA_u_EE, uBT_EE, &
            u, h_in, h_W, h_E, visc_rem_u, por_face_areaU, &
            IareaT, dy_Cu, IdxT, dxCu, visc_rem_max_in, &
            du_max_CFL_in, du_min_CFL_in, uh_tot_0_in, duhdu_tot_0_in, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
            tol_eta_base, tol_vel_val, vol_CFL_flag, use_visc_rem_flag, better_iter_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out)   :: FA_u_W0(n1, n2), FA_u_WW(n1, n2), uBT_WW(n1, n2)
        real(dp), intent(out)   :: FA_u_E0(n1, n2), FA_u_EE(n1, n2), uBT_EE(n1, n2)
        real(dp), intent(in)    :: u(n1, n2, n3), h_in(n1, n2, n3)
        real(dp), intent(in)    :: h_W(n1, n2, n3), h_E(n1, n2, n3)
        real(dp), intent(in)    :: visc_rem_u(n1, n2, n3), por_face_areaU(n1, n2, n3)
        real(dp), intent(in)    :: IareaT(n1, n2), dy_Cu(n1, n2), IdxT(n1, n2), dxCu(n1, n2)
        real(dp), intent(in)    :: visc_rem_max_in(n1, n2)
        real(dp), intent(in)    :: du_max_CFL_in(n1, n2), du_min_CFL_in(n1, n2)
        real(dp), intent(in)    :: uh_tot_0_in(n1, n2), duhdu_tot_0_in(n1, n2)
        real(dp), value, intent(in) :: dt, tol_eta_base, tol_vel_val
        integer, value, intent(in) :: vol_CFL_flag, use_visc_rem_flag, better_iter_flag

        integer :: i, j, k, itt
        integer, parameter :: max_itts = 20
        real(dp) :: du0_val, du_val, du_prev, ddu, uh_err_val, duhdu_tot_val
        real(dp) :: du_max_val, du_min_val, uh_err_best_val
        real(dp) :: tol_eta, tol_vel
        real(dp) :: duL_val, duR_val, du_CFL_val, Idt
        real(dp) :: visc_rem_lim, visc_rem_val
        real(dp) :: CFL, curv_3, h_marg, u_adj, uh_k, duhdu_k
        real(dp) :: FAmt_L_val, FAmt_R_val, FAmt_0_val
        real(dp) :: uhtot_L_val, uhtot_R_val
        real(dp) :: FA_0, FA_avg
        real(dp), parameter :: min_visc_rem = 0.1_dp
        real(dp), parameter :: CFL_min_val = 1.0e-6_dp
        logical :: do_more

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l) return

        Idt = 1.0_dp / dt

        ! --- Phase 1: Newton iteration to find du0 (zero-transport correction) ---
        du_val = 0.0_dp
        du_max_val = du_max_CFL_in(i, j)
        du_min_val = du_min_CFL_in(i, j)
        uh_err_val = uh_tot_0_in(i, j)  ! target is 0
        duhdu_tot_val = duhdu_tot_0_in(i, j)
        uh_err_best_val = abs(uh_err_val)
        do_more = .true.

        do itt = 1, max_itts
            if (itt <= 1) then
                tol_eta = 1.0e-6_dp * tol_eta_base
            elseif (itt == 2) then
                tol_eta = 1.0e-4_dp * tol_eta_base
            elseif (itt == 3) then
                tol_eta = 1.0e-2_dp * tol_eta_base
            else
                tol_eta = tol_eta_base
            end if
            tol_vel = tol_vel_val

            if (uh_err_val > 0.0_dp) then
                du_max_val = du_val
            elseif (uh_err_val < 0.0_dp) then
                du_min_val = du_val
            else
                do_more = .false.
            end if

            if (do_more) then
                if ((dt * min(IareaT(i, j), IareaT(i + 1, j)) * abs(uh_err_val) > tol_eta) .or. &
                    (better_iter_flag == 1 .and. ((abs(uh_err_val) > tol_vel * duhdu_tot_val) .or. &
                                                   (abs(uh_err_val) > uh_err_best_val)))) then
                    ddu = -uh_err_val / duhdu_tot_val
                    du_prev = du_val
                    du_val = du_val + ddu
                    if (abs(ddu) < 1.0e-15_dp * abs(du_val)) then
                        do_more = .false.
                    elseif (ddu > 0.0_dp) then
                        if (du_val >= du_max_val) then
                            du_val = 0.5_dp * (du_prev + du_max_val)
                            if (du_max_val - du_prev < 1.0e-15_dp * abs(du_val)) do_more = .false.
                        end if
                    else
                        if (du_val <= du_min_val) then
                            du_val = 0.5_dp * (du_prev + du_min_val)
                            if (du_prev - du_min_val < 1.0e-15_dp * abs(du_val)) do_more = .false.
                        end if
                    end if
                else
                    do_more = .false.
                end if
            end if

            if (.not. do_more) exit

            uh_err_val = 0.0_dp  ! target is zero transport
            duhdu_tot_val = 0.0_dp
            do k = 1, n3
                if (use_visc_rem_flag == 1) then
                    visc_rem_val = visc_rem_u(i, j, k)
                else
                    visc_rem_val = 1.0_dp
                end if
                u_adj = u(i, j, k) + du_val * visc_rem_val
                if (u_adj > 0.0_dp) then
                    if (vol_CFL_flag == 1) then
                        CFL = (u_adj * dt) * (dy_Cu(i, j) * IareaT(i, j))
                    else
                        CFL = u_adj * dt * IdxT(i, j)
                    end if
                    curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                    uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                        (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                         curv_3 * (CFL - 1.5_dp)))
                    h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + &
                             3.0_dp * curv_3 * (CFL - 1.0_dp))
                elseif (u_adj < 0.0_dp) then
                    if (vol_CFL_flag == 1) then
                        CFL = (-u_adj * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
                    else
                        CFL = -u_adj * dt * IdxT(i + 1, j)
                    end if
                    curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                    uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                        (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                         curv_3 * (CFL - 1.5_dp)))
                    h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                             3.0_dp * curv_3 * (CFL - 1.0_dp))
                else
                    uh_k = 0.0_dp
                    h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                end if
                duhdu_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * h_marg * visc_rem_val
                uh_err_val = uh_err_val + uh_k
                duhdu_tot_val = duhdu_tot_val + duhdu_k
            end do
            uh_err_best_val = min(uh_err_best_val, abs(uh_err_val))
        end do ! Newton for du0

        du0_val = du_val

        ! --- Phase 2: Determine test velocities duL, duR ---
        du_CFL_val = (CFL_min_val * Idt) * dxCu(i, j)
        duR_val = min(0.0_dp, du0_val - du_CFL_val)
        duL_val = max(0.0_dp, du0_val + du_CFL_val)

        ! Adjust duR, duL so test velocities are truly upwind
        do k = 1, n3
            if (use_visc_rem_flag == 1) then
                visc_rem_val = visc_rem_u(i, j, k)
            else
                visc_rem_val = 1.0_dp
            end if
            visc_rem_lim = max(visc_rem_val, min_visc_rem * visc_rem_max_in(i, j))
            if (visc_rem_lim > 0.0_dp) then
                if (u(i, j, k) + duR_val * visc_rem_lim > -du_CFL_val * visc_rem_val) &
                    duR_val = -(u(i, j, k) + du_CFL_val * visc_rem_val) / visc_rem_lim
                if (u(i, j, k) + duL_val * visc_rem_lim < du_CFL_val * visc_rem_val) &
                    duL_val = -(u(i, j, k) - du_CFL_val * visc_rem_val) / visc_rem_lim
            end if
        end do

        ! --- Phase 3: Evaluate fluxes at 3 test velocities ---
        FAmt_0_val = 0.0_dp; FAmt_L_val = 0.0_dp; FAmt_R_val = 0.0_dp
        uhtot_L_val = 0.0_dp; uhtot_R_val = 0.0_dp

        do k = 1, n3
            if (use_visc_rem_flag == 1) then
                visc_rem_val = visc_rem_u(i, j, k)
            else
                visc_rem_val = 1.0_dp
            end if

            ! u_0 test velocity
            u_adj = u(i, j, k) + du0_val * visc_rem_val
            if (u_adj > 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (u_adj * dt) * (dy_Cu(i, j) * IareaT(i, j))
                else
                    CFL = u_adj * dt * IdxT(i, j)
                end if
                curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + &
                         3.0_dp * curv_3 * (CFL - 1.0_dp))
            elseif (u_adj < 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (-u_adj * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
                else
                    CFL = -u_adj * dt * IdxT(i + 1, j)
                end if
                curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                         3.0_dp * curv_3 * (CFL - 1.0_dp))
            else
                h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
            end if
            FAmt_0_val = FAmt_0_val + (dy_Cu(i, j) * por_face_areaU(i, j, k)) * h_marg * visc_rem_val

            ! u_L test velocity (westerly, positive)
            u_adj = u(i, j, k) + duL_val * visc_rem_val
            if (u_adj > 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (u_adj * dt) * (dy_Cu(i, j) * IareaT(i, j))
                else
                    CFL = u_adj * dt * IdxT(i, j)
                end if
                curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                    (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                     curv_3 * (CFL - 1.5_dp)))
                h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + &
                         3.0_dp * curv_3 * (CFL - 1.0_dp))
            elseif (u_adj < 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (-u_adj * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
                else
                    CFL = -u_adj * dt * IdxT(i + 1, j)
                end if
                curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                    (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                     curv_3 * (CFL - 1.5_dp)))
                h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                         3.0_dp * curv_3 * (CFL - 1.0_dp))
            else
                uh_k = 0.0_dp
                h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
            end if
            FAmt_L_val = FAmt_L_val + (dy_Cu(i, j) * por_face_areaU(i, j, k)) * h_marg * visc_rem_val
            uhtot_L_val = uhtot_L_val + uh_k

            ! u_R test velocity (easterly, negative)
            u_adj = u(i, j, k) + duR_val * visc_rem_val
            if (u_adj > 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (u_adj * dt) * (dy_Cu(i, j) * IareaT(i, j))
                else
                    CFL = u_adj * dt * IdxT(i, j)
                end if
                curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                    (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                     curv_3 * (CFL - 1.5_dp)))
                h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + &
                         3.0_dp * curv_3 * (CFL - 1.0_dp))
            elseif (u_adj < 0.0_dp) then
                if (vol_CFL_flag == 1) then
                    CFL = (-u_adj * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
                else
                    CFL = -u_adj * dt * IdxT(i + 1, j)
                end if
                curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                uh_k = (dy_Cu(i, j) * por_face_areaU(i, j, k)) * u_adj * &
                    (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                     curv_3 * (CFL - 1.5_dp)))
                h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                         3.0_dp * curv_3 * (CFL - 1.0_dp))
            else
                uh_k = 0.0_dp
                h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
            end if
            FAmt_R_val = FAmt_R_val + (dy_Cu(i, j) * por_face_areaU(i, j, k)) * h_marg * visc_rem_val
            uhtot_R_val = uhtot_R_val + uh_k
        end do ! k

        ! --- Phase 4: Compute BT_cont fields ---
        ! Westerly (W0, WW)
        FA_0 = FAmt_0_val
        FA_avg = FAmt_0_val
        if ((duL_val - du0_val) /= 0.0_dp) &
            FA_avg = uhtot_L_val / (duL_val - du0_val)
        if (FA_avg > max(FA_0, FAmt_L_val)) then
            FA_avg = max(FA_0, FAmt_L_val)
        elseif (FA_avg < min(FA_0, FAmt_L_val)) then
            FA_0 = FA_avg
        end if
        FA_u_W0(i, j) = FA_0
        FA_u_WW(i, j) = FAmt_L_val
        if (abs(FA_0 - FAmt_L_val) <= 1.0e-12_dp * FA_0) then
            uBT_WW(i, j) = 0.0_dp
        else
            uBT_WW(i, j) = (1.5_dp * (duL_val - du0_val)) * &
                ((FAmt_L_val - FA_avg) / (FAmt_L_val - FA_0))
        end if

        ! Easterly (E0, EE)
        FA_0 = FAmt_0_val
        FA_avg = FAmt_0_val
        if ((duR_val - du0_val) /= 0.0_dp) &
            FA_avg = uhtot_R_val / (duR_val - du0_val)
        if (FA_avg > max(FA_0, FAmt_R_val)) then
            FA_avg = max(FA_0, FAmt_R_val)
        elseif (FA_avg < min(FA_0, FAmt_R_val)) then
            FA_0 = FA_avg
        end if
        FA_u_E0(i, j) = FA_0
        FA_u_EE(i, j) = FAmt_R_val
        if (abs(FAmt_R_val - FA_0) <= 1.0e-12_dp * FA_0) then
            uBT_EE(i, j) = 0.0_dp
        else
            uBT_EE(i, j) = (1.5_dp * (duR_val - du0_val)) * &
                ((FAmt_R_val - FA_avg) / (FAmt_R_val - FA_0))
        end if

    end subroutine set_zonal_BT_cont_kernel

    !> Compute effective face thickness for barotropic solver weights.
    !! Collapse(3): one thread per (I,j,k).
    attributes(global) subroutine zonal_flux_thickness_kernel( &
            h_u, u, h, h_W, h_E, &
            por_face_areaU, visc_rem_u, &
            IareaT, IdxT, dy_Cu, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
            vol_CFL_flag, marginal_flag, has_visc_rem_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: h_u(n1, n2, n3)
        real(dp), intent(in)  :: u(n1, n2, n3), h(n1, n2, n3)
        real(dp), intent(in)  :: h_W(n1, n2, n3), h_E(n1, n2, n3)
        real(dp), intent(in)  :: por_face_areaU(n1, n2, n3), visc_rem_u(n1, n2, n3)
        real(dp), intent(in)  :: IareaT(n1, n2), IdxT(n1, n2), dy_Cu(n1, n2)
        real(dp), value, intent(in) :: dt
        integer, value, intent(in) :: vol_CFL_flag, marginal_flag, has_visc_rem_flag

        integer :: i, j, k
        real(dp) :: CFL, curv_3, h_avg, h_marg

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        if (u(i, j, k) > 0.0_dp) then
            if (vol_CFL_flag == 1) then
                CFL = (u(i, j, k) * dt) * (dy_Cu(i, j) * IareaT(i, j))
            else
                CFL = u(i, j, k) * dt * IdxT(i, j)
            end if
            curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h(i, j, k)
            h_avg = h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                    curv_3 * (CFL - 1.5_dp))
            h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + &
                     3.0_dp * curv_3 * (CFL - 1.0_dp))
        elseif (u(i, j, k) < 0.0_dp) then
            if (vol_CFL_flag == 1) then
                CFL = (-u(i, j, k) * dt) * (dy_Cu(i, j) * IareaT(i + 1, j))
            else
                CFL = -u(i, j, k) * dt * IdxT(i + 1, j)
            end if
            curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h(i + 1, j, k)
            h_avg = h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                    curv_3 * (CFL - 1.5_dp))
            h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                     3.0_dp * curv_3 * (CFL - 1.0_dp))
        else
            h_avg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
            h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
        end if

        if (marginal_flag == 1) then
            h_u(i, j, k) = h_marg
        else
            h_u(i, j, k) = h_avg
        end if

        ! Scale by visc_rem and por_face_areaU
        if (has_visc_rem_flag == 1) then
            h_u(i, j, k) = h_u(i, j, k) * (visc_rem_u(i, j, k) * por_face_areaU(i, j, k))
        else
            h_u(i, j, k) = h_u(i, j, k) * por_face_areaU(i, j, k)
        end if

    end subroutine zonal_flux_thickness_kernel

    !> Compute corrected velocity: u_cor = u + du * visc_rem_u.
    !! Collapse(3): one thread per (I,j,k).
    attributes(global) subroutine u_cor_kernel( &
            u_cor, u, du, visc_rem_u, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, use_visc_rem_flag)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: u_cor(n1, n2, n3)
        real(dp), intent(in)  :: u(n1, n2, n3), visc_rem_u(n1, n2, n3)
        real(dp), intent(in)  :: du(n1, n2)
        integer, value, intent(in) :: use_visc_rem_flag

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        if (use_visc_rem_flag == 1) then
            u_cor(i, j, k) = u(i, j, k) + du(i, j) * visc_rem_u(i, j, k)
        else
            u_cor(i, j, k) = u(i, j, k) + du(i, j)
        end if

    end subroutine u_cor_kernel

    !=========================================================================
    ! Host routines
    !=========================================================================

    !> Initialize the CUDA continuity solver.
    subroutine continuity_init_cuda(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nk, &
                                     IareaT, IdxT, dy_Cu, mask2dT, monotonic, &
                                     dxT, areaT, dxCu, mask2dCu)
        type(continuity_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec, nk
        real(dp), intent(in) :: IareaT(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdxT(isd:ied, jsd:jed)
        real(dp), intent(in) :: dy_Cu(isd:ied, jsd:jed)
        real(dp), intent(in) :: mask2dT(isd:ied, jsd:jed)
        logical, intent(in)  :: monotonic
        real(dp), intent(in), optional :: dxT(isd:ied, jsd:jed)
        real(dp), intent(in), optional :: areaT(isd:ied, jsd:jed)
        real(dp), intent(in), optional :: dxCu(isd:ied, jsd:jed)
        real(dp), intent(in), optional :: mask2dCu(isd:ied, jsd:jed)

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec; CS%nz = nk

        ! Store PPM parameters
        CS%monotonic = monotonic
        CS%upwind_1st = .false.
        CS%simple_2nd = .false.

        ! Solver flags (match OpenACC defaults)
        CS%tol_eta = 1.0e-12_dp
        CS%tol_vel = 3.0e8_dp
        CS%CFL_limit_adjust = 0.5_dp
        CS%aggress_adjust = .false.
        CS%vol_CFL = .false.
        CS%better_iter = .true.
        CS%use_visc_rem_max = .true.
        CS%marginal_faces = .true.

        ! Allocate 3D device work arrays for PPM edge values
        allocate(CS%h_W(isd:ied, jsd:jed, nk))
        allocate(CS%h_E(isd:ied, jsd:jed, nk))

        ! Allocate 2D device grid metrics (existing) and copy from host
        allocate(CS%IareaT_d(isd:ied, jsd:jed));  CS%IareaT_d = IareaT
        allocate(CS%IdxT_d(isd:ied, jsd:jed));    CS%IdxT_d = IdxT
        allocate(CS%dy_Cu_d(isd:ied, jsd:jed));   CS%dy_Cu_d = dy_Cu
        allocate(CS%mask2dT_d(isd:ied, jsd:jed)); CS%mask2dT_d = mask2dT

        ! Allocate new 2D device grid metrics (optional)
        if (present(dxT)) then
            allocate(CS%dxT_d(isd:ied, jsd:jed));     CS%dxT_d = dxT
        end if
        if (present(areaT)) then
            allocate(CS%areaT_d(isd:ied, jsd:jed));   CS%areaT_d = areaT
        end if
        if (present(dxCu)) then
            allocate(CS%dxCu_d(isd:ied, jsd:jed));    CS%dxCu_d = dxCu
        end if
        if (present(mask2dCu)) then
            allocate(CS%mask2dCu_d(isd:ied, jsd:jed)); CS%mask2dCu_d = mask2dCu
        end if

        ! Allocate work arrays for full solver
        allocate(CS%duhdu(isd:ied, jsd:jed, nk))
        allocate(CS%du(isd:ied, jsd:jed))
        allocate(CS%du_max_CFL(isd:ied, jsd:jed))
        allocate(CS%du_min_CFL(isd:ied, jsd:jed))
        allocate(CS%duhdu_tot_0(isd:ied, jsd:jed))
        allocate(CS%uh_tot_0(isd:ied, jsd:jed))
        allocate(CS%visc_rem_max_arr(isd:ied, jsd:jed))

        ! Allocate default arrays (por_face = 1.0, visc_rem = 1.0)
        allocate(CS%por_face_areaU_def(isd:ied, jsd:jed, nk))
        allocate(CS%visc_rem_u_def(isd:ied, jsd:jed, nk))
        CS%por_face_areaU_def = 1.0_dp
        CS%visc_rem_u_def = 1.0_dp

        CS%initialized = .true.

    end subroutine continuity_init_cuda

    !> Run the full PPM continuity solver on the GPU.
    !! When optional args are absent, falls back to the basic 3-kernel path.
    subroutine continuity_PPM_cuda(u_d, hin_d, h_d, uh_d, dt, CS, &
            por_face_areaU_d, uhbt_d, visc_rem_u_d, u_cor_d, BT_cont, du_cor_d, bx_in, by_in)
        type(continuity_CS_cuda), intent(inout) :: CS
        real(dp), device, intent(in)    :: u_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in)    :: hin_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(out)   :: h_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(inout) :: uh_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), intent(in)            :: dt
        real(dp), device, intent(in), optional    :: por_face_areaU_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in), optional    :: uhbt_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(in), optional    :: visc_rem_u_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(out), optional   :: u_cor_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        type(BT_cont_type_cuda), intent(inout), optional :: BT_cont
        real(dp), device, intent(out), optional   :: du_cor_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, n3, is_l, ie_l, js_l, je_l, istat, bx, by
        integer :: mono_flag, upwind_flag, simple_flag
        integer :: vol_CFL_flag, use_visc_rem_flag, aggress_flag, better_flag, marginal_flag
        real(dp) :: h_min, CFL_dt, I_dt
        logical :: has_full_args, set_BT_cont
        type(dim3) :: grid3, grid2, tBlock

        ! Configurable block dimensions
        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        ! Array dimensions (1-based for kernel)
        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        n3 = CS%nz

        ! Compute-domain offsets in 1-based local coords
        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        ! PPM parameters
        h_min = 2.0e-10_dp
        mono_flag = 0; if (CS%monotonic) mono_flag = 1
        upwind_flag = 0; if (CS%upwind_1st) upwind_flag = 1
        simple_flag = 0; if (CS%simple_2nd) simple_flag = 1

        ! Solver flags
        vol_CFL_flag = 0; if (CS%vol_CFL) vol_CFL_flag = 1
        use_visc_rem_flag = 0; if (present(visc_rem_u_d)) use_visc_rem_flag = 1
        aggress_flag = 0; if (CS%aggress_adjust) aggress_flag = 1
        better_flag = 0; if (CS%better_iter) better_flag = 1
        marginal_flag = 0; if (CS%marginal_faces) marginal_flag = 1

        has_full_args = present(por_face_areaU_d)
        set_BT_cont = .false.
        if (present(BT_cont)) set_BT_cont = .true.

        CFL_dt = CS%CFL_limit_adjust / dt
        I_dt = 1.0_dp / dt
        if (CS%aggress_adjust) CFL_dt = I_dt

        ! Launch configurations
        tBlock = dim3(bx, by, 1)
        grid3 = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), n3)
        grid2 = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), 1)

        ! --- Kernel 1: PPM reconstruction ---
        call ppm_reconstruction_3d_kernel<<<grid3, tBlock>>>( &
            CS%h_W, CS%h_E, hin_d, CS%mask2dT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, h_min, mono_flag, &
            upwind_flag, simple_flag)

        ! --- Kernel 2: Zonal flux + duhdu ---
        if (has_full_args) then
            call zonal_flux_layer_3d_kernel<<<grid3, tBlock>>>( &
                uh_d, CS%duhdu, u_d, hin_d, CS%h_W, CS%h_E, CS%dy_Cu_d, CS%IdxT_d, CS%IareaT_d, &
                por_face_areaU_d, visc_rem_u_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, dt, vol_CFL_flag, use_visc_rem_flag)
        else
            call zonal_flux_layer_3d_kernel<<<grid3, tBlock>>>( &
                uh_d, CS%duhdu, u_d, hin_d, CS%h_W, CS%h_E, CS%dy_Cu_d, CS%IdxT_d, CS%IareaT_d, &
                CS%por_face_areaU_def, CS%visc_rem_u_def, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, dt, 0, 0)
        end if

        ! --- Steps 3-5: Adjustment kernels (only when uhbt or BT_cont present) ---
        if (present(uhbt_d) .or. set_BT_cont) then

            ! 3a. visc_rem_max
            if (has_full_args) then
                call visc_rem_max_kernel<<<grid2, tBlock>>>( &
                    CS%visc_rem_max_arr, visc_rem_u_d, &
                    n1, n2, n3, is_l, ie_l, js_l, je_l, &
                    merge(1, 0, use_visc_rem_flag == 1 .and. CS%use_visc_rem_max))
            else
                call visc_rem_max_kernel<<<grid2, tBlock>>>( &
                    CS%visc_rem_max_arr, CS%visc_rem_u_def, &
                    n1, n2, n3, is_l, ie_l, js_l, je_l, 0)
            end if

            ! 3b. uh_tot_0 + duhdu_tot_0
            call uh_duhdu_tot_kernel<<<grid2, tBlock>>>( &
                CS%uh_tot_0, CS%duhdu_tot_0, uh_d, CS%duhdu, &
                n1, n2, n3, is_l, ie_l, js_l, je_l)

            ! 3c. CFL limits
            if (has_full_args) then
                call zonal_CFL_limits_kernel<<<grid2, tBlock>>>( &
                    CS%du_max_CFL, CS%du_min_CFL, u_d, visc_rem_u_d, CS%visc_rem_max_arr, &
                    CS%dxT_d, CS%areaT_d, CS%dy_Cu_d, CS%mask2dCu_d, &
                    n1, n2, n3, is_l, ie_l, js_l, je_l, &
                    CFL_dt, I_dt, vol_CFL_flag, use_visc_rem_flag, aggress_flag)
            else
                call zonal_CFL_limits_kernel<<<grid2, tBlock>>>( &
                    CS%du_max_CFL, CS%du_min_CFL, u_d, CS%visc_rem_u_def, CS%visc_rem_max_arr, &
                    CS%dxT_d, CS%areaT_d, CS%dy_Cu_d, CS%mask2dCu_d, &
                    n1, n2, n3, is_l, ie_l, js_l, je_l, &
                    CFL_dt, I_dt, 0, 0, aggress_flag)
            end if

            ! 4. Newton flux adjustment (when uhbt present)
            if (present(uhbt_d)) then
                if (has_full_args) then
                    call zonal_flux_adjust_kernel<<<grid2, tBlock>>>( &
                        uh_d, CS%du, u_d, hin_d, CS%h_W, CS%h_E, uhbt_d, &
                        visc_rem_u_d, por_face_areaU_d, &
                        CS%IareaT_d, CS%dy_Cu_d, CS%IdxT_d, &
                        CS%du_max_CFL, CS%du_min_CFL, CS%uh_tot_0, CS%duhdu_tot_0, &
                        n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                        CS%tol_eta, CS%tol_vel, vol_CFL_flag, use_visc_rem_flag, better_flag)
                else
                    call zonal_flux_adjust_kernel<<<grid2, tBlock>>>( &
                        uh_d, CS%du, u_d, hin_d, CS%h_W, CS%h_E, uhbt_d, &
                        CS%visc_rem_u_def, CS%por_face_areaU_def, &
                        CS%IareaT_d, CS%dy_Cu_d, CS%IdxT_d, &
                        CS%du_max_CFL, CS%du_min_CFL, CS%uh_tot_0, CS%duhdu_tot_0, &
                        n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                        CS%tol_eta, CS%tol_vel, 0, 0, better_flag)
                end if

                ! u_cor kernel
                if (present(u_cor_d)) then
                    if (has_full_args) then
                        call u_cor_kernel<<<grid3, tBlock>>>( &
                            u_cor_d, u_d, CS%du, visc_rem_u_d, &
                            n1, n2, n3, is_l, ie_l, js_l, je_l, use_visc_rem_flag)
                    else
                        call u_cor_kernel<<<grid3, tBlock>>>( &
                            u_cor_d, u_d, CS%du, CS%visc_rem_u_def, &
                            n1, n2, n3, is_l, ie_l, js_l, je_l, 0)
                    end if
                end if

                ! du_cor: device-to-device copy
                if (present(du_cor_d)) then
                    du_cor_d = CS%du
                end if
            end if

            ! 5. BT_cont computation
            if (set_BT_cont) then
                if (has_full_args) then
                    call set_zonal_BT_cont_kernel<<<grid2, tBlock>>>( &
                        BT_cont%FA_u_W0, BT_cont%FA_u_WW, BT_cont%uBT_WW, &
                        BT_cont%FA_u_E0, BT_cont%FA_u_EE, BT_cont%uBT_EE, &
                        u_d, hin_d, CS%h_W, CS%h_E, visc_rem_u_d, por_face_areaU_d, &
                        CS%IareaT_d, CS%dy_Cu_d, CS%IdxT_d, CS%dxCu_d, CS%visc_rem_max_arr, &
                        CS%du_max_CFL, CS%du_min_CFL, CS%uh_tot_0, CS%duhdu_tot_0, &
                        n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                        CS%tol_eta, CS%tol_vel, vol_CFL_flag, use_visc_rem_flag, better_flag)
                else
                    call set_zonal_BT_cont_kernel<<<grid2, tBlock>>>( &
                        BT_cont%FA_u_W0, BT_cont%FA_u_WW, BT_cont%uBT_WW, &
                        BT_cont%FA_u_E0, BT_cont%FA_u_EE, BT_cont%uBT_EE, &
                        u_d, hin_d, CS%h_W, CS%h_E, CS%visc_rem_u_def, CS%por_face_areaU_def, &
                        CS%IareaT_d, CS%dy_Cu_d, CS%IdxT_d, CS%dxCu_d, CS%visc_rem_max_arr, &
                        CS%du_max_CFL, CS%du_min_CFL, CS%uh_tot_0, CS%duhdu_tot_0, &
                        n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                        CS%tol_eta, CS%tol_vel, 0, 0, better_flag)
                end if

                ! Flux thickness
                if (allocated(BT_cont%h_u)) then
                    if (present(u_cor_d) .and. has_full_args) then
                        call zonal_flux_thickness_kernel<<<grid3, tBlock>>>( &
                            BT_cont%h_u, u_cor_d, hin_d, CS%h_W, CS%h_E, &
                            por_face_areaU_d, visc_rem_u_d, &
                            CS%IareaT_d, CS%IdxT_d, CS%dy_Cu_d, &
                            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                            vol_CFL_flag, marginal_flag, use_visc_rem_flag)
                    elseif (has_full_args) then
                        call zonal_flux_thickness_kernel<<<grid3, tBlock>>>( &
                            BT_cont%h_u, u_d, hin_d, CS%h_W, CS%h_E, &
                            por_face_areaU_d, visc_rem_u_d, &
                            CS%IareaT_d, CS%IdxT_d, CS%dy_Cu_d, &
                            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                            vol_CFL_flag, marginal_flag, use_visc_rem_flag)
                    else
                        call zonal_flux_thickness_kernel<<<grid3, tBlock>>>( &
                            BT_cont%h_u, u_d, hin_d, CS%h_W, CS%h_E, &
                            CS%por_face_areaU_def, CS%visc_rem_u_def, &
                            CS%IareaT_d, CS%IdxT_d, CS%dy_Cu_d, &
                            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                            0, marginal_flag, 0)
                    end if
                end if
            end if
        end if

        ! --- Kernel: Zonal convergence (always) ---
        call zonal_convergence_kernel<<<grid3, tBlock>>>( &
            h_d, hin_d, uh_d, CS%IareaT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt)

        ! Sync
        istat = cudaDeviceSynchronize()

    end subroutine continuity_PPM_cuda

    !> Finalize the CUDA continuity solver. Deallocates all device arrays.
    subroutine continuity_end_cuda(CS)
        type(continuity_CS_cuda), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! PPM work arrays
        if (allocated(CS%h_W)) deallocate(CS%h_W)
        if (allocated(CS%h_E)) deallocate(CS%h_E)

        ! Existing grid metrics
        if (allocated(CS%IareaT_d)) deallocate(CS%IareaT_d)
        if (allocated(CS%IdxT_d)) deallocate(CS%IdxT_d)
        if (allocated(CS%dy_Cu_d)) deallocate(CS%dy_Cu_d)
        if (allocated(CS%mask2dT_d)) deallocate(CS%mask2dT_d)

        ! New grid metrics
        if (allocated(CS%dxT_d)) deallocate(CS%dxT_d)
        if (allocated(CS%areaT_d)) deallocate(CS%areaT_d)
        if (allocated(CS%dxCu_d)) deallocate(CS%dxCu_d)
        if (allocated(CS%mask2dCu_d)) deallocate(CS%mask2dCu_d)

        ! Solver work arrays
        if (allocated(CS%duhdu)) deallocate(CS%duhdu)
        if (allocated(CS%du)) deallocate(CS%du)
        if (allocated(CS%du_max_CFL)) deallocate(CS%du_max_CFL)
        if (allocated(CS%du_min_CFL)) deallocate(CS%du_min_CFL)
        if (allocated(CS%duhdu_tot_0)) deallocate(CS%duhdu_tot_0)
        if (allocated(CS%uh_tot_0)) deallocate(CS%uh_tot_0)
        if (allocated(CS%visc_rem_max_arr)) deallocate(CS%visc_rem_max_arr)

        ! Default arrays
        if (allocated(CS%por_face_areaU_def)) deallocate(CS%por_face_areaU_def)
        if (allocated(CS%visc_rem_u_def)) deallocate(CS%visc_rem_u_def)

        CS%initialized = .false.

    end subroutine continuity_end_cuda

    !> Allocate BT_cont_type_cuda arrays, initialized to 0.
    subroutine alloc_BT_cont_type_cuda(BT_cont, isd, ied, jsd, jed, nz)
        type(BT_cont_type_cuda), intent(inout) :: BT_cont
        integer, intent(in) :: isd, ied, jsd, jed, nz

        allocate(BT_cont%FA_u_W0(isd:ied, jsd:jed)); BT_cont%FA_u_W0 = 0.0_dp
        allocate(BT_cont%FA_u_WW(isd:ied, jsd:jed)); BT_cont%FA_u_WW = 0.0_dp
        allocate(BT_cont%FA_u_E0(isd:ied, jsd:jed)); BT_cont%FA_u_E0 = 0.0_dp
        allocate(BT_cont%FA_u_EE(isd:ied, jsd:jed)); BT_cont%FA_u_EE = 0.0_dp
        allocate(BT_cont%uBT_WW(isd:ied, jsd:jed));  BT_cont%uBT_WW = 0.0_dp
        allocate(BT_cont%uBT_EE(isd:ied, jsd:jed));  BT_cont%uBT_EE = 0.0_dp
        allocate(BT_cont%h_u(isd:ied, jsd:jed, nz)); BT_cont%h_u = 0.0_dp

    end subroutine alloc_BT_cont_type_cuda

    !> Deallocate BT_cont_type_cuda arrays.
    subroutine dealloc_BT_cont_type_cuda(BT_cont)
        type(BT_cont_type_cuda), intent(inout) :: BT_cont

        if (allocated(BT_cont%FA_u_W0)) deallocate(BT_cont%FA_u_W0)
        if (allocated(BT_cont%FA_u_WW)) deallocate(BT_cont%FA_u_WW)
        if (allocated(BT_cont%FA_u_E0)) deallocate(BT_cont%FA_u_E0)
        if (allocated(BT_cont%FA_u_EE)) deallocate(BT_cont%FA_u_EE)
        if (allocated(BT_cont%uBT_WW)) deallocate(BT_cont%uBT_WW)
        if (allocated(BT_cont%uBT_EE)) deallocate(BT_cont%uBT_EE)
        if (allocated(BT_cont%h_u)) deallocate(BT_cont%h_u)

    end subroutine dealloc_BT_cont_type_cuda

end module mom6_continuity_cuda
