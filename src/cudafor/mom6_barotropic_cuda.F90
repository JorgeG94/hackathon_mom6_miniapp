!> MOM6 Barotropic Solver Module (CUDA Fortran variant)
!!
!! Uses explicit attributes(global) CUDA kernels for GPU execution.
!! All 2D collapse(2) stencils with simple arithmetic and a host-side
!! substep loop.  1-based local indexing throughout.
!!
module mom6_barotropic_cuda
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: btstep_cuda, barotropic_init_cuda, barotropic_end_cuda
    public :: btstep_cuda_init_state, btstep_cuda_do_step, btstep_cuda_get_output
    public :: barotropic_CS_cuda

    !> Control structure for CUDA Fortran barotropic solver
    type :: barotropic_CS_cuda
        logical :: initialized = .false.
        real(dp) :: dtbt              ! Barotropic timestep [T]
        real(dp) :: bebt              ! Backward Euler parameter [nondim]
        real(dp) :: dgeo_de           ! Geopotential coefficient [nondim]
        integer :: nstep              ! Number of substeps
        integer :: is, ie, js, je, isd, ied, jsd, jed
        integer :: first_direction

        ! All device 2D arrays (isd:ied, jsd:jed) stored as 1-based (n1, n2)
        real(dp), device, allocatable :: eta(:,:), eta_pred(:,:)
        real(dp), device, allocatable :: ubt(:,:), vbt(:,:)
        real(dp), device, allocatable :: ubt_prev(:,:), vbt_prev(:,:)
        real(dp), device, allocatable :: uhbt(:,:), vhbt(:,:)
        real(dp), device, allocatable :: PFu(:,:), PFv(:,:)
        real(dp), device, allocatable :: Cor_u(:,:), Cor_v(:,:)
        real(dp), device, allocatable :: Datu(:,:), Datv(:,:)
        real(dp), device, allocatable :: gtot_E(:,:), gtot_W(:,:)
        real(dp), device, allocatable :: gtot_N(:,:), gtot_S(:,:)
        real(dp), device, allocatable :: bt_rem_u(:,:), bt_rem_v(:,:)
        real(dp), device, allocatable :: ubt_av(:,:), vbt_av(:,:)
        real(dp), device, allocatable :: uhbt_av(:,:), vhbt_av(:,:)

        ! Grid metrics on device
        real(dp), device, allocatable :: IareaT_d(:,:)
        real(dp), device, allocatable :: IdxCu_d(:,:), IdyCv_d(:,:)

        ! Coriolis coefficients: f_4_u(4, n1, n2) and f_4_v(4, n1, n2)
        real(dp), device, allocatable :: f_4_u(:,:,:), f_4_v(:,:,:)
    end type barotropic_CS_cuda

contains

    !=========================================================================
    ! CUDA Kernels (attributes(global))
    !
    ! All arrays use 1-based local indexing. The caller passes:
    !   n1 = ied - isd + 1   (array dim 1)
    !   n2 = jed - jsd + 1   (array dim 2)
    !   is_l = is - isd + 1  (compute-start offset in dim 1)
    !   ie_l = ie - isd + 1  (compute-end offset in dim 1)
    !   js_l = js - jsd + 1  (compute-start offset in dim 2)
    !   je_l = je - jsd + 1  (compute-end offset in dim 2)
    !=========================================================================

    !> Kernel 1: Initialize state from input arrays and zero time averages.
    !! Copies eta_in, ubt_in, vbt_in into CS state arrays over the full
    !! domain, and zeroes the accumulation arrays.
    attributes(global) subroutine bt_init_kernel( &
            eta, ubt, vbt, ubt_av, vbt_av, uhbt_av, vhbt_av, &
            eta_in, ubt_in, vbt_in, &
            n1, n2)
        integer, value, intent(in) :: n1, n2
        real(dp), intent(out) :: eta(n1, n2), ubt(n1, n2), vbt(n1, n2)
        real(dp), intent(out) :: ubt_av(n1, n2), vbt_av(n1, n2)
        real(dp), intent(out) :: uhbt_av(n1, n2), vhbt_av(n1, n2)
        real(dp), intent(in)  :: eta_in(n1, n2), ubt_in(n1, n2), vbt_in(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return

        eta(i, j) = eta_in(i, j)
        ubt(i, j) = ubt_in(i, j)
        vbt(i, j) = vbt_in(i, j)
        ubt_av(i, j) = 0.0_dp
        vbt_av(i, j) = 0.0_dp
        uhbt_av(i, j) = 0.0_dp
        vhbt_av(i, j) = 0.0_dp

    end subroutine bt_init_kernel

    !> Kernel 2: Store previous u-velocity for transport weighting.
    !! Range: i in [is_l-1 : ie_l+1], j in [js_l : je_l]
    attributes(global) subroutine bt_store_prev_u_kernel( &
            ubt_prev, ubt, &
            n1, n2, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: ubt_prev(n1, n2)
        real(dp), intent(in)  :: ubt(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l - 1 .or. i > ie_l + 1 .or. j < js_l .or. j > je_l) return

        ubt_prev(i, j) = ubt(i, j)

    end subroutine bt_store_prev_u_kernel

    !> Kernel 3: Store previous v-velocity for transport weighting.
    !! Range: i in [is_l : ie_l], j in [js_l-1 : je_l+1]
    attributes(global) subroutine bt_store_prev_v_kernel( &
            vbt_prev, vbt, &
            n1, n2, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: vbt_prev(n1, n2)
        real(dp), intent(in)  :: vbt(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l .or. i > ie_l .or. j < js_l - 1 .or. j > je_l + 1) return

        vbt_prev(i, j) = vbt(i, j)

    end subroutine bt_store_prev_v_kernel

    !> Kernel 4: Eta predictor from velocity divergence.
    !! Range: i in [is_l : ie_l], j in [js_l : je_l]
    attributes(global) subroutine bt_eta_pred_kernel( &
            eta_pred, eta, ubt, vbt, Datu, Datv, IareaT, &
            n1, n2, is_l, ie_l, js_l, je_l, dtbt)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: dtbt
        real(dp), intent(out) :: eta_pred(n1, n2)
        real(dp), intent(in)  :: eta(n1, n2), ubt(n1, n2), vbt(n1, n2)
        real(dp), intent(in)  :: Datu(n1, n2), Datv(n1, n2), IareaT(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l .or. i > ie_l .or. j < js_l .or. j > je_l) return

        eta_pred(i, j) = eta(i, j) + (dtbt * IareaT(i, j)) * &
            ((Datu(i - 1, j) * ubt(i - 1, j) - Datu(i, j) * ubt(i, j)) + &
             (Datv(i, j - 1) * vbt(i, j - 1) - Datv(i, j) * vbt(i, j)))

    end subroutine bt_eta_pred_kernel

    !> Kernel 5: Pressure force at u-points.
    !! Range: i in [is_l : ie_l-1], j in [js_l : je_l]
    attributes(global) subroutine bt_pressure_force_u_kernel( &
            PFu, eta_pred, gtot_E, gtot_W, IdxCu, &
            n1, n2, is_l, ie_l, js_l, je_l, dgeo_de)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: dgeo_de
        real(dp), intent(out) :: PFu(n1, n2)
        real(dp), intent(in)  :: eta_pred(n1, n2)
        real(dp), intent(in)  :: gtot_E(n1, n2), gtot_W(n1, n2)
        real(dp), intent(in)  :: IdxCu(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l .or. i > ie_l - 1 .or. j < js_l .or. j > je_l) return

        PFu(i, j) = (eta_pred(i, j) * gtot_E(i, j) - &
                      eta_pred(i + 1, j) * gtot_W(i + 1, j)) * &
                     dgeo_de * IdxCu(i, j)

    end subroutine bt_pressure_force_u_kernel

    !> Kernel 6: Pressure force at v-points.
    !! Range: i in [is_l : ie_l], j in [js_l : je_l-1]
    attributes(global) subroutine bt_pressure_force_v_kernel( &
            PFv, eta_pred, gtot_N, gtot_S, IdyCv, &
            n1, n2, is_l, ie_l, js_l, je_l, dgeo_de)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: dgeo_de
        real(dp), intent(out) :: PFv(n1, n2)
        real(dp), intent(in)  :: eta_pred(n1, n2)
        real(dp), intent(in)  :: gtot_N(n1, n2), gtot_S(n1, n2)
        real(dp), intent(in)  :: IdyCv(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l .or. i > ie_l .or. j < js_l .or. j > je_l - 1) return

        PFv(i, j) = (eta_pred(i, j) * gtot_N(i, j) - &
                      eta_pred(i, j + 1) * gtot_S(i, j + 1)) * &
                     dgeo_de * IdyCv(i, j)

    end subroutine bt_pressure_force_v_kernel

    !> Kernel 7: Coriolis update for u-points.
    !! Computes Cor_u from vbt and f_4_u, then updates ubt.
    !! Range: i in [is_l : ie_l-1], j in [js_l : je_l]
    attributes(global) subroutine bt_coriolis_update_u_kernel( &
            ubt, Cor_u, vbt, PFu, f_4_u, bt_rem_u, &
            n1, n2, is_l, ie_l, js_l, je_l, dtbt)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: dtbt
        real(dp), intent(inout) :: ubt(n1, n2)
        real(dp), intent(out)   :: Cor_u(n1, n2)
        real(dp), intent(in)    :: vbt(n1, n2)
        real(dp), intent(in)    :: PFu(n1, n2)
        real(dp), intent(in)    :: f_4_u(4, n1, n2)
        real(dp), intent(in)    :: bt_rem_u(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l .or. i > ie_l - 1 .or. j < js_l .or. j > je_l) return

        Cor_u(i, j) = (f_4_u(4, i, j) * vbt(i + 1, j)     + &
                        f_4_u(1, i, j) * vbt(i, j - 1))    + &
                       (f_4_u(3, i, j) * vbt(i, j)         + &
                        f_4_u(2, i, j) * vbt(i + 1, j - 1))

        ubt(i, j) = bt_rem_u(i, j) * (ubt(i, j) + dtbt * (Cor_u(i, j) + PFu(i, j)))

    end subroutine bt_coriolis_update_u_kernel

    !> Kernel 8: Coriolis update for v-points.
    !! Computes Cor_v from ubt and f_4_v, then updates vbt.
    !! Range: i in [is_l : ie_l], j in [js_l : je_l-1]
    attributes(global) subroutine bt_coriolis_update_v_kernel( &
            vbt, Cor_v, ubt, PFv, f_4_v, bt_rem_v, &
            n1, n2, is_l, ie_l, js_l, je_l, dtbt)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: dtbt
        real(dp), intent(inout) :: vbt(n1, n2)
        real(dp), intent(out)   :: Cor_v(n1, n2)
        real(dp), intent(in)    :: ubt(n1, n2)
        real(dp), intent(in)    :: PFv(n1, n2)
        real(dp), intent(in)    :: f_4_v(4, n1, n2)
        real(dp), intent(in)    :: bt_rem_v(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l .or. i > ie_l .or. j < js_l .or. j > je_l - 1) return

        Cor_v(i, j) = -1.0_dp * &
                      ((f_4_v(1, i, j) * ubt(i - 1, j)     + &
                        f_4_v(4, i, j) * ubt(i, j + 1))    + &
                       (f_4_v(2, i, j) * ubt(i, j)         + &
                        f_4_v(3, i, j) * ubt(i - 1, j + 1)))

        vbt(i, j) = bt_rem_v(i, j) * (vbt(i, j) + dtbt * (Cor_v(i, j) + PFv(i, j)))

    end subroutine bt_coriolis_update_v_kernel

    !> Kernel 9: Compute transports, update eta, and accumulate time averages.
    !! Fused kernel combining uhbt/vhbt computation, eta update, and
    !! accumulation of ubt_av/vbt_av into one launch.
    !!
    !! uhbt range: i in [is_l-1 : ie_l-1], j in [js_l : je_l]  (u-face)
    !! vhbt range: i in [is_l : ie_l], j in [js_l-1 : je_l-1]  (v-face)
    !! eta  range: i in [is_l : ie_l], j in [js_l : je_l]       (T-cell)
    !! ubt_av: same as uhbt; vbt_av: same as vhbt
    attributes(global) subroutine bt_transport_eta_accum_kernel( &
            uhbt, vhbt, eta, ubt_av, vbt_av, uhbt_av, vhbt_av, &
            ubt, vbt, ubt_prev, vbt_prev, &
            Datu, Datv, IareaT, &
            n1, n2, is_l, ie_l, js_l, je_l, &
            dtbt, trans_wt1, trans_wt2, inv_nstep)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: dtbt, trans_wt1, trans_wt2, inv_nstep
        real(dp), intent(inout) :: uhbt(n1, n2), vhbt(n1, n2)
        real(dp), intent(inout) :: eta(n1, n2)
        real(dp), intent(inout) :: ubt_av(n1, n2), vbt_av(n1, n2)
        real(dp), intent(inout) :: uhbt_av(n1, n2), vhbt_av(n1, n2)
        real(dp), intent(in)    :: ubt(n1, n2), vbt(n1, n2)
        real(dp), intent(in)    :: ubt_prev(n1, n2), vbt_prev(n1, n2)
        real(dp), intent(in)    :: Datu(n1, n2), Datv(n1, n2), IareaT(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return

        ! Compute u-transport: i in [is_l-1 : ie_l], j in [js_l : je_l]
        ! Extended range for MPI boundary correctness
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l .and. j <= je_l) then
            uhbt(i, j) = Datu(i, j) * (trans_wt1 * ubt(i, j) + trans_wt2 * ubt_prev(i, j))
        end if

        ! Compute v-transport: i in [is_l : ie_l], j in [js_l-1 : je_l]
        ! Extended range for MPI boundary correctness
        if (i >= is_l .and. i <= ie_l .and. j >= js_l - 1 .and. j <= je_l) then
            vhbt(i, j) = Datv(i, j) * (trans_wt1 * vbt(i, j) + trans_wt2 * vbt_prev(i, j))
        end if

        ! Sync needed: eta update reads uhbt/vhbt at neighbors computed above.
        ! Since different threads write different (i,j), we need a global sync.
        ! This is handled by splitting into a separate kernel or using atomics.
        ! For simplicity within a fused kernel, we rely on the fact that the
        ! transports at halo faces (is_l-1, js_l-1) are written by threads
        ! that are launched in the same grid. We add a grid-wide sync via
        ! cooperative groups, but for maximal compatibility we split eta
        ! update into a separate fence. Here we only accumulate velocity
        ! averages alongside the transport computation.

        ! Accumulate u time averages: same range as uhbt
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l .and. j <= je_l) then
            ubt_av(i, j) = ubt_av(i, j) + ubt(i, j) * inv_nstep
            uhbt_av(i, j) = uhbt_av(i, j) + uhbt(i, j) * inv_nstep
        end if

        ! Accumulate v time averages: same range as vhbt
        if (i >= is_l .and. i <= ie_l .and. j >= js_l - 1 .and. j <= je_l) then
            vbt_av(i, j) = vbt_av(i, j) + vbt(i, j) * inv_nstep
            vhbt_av(i, j) = vhbt_av(i, j) + vhbt(i, j) * inv_nstep
        end if

    end subroutine bt_transport_eta_accum_kernel

    !> Kernel 9b: Update eta from divergence of transports (uhbt, vhbt).
    !! Must be launched after bt_transport_eta_accum_kernel has completed
    !! (implicit sync between kernel launches on the same stream).
    !! Range: i in [is_l : ie_l], j in [js_l : je_l]
    attributes(global) subroutine bt_eta_update_kernel( &
            eta, uhbt, vhbt, IareaT, &
            n1, n2, is_l, ie_l, js_l, je_l, dtbt)
        integer, value, intent(in) :: n1, n2, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: dtbt
        real(dp), intent(inout) :: eta(n1, n2)
        real(dp), intent(in)    :: uhbt(n1, n2), vhbt(n1, n2)
        real(dp), intent(in)    :: IareaT(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return
        if (i < is_l .or. i > ie_l .or. j < js_l .or. j > je_l) return

        eta(i, j) = eta(i, j) - dtbt * IareaT(i, j) * &
            ((uhbt(i, j) - uhbt(i - 1, j)) + (vhbt(i, j) - vhbt(i, j - 1)))

    end subroutine bt_eta_update_kernel

    !> Kernel 10: Copy final output arrays.
    !! Copies ubt_av, vbt_av, eta to output arrays over the full domain.
    attributes(global) subroutine bt_copy_output_kernel( &
            u_av_out, v_av_out, eta_out, &
            ubt_av, vbt_av, eta, &
            n1, n2)
        integer, value, intent(in) :: n1, n2
        real(dp), intent(out) :: u_av_out(n1, n2), v_av_out(n1, n2), eta_out(n1, n2)
        real(dp), intent(in)  :: ubt_av(n1, n2), vbt_av(n1, n2), eta(n1, n2)

        integer :: i, j

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        if (i > n1 .or. j > n2) return

        u_av_out(i, j) = ubt_av(i, j)
        v_av_out(i, j) = vbt_av(i, j)
        eta_out(i, j) = eta(i, j)

    end subroutine bt_copy_output_kernel

    !=========================================================================
    ! Host routines
    !=========================================================================

    !> Initialize the CUDA barotropic solver.
    !! Stores dimensions and parameters, allocates device arrays,
    !! and copies time-invariant grid data to the device.
    subroutine barotropic_init_cuda(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, &
                                    nstep, dt, bebt, first_direction, &
                                    Datu, Datv, gtot_E, gtot_W, gtot_N, gtot_S, &
                                    f_4_u, f_4_v, bt_rem_u, bt_rem_v, &
                                    IareaT, IdxCu, IdyCv)
        type(barotropic_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec
        integer, intent(in) :: nstep, first_direction
        real(dp), intent(in) :: dt, bebt
        real(dp), intent(in) :: Datu(isd:ied, jsd:jed)
        real(dp), intent(in) :: Datv(isd:ied, jsd:jed)
        real(dp), intent(in) :: gtot_E(isd:ied, jsd:jed)
        real(dp), intent(in) :: gtot_W(isd:ied, jsd:jed)
        real(dp), intent(in) :: gtot_N(isd:ied, jsd:jed)
        real(dp), intent(in) :: gtot_S(isd:ied, jsd:jed)
        real(dp), intent(in) :: f_4_u(4, isd:ied, jsd:jed)
        real(dp), intent(in) :: f_4_v(4, isd:ied, jsd:jed)
        real(dp), intent(in) :: bt_rem_u(isd:ied, jsd:jed)
        real(dp), intent(in) :: bt_rem_v(isd:ied, jsd:jed)
        real(dp), intent(in) :: IareaT(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdxCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdyCv(isd:ied, jsd:jed)

        ! Store dimensions and parameters
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec
        CS%nstep = nstep
        CS%dtbt = dt / real(nstep, dp)
        CS%bebt = bebt
        CS%dgeo_de = 1.0_dp
        CS%first_direction = first_direction

        ! Allocate 2D device state/work arrays
        allocate(CS%eta(isd:ied, jsd:jed))
        allocate(CS%eta_pred(isd:ied, jsd:jed))
        allocate(CS%ubt(isd:ied, jsd:jed))
        allocate(CS%vbt(isd:ied, jsd:jed))
        allocate(CS%ubt_prev(isd:ied, jsd:jed))
        allocate(CS%vbt_prev(isd:ied, jsd:jed))
        allocate(CS%uhbt(isd:ied, jsd:jed))
        allocate(CS%vhbt(isd:ied, jsd:jed))
        allocate(CS%PFu(isd:ied, jsd:jed))
        allocate(CS%PFv(isd:ied, jsd:jed))
        allocate(CS%Cor_u(isd:ied, jsd:jed))
        allocate(CS%Cor_v(isd:ied, jsd:jed))
        allocate(CS%ubt_av(isd:ied, jsd:jed))
        allocate(CS%vbt_av(isd:ied, jsd:jed))
        allocate(CS%uhbt_av(isd:ied, jsd:jed))
        allocate(CS%vhbt_av(isd:ied, jsd:jed))

        ! Allocate and copy time-invariant grid data to device
        allocate(CS%Datu(isd:ied, jsd:jed));     CS%Datu = Datu
        allocate(CS%Datv(isd:ied, jsd:jed));     CS%Datv = Datv
        allocate(CS%gtot_E(isd:ied, jsd:jed));   CS%gtot_E = gtot_E
        allocate(CS%gtot_W(isd:ied, jsd:jed));   CS%gtot_W = gtot_W
        allocate(CS%gtot_N(isd:ied, jsd:jed));   CS%gtot_N = gtot_N
        allocate(CS%gtot_S(isd:ied, jsd:jed));   CS%gtot_S = gtot_S
        allocate(CS%bt_rem_u(isd:ied, jsd:jed)); CS%bt_rem_u = bt_rem_u
        allocate(CS%bt_rem_v(isd:ied, jsd:jed)); CS%bt_rem_v = bt_rem_v
        allocate(CS%IareaT_d(isd:ied, jsd:jed)); CS%IareaT_d = IareaT
        allocate(CS%IdxCu_d(isd:ied, jsd:jed));  CS%IdxCu_d = IdxCu
        allocate(CS%IdyCv_d(isd:ied, jsd:jed));  CS%IdyCv_d = IdyCv

        ! Coriolis coefficients: f_4_u(4, n1, n2) and f_4_v(4, n1, n2)
        allocate(CS%f_4_u(4, isd:ied, jsd:jed)); CS%f_4_u = f_4_u
        allocate(CS%f_4_v(4, isd:ied, jsd:jed)); CS%f_4_v = f_4_v

        CS%initialized = .true.

    end subroutine barotropic_init_cuda

    !> Main CUDA barotropic time-stepping routine.
    !! Performs nstep substeps of the barotropic equations on the GPU.
    !! All arguments are device arrays.
    subroutine btstep_cuda(eta_in_d, ubt_in_d, vbt_in_d, u_av_d, v_av_d, eta_av_d, &
                           CS, bx_in, by_in)
        type(barotropic_CS_cuda), intent(inout) :: CS
        real(dp), device, intent(in)  :: eta_in_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(in)  :: ubt_in_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(in)  :: vbt_in_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(out) :: u_av_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(out) :: v_av_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(out) :: eta_av_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, is_l, ie_l, js_l, je_l, istat, bx, by, n
        real(dp) :: trans_wt1, trans_wt2, inv_nstep
        logical :: v_first
        type(dim3) :: grid, tBlock

        ! Configurable block dimensions (default: 32 x 4 = 128 threads)
        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        ! Array dimensions (1-based for kernels)
        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1

        ! Compute-domain offsets in 1-based local coords
        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        ! Thread block and grid
        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), 1)

        ! Transport weights and averaging factor
        trans_wt1 = 1.0_dp + CS%bebt
        trans_wt2 = -CS%bebt
        inv_nstep = 1.0_dp / real(CS%nstep, dp)

        ! --- Initialize state from input ---
        call bt_init_kernel<<<grid, tBlock>>>( &
            CS%eta, CS%ubt, CS%vbt, CS%ubt_av, CS%vbt_av, CS%uhbt_av, CS%vhbt_av, &
            eta_in_d, ubt_in_d, vbt_in_d, &
            n1, n2)

        ! --- Substep loop (host-side) ---
        do n = 1, CS%nstep

            ! Store previous velocities
            call bt_store_prev_u_kernel<<<grid, tBlock>>>( &
                CS%ubt_prev, CS%ubt, &
                n1, n2, is_l, ie_l, js_l, je_l)

            call bt_store_prev_v_kernel<<<grid, tBlock>>>( &
                CS%vbt_prev, CS%vbt, &
                n1, n2, is_l, ie_l, js_l, je_l)

            ! Eta predictor
            call bt_eta_pred_kernel<<<grid, tBlock>>>( &
                CS%eta_pred, CS%eta, CS%ubt, CS%vbt, &
                CS%Datu, CS%Datv, CS%IareaT_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

            ! Pressure forces
            call bt_pressure_force_u_kernel<<<grid, tBlock>>>( &
                CS%PFu, CS%eta_pred, CS%gtot_E, CS%gtot_W, CS%IdxCu_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de)

            call bt_pressure_force_v_kernel<<<grid, tBlock>>>( &
                CS%PFv, CS%eta_pred, CS%gtot_N, CS%gtot_S, CS%IdyCv_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de)

            ! Alternating u/v Coriolis update order
            v_first = (mod(n + CS%first_direction, 2) == 1)

            if (v_first) then
                ! Update v first, then u
                call bt_coriolis_update_v_kernel<<<grid, tBlock>>>( &
                    CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v, CS%bt_rem_v, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

                call bt_coriolis_update_u_kernel<<<grid, tBlock>>>( &
                    CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u, CS%bt_rem_u, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)
            else
                ! Update u first, then v
                call bt_coriolis_update_u_kernel<<<grid, tBlock>>>( &
                    CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u, CS%bt_rem_u, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

                call bt_coriolis_update_v_kernel<<<grid, tBlock>>>( &
                    CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v, CS%bt_rem_v, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)
            end if

            ! Compute transports and accumulate time averages
            call bt_transport_eta_accum_kernel<<<grid, tBlock>>>( &
                CS%uhbt, CS%vhbt, CS%eta, CS%ubt_av, CS%vbt_av, &
                CS%uhbt_av, CS%vhbt_av, &
                CS%ubt, CS%vbt, CS%ubt_prev, CS%vbt_prev, &
                CS%Datu, CS%Datv, CS%IareaT_d, &
                n1, n2, is_l, ie_l, js_l, je_l, &
                CS%dtbt, trans_wt1, trans_wt2, inv_nstep)

            ! Update eta from transport divergence (must follow transport kernel)
            call bt_eta_update_kernel<<<grid, tBlock>>>( &
                CS%eta, CS%uhbt, CS%vhbt, CS%IareaT_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

        end do  ! substep loop

        ! --- Copy output ---
        call bt_copy_output_kernel<<<grid, tBlock>>>( &
            u_av_d, v_av_d, eta_av_d, &
            CS%ubt_av, CS%vbt_av, CS%eta, &
            n1, n2)

        ! Sync to ensure all output is ready before returning
        istat = cudaDeviceSynchronize()

    end subroutine btstep_cuda

    !> Finalize the CUDA barotropic solver. Deallocates all device arrays.
    subroutine barotropic_end_cuda(CS)
        type(barotropic_CS_cuda), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! Deallocate state/work arrays
        if (allocated(CS%eta)) deallocate(CS%eta)
        if (allocated(CS%eta_pred)) deallocate(CS%eta_pred)
        if (allocated(CS%ubt)) deallocate(CS%ubt)
        if (allocated(CS%vbt)) deallocate(CS%vbt)
        if (allocated(CS%ubt_prev)) deallocate(CS%ubt_prev)
        if (allocated(CS%vbt_prev)) deallocate(CS%vbt_prev)
        if (allocated(CS%uhbt)) deallocate(CS%uhbt)
        if (allocated(CS%vhbt)) deallocate(CS%vhbt)
        if (allocated(CS%PFu)) deallocate(CS%PFu)
        if (allocated(CS%PFv)) deallocate(CS%PFv)
        if (allocated(CS%Cor_u)) deallocate(CS%Cor_u)
        if (allocated(CS%Cor_v)) deallocate(CS%Cor_v)
        if (allocated(CS%ubt_av)) deallocate(CS%ubt_av)
        if (allocated(CS%vbt_av)) deallocate(CS%vbt_av)
        if (allocated(CS%uhbt_av)) deallocate(CS%uhbt_av)
        if (allocated(CS%vhbt_av)) deallocate(CS%vhbt_av)

        ! Deallocate grid/parameter arrays
        if (allocated(CS%Datu)) deallocate(CS%Datu)
        if (allocated(CS%Datv)) deallocate(CS%Datv)
        if (allocated(CS%gtot_E)) deallocate(CS%gtot_E)
        if (allocated(CS%gtot_W)) deallocate(CS%gtot_W)
        if (allocated(CS%gtot_N)) deallocate(CS%gtot_N)
        if (allocated(CS%gtot_S)) deallocate(CS%gtot_S)
        if (allocated(CS%bt_rem_u)) deallocate(CS%bt_rem_u)
        if (allocated(CS%bt_rem_v)) deallocate(CS%bt_rem_v)
        if (allocated(CS%IareaT_d)) deallocate(CS%IareaT_d)
        if (allocated(CS%IdxCu_d)) deallocate(CS%IdxCu_d)
        if (allocated(CS%IdyCv_d)) deallocate(CS%IdyCv_d)
        if (allocated(CS%f_4_u)) deallocate(CS%f_4_u)
        if (allocated(CS%f_4_v)) deallocate(CS%f_4_v)

        CS%initialized = .false.

    end subroutine barotropic_end_cuda

    !> Initialize CUDA barotropic state from input device arrays (split API for MPI drivers).
    subroutine btstep_cuda_init_state(CS, eta_in_d, ubt_in_d, vbt_in_d, bx_in, by_in)
        type(barotropic_CS_cuda), intent(inout) :: CS
        real(dp), device, intent(in) :: eta_in_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(in) :: ubt_in_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(in) :: vbt_in_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, bx, by
        type(dim3) :: grid, tBlock

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), 1)

        call bt_init_kernel<<<grid, tBlock>>>( &
            CS%eta, CS%ubt, CS%vbt, CS%ubt_av, CS%vbt_av, CS%uhbt_av, CS%vhbt_av, &
            eta_in_d, ubt_in_d, vbt_in_d, &
            n1, n2)

    end subroutine btstep_cuda_init_state

    !> Execute one CUDA barotropic substep (split API for MPI drivers).
    subroutine btstep_cuda_do_step(CS, n, bx_in, by_in)
        type(barotropic_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: n  ! substep number (1-based)
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, is_l, ie_l, js_l, je_l, bx, by
        real(dp) :: trans_wt1, trans_wt2, inv_nstep
        logical :: v_first
        type(dim3) :: grid, tBlock

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1

        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), 1)

        trans_wt1 = 1.0_dp + CS%bebt
        trans_wt2 = -CS%bebt
        inv_nstep = 1.0_dp / real(CS%nstep, dp)

        ! Store previous velocities
        call bt_store_prev_u_kernel<<<grid, tBlock>>>( &
            CS%ubt_prev, CS%ubt, &
            n1, n2, is_l, ie_l, js_l, je_l)

        call bt_store_prev_v_kernel<<<grid, tBlock>>>( &
            CS%vbt_prev, CS%vbt, &
            n1, n2, is_l, ie_l, js_l, je_l)

        ! Eta predictor
        call bt_eta_pred_kernel<<<grid, tBlock>>>( &
            CS%eta_pred, CS%eta, CS%ubt, CS%vbt, &
            CS%Datu, CS%Datv, CS%IareaT_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

        ! Pressure forces
        call bt_pressure_force_u_kernel<<<grid, tBlock>>>( &
            CS%PFu, CS%eta_pred, CS%gtot_E, CS%gtot_W, CS%IdxCu_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de)

        call bt_pressure_force_v_kernel<<<grid, tBlock>>>( &
            CS%PFv, CS%eta_pred, CS%gtot_N, CS%gtot_S, CS%IdyCv_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de)

        ! Alternating u/v Coriolis update order
        v_first = (mod(n + CS%first_direction, 2) == 1)

        if (v_first) then
            call bt_coriolis_update_v_kernel<<<grid, tBlock>>>( &
                CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v, CS%bt_rem_v, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

            call bt_coriolis_update_u_kernel<<<grid, tBlock>>>( &
                CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u, CS%bt_rem_u, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)
        else
            call bt_coriolis_update_u_kernel<<<grid, tBlock>>>( &
                CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u, CS%bt_rem_u, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

            call bt_coriolis_update_v_kernel<<<grid, tBlock>>>( &
                CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v, CS%bt_rem_v, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)
        end if

        ! Compute transports and accumulate time averages
        call bt_transport_eta_accum_kernel<<<grid, tBlock>>>( &
            CS%uhbt, CS%vhbt, CS%eta, CS%ubt_av, CS%vbt_av, &
            CS%uhbt_av, CS%vhbt_av, &
            CS%ubt, CS%vbt, CS%ubt_prev, CS%vbt_prev, &
            CS%Datu, CS%Datv, CS%IareaT_d, &
            n1, n2, is_l, ie_l, js_l, je_l, &
            CS%dtbt, trans_wt1, trans_wt2, inv_nstep)

        ! Update eta from transport divergence
        call bt_eta_update_kernel<<<grid, tBlock>>>( &
            CS%eta, CS%uhbt, CS%vhbt, CS%IareaT_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt)

    end subroutine btstep_cuda_do_step

    !> Copy CUDA barotropic output from CS to output device arrays (split API for MPI drivers).
    subroutine btstep_cuda_get_output(CS, u_av_d, v_av_d, eta_av_d, bx_in, by_in)
        type(barotropic_CS_cuda), intent(inout) :: CS
        real(dp), device, intent(out) :: u_av_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(out) :: v_av_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(out) :: eta_av_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, bx, by, istat
        type(dim3) :: grid, tBlock

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), 1)

        call bt_copy_output_kernel<<<grid, tBlock>>>( &
            u_av_d, v_av_d, eta_av_d, &
            CS%ubt_av, CS%vbt_av, CS%eta, &
            n1, n2)

        istat = cudaDeviceSynchronize()

    end subroutine btstep_cuda_get_output

end module mom6_barotropic_cuda
