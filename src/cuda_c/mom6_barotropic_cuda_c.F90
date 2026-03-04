!> MOM6 Barotropic Solver — Portable Fortran wrapper calling CUDA C kernels
!!
!! 1:1 port of mom6_barotropic_cuda.F90 to the CUDA C + Fortran wrapper pattern.
!! Uses ONLY iso_c_binding + iso_fortran_env (Fortran 2003 standard).
!! NO cudafor, NO device attribute, NO vendor-specific extensions.
!! All GPU memory is managed through C helper functions.
!!
!! Compilable with any Fortran compiler: gfortran, ifx, nvfortran, amdflang, etc.
!!
module mom6_barotropic_cuda_c
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding
    use mom6_cuda_c_common
    implicit none
    private

    public :: btstep_cuda_c, barotropic_init_cuda_c, barotropic_end_cuda_c
    public :: btstep_cuda_c_init_state, btstep_cuda_c_do_step, btstep_cuda_c_get_output
    public :: btstep_cuda_c_export_state, btstep_cuda_c_import_state
    public :: barotropic_CS_cuda_c

    !> Control structure — all device pointers are opaque type(c_ptr)
    type :: barotropic_CS_cuda_c
        logical :: initialized = .false.
        real(dp) :: dtbt              ! Barotropic timestep [T]
        real(dp) :: bebt              ! Backward Euler parameter [nondim]
        real(dp) :: dgeo_de           ! Geopotential coefficient [nondim]
        integer :: nstep              ! Number of substeps
        integer :: is, ie, js, je, isd, ied, jsd, jed
        integer :: first_direction

        ! State / work 2D device arrays
        type(c_ptr) :: eta      = c_null_ptr
        type(c_ptr) :: eta_pred = c_null_ptr
        type(c_ptr) :: ubt      = c_null_ptr
        type(c_ptr) :: vbt      = c_null_ptr
        type(c_ptr) :: ubt_prev = c_null_ptr
        type(c_ptr) :: vbt_prev = c_null_ptr
        type(c_ptr) :: uhbt     = c_null_ptr
        type(c_ptr) :: vhbt     = c_null_ptr
        type(c_ptr) :: PFu      = c_null_ptr
        type(c_ptr) :: PFv      = c_null_ptr
        type(c_ptr) :: Cor_u    = c_null_ptr
        type(c_ptr) :: Cor_v    = c_null_ptr
        type(c_ptr) :: ubt_av   = c_null_ptr
        type(c_ptr) :: vbt_av   = c_null_ptr
        type(c_ptr) :: uhbt_av  = c_null_ptr
        type(c_ptr) :: vhbt_av  = c_null_ptr

        ! Grid metric 2D device arrays
        type(c_ptr) :: Datu_d    = c_null_ptr
        type(c_ptr) :: Datv_d    = c_null_ptr
        type(c_ptr) :: gtot_E_d  = c_null_ptr
        type(c_ptr) :: gtot_W_d  = c_null_ptr
        type(c_ptr) :: gtot_N_d  = c_null_ptr
        type(c_ptr) :: gtot_S_d  = c_null_ptr
        type(c_ptr) :: bt_rem_u_d = c_null_ptr
        type(c_ptr) :: bt_rem_v_d = c_null_ptr
        type(c_ptr) :: IareaT_d  = c_null_ptr
        type(c_ptr) :: IdxCu_d   = c_null_ptr
        type(c_ptr) :: IdyCv_d   = c_null_ptr

        ! Coriolis coefficients: shape (4, n1, n2) on device
        type(c_ptr) :: f_4_u_d   = c_null_ptr
        type(c_ptr) :: f_4_v_d   = c_null_ptr
    end type barotropic_CS_cuda_c

    ! =====================================================================
    ! C kernel launch interfaces
    ! =====================================================================
    interface
        subroutine launch_bt_init_kernel( &
                eta, ubt, vbt, ubt_av, vbt_av, uhbt_av, vhbt_av, &
                eta_in, ubt_in, vbt_in, &
                n1, n2, grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_init_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: eta, ubt, vbt, ubt_av, vbt_av, uhbt_av, vhbt_av
            type(c_ptr), value :: eta_in, ubt_in, vbt_in
            integer(c_int), value :: n1, n2
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_store_prev_u_kernel( &
                ubt_prev, ubt, &
                n1, n2, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_store_prev_u_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: ubt_prev, ubt
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_store_prev_v_kernel( &
                vbt_prev, vbt, &
                n1, n2, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_store_prev_v_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: vbt_prev, vbt
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_eta_pred_kernel( &
                eta_pred, eta, ubt, vbt, &
                Datu, Datv, IareaT, &
                n1, n2, is_l, ie_l, js_l, je_l, dtbt, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_eta_pred_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: eta_pred, eta, ubt, vbt
            type(c_ptr), value :: Datu, Datv, IareaT
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            real(c_double), value :: dtbt
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_pressure_force_u_kernel( &
                PFu, eta_pred, gtot_E, gtot_W, IdxCu, &
                n1, n2, is_l, ie_l, js_l, je_l, dgeo_de, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_pressure_force_u_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: PFu, eta_pred, gtot_E, gtot_W, IdxCu
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            real(c_double), value :: dgeo_de
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_pressure_force_v_kernel( &
                PFv, eta_pred, gtot_N, gtot_S, IdyCv, &
                n1, n2, is_l, ie_l, js_l, je_l, dgeo_de, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_pressure_force_v_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: PFv, eta_pred, gtot_N, gtot_S, IdyCv
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            real(c_double), value :: dgeo_de
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_coriolis_update_u_kernel( &
                ubt, Cor_u, vbt, PFu, f_4_u, bt_rem_u, &
                n1, n2, is_l, ie_l, js_l, je_l, dtbt, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_coriolis_update_u_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: ubt, Cor_u, vbt, PFu, f_4_u, bt_rem_u
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            real(c_double), value :: dtbt
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_coriolis_update_v_kernel( &
                vbt, Cor_v, ubt, PFv, f_4_v, bt_rem_v, &
                n1, n2, is_l, ie_l, js_l, je_l, dtbt, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_coriolis_update_v_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: vbt, Cor_v, ubt, PFv, f_4_v, bt_rem_v
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            real(c_double), value :: dtbt
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_transport_eta_accum_kernel( &
                uhbt, vhbt, eta, ubt_av, vbt_av, uhbt_av, vhbt_av, &
                ubt, vbt, ubt_prev, vbt_prev, &
                Datu, Datv, IareaT, &
                n1, n2, is_l, ie_l, js_l, je_l, &
                dtbt, trans_wt1, trans_wt2, inv_nstep, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_transport_eta_accum_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: uhbt, vhbt, eta, ubt_av, vbt_av, uhbt_av, vhbt_av
            type(c_ptr), value :: ubt, vbt, ubt_prev, vbt_prev
            type(c_ptr), value :: Datu, Datv, IareaT
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            real(c_double), value :: dtbt, trans_wt1, trans_wt2, inv_nstep
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_eta_update_kernel( &
                eta, uhbt, vhbt, IareaT, &
                n1, n2, is_l, ie_l, js_l, je_l, dtbt, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_eta_update_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: eta, uhbt, vhbt, IareaT
            integer(c_int), value :: n1, n2, is_l, ie_l, js_l, je_l
            real(c_double), value :: dtbt
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_bt_copy_output_kernel( &
                u_av_out, v_av_out, eta_out, &
                ubt_av, vbt_av, eta, &
                n1, n2, grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_bt_copy_output_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: u_av_out, v_av_out, eta_out
            type(c_ptr), value :: ubt_av, vbt_av, eta
            integer(c_int), value :: n1, n2
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine
    end interface

contains

    ! =====================================================================
    ! Init
    ! =====================================================================
    subroutine barotropic_init_cuda_c(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, &
                                       nstep, dt, bebt, first_direction, &
                                       Datu, Datv, gtot_E, gtot_W, gtot_N, gtot_S, &
                                       f_4_u, f_4_v, bt_rem_u, bt_rem_v, &
                                       IareaT, IdxCu, IdyCv)
        type(barotropic_CS_cuda_c), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec
        integer, intent(in) :: nstep, first_direction
        real(dp), intent(in) :: dt, bebt
        real(dp), intent(in), target :: Datu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: Datv(isd:ied, jsd:jed)
        real(dp), intent(in), target :: gtot_E(isd:ied, jsd:jed)
        real(dp), intent(in), target :: gtot_W(isd:ied, jsd:jed)
        real(dp), intent(in), target :: gtot_N(isd:ied, jsd:jed)
        real(dp), intent(in), target :: gtot_S(isd:ied, jsd:jed)
        real(dp), intent(in), target :: f_4_u(4, isd:ied, jsd:jed)
        real(dp), intent(in), target :: f_4_v(4, isd:ied, jsd:jed)
        real(dp), intent(in), target :: bt_rem_u(isd:ied, jsd:jed)
        real(dp), intent(in), target :: bt_rem_v(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IareaT(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdxCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdyCv(isd:ied, jsd:jed)

        integer :: n1, n2

        ! Store dimensions and parameters
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec
        CS%nstep = nstep
        CS%dtbt = dt / real(nstep, dp)
        CS%bebt = bebt
        CS%dgeo_de = 1.0_dp
        CS%first_direction = first_direction

        n1 = ied - isd + 1
        n2 = jed - jsd + 1

        ! Allocate 2D state/work arrays (zeroed)
        CS%eta      = alloc_zero_2d(n1, n2)
        CS%eta_pred = alloc_zero_2d(n1, n2)
        CS%ubt      = alloc_zero_2d(n1, n2)
        CS%vbt      = alloc_zero_2d(n1, n2)
        CS%ubt_prev = alloc_zero_2d(n1, n2)
        CS%vbt_prev = alloc_zero_2d(n1, n2)
        CS%uhbt     = alloc_zero_2d(n1, n2)
        CS%vhbt     = alloc_zero_2d(n1, n2)
        CS%PFu      = alloc_zero_2d(n1, n2)
        CS%PFv      = alloc_zero_2d(n1, n2)
        CS%Cor_u    = alloc_zero_2d(n1, n2)
        CS%Cor_v    = alloc_zero_2d(n1, n2)
        CS%ubt_av   = alloc_zero_2d(n1, n2)
        CS%vbt_av   = alloc_zero_2d(n1, n2)
        CS%uhbt_av  = alloc_zero_2d(n1, n2)
        CS%vhbt_av  = alloc_zero_2d(n1, n2)

        ! Copy time-invariant 2D grid data to device
        CS%Datu_d    = alloc_copy_2d(Datu, n1, n2)
        CS%Datv_d    = alloc_copy_2d(Datv, n1, n2)
        CS%gtot_E_d  = alloc_copy_2d(gtot_E, n1, n2)
        CS%gtot_W_d  = alloc_copy_2d(gtot_W, n1, n2)
        CS%gtot_N_d  = alloc_copy_2d(gtot_N, n1, n2)
        CS%gtot_S_d  = alloc_copy_2d(gtot_S, n1, n2)
        CS%bt_rem_u_d = alloc_copy_2d(bt_rem_u, n1, n2)
        CS%bt_rem_v_d = alloc_copy_2d(bt_rem_v, n1, n2)
        CS%IareaT_d  = alloc_copy_2d(IareaT, n1, n2)
        CS%IdxCu_d   = alloc_copy_2d(IdxCu, n1, n2)
        CS%IdyCv_d   = alloc_copy_2d(IdyCv, n1, n2)

        ! Coriolis coefficients: shape (4, n1, n2) — use alloc_copy_3d
        CS%f_4_u_d = alloc_copy_3d(f_4_u, 4, n1, n2)
        CS%f_4_v_d = alloc_copy_3d(f_4_v, 4, n1, n2)

        CS%initialized = .true.

    end subroutine barotropic_init_cuda_c

    ! =====================================================================
    ! Main btstep — full nstep substep loop
    ! =====================================================================
    subroutine btstep_cuda_c(eta_in_p, ubt_in_p, vbt_in_p, u_av_p, v_av_p, eta_av_p, &
                              CS, bx_in, by_in)
        type(barotropic_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in)  :: eta_in_p, ubt_in_p, vbt_in_p
        type(c_ptr), intent(in)  :: u_av_p, v_av_p, eta_av_p
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, is_l, ie_l, js_l, je_l, gx, gy, bx, by
        integer(c_int) :: istat, n
        real(dp) :: trans_wt1, trans_wt2, inv_nstep
        logical :: v_first

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

        ! Grid dimensions
        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by

        ! Transport weights and averaging factor
        trans_wt1 = 1.0_dp + CS%bebt
        trans_wt2 = -CS%bebt
        inv_nstep = 1.0_dp / real(CS%nstep, dp)

        ! --- Initialize state from input ---
        call launch_bt_init_kernel( &
            CS%eta, CS%ubt, CS%vbt, CS%ubt_av, CS%vbt_av, CS%uhbt_av, CS%vhbt_av, &
            eta_in_p, ubt_in_p, vbt_in_p, &
            n1, n2, gx, gy, bx, by, c_null_ptr)

        ! --- Substep loop (host-side) ---
        do n = 1, CS%nstep

            ! Store previous velocities
            call launch_bt_store_prev_u_kernel( &
                CS%ubt_prev, CS%ubt, &
                n1, n2, is_l, ie_l, js_l, je_l, &
                gx, gy, bx, by, c_null_ptr)

            call launch_bt_store_prev_v_kernel( &
                CS%vbt_prev, CS%vbt, &
                n1, n2, is_l, ie_l, js_l, je_l, &
                gx, gy, bx, by, c_null_ptr)

            ! Eta predictor
            call launch_bt_eta_pred_kernel( &
                CS%eta_pred, CS%eta, CS%ubt, CS%vbt, &
                CS%Datu_d, CS%Datv_d, CS%IareaT_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                gx, gy, bx, by, c_null_ptr)

            ! Pressure forces
            call launch_bt_pressure_force_u_kernel( &
                CS%PFu, CS%eta_pred, CS%gtot_E_d, CS%gtot_W_d, CS%IdxCu_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de, &
                gx, gy, bx, by, c_null_ptr)

            call launch_bt_pressure_force_v_kernel( &
                CS%PFv, CS%eta_pred, CS%gtot_N_d, CS%gtot_S_d, CS%IdyCv_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de, &
                gx, gy, bx, by, c_null_ptr)

            ! Alternating u/v Coriolis update order
            v_first = (mod(n + CS%first_direction, 2) == 1)

            if (v_first) then
                ! Update v first, then u
                call launch_bt_coriolis_update_v_kernel( &
                    CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v_d, CS%bt_rem_v_d, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                    gx, gy, bx, by, c_null_ptr)

                call launch_bt_coriolis_update_u_kernel( &
                    CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u_d, CS%bt_rem_u_d, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                    gx, gy, bx, by, c_null_ptr)
            else
                ! Update u first, then v
                call launch_bt_coriolis_update_u_kernel( &
                    CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u_d, CS%bt_rem_u_d, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                    gx, gy, bx, by, c_null_ptr)

                call launch_bt_coriolis_update_v_kernel( &
                    CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v_d, CS%bt_rem_v_d, &
                    n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                    gx, gy, bx, by, c_null_ptr)
            end if

            ! Compute transports and accumulate time averages
            call launch_bt_transport_eta_accum_kernel( &
                CS%uhbt, CS%vhbt, CS%eta, CS%ubt_av, CS%vbt_av, &
                CS%uhbt_av, CS%vhbt_av, &
                CS%ubt, CS%vbt, CS%ubt_prev, CS%vbt_prev, &
                CS%Datu_d, CS%Datv_d, CS%IareaT_d, &
                n1, n2, is_l, ie_l, js_l, je_l, &
                CS%dtbt, trans_wt1, trans_wt2, inv_nstep, &
                gx, gy, bx, by, c_null_ptr)

            ! Update eta from transport divergence (must follow transport kernel)
            call launch_bt_eta_update_kernel( &
                CS%eta, CS%uhbt, CS%vhbt, CS%IareaT_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                gx, gy, bx, by, c_null_ptr)

        end do  ! substep loop

        ! --- Copy output ---
        call launch_bt_copy_output_kernel( &
            u_av_p, v_av_p, eta_av_p, &
            CS%ubt_av, CS%vbt_av, CS%eta, &
            n1, n2, gx, gy, bx, by, c_null_ptr)

        ! Sync to ensure all output is ready before returning
        istat = cuda_device_synchronize_c()

    end subroutine btstep_cuda_c

    ! =====================================================================
    ! Split API: init_state (for MPI drivers)
    ! =====================================================================
    subroutine btstep_cuda_c_init_state(CS, eta_in_p, ubt_in_p, vbt_in_p, bx_in, by_in)
        type(barotropic_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in) :: eta_in_p, ubt_in_p, vbt_in_p
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, gx, gy, bx, by

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by

        call launch_bt_init_kernel( &
            CS%eta, CS%ubt, CS%vbt, CS%ubt_av, CS%vbt_av, CS%uhbt_av, CS%vhbt_av, &
            eta_in_p, ubt_in_p, vbt_in_p, &
            n1, n2, gx, gy, bx, by, c_null_ptr)

    end subroutine btstep_cuda_c_init_state

    ! =====================================================================
    ! Split API: do_step (for MPI drivers)
    ! =====================================================================
    subroutine btstep_cuda_c_do_step(CS, n, bx_in, by_in)
        type(barotropic_CS_cuda_c), intent(inout) :: CS
        integer, intent(in) :: n  ! substep number (1-based)
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, is_l, ie_l, js_l, je_l, gx, gy, bx, by
        real(dp) :: trans_wt1, trans_wt2, inv_nstep
        logical :: v_first

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1

        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by

        trans_wt1 = 1.0_dp + CS%bebt
        trans_wt2 = -CS%bebt
        inv_nstep = 1.0_dp / real(CS%nstep, dp)

        ! Store previous velocities
        call launch_bt_store_prev_u_kernel( &
            CS%ubt_prev, CS%ubt, &
            n1, n2, is_l, ie_l, js_l, je_l, &
            gx, gy, bx, by, c_null_ptr)

        call launch_bt_store_prev_v_kernel( &
            CS%vbt_prev, CS%vbt, &
            n1, n2, is_l, ie_l, js_l, je_l, &
            gx, gy, bx, by, c_null_ptr)

        ! Eta predictor
        call launch_bt_eta_pred_kernel( &
            CS%eta_pred, CS%eta, CS%ubt, CS%vbt, &
            CS%Datu_d, CS%Datv_d, CS%IareaT_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
            gx, gy, bx, by, c_null_ptr)

        ! Pressure forces
        call launch_bt_pressure_force_u_kernel( &
            CS%PFu, CS%eta_pred, CS%gtot_E_d, CS%gtot_W_d, CS%IdxCu_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de, &
            gx, gy, bx, by, c_null_ptr)

        call launch_bt_pressure_force_v_kernel( &
            CS%PFv, CS%eta_pred, CS%gtot_N_d, CS%gtot_S_d, CS%IdyCv_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dgeo_de, &
            gx, gy, bx, by, c_null_ptr)

        ! Alternating u/v Coriolis update order
        v_first = (mod(n + CS%first_direction, 2) == 1)

        if (v_first) then
            call launch_bt_coriolis_update_v_kernel( &
                CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v_d, CS%bt_rem_v_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                gx, gy, bx, by, c_null_ptr)

            call launch_bt_coriolis_update_u_kernel( &
                CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u_d, CS%bt_rem_u_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                gx, gy, bx, by, c_null_ptr)
        else
            call launch_bt_coriolis_update_u_kernel( &
                CS%ubt, CS%Cor_u, CS%vbt, CS%PFu, CS%f_4_u_d, CS%bt_rem_u_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                gx, gy, bx, by, c_null_ptr)

            call launch_bt_coriolis_update_v_kernel( &
                CS%vbt, CS%Cor_v, CS%ubt, CS%PFv, CS%f_4_v_d, CS%bt_rem_v_d, &
                n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
                gx, gy, bx, by, c_null_ptr)
        end if

        ! Compute transports and accumulate time averages
        call launch_bt_transport_eta_accum_kernel( &
            CS%uhbt, CS%vhbt, CS%eta, CS%ubt_av, CS%vbt_av, &
            CS%uhbt_av, CS%vhbt_av, &
            CS%ubt, CS%vbt, CS%ubt_prev, CS%vbt_prev, &
            CS%Datu_d, CS%Datv_d, CS%IareaT_d, &
            n1, n2, is_l, ie_l, js_l, je_l, &
            CS%dtbt, trans_wt1, trans_wt2, inv_nstep, &
            gx, gy, bx, by, c_null_ptr)

        ! Update eta from transport divergence
        call launch_bt_eta_update_kernel( &
            CS%eta, CS%uhbt, CS%vhbt, CS%IareaT_d, &
            n1, n2, is_l, ie_l, js_l, je_l, CS%dtbt, &
            gx, gy, bx, by, c_null_ptr)

    end subroutine btstep_cuda_c_do_step

    ! =====================================================================
    ! Split API: get_output (for MPI drivers)
    ! =====================================================================
    subroutine btstep_cuda_c_get_output(CS, u_av_p, v_av_p, eta_av_p, bx_in, by_in)
        type(barotropic_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in) :: u_av_p, v_av_p, eta_av_p
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, gx, gy, bx, by, istat

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by

        call launch_bt_copy_output_kernel( &
            u_av_p, v_av_p, eta_av_p, &
            CS%ubt_av, CS%vbt_av, CS%eta, &
            n1, n2, gx, gy, bx, by, c_null_ptr)

        istat = cuda_device_synchronize_c()

    end subroutine btstep_cuda_c_get_output

    ! =====================================================================
    ! Export internal state to external device pointers (for MPI halo exchange)
    ! =====================================================================
    subroutine btstep_cuda_c_export_state(CS, ubt_ext, vbt_ext, eta_ext)
        type(barotropic_CS_cuda_c), intent(in) :: CS
        type(c_ptr), intent(in) :: ubt_ext, vbt_ext, eta_ext

        integer(c_size_t) :: nbytes
        integer(c_int) :: ierr

        nbytes = int(CS%ied - CS%isd + 1, c_size_t) * &
                 int(CS%jed - CS%jsd + 1, c_size_t) * 8_c_size_t
        ierr = cuda_memcpy_d2d_c(ubt_ext, CS%ubt, nbytes)
        ierr = cuda_memcpy_d2d_c(vbt_ext, CS%vbt, nbytes)
        ierr = cuda_memcpy_d2d_c(eta_ext, CS%eta, nbytes)
    end subroutine btstep_cuda_c_export_state

    ! =====================================================================
    ! Import external device pointers back to internal state (after halo exchange)
    ! =====================================================================
    subroutine btstep_cuda_c_import_state(CS, ubt_ext, vbt_ext, eta_ext)
        type(barotropic_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in) :: ubt_ext, vbt_ext, eta_ext

        integer(c_size_t) :: nbytes
        integer(c_int) :: ierr

        nbytes = int(CS%ied - CS%isd + 1, c_size_t) * &
                 int(CS%jed - CS%jsd + 1, c_size_t) * 8_c_size_t
        ierr = cuda_memcpy_d2d_c(CS%ubt, ubt_ext, nbytes)
        ierr = cuda_memcpy_d2d_c(CS%vbt, vbt_ext, nbytes)
        ierr = cuda_memcpy_d2d_c(CS%eta, eta_ext, nbytes)
    end subroutine btstep_cuda_c_import_state

    ! =====================================================================
    ! End — free all device memory
    ! =====================================================================
    subroutine barotropic_end_cuda_c(CS)
        type(barotropic_CS_cuda_c), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! Free state/work arrays
        call cuda_free_c(CS%eta);       CS%eta = c_null_ptr
        call cuda_free_c(CS%eta_pred);  CS%eta_pred = c_null_ptr
        call cuda_free_c(CS%ubt);       CS%ubt = c_null_ptr
        call cuda_free_c(CS%vbt);       CS%vbt = c_null_ptr
        call cuda_free_c(CS%ubt_prev);  CS%ubt_prev = c_null_ptr
        call cuda_free_c(CS%vbt_prev);  CS%vbt_prev = c_null_ptr
        call cuda_free_c(CS%uhbt);      CS%uhbt = c_null_ptr
        call cuda_free_c(CS%vhbt);      CS%vhbt = c_null_ptr
        call cuda_free_c(CS%PFu);       CS%PFu = c_null_ptr
        call cuda_free_c(CS%PFv);       CS%PFv = c_null_ptr
        call cuda_free_c(CS%Cor_u);     CS%Cor_u = c_null_ptr
        call cuda_free_c(CS%Cor_v);     CS%Cor_v = c_null_ptr
        call cuda_free_c(CS%ubt_av);    CS%ubt_av = c_null_ptr
        call cuda_free_c(CS%vbt_av);    CS%vbt_av = c_null_ptr
        call cuda_free_c(CS%uhbt_av);   CS%uhbt_av = c_null_ptr
        call cuda_free_c(CS%vhbt_av);   CS%vhbt_av = c_null_ptr

        ! Free grid/parameter arrays
        call cuda_free_c(CS%Datu_d);     CS%Datu_d = c_null_ptr
        call cuda_free_c(CS%Datv_d);     CS%Datv_d = c_null_ptr
        call cuda_free_c(CS%gtot_E_d);   CS%gtot_E_d = c_null_ptr
        call cuda_free_c(CS%gtot_W_d);   CS%gtot_W_d = c_null_ptr
        call cuda_free_c(CS%gtot_N_d);   CS%gtot_N_d = c_null_ptr
        call cuda_free_c(CS%gtot_S_d);   CS%gtot_S_d = c_null_ptr
        call cuda_free_c(CS%bt_rem_u_d); CS%bt_rem_u_d = c_null_ptr
        call cuda_free_c(CS%bt_rem_v_d); CS%bt_rem_v_d = c_null_ptr
        call cuda_free_c(CS%IareaT_d);   CS%IareaT_d = c_null_ptr
        call cuda_free_c(CS%IdxCu_d);    CS%IdxCu_d = c_null_ptr
        call cuda_free_c(CS%IdyCv_d);    CS%IdyCv_d = c_null_ptr
        call cuda_free_c(CS%f_4_u_d);    CS%f_4_u_d = c_null_ptr
        call cuda_free_c(CS%f_4_v_d);    CS%f_4_v_d = c_null_ptr

        CS%initialized = .false.

    end subroutine barotropic_end_cuda_c

end module mom6_barotropic_cuda_c
