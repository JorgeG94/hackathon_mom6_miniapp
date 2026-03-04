!> MOM6 Continuity PPM — Portable Fortran wrapper calling CUDA C kernels
!!
!! This module uses ONLY iso_c_binding + iso_fortran_env (Fortran 2003 standard).
!! NO cudafor, NO device attribute, NO vendor-specific extensions.
!! All GPU memory is managed through C helper functions in mom6_continuity_kernels.cu.
!!
!! Compilable with any Fortran compiler: gfortran, ifx, nvfortran, amdflang, etc.
!!
module mom6_continuity_cuda_c
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding
    use mom6_cuda_c_common
    implicit none
    private

    public :: continuity_init_cuda_c, continuity_PPM_cuda_c, continuity_end_cuda_c
    public :: continuity_CS_cuda_c
    public :: BT_cont_type_cuda_c, alloc_BT_cont_type_cuda_c, dealloc_BT_cont_type_cuda_c

    !> Control structure — all device pointers are opaque type(c_ptr)
    type :: continuity_CS_cuda_c
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

        !> Device 2D grid metrics (persistent, copied once at init)
        type(c_ptr) :: IareaT_d = c_null_ptr
        type(c_ptr) :: IdxT_d = c_null_ptr
        type(c_ptr) :: dy_Cu_d = c_null_ptr
        type(c_ptr) :: mask2dT_d = c_null_ptr
        type(c_ptr) :: dxT_d = c_null_ptr
        type(c_ptr) :: areaT_d = c_null_ptr
        type(c_ptr) :: dxCu_d = c_null_ptr
        type(c_ptr) :: mask2dCu_d = c_null_ptr

        !> 3D scratch (h_W, h_E, duhdu)
        type(c_ptr) :: s3d(3) = c_null_ptr

        !> 2D scratch (du, du_max_CFL, du_min_CFL, duhdu_tot_0, uh_tot_0, visc_rem_max)
        type(c_ptr) :: s2d(6) = c_null_ptr
    end type continuity_CS_cuda_c

    !> BT_cont type with device arrays
    type :: BT_cont_type_cuda_c
        type(c_ptr) :: FA_u_EE = c_null_ptr
        type(c_ptr) :: FA_u_E0 = c_null_ptr
        type(c_ptr) :: FA_u_W0 = c_null_ptr
        type(c_ptr) :: FA_u_WW = c_null_ptr
        type(c_ptr) :: uBT_WW = c_null_ptr
        type(c_ptr) :: uBT_EE = c_null_ptr
        type(c_ptr) :: h_u = c_null_ptr
    end type BT_cont_type_cuda_c

    ! =====================================================================
    ! C kernel launch interfaces
    ! =====================================================================
    interface
        subroutine launch_ppm_reconstruction_3d_kernel( &
                h_W, h_E, h_in, mask2dT, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                h_min, monotonic, upwind_1st_flag, simple_2nd_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_ppm_reconstruction_3d_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: h_W, h_E, h_in, mask2dT
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            real(c_double), value :: h_min
            integer(c_int), value :: monotonic, upwind_1st_flag, simple_2nd_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_zonal_flux_layer_3d_kernel( &
                uh, duhdu, u, h_in, h_W, h_E, &
                dy_Cu, IdxT, IareaT, por_face_areaU, visc_rem_u, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                dt, vol_CFL_flag, use_visc_rem_flag, use_por_face_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_zonal_flux_layer_3d_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: uh, duhdu, u, h_in, h_W, h_E
            type(c_ptr), value :: dy_Cu, IdxT, IareaT, por_face_areaU, visc_rem_u
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            real(c_double), value :: dt
            integer(c_int), value :: vol_CFL_flag, use_visc_rem_flag, use_por_face_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_zonal_convergence_kernel( &
                h, hin, uh, IareaT, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_zonal_convergence_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: h, hin, uh, IareaT
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            real(c_double), value :: dt
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_visc_rem_max_kernel( &
                visc_rem_max_out, visc_rem_u, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                use_vrm_max_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_visc_rem_max_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: visc_rem_max_out, visc_rem_u
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: use_vrm_max_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_uh_duhdu_tot_kernel( &
                uh_tot_0, duhdu_tot_0, uh, duhdu, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_uh_duhdu_tot_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: uh_tot_0, duhdu_tot_0, uh, duhdu
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_zonal_CFL_limits_kernel( &
                du_max_CFL_out, du_min_CFL_out, &
                u, visc_rem_u, visc_rem_max_in, &
                dxT, areaT, dy_Cu, mask2dCu, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                CFL_dt, I_dt, &
                vol_CFL_flag, use_visc_rem_flag, aggress_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_zonal_CFL_limits_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: du_max_CFL_out, du_min_CFL_out
            type(c_ptr), value :: u, visc_rem_u, visc_rem_max_in
            type(c_ptr), value :: dxT, areaT, dy_Cu, mask2dCu
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            real(c_double), value :: CFL_dt, I_dt
            integer(c_int), value :: vol_CFL_flag, use_visc_rem_flag, aggress_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_zonal_flux_adjust_kernel( &
                uh, du_out, u, h_in, h_W, h_E, uhbt, &
                visc_rem_u, por_face_areaU, &
                IareaT, dy_Cu, IdxT, &
                du_max_CFL_in, du_min_CFL_in, &
                uh_tot_0_in, duhdu_tot_0_in, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                dt, tol_eta_base, tol_vel_val, &
                vol_CFL_flag, use_visc_rem_flag, better_iter_flag, &
                use_por_face_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_zonal_flux_adjust_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: uh, du_out, u, h_in, h_W, h_E, uhbt
            type(c_ptr), value :: visc_rem_u, por_face_areaU
            type(c_ptr), value :: IareaT, dy_Cu, IdxT
            type(c_ptr), value :: du_max_CFL_in, du_min_CFL_in
            type(c_ptr), value :: uh_tot_0_in, duhdu_tot_0_in
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            real(c_double), value :: dt, tol_eta_base, tol_vel_val
            integer(c_int), value :: vol_CFL_flag, use_visc_rem_flag, better_iter_flag
            integer(c_int), value :: use_por_face_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_set_zonal_BT_cont_kernel( &
                FA_u_W0, FA_u_WW, uBT_WW, FA_u_E0, FA_u_EE, uBT_EE, &
                u, h_in, h_W, h_E, &
                visc_rem_u, por_face_areaU, &
                IareaT, dy_Cu, IdxT, dxCu, visc_rem_max_in, &
                du_max_CFL_in, du_min_CFL_in, &
                uh_tot_0_in, duhdu_tot_0_in, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                dt, tol_eta_base, tol_vel_val, &
                vol_CFL_flag, use_visc_rem_flag, better_iter_flag, &
                use_por_face_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_set_zonal_BT_cont_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: FA_u_W0, FA_u_WW, uBT_WW
            type(c_ptr), value :: FA_u_E0, FA_u_EE, uBT_EE
            type(c_ptr), value :: u, h_in, h_W, h_E
            type(c_ptr), value :: visc_rem_u, por_face_areaU
            type(c_ptr), value :: IareaT, dy_Cu, IdxT, dxCu, visc_rem_max_in
            type(c_ptr), value :: du_max_CFL_in, du_min_CFL_in
            type(c_ptr), value :: uh_tot_0_in, duhdu_tot_0_in
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            real(c_double), value :: dt, tol_eta_base, tol_vel_val
            integer(c_int), value :: vol_CFL_flag, use_visc_rem_flag, better_iter_flag
            integer(c_int), value :: use_por_face_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_zonal_flux_thickness_kernel( &
                h_u, u, h, h_W, h_E, &
                por_face_areaU, visc_rem_u, &
                IareaT, IdxT, dy_Cu, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                dt, &
                vol_CFL_flag, marginal_flag, has_visc_rem_flag, use_por_face_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_zonal_flux_thickness_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: h_u, u, h, h_W, h_E
            type(c_ptr), value :: por_face_areaU, visc_rem_u
            type(c_ptr), value :: IareaT, IdxT, dy_Cu
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            real(c_double), value :: dt
            integer(c_int), value :: vol_CFL_flag, marginal_flag, has_visc_rem_flag, use_por_face_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_u_cor_kernel( &
                u_cor, u, du, visc_rem_u, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                use_visc_rem_flag, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_u_cor_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: u_cor, u, du, visc_rem_u
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: use_visc_rem_flag
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine
    end interface

contains

    ! =====================================================================
    ! Init
    ! =====================================================================
    subroutine continuity_init_cuda_c(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nk, &
                                       IareaT, IdxT, dy_Cu, mask2dT, monotonic, &
                                       dxT, areaT, dxCu, mask2dCu)
        type(continuity_CS_cuda_c), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec, nk
        real(dp), intent(in), target :: IareaT(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdxT(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dy_Cu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: mask2dT(isd:ied, jsd:jed)
        logical, intent(in) :: monotonic
        real(dp), intent(in), target, optional :: dxT(isd:ied, jsd:jed)
        real(dp), intent(in), target, optional :: areaT(isd:ied, jsd:jed)
        real(dp), intent(in), target, optional :: dxCu(isd:ied, jsd:jed)
        real(dp), intent(in), target, optional :: mask2dCu(isd:ied, jsd:jed)

        integer :: n1, n2, slot

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

        n1 = ied - isd + 1
        n2 = jed - jsd + 1

        ! Copy 2D grid metrics to device
        CS%IareaT_d  = alloc_copy_2d(IareaT, n1, n2)
        CS%IdxT_d    = alloc_copy_2d(IdxT, n1, n2)
        CS%dy_Cu_d   = alloc_copy_2d(dy_Cu, n1, n2)
        CS%mask2dT_d = alloc_copy_2d(mask2dT, n1, n2)

        ! Optional 2D grid metrics
        if (present(dxT)) then
            CS%dxT_d = alloc_copy_2d(dxT, n1, n2)
        end if
        if (present(areaT)) then
            CS%areaT_d = alloc_copy_2d(areaT, n1, n2)
        end if
        if (present(dxCu)) then
            CS%dxCu_d = alloc_copy_2d(dxCu, n1, n2)
        end if
        if (present(mask2dCu)) then
            CS%mask2dCu_d = alloc_copy_2d(mask2dCu, n1, n2)
        end if

        ! Allocate 3D scratch (h_W, h_E, duhdu)
        do slot = 1, 3
            CS%s3d(slot) = alloc_zero_3d(n1, n2, nk)
        end do

        ! Allocate 2D scratch (du, du_max_CFL, du_min_CFL, duhdu_tot_0, uh_tot_0, visc_rem_max)
        do slot = 1, 6
            CS%s2d(slot) = alloc_zero_2d(n1, n2)
        end do

        CS%initialized = .true.

    end subroutine continuity_init_cuda_c

    ! =====================================================================
    ! End
    ! =====================================================================
    subroutine continuity_end_cuda_c(CS)
        type(continuity_CS_cuda_c), intent(inout) :: CS
        integer :: slot

        if (.not. CS%initialized) return

        ! Free 2D metric arrays
        call cuda_free_c(CS%IareaT_d);  CS%IareaT_d = c_null_ptr
        call cuda_free_c(CS%IdxT_d);    CS%IdxT_d = c_null_ptr
        call cuda_free_c(CS%dy_Cu_d);   CS%dy_Cu_d = c_null_ptr
        call cuda_free_c(CS%mask2dT_d); CS%mask2dT_d = c_null_ptr

        if (c_associated(CS%dxT_d)) then
            call cuda_free_c(CS%dxT_d); CS%dxT_d = c_null_ptr
        end if
        if (c_associated(CS%areaT_d)) then
            call cuda_free_c(CS%areaT_d); CS%areaT_d = c_null_ptr
        end if
        if (c_associated(CS%dxCu_d)) then
            call cuda_free_c(CS%dxCu_d); CS%dxCu_d = c_null_ptr
        end if
        if (c_associated(CS%mask2dCu_d)) then
            call cuda_free_c(CS%mask2dCu_d); CS%mask2dCu_d = c_null_ptr
        end if

        ! Free 3D scratch
        do slot = 1, 3
            call cuda_free_c(CS%s3d(slot))
            CS%s3d(slot) = c_null_ptr
        end do

        ! Free 2D scratch
        do slot = 1, 6
            call cuda_free_c(CS%s2d(slot))
            CS%s2d(slot) = c_null_ptr
        end do

        CS%initialized = .false.

    end subroutine continuity_end_cuda_c

    ! =====================================================================
    ! Allocate BT_cont_type_cuda_c
    ! =====================================================================
    subroutine alloc_BT_cont_type_cuda_c(BT_cont, isd, ied, jsd, jed, nz)
        type(BT_cont_type_cuda_c), intent(inout) :: BT_cont
        integer, intent(in) :: isd, ied, jsd, jed, nz
        integer :: n1, n2

        n1 = ied - isd + 1
        n2 = jed - jsd + 1

        BT_cont%FA_u_W0 = alloc_zero_2d(n1, n2)
        BT_cont%FA_u_WW = alloc_zero_2d(n1, n2)
        BT_cont%FA_u_E0 = alloc_zero_2d(n1, n2)
        BT_cont%FA_u_EE = alloc_zero_2d(n1, n2)
        BT_cont%uBT_WW  = alloc_zero_2d(n1, n2)
        BT_cont%uBT_EE  = alloc_zero_2d(n1, n2)
        BT_cont%h_u     = alloc_zero_3d(n1, n2, nz)

    end subroutine alloc_BT_cont_type_cuda_c

    ! =====================================================================
    ! Deallocate BT_cont_type_cuda_c
    ! =====================================================================
    subroutine dealloc_BT_cont_type_cuda_c(BT_cont)
        type(BT_cont_type_cuda_c), intent(inout) :: BT_cont

        if (c_associated(BT_cont%FA_u_W0)) then
            call cuda_free_c(BT_cont%FA_u_W0); BT_cont%FA_u_W0 = c_null_ptr
        end if
        if (c_associated(BT_cont%FA_u_WW)) then
            call cuda_free_c(BT_cont%FA_u_WW); BT_cont%FA_u_WW = c_null_ptr
        end if
        if (c_associated(BT_cont%FA_u_E0)) then
            call cuda_free_c(BT_cont%FA_u_E0); BT_cont%FA_u_E0 = c_null_ptr
        end if
        if (c_associated(BT_cont%FA_u_EE)) then
            call cuda_free_c(BT_cont%FA_u_EE); BT_cont%FA_u_EE = c_null_ptr
        end if
        if (c_associated(BT_cont%uBT_WW)) then
            call cuda_free_c(BT_cont%uBT_WW); BT_cont%uBT_WW = c_null_ptr
        end if
        if (c_associated(BT_cont%uBT_EE)) then
            call cuda_free_c(BT_cont%uBT_EE); BT_cont%uBT_EE = c_null_ptr
        end if
        if (c_associated(BT_cont%h_u)) then
            call cuda_free_c(BT_cont%h_u); BT_cont%h_u = c_null_ptr
        end if

    end subroutine dealloc_BT_cont_type_cuda_c

    ! =====================================================================
    ! continuity_PPM_cuda_c — orchestrates all 10 kernels
    !
    ! Input/output arrays are passed as type(c_ptr) device pointers.
    ! uhbt_p = c_null_ptr when barotropic transport is not being matched.
    ! BT_cont is optional; when absent, BT_cont face-area computation is skipped.
    ! =====================================================================
    subroutine continuity_PPM_cuda_c(u_p, hin_p, h_p, uh_p, dt, CS, &
                                      uhbt_p, BT_cont, bx_in, by_in)
        type(continuity_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in)  :: u_p, hin_p
        type(c_ptr), intent(in)  :: h_p        ! output
        type(c_ptr), intent(in)  :: uh_p       ! output (inout for adjust)
        real(dp), intent(in)     :: dt
        type(c_ptr), intent(in), optional :: uhbt_p
        type(BT_cont_type_cuda_c), intent(inout), optional :: BT_cont
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        integer(c_int) :: gx3, gy3, gz3, gx2, gy2
        integer(c_int) :: bx, by
        integer(c_int) :: mono_flag, upwind_flag, simple_flag
        integer(c_int) :: vol_CFL_flag, aggress_flag, better_flag, marginal_flag
        integer(c_int) :: istat
        real(dp) :: h_min, CFL_dt, I_dt
        logical :: has_uhbt, set_BT_cont
        type(c_ptr) :: uhbt_ptr

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

        ! Solver flags — visc_rem and por_face_areaU are always 1.0, use flags=0
        vol_CFL_flag = 0; if (CS%vol_CFL) vol_CFL_flag = 1
        aggress_flag = 0; if (CS%aggress_adjust) aggress_flag = 1
        better_flag = 0; if (CS%better_iter) better_flag = 1
        marginal_flag = 0; if (CS%marginal_faces) marginal_flag = 1

        ! uhbt handling
        has_uhbt = .false.
        uhbt_ptr = c_null_ptr
        if (present(uhbt_p)) then
            if (c_associated(uhbt_p)) then
                has_uhbt = .true.
                uhbt_ptr = uhbt_p
            end if
        end if

        set_BT_cont = .false.
        if (present(BT_cont)) set_BT_cont = .true.

        CFL_dt = CS%CFL_limit_adjust / dt
        I_dt = 1.0_dp / dt
        if (CS%aggress_adjust) CFL_dt = I_dt

        ! Grid configurations
        gx3 = (n1 + bx - 1) / bx
        gy3 = (n2 + by - 1) / by
        gz3 = n3
        gx2 = gx3
        gy2 = gy3

        ! Scratch slot aliases:
        ! 3D: s3d(1) = h_W, s3d(2) = h_E, s3d(3) = duhdu
        ! 2D: s2d(1) = du, s2d(2) = du_max_CFL, s2d(3) = du_min_CFL
        !     s2d(4) = duhdu_tot_0, s2d(5) = uh_tot_0, s2d(6) = visc_rem_max

        ! --- Kernel 1: PPM reconstruction ---
        call launch_ppm_reconstruction_3d_kernel( &
            CS%s3d(1), CS%s3d(2), hin_p, CS%mask2dT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            h_min, mono_flag, upwind_flag, simple_flag, &
            gx3, gy3, gz3, bx, by, 1_c_int, c_null_ptr)

        ! --- Kernel 2: Zonal flux + duhdu ---
        ! por_face_areaU/visc_rem_u dummy (flags=0, not read) — pass s3d(1) as placeholder
        call launch_zonal_flux_layer_3d_kernel( &
            uh_p, CS%s3d(3), u_p, hin_p, &
            CS%s3d(1), CS%s3d(2), &
            CS%dy_Cu_d, CS%IdxT_d, CS%IareaT_d, &
            CS%s3d(1), CS%s3d(1), &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            dt, vol_CFL_flag, 0_c_int, 0_c_int, &
            gx3, gy3, gz3, bx, by, 1_c_int, c_null_ptr)

        ! --- Steps 3-5: Adjustment kernels (only when uhbt or BT_cont present) ---
        if (has_uhbt .or. set_BT_cont) then

            ! 3a. visc_rem_max — flag=0 writes 1.0 (no visc_rem)
            call launch_visc_rem_max_kernel( &
                CS%s2d(6), CS%s3d(1), &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                0_c_int, &
                gx2, gy2, 1_c_int, bx, by, 1_c_int, c_null_ptr)

            ! 3b. uh_tot_0 + duhdu_tot_0
            call launch_uh_duhdu_tot_kernel( &
                CS%s2d(5), CS%s2d(4), uh_p, CS%s3d(3), &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                gx2, gy2, 1_c_int, bx, by, 1_c_int, c_null_ptr)

            ! 3c. CFL limits
            call launch_zonal_CFL_limits_kernel( &
                CS%s2d(2), CS%s2d(3), u_p, &
                CS%s3d(1), CS%s2d(6), &
                CS%dxT_d, CS%areaT_d, CS%dy_Cu_d, CS%mask2dCu_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                CFL_dt, I_dt, vol_CFL_flag, 0_c_int, aggress_flag, &
                gx2, gy2, 1_c_int, bx, by, 1_c_int, c_null_ptr)

            ! 4. Newton flux adjustment (when uhbt present)
            if (has_uhbt) then
                call launch_zonal_flux_adjust_kernel( &
                    uh_p, CS%s2d(1), u_p, hin_p, &
                    CS%s3d(1), CS%s3d(2), uhbt_ptr, &
                    CS%s3d(1), CS%s3d(1), &
                    CS%IareaT_d, CS%dy_Cu_d, CS%IdxT_d, &
                    CS%s2d(2), CS%s2d(3), CS%s2d(5), CS%s2d(4), &
                    n1, n2, n3, is_l, ie_l, js_l, je_l, &
                    dt, CS%tol_eta, CS%tol_vel, &
                    vol_CFL_flag, 0_c_int, better_flag, 0_c_int, &
                    gx2, gy2, 1_c_int, bx, by, 1_c_int, c_null_ptr)
            end if

            ! 5. BT_cont computation
            if (set_BT_cont) then
                call launch_set_zonal_BT_cont_kernel( &
                    BT_cont%FA_u_W0, BT_cont%FA_u_WW, BT_cont%uBT_WW, &
                    BT_cont%FA_u_E0, BT_cont%FA_u_EE, BT_cont%uBT_EE, &
                    u_p, hin_p, CS%s3d(1), CS%s3d(2), &
                    CS%s3d(1), CS%s3d(1), &
                    CS%IareaT_d, CS%dy_Cu_d, CS%IdxT_d, CS%dxCu_d, CS%s2d(6), &
                    CS%s2d(2), CS%s2d(3), CS%s2d(5), CS%s2d(4), &
                    n1, n2, n3, is_l, ie_l, js_l, je_l, &
                    dt, CS%tol_eta, CS%tol_vel, &
                    vol_CFL_flag, 0_c_int, better_flag, 0_c_int, &
                    gx2, gy2, 1_c_int, bx, by, 1_c_int, c_null_ptr)

                ! Flux thickness
                if (c_associated(BT_cont%h_u)) then
                    call launch_zonal_flux_thickness_kernel( &
                        BT_cont%h_u, u_p, hin_p, &
                        CS%s3d(1), CS%s3d(2), &
                        CS%s3d(1), CS%s3d(1), &
                        CS%IareaT_d, CS%IdxT_d, CS%dy_Cu_d, &
                        n1, n2, n3, is_l, ie_l, js_l, je_l, &
                        dt, &
                        vol_CFL_flag, marginal_flag, 0_c_int, 0_c_int, &
                        gx3, gy3, gz3, bx, by, 1_c_int, c_null_ptr)
                end if
            end if
        end if

        ! --- Kernel: Zonal convergence (always) ---
        call launch_zonal_convergence_kernel( &
            h_p, hin_p, uh_p, CS%IareaT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt, &
            gx3, gy3, gz3, bx, by, 1_c_int, c_null_ptr)

        ! Sync
        istat = cuda_device_synchronize_c()

    end subroutine continuity_PPM_cuda_c

end module mom6_continuity_cuda_c
