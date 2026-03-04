!> MOM6 Vertical Viscosity — Portable Fortran wrapper calling CUDA C kernels
!!
!! 1:1 port of mom6_vert_visc_cuda.F90 to the CUDA C + iso_c_binding pattern.
!! Uses ONLY iso_c_binding + iso_fortran_env (Fortran 2003 standard).
!! NO cudafor, NO device attribute, NO vendor-specific extensions.
!!
!! Compilable with any Fortran compiler: gfortran, ifx, nvfortran, amdflang, etc.
!!
module mom6_vert_visc_cuda_c
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding
    use mom6_cuda_c_common
    implicit none
    private

    public :: vert_visc_init_cuda_c, vert_visc_cra_cuda_c, vert_visc_end_cuda_c
    public :: vert_visc_CS_cuda_c

    real(dp), parameter :: RHO_0 = 1035.0_dp

    !> Control structure — all device pointers are opaque type(c_ptr)
    type :: vert_visc_CS_cuda_c
        logical :: initialized = .false.
        integer :: is, ie, js, je, nz
        integer :: isd, ied, jsd, jed

        ! Viscosity parameters (scalars, host-side)
        real(dp) :: Kv, Kv_extra_bbl, Hmix, Hbbl, Kv_ml

        ! Grid metrics on device (2D, copied once at init)
        type(c_ptr) :: mask2dCu_d = c_null_ptr
        type(c_ptr) :: mask2dCv_d = c_null_ptr

        ! Scratch workspace (allocated once at init)
        type(c_ptr) :: hvel_d  = c_null_ptr   ! 3D: n1 x n2 x nz
        type(c_ptr) :: z_i_d   = c_null_ptr   ! 3D: n1 x n2 x (nz+1)

        ! Discard outputs (workspace, never read by driver)
        type(c_ptr) :: visc_rem_d = c_null_ptr ! 3D: n1 x n2 x nz
        type(c_ptr) :: tau_bot_d  = c_null_ptr ! 2D: n1 x n2
    end type vert_visc_CS_cuda_c

    ! =====================================================================
    ! C kernel launch interfaces
    ! =====================================================================
    interface
        subroutine launch_vert_visc_cra_u_kernel( &
                u, h, mask2dCu, visc_rem_u, taux_bot, taux, &
                Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl, &
                dt, dt_Rho0, h_neglect, I_Hbbl, &
                n1, n2, nz, is_l, ie_l, js_l, je_l, &
                hvel, z_i, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_vert_visc_cra_u_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: u, h, mask2dCu, visc_rem_u, taux_bot, taux
            real(c_double), value :: Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl
            real(c_double), value :: dt, dt_Rho0, h_neglect, I_Hbbl
            integer(c_int), value :: n1, n2, nz, is_l, ie_l, js_l, je_l
            type(c_ptr), value :: hvel, z_i
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_vert_visc_cra_v_kernel( &
                v, h, mask2dCv, visc_rem_v, tauy_bot, tauy, &
                Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl, &
                dt, dt_Rho0, h_neglect, I_Hbbl, &
                n1, n2, nz, is_l, ie_l, js_l, je_l, &
                hvel, z_i, &
                grid_x, grid_y, block_x, block_y, stream) &
                bind(c, name='launch_vert_visc_cra_v_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: v, h, mask2dCv, visc_rem_v, tauy_bot, tauy
            real(c_double), value :: Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl
            real(c_double), value :: dt, dt_Rho0, h_neglect, I_Hbbl
            integer(c_int), value :: n1, n2, nz, is_l, ie_l, js_l, je_l
            type(c_ptr), value :: hvel, z_i
            integer(c_int), value :: grid_x, grid_y, block_x, block_y
            type(c_ptr), value :: stream
        end subroutine
    end interface

contains

    ! =====================================================================
    ! Init
    ! =====================================================================
    subroutine vert_visc_init_cuda_c(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nk, &
                                      mask2dCu, mask2dCv, Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl)
        type(vert_visc_CS_cuda_c), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec, nk
        real(dp), intent(in), target :: mask2dCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: mask2dCv(isd:ied, jsd:jed)
        real(dp), intent(in) :: Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl

        integer :: n1, n2

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec; CS%nz = nk

        ! Store viscosity parameters
        CS%Kv = Kv
        CS%Kv_ml = Kv_ml
        CS%Kv_extra_bbl = Kv_extra_bbl
        CS%Hmix = Hmix
        CS%Hbbl = Hbbl

        n1 = ied - isd + 1
        n2 = jed - jsd + 1

        ! Copy 2D grid metrics to device
        CS%mask2dCu_d = alloc_copy_2d(mask2dCu, n1, n2)
        CS%mask2dCv_d = alloc_copy_2d(mask2dCv, n1, n2)

        ! Allocate scratch workspace
        CS%hvel_d     = alloc_zero_3d(n1, n2, nk)        ! 3D: nz layers
        CS%z_i_d      = alloc_zero_3d(n1, n2, nk + 1)    ! 3D: nz+1 layers
        CS%visc_rem_d = alloc_zero_3d(n1, n2, nk)        ! 3D: discard output
        CS%tau_bot_d  = alloc_zero_2d(n1, n2)             ! 2D: discard output

        CS%initialized = .true.

    end subroutine vert_visc_init_cuda_c

    ! =====================================================================
    ! Compute: launch fused coef+remnant+apply kernels for u and v
    ! =====================================================================
    subroutine vert_visc_cra_cuda_c(u_p, v_p, h_p, dt, CS, taux_p, tauy_p, bx_in, by_in)
        type(vert_visc_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in) :: u_p, v_p, h_p
        real(dp), intent(in) :: dt
        type(c_ptr), intent(in) :: taux_p, tauy_p
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, nz, is_l, ie_l, js_l, je_l
        integer(c_int) :: gx, gy, bx, by
        integer(c_int) :: istat
        real(dp) :: dt_Rho0, h_neglect, I_Hbbl

        ! Configurable block dimensions (default: 32 x 4 = 128 threads)
        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        ! Array dimensions (1-based for kernel)
        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        nz = CS%nz

        ! Compute-domain offsets in 1-based local coords
        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        ! Derived constants
        h_neglect = 1.0e-30_dp
        I_Hbbl = 1.0_dp / (CS%Hbbl + h_neglect)
        dt_Rho0 = dt / RHO_0

        ! 2D grid (no z-dimension)
        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by

        ! Launch u-points kernel
        call launch_vert_visc_cra_u_kernel( &
            u_p, h_p, CS%mask2dCu_d, CS%visc_rem_d, CS%tau_bot_d, taux_p, &
            CS%Kv, CS%Kv_ml, CS%Kv_extra_bbl, CS%Hmix, CS%Hbbl, &
            dt, dt_Rho0, h_neglect, I_Hbbl, &
            n1, n2, nz, is_l, ie_l, js_l, je_l, &
            CS%hvel_d, CS%z_i_d, &
            gx, gy, bx, by, c_null_ptr)

        ! Launch v-points kernel (reuse hvel and z_i scratch)
        call launch_vert_visc_cra_v_kernel( &
            v_p, h_p, CS%mask2dCv_d, CS%visc_rem_d, CS%tau_bot_d, tauy_p, &
            CS%Kv, CS%Kv_ml, CS%Kv_extra_bbl, CS%Hmix, CS%Hbbl, &
            dt, dt_Rho0, h_neglect, I_Hbbl, &
            n1, n2, nz, is_l, ie_l, js_l, je_l, &
            CS%hvel_d, CS%z_i_d, &
            gx, gy, bx, by, c_null_ptr)

        ! Sync to ensure all output is ready before returning
        istat = cuda_device_synchronize_c()

    end subroutine vert_visc_cra_cuda_c

    ! =====================================================================
    ! End
    ! =====================================================================
    subroutine vert_visc_end_cuda_c(CS)
        type(vert_visc_CS_cuda_c), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! Free 2D metric arrays
        call cuda_free_c(CS%mask2dCu_d);  CS%mask2dCu_d = c_null_ptr
        call cuda_free_c(CS%mask2dCv_d);  CS%mask2dCv_d = c_null_ptr

        ! Free scratch workspace
        call cuda_free_c(CS%hvel_d);      CS%hvel_d = c_null_ptr
        call cuda_free_c(CS%z_i_d);       CS%z_i_d = c_null_ptr
        call cuda_free_c(CS%visc_rem_d);  CS%visc_rem_d = c_null_ptr
        call cuda_free_c(CS%tau_bot_d);   CS%tau_bot_d = c_null_ptr

        CS%initialized = .false.

    end subroutine vert_visc_end_cuda_c

end module mom6_vert_visc_cuda_c
