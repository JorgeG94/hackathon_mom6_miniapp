!> MOM6 Horizontal Viscosity — Portable Fortran wrapper calling CUDA C kernels
!!
!! This module uses ONLY iso_c_binding + iso_fortran_env (Fortran 2003 standard).
!! NO cudafor, NO device attribute, NO vendor-specific extensions.
!! All GPU memory is managed through C helper functions.
!!
!! 1:1 port of mom6_hor_visc_cuda.F90 to the CUDA C + Fortran wrapper pattern.
!!
!! Two kernels:
!!   1. stress_kernel   — vel_grad + strain + stress (str_xx, str_xy)
!!   2. divergence_kernel — viscous acceleration (diffu, diffv)
!!
!! Compilable with any Fortran compiler: gfortran, ifx, nvfortran, amdflang, etc.
!!
module mom6_hor_visc_cuda_c
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding
    use mom6_cuda_c_common
    implicit none
    private

    public :: hor_visc_cuda_c, hor_visc_init_cuda_c, hor_visc_end_cuda_c
    public :: hor_visc_CS_cuda_c

    integer, parameter :: NUM_SCRATCH_3D = 2  ! str_xx, str_xy

    !> Control structure — all device pointers are opaque type(c_ptr)
    type :: hor_visc_CS_cuda_c
        logical :: initialized = .false.
        integer :: is, ie, js, je, nz
        integer :: isd, ied, jsd, jed

        ! Scalar parameters
        real(dp) :: Kh_bg                    ! Background Laplacian viscosity
        real(dp) :: h_neglect

        ! 2D device metric arrays (copied once at init)
        type(c_ptr) :: DY_dxT_d   = c_null_ptr  ! dy/dx at h-points
        type(c_ptr) :: DX_dyT_d   = c_null_ptr  ! dx/dy at h-points
        type(c_ptr) :: DY_dxBu_d  = c_null_ptr  ! dy/dx at q-points
        type(c_ptr) :: DX_dyBu_d  = c_null_ptr  ! dx/dy at q-points
        type(c_ptr) :: IdyCu_d    = c_null_ptr
        type(c_ptr) :: IdxCu_d    = c_null_ptr
        type(c_ptr) :: IdyCv_d    = c_null_ptr
        type(c_ptr) :: IdxCv_d    = c_null_ptr
        type(c_ptr) :: IareaCu_d  = c_null_ptr
        type(c_ptr) :: IareaCv_d  = c_null_ptr
        type(c_ptr) :: mask2dT_d  = c_null_ptr
        type(c_ptr) :: mask2dBu_d = c_null_ptr
        type(c_ptr) :: reduction_xx_d = c_null_ptr
        type(c_ptr) :: reduction_xy_d = c_null_ptr
        type(c_ptr) :: dy2h_d     = c_null_ptr
        type(c_ptr) :: dx2h_d     = c_null_ptr
        type(c_ptr) :: dy2q_d     = c_null_ptr
        type(c_ptr) :: dx2q_d     = c_null_ptr

        ! Scratch workspace (2 x 3D slots: str_xx, str_xy)
        type(c_ptr) :: scratch(NUM_SCRATCH_3D) = c_null_ptr
    end type hor_visc_CS_cuda_c

    ! =====================================================================
    ! C kernel launch interfaces
    ! =====================================================================
    interface
        subroutine launch_stress_kernel( &
                str_xx, str_xy, u, v, h, &
                DY_dxT, DX_dyT, DY_dxBu, DX_dyBu, &
                IdyCu, IdxCu, IdyCv, IdxCv, &
                mask2dBu, reduction_xx, reduction_xy, &
                Kh_bg, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_stress_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: str_xx, str_xy, u, v, h
            type(c_ptr), value :: DY_dxT, DX_dyT, DY_dxBu, DX_dyBu
            type(c_ptr), value :: IdyCu, IdxCu, IdyCv, IdxCv
            type(c_ptr), value :: mask2dBu, reduction_xx, reduction_xy
            real(c_double), value :: Kh_bg
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_divergence_kernel( &
                diffu, diffv, &
                str_xx, str_xy, h, &
                mask2dT, &
                IdyCu, IdxCu, IdyCv, IdxCv, &
                IareaCu, IareaCv, &
                dy2h, dx2h, dy2q, dx2q, &
                h_neglect, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_divergence_kernel')
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: diffu, diffv
            type(c_ptr), value :: str_xx, str_xy, h
            type(c_ptr), value :: mask2dT
            type(c_ptr), value :: IdyCu, IdxCu, IdyCv, IdxCv
            type(c_ptr), value :: IareaCu, IareaCv
            type(c_ptr), value :: dy2h, dx2h, dy2q, dx2q
            real(c_double), value :: h_neglect
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine
    end interface

contains

    ! =====================================================================
    ! Init — copy metric arrays to device, allocate scratch workspace
    ! =====================================================================
    subroutine hor_visc_init_cuda_c(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nz, &
                                     Kh, h_neglect, &
                                     DY_dxT, DX_dyT, DY_dxBu, DX_dyBu, &
                                     IdyCu, IdxCu, IdyCv, IdxCv, &
                                     IareaCu, IareaCv, &
                                     mask2dT, mask2dBu, &
                                     reduction_xx, reduction_xy, &
                                     dy2h, dx2h, dy2q, dx2q)
        type(hor_visc_CS_cuda_c), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed
        integer, intent(in) :: isc, iec, jsc, jec, nz
        real(dp), intent(in) :: Kh
        real(dp), intent(in) :: h_neglect
        real(dp), intent(in), target :: DY_dxT(isd:ied, jsd:jed)
        real(dp), intent(in), target :: DX_dyT(isd:ied, jsd:jed)
        real(dp), intent(in), target :: DY_dxBu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: DX_dyBu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdyCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdxCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdyCv(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdxCv(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IareaCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IareaCv(isd:ied, jsd:jed)
        real(dp), intent(in), target :: mask2dT(isd:ied, jsd:jed)
        real(dp), intent(in), target :: mask2dBu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: reduction_xx(isd:ied, jsd:jed)
        real(dp), intent(in), target :: reduction_xy(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dy2h(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dx2h(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dy2q(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dx2q(isd:ied, jsd:jed)

        integer :: n1, n2, slot

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec
        CS%nz = nz

        ! Store scalar parameters
        CS%Kh_bg = Kh
        CS%h_neglect = h_neglect

        n1 = ied - isd + 1
        n2 = jed - jsd + 1

        ! Copy 2D metric arrays to device
        CS%DY_dxT_d      = alloc_copy_2d(DY_dxT, n1, n2)
        CS%DX_dyT_d      = alloc_copy_2d(DX_dyT, n1, n2)
        CS%DY_dxBu_d     = alloc_copy_2d(DY_dxBu, n1, n2)
        CS%DX_dyBu_d     = alloc_copy_2d(DX_dyBu, n1, n2)
        CS%IdyCu_d       = alloc_copy_2d(IdyCu, n1, n2)
        CS%IdxCu_d       = alloc_copy_2d(IdxCu, n1, n2)
        CS%IdyCv_d       = alloc_copy_2d(IdyCv, n1, n2)
        CS%IdxCv_d       = alloc_copy_2d(IdxCv, n1, n2)
        CS%IareaCu_d     = alloc_copy_2d(IareaCu, n1, n2)
        CS%IareaCv_d     = alloc_copy_2d(IareaCv, n1, n2)
        CS%mask2dT_d     = alloc_copy_2d(mask2dT, n1, n2)
        CS%mask2dBu_d    = alloc_copy_2d(mask2dBu, n1, n2)
        CS%reduction_xx_d = alloc_copy_2d(reduction_xx, n1, n2)
        CS%reduction_xy_d = alloc_copy_2d(reduction_xy, n1, n2)
        CS%dy2h_d        = alloc_copy_2d(dy2h, n1, n2)
        CS%dx2h_d        = alloc_copy_2d(dx2h, n1, n2)
        CS%dy2q_d        = alloc_copy_2d(dy2q, n1, n2)
        CS%dx2q_d        = alloc_copy_2d(dx2q, n1, n2)

        ! Allocate scratch workspace (2 x 3D slots for str_xx, str_xy)
        do slot = 1, NUM_SCRATCH_3D
            CS%scratch(slot) = alloc_zero_3d(n1, n2, nz)
        end do

        CS%initialized = .true.

    end subroutine hor_visc_init_cuda_c

    ! =====================================================================
    ! End — free all device arrays
    ! =====================================================================
    subroutine hor_visc_end_cuda_c(CS)
        type(hor_visc_CS_cuda_c), intent(inout) :: CS
        integer :: slot

        if (.not. CS%initialized) return

        ! Free 2D metric arrays
        call cuda_free_c(CS%DY_dxT_d);      CS%DY_dxT_d = c_null_ptr
        call cuda_free_c(CS%DX_dyT_d);      CS%DX_dyT_d = c_null_ptr
        call cuda_free_c(CS%DY_dxBu_d);     CS%DY_dxBu_d = c_null_ptr
        call cuda_free_c(CS%DX_dyBu_d);     CS%DX_dyBu_d = c_null_ptr
        call cuda_free_c(CS%IdyCu_d);       CS%IdyCu_d = c_null_ptr
        call cuda_free_c(CS%IdxCu_d);       CS%IdxCu_d = c_null_ptr
        call cuda_free_c(CS%IdyCv_d);       CS%IdyCv_d = c_null_ptr
        call cuda_free_c(CS%IdxCv_d);       CS%IdxCv_d = c_null_ptr
        call cuda_free_c(CS%IareaCu_d);     CS%IareaCu_d = c_null_ptr
        call cuda_free_c(CS%IareaCv_d);     CS%IareaCv_d = c_null_ptr
        call cuda_free_c(CS%mask2dT_d);     CS%mask2dT_d = c_null_ptr
        call cuda_free_c(CS%mask2dBu_d);    CS%mask2dBu_d = c_null_ptr
        call cuda_free_c(CS%reduction_xx_d); CS%reduction_xx_d = c_null_ptr
        call cuda_free_c(CS%reduction_xy_d); CS%reduction_xy_d = c_null_ptr
        call cuda_free_c(CS%dy2h_d);        CS%dy2h_d = c_null_ptr
        call cuda_free_c(CS%dx2h_d);        CS%dx2h_d = c_null_ptr
        call cuda_free_c(CS%dy2q_d);        CS%dy2q_d = c_null_ptr
        call cuda_free_c(CS%dx2q_d);        CS%dx2q_d = c_null_ptr

        ! Free scratch workspace
        do slot = 1, NUM_SCRATCH_3D
            call cuda_free_c(CS%scratch(slot))
            CS%scratch(slot) = c_null_ptr
        end do

        CS%initialized = .false.

    end subroutine hor_visc_end_cuda_c

    ! =====================================================================
    ! Compute horizontal viscous accelerations on the GPU.
    ! Launches 2 fused 3D kernels covering all layers simultaneously.
    !
    ! Input/output arrays are passed as type(c_ptr) device pointers.
    ! =====================================================================
    subroutine hor_visc_cuda_c(u_p, v_p, h_p, diffu_p, diffv_p, CS, bx_in, by_in)
        type(hor_visc_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in)  :: u_p, v_p, h_p
        type(c_ptr), intent(in)  :: diffu_p, diffv_p
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        integer(c_int) :: gx, gy, gz, bx, by, bz
        integer(c_int) :: istat

        ! Block dimensions (default 32x4 for good occupancy)
        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in
        bz = 1

        ! Array dimensions (1-based for kernel)
        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        n3 = CS%nz

        ! Compute-domain offsets in 1-based local coords
        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        ! Grid dimensions: cover full data domain, k in blockIdx%z
        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by
        gz = n3

        ! Scratch slot aliases:
        !   scratch(1) = str_xx (3D)
        !   scratch(2) = str_xy (3D)

        ! Kernel 1: Fused stress (vel_grad + strain + stress, all layers)
        call launch_stress_kernel( &
            CS%scratch(1), CS%scratch(2), u_p, v_p, h_p, &
            CS%DY_dxT_d, CS%DX_dyT_d, CS%DY_dxBu_d, CS%DX_dyBu_d, &
            CS%IdyCu_d, CS%IdxCu_d, CS%IdyCv_d, CS%IdxCv_d, &
            CS%mask2dBu_d, CS%reduction_xx_d, CS%reduction_xy_d, &
            CS%Kh_bg, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            gx, gy, gz, bx, by, bz, c_null_ptr)

        ! Kernel 2: Stress divergence -> viscous acceleration (all layers)
        call launch_divergence_kernel( &
            diffu_p, diffv_p, &
            CS%scratch(1), CS%scratch(2), h_p, &
            CS%mask2dT_d, &
            CS%IdyCu_d, CS%IdxCu_d, CS%IdyCv_d, CS%IdxCv_d, &
            CS%IareaCu_d, CS%IareaCv_d, &
            CS%dy2h_d, CS%dx2h_d, CS%dy2q_d, CS%dx2q_d, &
            CS%h_neglect, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            gx, gy, gz, bx, by, bz, c_null_ptr)

        ! Sync to ensure all output is ready before returning
        istat = cuda_device_synchronize_c()

    end subroutine hor_visc_cuda_c

end module mom6_hor_visc_cuda_c
