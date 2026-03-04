!> MOM6 Coriolis — Portable Fortran wrapper calling CUDA C kernels
!!
!! This module uses ONLY iso_c_binding + iso_fortran_env (Fortran 2003 standard).
!! NO cudafor, NO device attribute, NO vendor-specific extensions.
!! All GPU memory is managed through C helper functions in mom6_coriolis_kernels.cu.
!!
!! Compilable with any Fortran compiler: gfortran, ifx, nvfortran, amdflang, etc.
!!
module mom6_coriolis_cuda_c
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding
    implicit none
    private

    public :: CorAdCalc_cuda_c, CorAdCalc_cuda_c_fused
    public :: coriolis_init_cuda_c, coriolis_end_cuda_c
    public :: coriolis_CS_cuda_c
    public :: SADOURNY75_ENERGY_CUDA_C, ARAKAWA_HSU90_CUDA_C, ARAKAWA_LAMB81_CUDA_C

    integer, parameter :: SADOURNY75_ENERGY_CUDA_C = 1
    integer, parameter :: ARAKAWA_HSU90_CUDA_C = 2
    integer, parameter :: ARAKAWA_LAMB81_CUDA_C = 3

    integer, parameter :: NUM_SCRATCH_3D = 9  ! workspace slots needed

    !> Control structure — all device pointers are opaque type(c_ptr)
    type :: coriolis_CS_cuda_c
        logical :: initialized = .false.
        integer :: Coriolis_Scheme
        integer :: is, ie, js, je, nz
        integer :: isd, ied, jsd, jed

        ! Persistent 3D device array (computed once at init)
        type(c_ptr) :: Area_q = c_null_ptr

        ! 2D device grid metrics (copied once at init)
        type(c_ptr) :: dyCv_d = c_null_ptr
        type(c_ptr) :: dxCu_d = c_null_ptr
        type(c_ptr) :: areaT_d = c_null_ptr
        type(c_ptr) :: IareaBu_d = c_null_ptr
        type(c_ptr) :: CoriolisBu_d = c_null_ptr
        type(c_ptr) :: mask2dBu_d = c_null_ptr
        type(c_ptr) :: dyCu_d = c_null_ptr
        type(c_ptr) :: dxCv_d = c_null_ptr
        type(c_ptr) :: IdxCu_d = c_null_ptr
        type(c_ptr) :: IdyCv_d = c_null_ptr

        ! Scratch workspace (9 x 3D slots, allocated once)
        type(c_ptr) :: scratch(NUM_SCRATCH_3D) = c_null_ptr
    end type coriolis_CS_cuda_c

    ! =====================================================================
    ! C helper interfaces (CUDA runtime wrappers)
    ! =====================================================================
    interface
        function cuda_malloc_c(bytes) result(ptr) bind(c, name='cuda_malloc_c')
            import :: c_ptr, c_size_t
            integer(c_size_t), value :: bytes
            type(c_ptr) :: ptr
        end function

        subroutine cuda_free_c(ptr) bind(c, name='cuda_free_c')
            import :: c_ptr
            type(c_ptr), value :: ptr
        end subroutine

        function cuda_memcpy_h2d_c(dst, src, bytes) result(ierr) bind(c, name='cuda_memcpy_h2d_c')
            import :: c_ptr, c_int, c_size_t
            type(c_ptr), value :: dst, src
            integer(c_size_t), value :: bytes
            integer(c_int) :: ierr
        end function

        function cuda_memcpy_d2h_c(dst, src, bytes) result(ierr) bind(c, name='cuda_memcpy_d2h_c')
            import :: c_ptr, c_int, c_size_t
            type(c_ptr), value :: dst, src
            integer(c_size_t), value :: bytes
            integer(c_int) :: ierr
        end function

        function cuda_memset_c(ptr, val, bytes) result(ierr) bind(c, name='cuda_memset_c')
            import :: c_ptr, c_int, c_size_t
            type(c_ptr), value :: ptr
            integer(c_int), value :: val
            integer(c_size_t), value :: bytes
            integer(c_int) :: ierr
        end function

        function cuda_device_synchronize_c() result(ierr) bind(c, name='cuda_device_synchronize_c')
            import :: c_int
            integer(c_int) :: ierr
        end function
    end interface

    ! =====================================================================
    ! C kernel launch interfaces
    ! =====================================================================
    interface
        subroutine launch_phase1_kernel( &
                dvdx, dudy, hArea_u, hArea_v, KE, &
                u, v, h, dyCv, dxCu, dyCu, dxCv, areaT, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_phase1_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: dvdx, dudy, hArea_u, hArea_v, KE
            type(c_ptr), value :: u, v, h, dyCv, dxCu, dyCu, dxCv, areaT
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_vorticity_pv_kernel( &
                rel_vort, abs_vort, q, Ih_q, &
                dvdx, dudy, hArea_u, hArea_v, Area_q, &
                IareaBu, CoriolisBu, mask2dBu, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_vorticity_pv_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: rel_vort, abs_vort, q, Ih_q
            type(c_ptr), value :: dvdx, dudy, hArea_u, hArea_v, Area_q
            type(c_ptr), value :: IareaBu, CoriolisBu, mask2dBu
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_coriolis_sadourny_kernel( &
                CAu, CAv, q, KE, uh, vh, IdxCu, IdyCv, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_coriolis_sadourny_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: CAu, CAv, q, KE, uh, vh, IdxCu, IdyCv
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_arakawa_hsu90_coef_kernel( &
                a, b, c, d, q, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_arakawa_hsu90_coef_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: a, b, c, d, q
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_arakawa_lamb81_coef_kernel( &
                a, b, c, d, q, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_arakawa_lamb81_coef_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: a, b, c, d, q
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_coriolis_arakawa_kernel( &
                CAu, CAv, a, b, c, d, KE, uh, vh, IdxCu, IdyCv, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_coriolis_arakawa_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: CAu, CAv, a, b, c, d, KE, uh, vh, IdxCu, IdyCv
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine

        subroutine launch_coriolis_fused_sadourny_kernel( &
                CAu, CAv, u, v, h, uh, vh, &
                dyCv, dxCu, dyCu, dxCv, areaT, &
                IareaBu, CoriolisBu, mask2dBu, IdxCu, IdyCv, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                grid_x, grid_y, grid_z, block_x, block_y, block_z, stream) &
                bind(c, name='launch_coriolis_fused_sadourny_kernel')
            import :: c_ptr, c_int
            type(c_ptr), value :: CAu, CAv, u, v, h, uh, vh
            type(c_ptr), value :: dyCv, dxCu, dyCu, dxCv, areaT
            type(c_ptr), value :: IareaBu, CoriolisBu, mask2dBu, IdxCu, IdyCv
            integer(c_int), value :: n1, n2, n3, is_l, ie_l, js_l, je_l
            integer(c_int), value :: grid_x, grid_y, grid_z
            integer(c_int), value :: block_x, block_y, block_z
            type(c_ptr), value :: stream
        end subroutine
    end interface

contains

    ! =====================================================================
    ! Internal helpers for GPU memory management
    ! =====================================================================

    !> Allocate n bytes on GPU, return c_ptr. Fatal stop on failure.
    function gpu_alloc(n) result(ptr)
        integer(c_size_t), intent(in) :: n
        type(c_ptr) :: ptr
        ptr = cuda_malloc_c(n)
        if (.not. c_associated(ptr)) then
            print '(A,I0,A)', 'FATAL: cuda_malloc_c failed for ', n, ' bytes'
            stop 1
        end if
    end function gpu_alloc

    !> Allocate a 2D device array (n1 x n2 doubles) and copy host data to it.
    function alloc_copy_2d(host_arr, n1, n2) result(dptr)
        integer, intent(in) :: n1, n2
        real(dp), intent(in), target :: host_arr(n1, n2)
        type(c_ptr) :: dptr
        integer(c_size_t) :: nbytes
        integer(c_int) :: ierr

        nbytes = int(n1, c_size_t) * int(n2, c_size_t) * 8_c_size_t
        dptr = gpu_alloc(nbytes)
        ierr = cuda_memcpy_h2d_c(dptr, c_loc(host_arr), nbytes)
    end function alloc_copy_2d

    !> Allocate a 3D device array (n1 x n2 x n3 doubles) and copy host data.
    function alloc_copy_3d(host_arr, n1, n2, n3) result(dptr)
        integer, intent(in) :: n1, n2, n3
        real(dp), intent(in), target :: host_arr(n1, n2, n3)
        type(c_ptr) :: dptr
        integer(c_size_t) :: nbytes
        integer(c_int) :: ierr

        nbytes = int(n1, c_size_t) * int(n2, c_size_t) * int(n3, c_size_t) * 8_c_size_t
        dptr = gpu_alloc(nbytes)
        ierr = cuda_memcpy_h2d_c(dptr, c_loc(host_arr), nbytes)
    end function alloc_copy_3d

    !> Allocate a 3D device array (n1 x n2 x n3 doubles), zeroed.
    function alloc_zero_3d(n1, n2, n3) result(dptr)
        integer, intent(in) :: n1, n2, n3
        type(c_ptr) :: dptr
        integer(c_size_t) :: nbytes
        integer(c_int) :: ierr

        nbytes = int(n1, c_size_t) * int(n2, c_size_t) * int(n3, c_size_t) * 8_c_size_t
        dptr = gpu_alloc(nbytes)
        ierr = cuda_memset_c(dptr, 0, nbytes)
    end function alloc_zero_3d

    ! =====================================================================
    ! Init
    ! =====================================================================
    subroutine coriolis_init_cuda_c(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nk, &
                                     areaT, IareaBu, CoriolisBu, mask2dBu, &
                                     dyCv, dxCu, dyCu, dxCv, IdxCu, IdyCv, scheme)
        type(coriolis_CS_cuda_c), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec, nk
        real(dp), intent(in), target :: areaT(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IareaBu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: CoriolisBu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: mask2dBu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dyCv(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dxCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dyCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: dxCv(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdxCu(isd:ied, jsd:jed)
        real(dp), intent(in), target :: IdyCv(isd:ied, jsd:jed)
        integer, intent(in), optional :: scheme

        integer :: i, j, k, n1, n2, slot
        real(dp), allocatable, target :: Area_q_h(:,:,:)

        CS%Coriolis_Scheme = SADOURNY75_ENERGY_CUDA_C
        if (present(scheme)) CS%Coriolis_Scheme = scheme

        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec; CS%nz = nk

        n1 = ied - isd + 1
        n2 = jed - jsd + 1

        ! Copy 2D grid metrics to device
        CS%dyCv_d      = alloc_copy_2d(dyCv, n1, n2)
        CS%dxCu_d      = alloc_copy_2d(dxCu, n1, n2)
        CS%areaT_d     = alloc_copy_2d(areaT, n1, n2)
        CS%IareaBu_d   = alloc_copy_2d(IareaBu, n1, n2)
        CS%CoriolisBu_d = alloc_copy_2d(CoriolisBu, n1, n2)
        CS%mask2dBu_d  = alloc_copy_2d(mask2dBu, n1, n2)
        CS%dyCu_d      = alloc_copy_2d(dyCu, n1, n2)
        CS%dxCv_d      = alloc_copy_2d(dxCv, n1, n2)
        CS%IdxCu_d     = alloc_copy_2d(IdxCu, n1, n2)
        CS%IdyCv_d     = alloc_copy_2d(IdyCv, n1, n2)

        ! Precompute Area_q on host, then copy to device
        allocate(Area_q_h(n1, n2, nk))
        Area_q_h = 0.0_dp
        do k = 1, nk
            do j = 1, n2 - 1
                do i = 1, n1 - 1
                    Area_q_h(i, j, k) = (areaT(isd + i - 1, jsd + j - 1) &
                        + areaT(isd + i, jsd + j)) + &
                        (areaT(isd + i, jsd + j - 1) + areaT(isd + i - 1, jsd + j))
                end do
            end do
        end do
        CS%Area_q = alloc_copy_3d(Area_q_h, n1, n2, nk)
        deallocate(Area_q_h)

        ! Allocate scratch workspace (9 x 3D slots for intermediate arrays)
        do slot = 1, NUM_SCRATCH_3D
            CS%scratch(slot) = alloc_zero_3d(n1, n2, nk)
        end do

        CS%initialized = .true.

    end subroutine coriolis_init_cuda_c

    ! =====================================================================
    ! End
    ! =====================================================================
    subroutine coriolis_end_cuda_c(CS)
        type(coriolis_CS_cuda_c), intent(inout) :: CS
        integer :: slot

        if (.not. CS%initialized) return

        ! Free 2D metric arrays
        call cuda_free_c(CS%dyCv_d);      CS%dyCv_d = c_null_ptr
        call cuda_free_c(CS%dxCu_d);      CS%dxCu_d = c_null_ptr
        call cuda_free_c(CS%areaT_d);     CS%areaT_d = c_null_ptr
        call cuda_free_c(CS%IareaBu_d);   CS%IareaBu_d = c_null_ptr
        call cuda_free_c(CS%CoriolisBu_d); CS%CoriolisBu_d = c_null_ptr
        call cuda_free_c(CS%mask2dBu_d);  CS%mask2dBu_d = c_null_ptr
        call cuda_free_c(CS%dyCu_d);      CS%dyCu_d = c_null_ptr
        call cuda_free_c(CS%dxCv_d);      CS%dxCv_d = c_null_ptr
        call cuda_free_c(CS%IdxCu_d);     CS%IdxCu_d = c_null_ptr
        call cuda_free_c(CS%IdyCv_d);     CS%IdyCv_d = c_null_ptr

        ! Free 3D persistent array
        call cuda_free_c(CS%Area_q);      CS%Area_q = c_null_ptr

        ! Free scratch workspace
        do slot = 1, NUM_SCRATCH_3D
            call cuda_free_c(CS%scratch(slot))
            CS%scratch(slot) = c_null_ptr
        end do

        CS%initialized = .false.

    end subroutine coriolis_end_cuda_c

    ! =====================================================================
    ! CorAdCalc — multi-phase Coriolis (calls C launch wrappers)
    !
    ! Input/output arrays are passed as type(c_ptr) device pointers.
    ! The driver is responsible for extracting device pointers from
    ! whatever memory system it uses (CUDA Fortran c_devloc, HIP, etc.)
    ! =====================================================================
    subroutine CorAdCalc_cuda_c(u_p, v_p, h_p, uh_p, vh_p, CAu_p, CAv_p, CS, bx_in, by_in)
        type(coriolis_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in)  :: u_p, v_p, h_p, uh_p, vh_p
        type(c_ptr), intent(in)  :: CAu_p, CAv_p
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        integer(c_int) :: gx, gy, gz, bx, by, bz
        integer(c_int) :: istat

        bx = 32; by = 8
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in
        bz = 1

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        n3 = CS%nz

        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by
        gz = n3

        ! Scratch slot aliases:
        ! Phase 1: dvdx→1, dudy→2, hArea_u→3, hArea_v→4, KE→5
        ! Phase 2: rel_vort→6, abs_vort→7, q_d→8, Ih_q→9
        ! Phase 3a: a→1, b→2, c→3, d→4 (reuse dead Phase 1 slots)

        ! Phase 1: circulation, area-weighted thickness, KE
        call launch_phase1_kernel( &
            CS%scratch(1), CS%scratch(2), CS%scratch(3), CS%scratch(4), CS%scratch(5), &
            u_p, v_p, h_p, &
            CS%dyCv_d, CS%dxCu_d, CS%dyCu_d, CS%dxCv_d, CS%areaT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            gx, gy, gz, bx, by, bz, c_null_ptr)

        ! Phase 2: vorticity and PV
        call launch_vorticity_pv_kernel( &
            CS%scratch(6), CS%scratch(7), CS%scratch(8), CS%scratch(9), &
            CS%scratch(1), CS%scratch(2), CS%scratch(3), CS%scratch(4), CS%Area_q, &
            CS%IareaBu_d, CS%CoriolisBu_d, CS%mask2dBu_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            gx, gy, gz, bx, by, bz, c_null_ptr)

        ! Phase 3: Coriolis accelerations
        if (CS%Coriolis_Scheme == SADOURNY75_ENERGY_CUDA_C) then
            call launch_coriolis_sadourny_kernel( &
                CAu_p, CAv_p, CS%scratch(8), CS%scratch(5), uh_p, vh_p, &
                CS%IdxCu_d, CS%IdyCv_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                gx, gy, gz, bx, by, bz, c_null_ptr)

        else if (CS%Coriolis_Scheme == ARAKAWA_HSU90_CUDA_C) then
            call launch_arakawa_hsu90_coef_kernel( &
                CS%scratch(1), CS%scratch(2), CS%scratch(3), CS%scratch(4), CS%scratch(8), &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                gx, gy, gz, bx, by, bz, c_null_ptr)
            call launch_coriolis_arakawa_kernel( &
                CAu_p, CAv_p, &
                CS%scratch(1), CS%scratch(2), CS%scratch(3), CS%scratch(4), &
                CS%scratch(5), uh_p, vh_p, &
                CS%IdxCu_d, CS%IdyCv_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                gx, gy, gz, bx, by, bz, c_null_ptr)

        else if (CS%Coriolis_Scheme == ARAKAWA_LAMB81_CUDA_C) then
            call launch_arakawa_lamb81_coef_kernel( &
                CS%scratch(1), CS%scratch(2), CS%scratch(3), CS%scratch(4), CS%scratch(8), &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                gx, gy, gz, bx, by, bz, c_null_ptr)
            call launch_coriolis_arakawa_kernel( &
                CAu_p, CAv_p, &
                CS%scratch(1), CS%scratch(2), CS%scratch(3), CS%scratch(4), &
                CS%scratch(5), uh_p, vh_p, &
                CS%IdxCu_d, CS%IdyCv_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l, &
                gx, gy, gz, bx, by, bz, c_null_ptr)
        end if

        istat = cuda_device_synchronize_c()

    end subroutine CorAdCalc_cuda_c

    ! =====================================================================
    ! Fused single-kernel Sadourny (no scratch needed)
    ! =====================================================================
    subroutine CorAdCalc_cuda_c_fused(u_p, v_p, h_p, uh_p, vh_p, CAu_p, CAv_p, CS, bx_in, by_in)
        type(coriolis_CS_cuda_c), intent(inout) :: CS
        type(c_ptr), intent(in)  :: u_p, v_p, h_p, uh_p, vh_p
        type(c_ptr), intent(in)  :: CAu_p, CAv_p
        integer, intent(in), optional :: bx_in, by_in

        integer(c_int) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        integer(c_int) :: gx, gy, gz, bx, by, bz
        integer(c_int) :: istat

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in
        bz = 1

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        n3 = CS%nz

        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        gx = (n1 + bx - 1) / bx
        gy = (n2 + by - 1) / by
        gz = n3

        call launch_coriolis_fused_sadourny_kernel( &
            CAu_p, CAv_p, u_p, v_p, h_p, uh_p, vh_p, &
            CS%dyCv_d, CS%dxCu_d, CS%dyCu_d, CS%dxCv_d, CS%areaT_d, &
            CS%IareaBu_d, CS%CoriolisBu_d, CS%mask2dBu_d, &
            CS%IdxCu_d, CS%IdyCv_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, &
            gx, gy, gz, bx, by, bz, c_null_ptr)

        istat = cuda_device_synchronize_c()

    end subroutine CorAdCalc_cuda_c_fused

end module mom6_coriolis_cuda_c
