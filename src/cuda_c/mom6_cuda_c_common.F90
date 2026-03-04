!> Shared infrastructure for CUDA C kernel wrappers.
!!
!! Provides Fortran interfaces to CUDA runtime helpers and
!! convenience functions for GPU memory management.
!! Uses ONLY iso_c_binding + iso_fortran_env (Fortran 2003 standard).
!!
module mom6_cuda_c_common
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding
    implicit none
    private

    public :: gpu_alloc, alloc_copy_2d, alloc_copy_3d, alloc_zero_3d, alloc_zero_2d
    public :: cuda_malloc_c, cuda_free_c, cuda_memcpy_h2d_c, cuda_memcpy_d2h_c
    public :: cuda_memset_c, cuda_device_synchronize_c

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

contains

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

    !> Allocate a 2D device array (n1 x n2 doubles), zeroed.
    function alloc_zero_2d(n1, n2) result(dptr)
        integer, intent(in) :: n1, n2
        type(c_ptr) :: dptr
        integer(c_size_t) :: nbytes
        integer(c_int) :: ierr
        nbytes = int(n1, c_size_t) * int(n2, c_size_t) * 8_c_size_t
        dptr = gpu_alloc(nbytes)
        ierr = cuda_memset_c(dptr, 0, nbytes)
    end function alloc_zero_2d

end module mom6_cuda_c_common
