!> Shared GPU workspace pool for CUDA Fortran solvers.
!!
!! Provides a cuda_workspace_type with pre-allocated 3D and 2D device scratch
!! arrays that all solvers reuse sequentially. This replaces per-module scratch
!! arrays that were previously allocated in each CS type, reducing total device
!! memory from ~36 GB to ~14-19 GB at 1024x1024x100.
!!
!! Usage:
!!   type(cuda_workspace_type) :: ws
!!   call workspace_init(ws, isd, ied, jsd, jed, nk, 9, 6)
!!   ! Pass ws to solver calls; each solver uses ws%s3d(:,:,:,slot)
!!   call workspace_end(ws)
!!
!! The 3rd dimension of s3d is nk+1 to accommodate z_i_d from vert_visc.
!! Other users pass ws%s3d(:,:,1:nk,slot).
!!
module cuda_workspace
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: cuda_workspace_type, workspace_init, workspace_end, workspace_copy_3d

    !> Shared workspace pool holding reusable device scratch arrays.
    type :: cuda_workspace_type
        logical :: initialized = .false.
        integer :: isd, ied, jsd, jed, nk
        integer :: n3d_slots  !< number of 3D slots allocated
        integer :: n2d_slots  !< number of 2D slots allocated

        !> 4D device array: s3d(isd:ied, jsd:jed, nk+1, n3d_slots)
        !! Each s3d(:,:,:,slot) is a contiguous 3D scratch array.
        real(dp), device, allocatable :: s3d(:,:,:,:)

        !> 3D device array: s2d(isd:ied, jsd:jed, n2d_slots)
        !! Each s2d(:,:,slot) is a contiguous 2D scratch array.
        real(dp), device, allocatable :: s2d(:,:,:)
    end type cuda_workspace_type

contains

    !> Allocate the workspace pool on device.
    subroutine workspace_init(ws, isd, ied, jsd, jed, nk, n3d, n2d)
        type(cuda_workspace_type), intent(inout) :: ws
        integer, intent(in) :: isd, ied, jsd, jed, nk
        integer, intent(in) :: n3d  !< number of 3D slots
        integer, intent(in) :: n2d  !< number of 2D slots

        ws%isd = isd; ws%ied = ied
        ws%jsd = jsd; ws%jed = jed
        ws%nk = nk
        ws%n3d_slots = n3d
        ws%n2d_slots = n2d

        if (n3d > 0) then
            allocate(ws%s3d(isd:ied, jsd:jed, nk+1, n3d))
        end if

        if (n2d > 0) then
            allocate(ws%s2d(isd:ied, jsd:jed, n2d))
        end if

        ws%initialized = .true.

    end subroutine workspace_init

    !> Deallocate all workspace device arrays.
    subroutine workspace_end(ws)
        type(cuda_workspace_type), intent(inout) :: ws

        if (.not. ws%initialized) return

        if (allocated(ws%s3d)) deallocate(ws%s3d)
        if (allocated(ws%s2d)) deallocate(ws%s2d)

        ws%initialized = .false.

    end subroutine workspace_end

    !> Device-to-device 3D array copy via CUF kernel.
    !! Call as: workspace_copy_3d(ws%s3d(:,:,1:nk,slot), src_d, n1, n2, n3)
    !! The caller passes the workspace slice as an actual argument so nvfortran
    !! resolves the host-side descriptor before entering the kernel.
    subroutine workspace_copy_3d(dst, src, n1, n2, n3)
        real(dp), device, intent(out) :: dst(n1, n2, n3)
        real(dp), device, intent(in)  :: src(n1, n2, n3)
        integer, intent(in) :: n1, n2, n3
        integer :: i, j, k

        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, n3
            do j = 1, n2
                do i = 1, n1
                    dst(i,j,k) = src(i,j,k)
                end do
            end do
        end do
    end subroutine workspace_copy_3d

end module cuda_workspace
