!> GPU-resident halo exchange for CUDA Fortran device arrays
!!
!! Provides halo exchange routines for 2D and 3D device arrays.
!! GPU-aware MPI: pass device pointers directly to MPI_Isend/Irecv.
!! Fallback: host-staging with cudaMemcpy D2H/H2D.
!!
!! Exchange pattern: 4 cardinal + 4 corner directions (8 total).
!!
!! Buffers are persistent (allocated once, reused across calls) to avoid
!! per-call cudaMalloc/cudaFree overhead.
!!
module mom6_mpi_halo_cuda
    use cudafor
    use mpi
    use iso_fortran_env, only: dp => real64
    use mom6_mpi_domain, only: mpi_domain_type
    implicit none
    private

    public :: halo_exchange_3d_cuda, halo_exchange_2d_cuda, halo_cleanup_cuda

    ! GPU-aware MPI: pass device pointers directly to MPI, skip host staging.
    ! Enabled by setting environment variable MOM6_GPU_AWARE_MPI=1
    logical, save :: gpu_aware_mpi = .false.
    logical, save :: gpu_aware_checked = .false.

    ! Persistent device buffers — allocated once, reused across calls.
    ! Eliminates per-call cudaMalloc/cudaFree.
    real(dp), device, allocatable, save :: d_send_E(:), d_send_W(:)
    real(dp), device, allocatable, save :: d_send_N(:), d_send_S(:)
    real(dp), device, allocatable, save :: d_send_NE(:), d_send_NW(:)
    real(dp), device, allocatable, save :: d_send_SE(:), d_send_SW(:)
    real(dp), device, allocatable, save :: d_recv_E(:), d_recv_W(:)
    real(dp), device, allocatable, save :: d_recv_N(:), d_recv_S(:)
    real(dp), device, allocatable, save :: d_recv_NE(:), d_recv_NW(:)
    real(dp), device, allocatable, save :: d_recv_SE(:), d_recv_SW(:)

    ! Persistent host buffers for host-staging fallback.
    ! Only used when gpu_aware_mpi = .false.
    real(dp), allocatable, save :: h_send_E(:), h_send_W(:)
    real(dp), allocatable, save :: h_send_N(:), h_send_S(:)
    real(dp), allocatable, save :: h_send_NE(:), h_send_NW(:)
    real(dp), allocatable, save :: h_send_SE(:), h_send_SW(:)
    real(dp), allocatable, save :: h_recv_E(:), h_recv_W(:)
    real(dp), allocatable, save :: h_recv_N(:), h_recv_S(:)
    real(dp), allocatable, save :: h_recv_NE(:), h_recv_NW(:)
    real(dp), allocatable, save :: h_recv_SE(:), h_recv_SW(:)

    integer, save :: alloc_ew = -1, alloc_ns = -1, alloc_co = -1
    logical, save :: host_bufs_allocated = .false.

contains

    subroutine check_gpu_aware_mpi()
        character(len=32) :: env_val
        integer :: env_len, env_stat
        integer :: rank, ierr
        if (gpu_aware_checked) return
        call get_environment_variable("MOM6_GPU_AWARE_MPI", env_val, env_len, env_stat)
        if (env_stat == 0 .and. env_len > 0) then
            if (trim(env_val) == "1" .or. trim(env_val) == "true") then
                gpu_aware_mpi = .true.
            end if
        end if
        gpu_aware_checked = .true.
        ! Diagnostic output on rank 0
        call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
        if (rank == 0) then
            if (gpu_aware_mpi) then
                write(*,'(A)') "[MPI Halo CUDA] GPU-aware MPI: ENABLED"
            else
                write(*,'(A)') "[MPI Halo CUDA] GPU-aware MPI: DISABLED (host staging)"
            end if
        end if
    end subroutine

    !> Ensure persistent device buffers are allocated with at least the given sizes.
    subroutine ensure_device_buffers(ew_size, ns_size, corner_size)
        integer, intent(in) :: ew_size, ns_size, corner_size

        if (ew_size <= alloc_ew .and. ns_size <= alloc_ns .and. &
            corner_size <= alloc_co) return

        ! Free old if allocated
        if (allocated(d_send_E)) then
            deallocate(d_send_E, d_recv_E, d_send_W, d_recv_W)
            deallocate(d_send_N, d_recv_N, d_send_S, d_recv_S)
            deallocate(d_send_NE, d_recv_NE, d_send_NW, d_recv_NW)
            deallocate(d_send_SE, d_recv_SE, d_send_SW, d_recv_SW)
        end if
        if (allocated(h_send_E)) then
            deallocate(h_send_E, h_recv_E, h_send_W, h_recv_W)
            deallocate(h_send_N, h_recv_N, h_send_S, h_recv_S)
            deallocate(h_send_NE, h_recv_NE, h_send_NW, h_recv_NW)
            deallocate(h_send_SE, h_recv_SE, h_send_SW, h_recv_SW)
            host_bufs_allocated = .false.
        end if

        ! Allocate device buffers
        allocate(d_send_E(ew_size), d_recv_E(ew_size))
        allocate(d_send_W(ew_size), d_recv_W(ew_size))
        allocate(d_send_N(ns_size), d_recv_N(ns_size))
        allocate(d_send_S(ns_size), d_recv_S(ns_size))
        allocate(d_send_NE(corner_size), d_recv_NE(corner_size))
        allocate(d_send_NW(corner_size), d_recv_NW(corner_size))
        allocate(d_send_SE(corner_size), d_recv_SE(corner_size))
        allocate(d_send_SW(corner_size), d_recv_SW(corner_size))

        alloc_ew = ew_size
        alloc_ns = ns_size
        alloc_co = corner_size
    end subroutine

    !> Ensure host buffers are allocated (lazy — only when host staging is needed).
    subroutine ensure_host_buffers(ew_size, ns_size, corner_size)
        integer, intent(in) :: ew_size, ns_size, corner_size

        if (host_bufs_allocated .and. &
            ew_size <= alloc_ew .and. ns_size <= alloc_ns .and. &
            corner_size <= alloc_co) return

        if (allocated(h_send_E)) then
            deallocate(h_send_E, h_recv_E, h_send_W, h_recv_W)
            deallocate(h_send_N, h_recv_N, h_send_S, h_recv_S)
            deallocate(h_send_NE, h_recv_NE, h_send_NW, h_recv_NW)
            deallocate(h_send_SE, h_recv_SE, h_send_SW, h_recv_SW)
        end if

        allocate(h_send_E(ew_size), h_recv_E(ew_size))
        allocate(h_send_W(ew_size), h_recv_W(ew_size))
        allocate(h_send_N(ns_size), h_recv_N(ns_size))
        allocate(h_send_S(ns_size), h_recv_S(ns_size))
        allocate(h_send_NE(corner_size), h_recv_NE(corner_size))
        allocate(h_send_NW(corner_size), h_recv_NW(corner_size))
        allocate(h_send_SE(corner_size), h_recv_SE(corner_size))
        allocate(h_send_SW(corner_size), h_recv_SW(corner_size))
        host_bufs_allocated = .true.
    end subroutine

    !> 3D halo exchange for CUDA device arrays
    !!
    !! d_array(isd:ied, jsd:jed, nz) is a device array.
    !! Uses CUF kernels for pack/unpack and GPU-aware MPI or host staging.
    subroutine halo_exchange_3d_cuda(d_array, isd, ied, jsd, jed, nz, MD, halo_width)
        integer, intent(in) :: isd, ied, jsd, jed, nz, halo_width
        real(dp), device, intent(inout) :: d_array(isd:ied, jsd:jed, nz)
        type(mpi_domain_type), intent(in) :: MD

        integer :: ni_total, nj_total, hw
        integer :: ew_size, ns_size, corner_size
        integer :: nreqs, i, j, k, idx, istat
        integer :: ierr

        integer :: reqs(16)
        integer :: stats(MPI_STATUS_SIZE, 16)

        call check_gpu_aware_mpi()

        hw = halo_width
        ni_total = ied - isd + 1
        nj_total = jed - jsd + 1

        ! East/West: hw columns x compute rows x nz
        ew_size = hw * (nj_total - 2 * hw) * nz
        ! North/South: full width x hw rows x nz
        ns_size = ni_total * hw * nz
        ! Corners: hw x hw x nz
        corner_size = hw * hw * nz

        ! Ensure persistent buffers (no-op after first call)
        call ensure_device_buffers(ew_size, ns_size, corner_size)

        ! === PACK send buffers on GPU using CUF kernels ===

        ! East: rightmost hw compute columns
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    d_send_E(idx) = d_array(ied - 2 * hw + i, j, k)
                end do
            end do
        end do

        ! West: leftmost hw compute columns
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    d_send_W(idx) = d_array(isd + hw + i - 1, j, k)
                end do
            end do
        end do

        ! North: topmost hw compute rows
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    d_send_N(idx) = d_array(isd + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        ! South: bottommost hw compute rows
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    d_send_S(idx) = d_array(isd + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! NE corner
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    d_send_NE(idx) = d_array(ied - 2 * hw + i, jed - 2 * hw + j, k)
                end do
            end do
        end do

        ! NW corner
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    d_send_NW(idx) = d_array(isd + hw + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        ! SE corner
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    d_send_SE(idx) = d_array(ied - 2 * hw + i, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! SW corner
        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    d_send_SW(idx) = d_array(isd + hw + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! Sync before MPI
        istat = cudaDeviceSynchronize()

        ! === MPI communication ===
        nreqs = 0
        if (gpu_aware_mpi) then
            ! GPU-aware: pass device buffers directly to MPI
            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 300, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 301, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 302, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 303, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 304, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 305, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 306, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 307, MD%comm, reqs(nreqs), ierr); end if
            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 301, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 300, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 303, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 302, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 307, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 306, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 305, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 304, MD%comm, reqs(nreqs), ierr); end if
        else
            ! Host staging: D2H -> MPI -> H2D
            call ensure_host_buffers(ew_size, ns_size, corner_size)

            h_send_E(1:ew_size) = d_send_E(1:ew_size)
            h_send_W(1:ew_size) = d_send_W(1:ew_size)
            h_send_N(1:ns_size) = d_send_N(1:ns_size)
            h_send_S(1:ns_size) = d_send_S(1:ns_size)
            h_send_NE(1:corner_size) = d_send_NE(1:corner_size)
            h_send_NW(1:corner_size) = d_send_NW(1:corner_size)
            h_send_SE(1:corner_size) = d_send_SE(1:corner_size)
            h_send_SW(1:corner_size) = d_send_SW(1:corner_size)

            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 300, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 301, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 302, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 303, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 304, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 305, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 306, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 307, MD%comm, reqs(nreqs), ierr); end if
            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 301, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 300, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 303, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 302, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 307, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 306, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 305, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 304, MD%comm, reqs(nreqs), ierr); end if
        end if
        if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

        if (.not. gpu_aware_mpi) then
            if (MD%east /= MPI_PROC_NULL) d_recv_E(1:ew_size) = h_recv_E(1:ew_size)
            if (MD%west /= MPI_PROC_NULL) d_recv_W(1:ew_size) = h_recv_W(1:ew_size)
            if (MD%north /= MPI_PROC_NULL) d_recv_N(1:ns_size) = h_recv_N(1:ns_size)
            if (MD%south /= MPI_PROC_NULL) d_recv_S(1:ns_size) = h_recv_S(1:ns_size)
            if (MD%ne /= MPI_PROC_NULL) d_recv_NE(1:corner_size) = h_recv_NE(1:corner_size)
            if (MD%nw /= MPI_PROC_NULL) d_recv_NW(1:corner_size) = h_recv_NW(1:corner_size)
            if (MD%se /= MPI_PROC_NULL) d_recv_SE(1:corner_size) = h_recv_SE(1:corner_size)
            if (MD%sw /= MPI_PROC_NULL) d_recv_SW(1:corner_size) = h_recv_SW(1:corner_size)
        end if

        ! === UNPACK recv buffers on GPU ===

        if (MD%east /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = jsd + hw, jed - hw
                    do i = 1, hw
                        idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                        d_array(ied - hw + i, j, k) = d_recv_E(idx)
                    end do
                end do
            end do
        end if

        if (MD%west /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = jsd + hw, jed - hw
                    do i = 1, hw
                        idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                        d_array(isd + i - 1, j, k) = d_recv_W(idx)
                    end do
                end do
            end do
        end if

        if (MD%north /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = 1, hw
                    do i = 1, ni_total
                        idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                        d_array(isd + i - 1, jed - hw + j, k) = d_recv_N(idx)
                    end do
                end do
            end do
        end if

        if (MD%south /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = 1, hw
                    do i = 1, ni_total
                        idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                        d_array(isd + i - 1, jsd + j - 1, k) = d_recv_S(idx)
                    end do
                end do
            end do
        end if

        if (MD%ne /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        d_array(ied - hw + i, jed - hw + j, k) = d_recv_NE(idx)
                    end do
                end do
            end do
        end if

        if (MD%nw /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        d_array(isd + i - 1, jed - hw + j, k) = d_recv_NW(idx)
                    end do
                end do
            end do
        end if

        if (MD%se /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        d_array(ied - hw + i, jsd + j - 1, k) = d_recv_SE(idx)
                    end do
                end do
            end do
        end if

        if (MD%sw /= MPI_PROC_NULL) then
            !$cuf kernel do(3) <<< *, * >>>
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        d_array(isd + i - 1, jsd + j - 1, k) = d_recv_SW(idx)
                    end do
                end do
            end do
        end if

        istat = cudaDeviceSynchronize()

        ! No per-call cleanup — buffers persist for reuse

    end subroutine halo_exchange_3d_cuda

    !> 2D halo exchange for CUDA device arrays
    subroutine halo_exchange_2d_cuda(d_array, isd, ied, jsd, jed, MD, halo_width)
        integer, intent(in) :: isd, ied, jsd, jed, halo_width
        real(dp), device, intent(inout) :: d_array(isd:ied, jsd:jed)
        type(mpi_domain_type), intent(in) :: MD

        integer :: ni_total, nj_total, hw
        integer :: ew_size, ns_size, corner_size
        integer :: nreqs, i, j, idx, istat
        integer :: ierr

        integer :: reqs(16)
        integer :: stats(MPI_STATUS_SIZE, 16)

        call check_gpu_aware_mpi()

        hw = halo_width
        ni_total = ied - isd + 1
        nj_total = jed - jsd + 1

        ew_size = hw * (nj_total - 2 * hw)
        ns_size = ni_total * hw
        corner_size = hw * hw

        ! Ensure persistent buffers (no-op after first call)
        call ensure_device_buffers(ew_size, ns_size, corner_size)

        ! === PACK ===
        !$cuf kernel do(2) <<< *, * >>>
        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                d_send_E(idx) = d_array(ied - 2 * hw + i, j)
            end do
        end do

        !$cuf kernel do(2) <<< *, * >>>
        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                d_send_W(idx) = d_array(isd + hw + i - 1, j)
            end do
        end do

        !$cuf kernel do(2) <<< *, * >>>
        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                d_send_N(idx) = d_array(isd + i - 1, jed - 2 * hw + j)
            end do
        end do

        !$cuf kernel do(2) <<< *, * >>>
        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                d_send_S(idx) = d_array(isd + i - 1, jsd + hw + j - 1)
            end do
        end do

        !$cuf kernel do(2) <<< *, * >>>
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                d_send_NE(idx) = d_array(ied - 2 * hw + i, jed - 2 * hw + j)
            end do
        end do

        !$cuf kernel do(2) <<< *, * >>>
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                d_send_NW(idx) = d_array(isd + hw + i - 1, jed - 2 * hw + j)
            end do
        end do

        !$cuf kernel do(2) <<< *, * >>>
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                d_send_SE(idx) = d_array(ied - 2 * hw + i, jsd + hw + j - 1)
            end do
        end do

        !$cuf kernel do(2) <<< *, * >>>
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                d_send_SW(idx) = d_array(isd + hw + i - 1, jsd + hw + j - 1)
            end do
        end do

        istat = cudaDeviceSynchronize()

        ! === MPI communication ===
        nreqs = 0
        if (gpu_aware_mpi) then
            ! GPU-aware: pass device buffers directly to MPI
            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 400, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 401, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 402, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 403, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 404, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 405, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 406, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(d_recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 407, MD%comm, reqs(nreqs), ierr); end if
            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 401, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 400, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 403, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 402, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 407, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 406, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 405, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(d_send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 404, MD%comm, reqs(nreqs), ierr); end if
        else
            ! Host staging: D2H -> MPI -> H2D
            call ensure_host_buffers(ew_size, ns_size, corner_size)

            h_send_E(1:ew_size) = d_send_E(1:ew_size)
            h_send_W(1:ew_size) = d_send_W(1:ew_size)
            h_send_N(1:ns_size) = d_send_N(1:ns_size)
            h_send_S(1:ns_size) = d_send_S(1:ns_size)
            h_send_NE(1:corner_size) = d_send_NE(1:corner_size)
            h_send_NW(1:corner_size) = d_send_NW(1:corner_size)
            h_send_SE(1:corner_size) = d_send_SE(1:corner_size)
            h_send_SW(1:corner_size) = d_send_SW(1:corner_size)

            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 400, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 401, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 402, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 403, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 404, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 405, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 406, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Irecv(h_recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 407, MD%comm, reqs(nreqs), ierr); end if
            if (MD%east /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 401, MD%comm, reqs(nreqs), ierr); end if
            if (MD%west /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 400, MD%comm, reqs(nreqs), ierr); end if
            if (MD%north /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 403, MD%comm, reqs(nreqs), ierr); end if
            if (MD%south /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 402, MD%comm, reqs(nreqs), ierr); end if
            if (MD%ne /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 407, MD%comm, reqs(nreqs), ierr); end if
            if (MD%nw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 406, MD%comm, reqs(nreqs), ierr); end if
            if (MD%se /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 405, MD%comm, reqs(nreqs), ierr); end if
            if (MD%sw /= MPI_PROC_NULL) then; nreqs=nreqs+1
                call MPI_Isend(h_send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 404, MD%comm, reqs(nreqs), ierr); end if
        end if
        if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

        if (.not. gpu_aware_mpi) then
            if (MD%east /= MPI_PROC_NULL) d_recv_E(1:ew_size) = h_recv_E(1:ew_size)
            if (MD%west /= MPI_PROC_NULL) d_recv_W(1:ew_size) = h_recv_W(1:ew_size)
            if (MD%north /= MPI_PROC_NULL) d_recv_N(1:ns_size) = h_recv_N(1:ns_size)
            if (MD%south /= MPI_PROC_NULL) d_recv_S(1:ns_size) = h_recv_S(1:ns_size)
            if (MD%ne /= MPI_PROC_NULL) d_recv_NE(1:corner_size) = h_recv_NE(1:corner_size)
            if (MD%nw /= MPI_PROC_NULL) d_recv_NW(1:corner_size) = h_recv_NW(1:corner_size)
            if (MD%se /= MPI_PROC_NULL) d_recv_SE(1:corner_size) = h_recv_SE(1:corner_size)
            if (MD%sw /= MPI_PROC_NULL) d_recv_SW(1:corner_size) = h_recv_SW(1:corner_size)
        end if

        ! === UNPACK ===
        if (MD%east /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    d_array(ied - hw + i, j) = d_recv_E(idx)
                end do
            end do
        end if

        if (MD%west /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    d_array(isd + i - 1, j) = d_recv_W(idx)
                end do
            end do
        end if

        if (MD%north /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    d_array(isd + i - 1, jed - hw + j) = d_recv_N(idx)
                end do
            end do
        end if

        if (MD%south /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    d_array(isd + i - 1, jsd + j - 1) = d_recv_S(idx)
                end do
            end do
        end if

        if (MD%ne /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    d_array(ied - hw + i, jed - hw + j) = d_recv_NE(idx)
                end do
            end do
        end if

        if (MD%nw /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    d_array(isd + i - 1, jed - hw + j) = d_recv_NW(idx)
                end do
            end do
        end if

        if (MD%se /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    d_array(ied - hw + i, jsd + j - 1) = d_recv_SE(idx)
                end do
            end do
        end if

        if (MD%sw /= MPI_PROC_NULL) then
            !$cuf kernel do(2) <<< *, * >>>
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    d_array(isd + i - 1, jsd + j - 1) = d_recv_SW(idx)
                end do
            end do
        end if

        istat = cudaDeviceSynchronize()

        ! No per-call cleanup — buffers persist for reuse

    end subroutine halo_exchange_2d_cuda

    !> Free persistent halo buffers. Call at program finalization.
    subroutine halo_cleanup_cuda()
        if (allocated(d_send_E)) then
            deallocate(d_send_E, d_recv_E, d_send_W, d_recv_W)
            deallocate(d_send_N, d_recv_N, d_send_S, d_recv_S)
            deallocate(d_send_NE, d_recv_NE, d_send_NW, d_recv_NW)
            deallocate(d_send_SE, d_recv_SE, d_send_SW, d_recv_SW)
        end if
        if (allocated(h_send_E)) then
            deallocate(h_send_E, h_recv_E, h_send_W, h_recv_W)
            deallocate(h_send_N, h_recv_N, h_send_S, h_recv_S)
            deallocate(h_send_NE, h_recv_NE, h_send_NW, h_recv_NW)
            deallocate(h_send_SE, h_recv_SE, h_send_SW, h_recv_SW)
        end if
        alloc_ew = -1
        alloc_ns = -1
        alloc_co = -1
        host_bufs_allocated = .false.
    end subroutine

end module mom6_mpi_halo_cuda
