!> GPU-resident halo exchange for OpenMP target arrays
!!
!! Provides halo exchange routines for 2D and 3D arrays that are
!! resident on the GPU via OpenMP target. Uses !$omp target data
!! use_device_addr(...) for GPU-aware MPI when available, with
!! host-staging fallback.
!!
!! Exchange pattern: 4 cardinal + 4 corner directions (8 total).
!! For boundary PEs (neighbor = MPI_PROC_NULL), MPI handles skip automatically.
!!
!! Buffers are persistent (allocated once, reused across calls) to avoid
!! per-call device allocation overhead.
!!
!! GPU-aware path: MPI calls are inlined directly within !$omp target data
!! use_device_addr blocks to ensure device pointers are passed to MPI.
!!
module mom6_mpi_halo_omp
    use mpi
    use iso_fortran_env, only: dp => real64
    use mom6_mpi_domain, only: mpi_domain_type
    implicit none
    private

    public :: halo_exchange_3d, halo_exchange_2d, halo_cleanup

    ! GPU-aware MPI: pass device pointers directly to MPI, skip host staging.
    ! Enabled by setting environment variable MOM6_GPU_AWARE_MPI=1
    logical, save :: gpu_aware_mpi = .false.
    logical, save :: gpu_aware_checked = .false.

    ! Persistent send/recv buffers — allocated once on first use (or when
    ! sizes change), kept alive across calls.
    real(dp), allocatable, save :: send_E(:), recv_E(:), send_W(:), recv_W(:)
    real(dp), allocatable, save :: send_N(:), recv_N(:), send_S(:), recv_S(:)
    real(dp), allocatable, save :: send_NE(:), recv_NE(:), send_NW(:), recv_NW(:)
    real(dp), allocatable, save :: send_SE(:), recv_SE(:), send_SW(:), recv_SW(:)
    integer, save :: alloc_ew = -1, alloc_ns = -1, alloc_co = -1

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
                write(*,'(A)') "[MPI Halo OpenMP] GPU-aware MPI: ENABLED"
            else
                write(*,'(A)') "[MPI Halo OpenMP] GPU-aware MPI: DISABLED (host staging)"
            end if
        end if
    end subroutine

    !> Ensure persistent buffers are allocated with at least the given sizes.
    !! Allocates on first call; reallocates only if sizes increase.
    subroutine ensure_buffers(ew_size, ns_size, corner_size)
        integer, intent(in) :: ew_size, ns_size, corner_size

        if (ew_size <= alloc_ew .and. ns_size <= alloc_ns .and. &
            corner_size <= alloc_co) return

        ! Free old buffers if previously allocated
        if (allocated(send_E)) then
            !$omp target exit data map(delete: send_E, recv_E, send_W, recv_W)
            !$omp target exit data map(delete: send_N, recv_N, send_S, recv_S)
            !$omp target exit data map(delete: send_NE, recv_NE, send_NW, recv_NW)
            !$omp target exit data map(delete: send_SE, recv_SE, send_SW, recv_SW)
            deallocate(send_E, recv_E, send_W, recv_W)
            deallocate(send_N, recv_N, send_S, recv_S)
            deallocate(send_NE, recv_NE, send_NW, recv_NW)
            deallocate(send_SE, recv_SE, send_SW, recv_SW)
        end if

        ! Allocate new buffers
        allocate(send_E(ew_size), recv_E(ew_size))
        allocate(send_W(ew_size), recv_W(ew_size))
        allocate(send_N(ns_size), recv_N(ns_size))
        allocate(send_S(ns_size), recv_S(ns_size))
        allocate(send_NE(corner_size), recv_NE(corner_size))
        allocate(send_NW(corner_size), recv_NW(corner_size))
        allocate(send_SE(corner_size), recv_SE(corner_size))
        allocate(send_SW(corner_size), recv_SW(corner_size))

        ! Create persistent device copies (one-time cost)
        !$omp target enter data map(alloc: send_E, recv_E, send_W, recv_W)
        !$omp target enter data map(alloc: send_N, recv_N, send_S, recv_S)
        !$omp target enter data map(alloc: send_NE, recv_NE, send_NW, recv_NW)
        !$omp target enter data map(alloc: send_SE, recv_SE, send_SW, recv_SW)

        alloc_ew = ew_size
        alloc_ns = ns_size
        alloc_co = corner_size
    end subroutine

    !> 3D halo exchange for OpenMP target arrays
    subroutine halo_exchange_3d(array, isd, ied, jsd, jed, nz, MD, halo_width)
        integer, intent(in) :: isd, ied, jsd, jed, nz, halo_width
        real(dp), intent(inout) :: array(isd:ied, jsd:jed, nz)
        type(mpi_domain_type), intent(in) :: MD

        integer :: ni_total, nj_total, hw
        integer :: ew_size, ns_size, corner_size
        integer :: nreqs, i, j, k, idx
        integer :: ierr

        integer :: reqs(16)
        integer :: stats(MPI_STATUS_SIZE, 16)

        call check_gpu_aware_mpi()

        hw = halo_width
        ni_total = ied - isd + 1
        nj_total = jed - jsd + 1

        ew_size = hw * (nj_total - 2 * hw) * nz
        ns_size = ni_total * hw * nz
        corner_size = hw * hw * nz

        call ensure_buffers(ew_size, ns_size, corner_size)

        ! === PACK send buffers on GPU ===

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    send_E(idx) = array(ied - 2 * hw + i, j, k)
                end do
            end do
        end do

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    send_W(idx) = array(isd + hw + i - 1, j, k)
                end do
            end do
        end do

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    send_N(idx) = array(isd + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    send_S(idx) = array(isd + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_NE(idx) = array(ied - 2 * hw + i, jed - 2 * hw + j, k)
                end do
            end do
        end do

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_NW(idx) = array(isd + hw + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_SE(idx) = array(ied - 2 * hw + i, jsd + hw + j - 1, k)
                end do
            end do
        end do

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_SW(idx) = array(isd + hw + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! === MPI communication ===
        if (gpu_aware_mpi) then
            ! GPU-aware MPI: device pointers passed directly to MPI.
            ! MPI calls MUST be inline within target data use_device_addr.
            !$omp target data use_device_addr(send_E, recv_E, send_W, recv_W, &
            !$omp&    send_N, recv_N, send_S, recv_S, &
            !$omp&    send_NE, recv_NE, send_NW, recv_NW, &
            !$omp&    send_SE, recv_SE, send_SW, recv_SW)

            nreqs = 0
            ! Receives
            if (MD%east /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, &
                               100, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%west /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, &
                               101, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%north /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, &
                               102, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%south /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, &
                               103, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%ne /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, &
                               104, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%nw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, &
                               105, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%se /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, &
                               106, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%sw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, &
                               107, MD%comm, reqs(nreqs), ierr)
            end if
            ! Sends (tag swapped so east send matches west recv, etc.)
            if (MD%east /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, &
                               101, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%west /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, &
                               100, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%north /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, &
                               103, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%south /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, &
                               102, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%ne /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, &
                               107, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%nw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, &
                               106, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%se /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, &
                               105, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%sw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, &
                               104, MD%comm, reqs(nreqs), ierr)
            end if

            if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

            !$omp end target data
        else
            ! Host staging: GPU -> host -> MPI -> host -> GPU
            !$omp target update from(send_E(1:ew_size), send_W(1:ew_size))
            !$omp target update from(send_N(1:ns_size), send_S(1:ns_size))
            !$omp target update from(send_NE(1:corner_size), send_NW(1:corner_size))
            !$omp target update from(send_SE(1:corner_size), send_SW(1:corner_size))

            nreqs = 0
            call post_irecv_isend(MD, reqs, nreqs, &
                send_E, recv_E, send_W, recv_W, send_N, recv_N, send_S, recv_S, &
                send_NE, recv_NE, send_NW, recv_NW, send_SE, recv_SE, send_SW, recv_SW, &
                ew_size, ns_size, corner_size, 100)
            if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

            !$omp target update to(recv_E(1:ew_size), recv_W(1:ew_size))
            !$omp target update to(recv_N(1:ns_size), recv_S(1:ns_size))
            !$omp target update to(recv_NE(1:corner_size), recv_NW(1:corner_size))
            !$omp target update to(recv_SE(1:corner_size), recv_SW(1:corner_size))
        end if

        ! === UNPACK recv buffers on GPU ===

        if (MD%east /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = jsd + hw, jed - hw
                    do i = 1, hw
                        idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                        array(ied - hw + i, j, k) = recv_E(idx)
                    end do
                end do
            end do
        end if

        if (MD%west /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = jsd + hw, jed - hw
                    do i = 1, hw
                        idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                        array(isd + i - 1, j, k) = recv_W(idx)
                    end do
                end do
            end do
        end if

        if (MD%north /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, ni_total
                        idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                        array(isd + i - 1, jed - hw + j, k) = recv_N(idx)
                    end do
                end do
            end do
        end if

        if (MD%south /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, ni_total
                        idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                        array(isd + i - 1, jsd + j - 1, k) = recv_S(idx)
                    end do
                end do
            end do
        end if

        if (MD%ne /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(ied - hw + i, jed - hw + j, k) = recv_NE(idx)
                    end do
                end do
            end do
        end if

        if (MD%nw /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(isd + i - 1, jed - hw + j, k) = recv_NW(idx)
                    end do
                end do
            end do
        end if

        if (MD%se /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(ied - hw + i, jsd + j - 1, k) = recv_SE(idx)
                    end do
                end do
            end do
        end if

        if (MD%sw /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(isd + i - 1, jsd + j - 1, k) = recv_SW(idx)
                    end do
                end do
            end do
        end if

    end subroutine halo_exchange_3d

    !> 2D halo exchange for OpenMP target arrays
    subroutine halo_exchange_2d(array, isd, ied, jsd, jed, MD, halo_width)
        integer, intent(in) :: isd, ied, jsd, jed, halo_width
        real(dp), intent(inout) :: array(isd:ied, jsd:jed)
        type(mpi_domain_type), intent(in) :: MD

        integer :: ni_total, nj_total, hw
        integer :: ew_size, ns_size, corner_size
        integer :: nreqs, i, j, idx
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

        call ensure_buffers(ew_size, ns_size, corner_size)

        ! === PACK ===
        !$omp target teams distribute parallel do collapse(2)
        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                send_E(idx) = array(ied - 2 * hw + i, j)
            end do
        end do

        !$omp target teams distribute parallel do collapse(2)
        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                send_W(idx) = array(isd + hw + i - 1, j)
            end do
        end do

        !$omp target teams distribute parallel do collapse(2)
        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                send_N(idx) = array(isd + i - 1, jed - 2 * hw + j)
            end do
        end do

        !$omp target teams distribute parallel do collapse(2)
        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                send_S(idx) = array(isd + i - 1, jsd + hw + j - 1)
            end do
        end do

        !$omp target teams distribute parallel do collapse(2)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_NE(idx) = array(ied - 2 * hw + i, jed - 2 * hw + j)
            end do
        end do

        !$omp target teams distribute parallel do collapse(2)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_NW(idx) = array(isd + hw + i - 1, jed - 2 * hw + j)
            end do
        end do

        !$omp target teams distribute parallel do collapse(2)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_SE(idx) = array(ied - 2 * hw + i, jsd + hw + j - 1)
            end do
        end do

        !$omp target teams distribute parallel do collapse(2)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_SW(idx) = array(isd + hw + i - 1, jsd + hw + j - 1)
            end do
        end do

        ! === MPI communication ===
        if (gpu_aware_mpi) then
            ! GPU-aware MPI: inline MPI calls within target data use_device_addr
            !$omp target data use_device_addr(send_E, recv_E, send_W, recv_W, &
            !$omp&    send_N, recv_N, send_S, recv_S, &
            !$omp&    send_NE, recv_NE, send_NW, recv_NW, &
            !$omp&    send_SE, recv_SE, send_SW, recv_SW)

            nreqs = 0
            ! Receives
            if (MD%east /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, &
                               200, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%west /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, &
                               201, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%north /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, &
                               202, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%south /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, &
                               203, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%ne /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, &
                               204, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%nw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, &
                               205, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%se /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, &
                               206, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%sw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Irecv(recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, &
                               207, MD%comm, reqs(nreqs), ierr)
            end if
            ! Sends
            if (MD%east /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, &
                               201, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%west /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, &
                               200, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%north /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, &
                               203, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%south /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, &
                               202, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%ne /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, &
                               207, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%nw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, &
                               206, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%se /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, &
                               205, MD%comm, reqs(nreqs), ierr)
            end if
            if (MD%sw /= MPI_PROC_NULL) then
                nreqs = nreqs + 1
                call MPI_Isend(send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, &
                               204, MD%comm, reqs(nreqs), ierr)
            end if

            if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

            !$omp end target data
        else
            !$omp target update from(send_E(1:ew_size), send_W(1:ew_size))
            !$omp target update from(send_N(1:ns_size), send_S(1:ns_size))
            !$omp target update from(send_NE(1:corner_size), send_NW(1:corner_size))
            !$omp target update from(send_SE(1:corner_size), send_SW(1:corner_size))

            nreqs = 0
            call post_irecv_isend(MD, reqs, nreqs, &
                send_E, recv_E, send_W, recv_W, send_N, recv_N, send_S, recv_S, &
                send_NE, recv_NE, send_NW, recv_NW, send_SE, recv_SE, send_SW, recv_SW, &
                ew_size, ns_size, corner_size, 200)
            if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

            !$omp target update to(recv_E(1:ew_size), recv_W(1:ew_size))
            !$omp target update to(recv_N(1:ns_size), recv_S(1:ns_size))
            !$omp target update to(recv_NE(1:corner_size), recv_NW(1:corner_size))
            !$omp target update to(recv_SE(1:corner_size), recv_SW(1:corner_size))
        end if

        ! === UNPACK ===
        if (MD%east /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    array(ied - hw + i, j) = recv_E(idx)
                end do
            end do
        end if

        if (MD%west /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    array(isd + i - 1, j) = recv_W(idx)
                end do
            end do
        end if

        if (MD%north /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    array(isd + i - 1, jed - hw + j) = recv_N(idx)
                end do
            end do
        end if

        if (MD%south /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    array(isd + i - 1, jsd + j - 1) = recv_S(idx)
                end do
            end do
        end if

        if (MD%ne /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(ied - hw + i, jed - hw + j) = recv_NE(idx)
                end do
            end do
        end if

        if (MD%nw /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(isd + i - 1, jed - hw + j) = recv_NW(idx)
                end do
            end do
        end if

        if (MD%se /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(ied - hw + i, jsd + j - 1) = recv_SE(idx)
                end do
            end do
        end if

        if (MD%sw /= MPI_PROC_NULL) then
            !$omp target teams distribute parallel do collapse(2)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(isd + i - 1, jsd + j - 1) = recv_SW(idx)
                end do
            end do
        end if

    end subroutine halo_exchange_2d

    !> Free persistent halo buffers. Call at program finalization.
    subroutine halo_cleanup()
        if (allocated(send_E)) then
            !$omp target exit data map(delete: send_E, recv_E, send_W, recv_W)
            !$omp target exit data map(delete: send_N, recv_N, send_S, recv_S)
            !$omp target exit data map(delete: send_NE, recv_NE, send_NW, recv_NW)
            !$omp target exit data map(delete: send_SE, recv_SE, send_SW, recv_SW)
            deallocate(send_E, recv_E, send_W, recv_W)
            deallocate(send_N, recv_N, send_S, recv_S)
            deallocate(send_NE, recv_NE, send_NW, recv_NW)
            deallocate(send_SE, recv_SE, send_SW, recv_SW)
        end if
        alloc_ew = -1
        alloc_ns = -1
        alloc_co = -1
    end subroutine

    !> Post all Irecv and Isend for 8-direction halo exchange.
    !! Used ONLY by the host-staging fallback path (not the GPU-aware path,
    !! which has MPI calls inlined within target data use_device_addr).
    subroutine post_irecv_isend(MD, reqs, nreqs, &
            send_E, recv_E, send_W, recv_W, send_N, recv_N, send_S, recv_S, &
            send_NE, recv_NE, send_NW, recv_NW, send_SE, recv_SE, send_SW, recv_SW, &
            ew_size, ns_size, corner_size, tag_base)
        type(mpi_domain_type), intent(in) :: MD
        integer, intent(inout) :: reqs(:)
        integer, intent(inout) :: nreqs
        real(dp), intent(inout) :: send_E(*), recv_E(*), send_W(*), recv_W(*)
        real(dp), intent(inout) :: send_N(*), recv_N(*), send_S(*), recv_S(*)
        real(dp), intent(inout) :: send_NE(*), recv_NE(*), send_NW(*), recv_NW(*)
        real(dp), intent(inout) :: send_SE(*), recv_SE(*), send_SW(*), recv_SW(*)
        integer, intent(in) :: ew_size, ns_size, corner_size, tag_base
        integer :: ierr

        ! Receives
        if (MD%east /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, &
                           tag_base, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%west /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, &
                           tag_base+1, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%north /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, &
                           tag_base+2, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%south /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, &
                           tag_base+3, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%ne /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, &
                           tag_base+4, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%nw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, &
                           tag_base+5, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%se /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, &
                           tag_base+6, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%sw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, &
                           tag_base+7, MD%comm, reqs(nreqs), ierr)
        end if

        ! Sends (tag swapped so east send matches west recv, etc.)
        if (MD%east /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, &
                           tag_base+1, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%west /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, &
                           tag_base, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%north /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, &
                           tag_base+3, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%south /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, &
                           tag_base+2, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%ne /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, &
                           tag_base+7, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%nw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, &
                           tag_base+6, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%se /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, &
                           tag_base+5, MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%sw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, &
                           tag_base+4, MD%comm, reqs(nreqs), ierr)
        end if
    end subroutine

end module mom6_mpi_halo_omp
