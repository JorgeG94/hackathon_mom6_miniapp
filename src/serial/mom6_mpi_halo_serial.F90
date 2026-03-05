!> Pure-CPU halo exchange for serial (non-GPU) MPI drivers
!!
!! Drop-in replacement for mom6_mpi_halo with all OpenACC directives
!! removed.  Pack/unpack loops run on the host; MPI communication uses
!! standard host buffers — no GPU-aware path, no device staging.
!!
!! Same public API: halo_exchange_3d, halo_exchange_2d, halo_cleanup
!!
module mom6_mpi_halo
    use mpi
    use iso_fortran_env, only: dp => real64
    use mom6_mpi_domain, only: mpi_domain_type
    implicit none
    private

    public :: halo_exchange_3d, halo_exchange_2d, halo_cleanup

    ! Persistent send/recv buffers — allocated once on first use (or when
    ! sizes change), kept alive across calls.
    real(dp), allocatable, save :: send_E(:), recv_E(:), send_W(:), recv_W(:)
    real(dp), allocatable, save :: send_N(:), recv_N(:), send_S(:), recv_S(:)
    real(dp), allocatable, save :: send_NE(:), recv_NE(:), send_NW(:), recv_NW(:)
    real(dp), allocatable, save :: send_SE(:), recv_SE(:), send_SW(:), recv_SW(:)
    integer, save :: alloc_ew = -1, alloc_ns = -1, alloc_co = -1

contains

    !> Ensure persistent buffers are allocated with at least the given sizes.
    subroutine ensure_buffers(ew_size, ns_size, corner_size)
        integer, intent(in) :: ew_size, ns_size, corner_size

        if (ew_size <= alloc_ew .and. ns_size <= alloc_ns .and. &
            corner_size <= alloc_co) return

        ! Free old buffers if previously allocated
        if (allocated(send_E)) then
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

        alloc_ew = ew_size
        alloc_ns = ns_size
        alloc_co = corner_size
    end subroutine

    !> 3D halo exchange for host-resident arrays (pure CPU)
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

        hw = halo_width
        ni_total = ied - isd + 1
        nj_total = jed - jsd + 1

        ew_size = hw * (nj_total - 2 * hw) * nz
        ns_size = ni_total * hw * nz
        corner_size = hw * hw * nz

        call ensure_buffers(ew_size, ns_size, corner_size)

        ! === PACK send buffers ===

        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    send_E(idx) = array(ied - 2 * hw + i, j, k)
                end do
            end do
        end do

        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    send_W(idx) = array(isd + hw + i - 1, j, k)
                end do
            end do
        end do

        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    send_N(idx) = array(isd + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    send_S(idx) = array(isd + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_NE(idx) = array(ied - 2 * hw + i, jed - 2 * hw + j, k)
                end do
            end do
        end do

        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_NW(idx) = array(isd + hw + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_SE(idx) = array(ied - 2 * hw + i, jsd + hw + j - 1, k)
                end do
            end do
        end do

        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_SW(idx) = array(isd + hw + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! === MPI communication (host buffers only) ===
        nreqs = 0
        call post_irecv_isend(MD, reqs, nreqs, &
            send_E, recv_E, send_W, recv_W, send_N, recv_N, send_S, recv_S, &
            send_NE, recv_NE, send_NW, recv_NW, send_SE, recv_SE, send_SW, recv_SW, &
            ew_size, ns_size, corner_size, 100)
        if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

        ! === UNPACK recv buffers ===

        if (MD%east /= MPI_PROC_NULL) then
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

    !> 2D halo exchange for host-resident arrays (pure CPU)
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

        hw = halo_width
        ni_total = ied - isd + 1
        nj_total = jed - jsd + 1

        ew_size = hw * (nj_total - 2 * hw)
        ns_size = ni_total * hw
        corner_size = hw * hw

        call ensure_buffers(ew_size, ns_size, corner_size)

        ! === PACK ===
        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                send_E(idx) = array(ied - 2 * hw + i, j)
            end do
        end do

        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                send_W(idx) = array(isd + hw + i - 1, j)
            end do
        end do

        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                send_N(idx) = array(isd + i - 1, jed - 2 * hw + j)
            end do
        end do

        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                send_S(idx) = array(isd + i - 1, jsd + hw + j - 1)
            end do
        end do

        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_NE(idx) = array(ied - 2 * hw + i, jed - 2 * hw + j)
            end do
        end do

        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_NW(idx) = array(isd + hw + i - 1, jed - 2 * hw + j)
            end do
        end do

        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_SE(idx) = array(ied - 2 * hw + i, jsd + hw + j - 1)
            end do
        end do

        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_SW(idx) = array(isd + hw + i - 1, jsd + hw + j - 1)
            end do
        end do

        ! === MPI communication (host buffers only) ===
        nreqs = 0
        call post_irecv_isend(MD, reqs, nreqs, &
            send_E, recv_E, send_W, recv_W, send_N, recv_N, send_S, recv_S, &
            send_NE, recv_NE, send_NW, recv_NW, send_SE, recv_SE, send_SW, recv_SW, &
            ew_size, ns_size, corner_size, 200)
        if (nreqs > 0) call MPI_Waitall(nreqs, reqs(1:nreqs), stats(:, 1:nreqs), ierr)

        ! === UNPACK ===
        if (MD%east /= MPI_PROC_NULL) then
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    array(ied - hw + i, j) = recv_E(idx)
                end do
            end do
        end if

        if (MD%west /= MPI_PROC_NULL) then
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    array(isd + i - 1, j) = recv_W(idx)
                end do
            end do
        end if

        if (MD%north /= MPI_PROC_NULL) then
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    array(isd + i - 1, jed - hw + j) = recv_N(idx)
                end do
            end do
        end if

        if (MD%south /= MPI_PROC_NULL) then
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    array(isd + i - 1, jsd + j - 1) = recv_S(idx)
                end do
            end do
        end if

        if (MD%ne /= MPI_PROC_NULL) then
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(ied - hw + i, jed - hw + j) = recv_NE(idx)
                end do
            end do
        end if

        if (MD%nw /= MPI_PROC_NULL) then
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(isd + i - 1, jed - hw + j) = recv_NW(idx)
                end do
            end do
        end if

        if (MD%se /= MPI_PROC_NULL) then
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(ied - hw + i, jsd + j - 1) = recv_SE(idx)
                end do
            end do
        end if

        if (MD%sw /= MPI_PROC_NULL) then
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

end module mom6_mpi_halo
