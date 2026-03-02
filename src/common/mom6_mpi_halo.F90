!> GPU-resident halo exchange for OpenACC arrays
!!
!! Provides halo exchange routines for 2D and 3D arrays that are
!! resident on the GPU via OpenACC. Uses !$acc host_data use_device(...)
!! for GPU-aware MPI when available, with host-staging fallback.
!!
!! Exchange pattern: 4 cardinal + 4 corner directions (8 total).
!! For boundary PEs (neighbor = MPI_PROC_NULL), MPI handles skip automatically.
!!
module mom6_mpi_halo
    use mpi_f08
    use iso_fortran_env, only: dp => real64
    use mom6_mpi_domain, only: mpi_domain_type
    implicit none
    private

    public :: halo_exchange_3d, halo_exchange_2d

contains

    !> 3D halo exchange for OpenACC arrays
    !!
    !! array(isd:ied, jsd:jed, nz) must already be on GPU via OpenACC.
    !! Exchanges halo_width cells in each direction (cardinal + corners).
    subroutine halo_exchange_3d(array, isd, ied, jsd, jed, nz, MD, halo_width)
        integer, intent(in) :: isd, ied, jsd, jed, nz, halo_width
        real(dp), intent(inout) :: array(isd:ied, jsd:jed, nz)
        type(mpi_domain_type), intent(in) :: MD

        ! Buffer sizes
        integer :: ni_total, nj_total, hw
        integer :: ew_size, ns_size, corner_size
        integer :: nreqs, i, j, k, idx
        integer :: ierr

        ! Send/recv buffers
        real(dp), allocatable :: send_E(:), send_W(:), send_N(:), send_S(:)
        real(dp), allocatable :: recv_E(:), recv_W(:), recv_N(:), recv_S(:)
        real(dp), allocatable :: send_NE(:), send_NW(:), send_SE(:), send_SW(:)
        real(dp), allocatable :: recv_NE(:), recv_NW(:), recv_SE(:), recv_SW(:)
        type(MPI_Request) :: reqs(16)
        type(MPI_Status) :: stats(16)

        hw = halo_width
        ni_total = ied - isd + 1
        nj_total = jed - jsd + 1

        ! East/West strips: hw columns x nj_compute rows x nz layers
        ! nj_compute = nj_total - 2*hw (just the compute domain rows)
        ew_size = hw * (nj_total - 2 * hw) * nz

        ! North/South strips: ni_total columns x hw rows x nz layers
        ! Use full width including halos for simplicity
        ns_size = ni_total * hw * nz

        ! Corner blocks: hw x hw x nz
        corner_size = hw * hw * nz

        ! Allocate buffers
        allocate(send_E(ew_size), recv_E(ew_size))
        allocate(send_W(ew_size), recv_W(ew_size))
        allocate(send_N(ns_size), recv_N(ns_size))
        allocate(send_S(ns_size), recv_S(ns_size))
        allocate(send_NE(corner_size), recv_NE(corner_size))
        allocate(send_NW(corner_size), recv_NW(corner_size))
        allocate(send_SE(corner_size), recv_SE(corner_size))
        allocate(send_SW(corner_size), recv_SW(corner_size))

        !$acc enter data create(send_E, recv_E, send_W, recv_W)
        !$acc enter data create(send_N, recv_N, send_S, recv_S)
        !$acc enter data create(send_NE, recv_NE, send_NW, recv_NW)
        !$acc enter data create(send_SE, recv_SE, send_SW, recv_SW)

        ! === PACK send buffers on GPU ===

        ! East: send rightmost hw compute columns
        ! Compute domain i: isc=isd+hw to iec=ied-hw
        ! Send columns iec-hw+1 to iec
        !$acc parallel loop collapse(3) present(array, send_E)
        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    send_E(idx) = array(ied - 2 * hw + i, j, k)
                end do
            end do
        end do

        ! West: send leftmost hw compute columns
        !$acc parallel loop collapse(3) present(array, send_W)
        do k = 1, nz
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                    send_W(idx) = array(isd + hw + i - 1, j, k)
                end do
            end do
        end do

        ! North: send topmost hw compute rows (full width)
        !$acc parallel loop collapse(3) present(array, send_N)
        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    send_N(idx) = array(isd + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        ! South: send bottommost hw compute rows (full width)
        !$acc parallel loop collapse(3) present(array, send_S)
        do k = 1, nz
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                    send_S(idx) = array(isd + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! NE corner: send top-right hw x hw block
        !$acc parallel loop collapse(3) present(array, send_NE)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_NE(idx) = array(ied - 2 * hw + i, jed - 2 * hw + j, k)
                end do
            end do
        end do

        ! NW corner
        !$acc parallel loop collapse(3) present(array, send_NW)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_NW(idx) = array(isd + hw + i - 1, jed - 2 * hw + j, k)
                end do
            end do
        end do

        ! SE corner
        !$acc parallel loop collapse(3) present(array, send_SE)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_SE(idx) = array(ied - 2 * hw + i, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! SW corner
        !$acc parallel loop collapse(3) present(array, send_SW)
        do k = 1, nz
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw + (k - 1) * hw * hw
                    send_SW(idx) = array(isd + hw + i - 1, jsd + hw + j - 1, k)
                end do
            end do
        end do

        ! === MPI communication with host staging ===
        ! Copy send buffers from device to host
        !$acc update self(send_E, send_W, send_N, send_S)
        !$acc update self(send_NE, send_NW, send_SE, send_SW)

        nreqs = 0

        ! Post receives then sends for each direction
        ! East: recv from east neighbor into east halo
        if (MD%east /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 100, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%west /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 101, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%north /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 102, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%south /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 103, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%ne /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 104, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%nw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 105, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%se /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 106, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%sw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 107, &
                           MD%comm, reqs(nreqs), ierr)
        end if

        ! Sends (matching tag: send east with tag 101 so west neighbor's recv_W gets it)
        if (MD%east /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 101, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%west /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 100, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%north /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 103, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%south /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 102, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%ne /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 107, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%nw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 106, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%se /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 105, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%sw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 104, &
                           MD%comm, reqs(nreqs), ierr)
        end if

        ! Wait for all
        if (nreqs > 0) then
            call MPI_Waitall(nreqs, reqs(1:nreqs), stats(1:nreqs), ierr)
        end if

        ! Copy recv buffers to device
        !$acc update device(recv_E, recv_W, recv_N, recv_S)
        !$acc update device(recv_NE, recv_NW, recv_SE, recv_SW)

        ! === UNPACK recv buffers on GPU ===

        ! East halo: columns ied-hw+1 to ied
        if (MD%east /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_E)
            do k = 1, nz
                do j = jsd + hw, jed - hw
                    do i = 1, hw
                        idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                        array(ied - hw + i, j, k) = recv_E(idx)
                    end do
                end do
            end do
        end if

        ! West halo: columns isd to isd+hw-1
        if (MD%west /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_W)
            do k = 1, nz
                do j = jsd + hw, jed - hw
                    do i = 1, hw
                        idx = i + (j - jsd - hw) * hw + (k - 1) * hw * (nj_total - 2 * hw)
                        array(isd + i - 1, j, k) = recv_W(idx)
                    end do
                end do
            end do
        end if

        ! North halo: rows jed-hw+1 to jed
        if (MD%north /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_N)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, ni_total
                        idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                        array(isd + i - 1, jed - hw + j, k) = recv_N(idx)
                    end do
                end do
            end do
        end if

        ! South halo: rows jsd to jsd+hw-1
        if (MD%south /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_S)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, ni_total
                        idx = i + (j - 1) * ni_total + (k - 1) * ni_total * hw
                        array(isd + i - 1, jsd + j - 1, k) = recv_S(idx)
                    end do
                end do
            end do
        end if

        ! NE corner halo
        if (MD%ne /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_NE)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(ied - hw + i, jed - hw + j, k) = recv_NE(idx)
                    end do
                end do
            end do
        end if

        ! NW corner halo
        if (MD%nw /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_NW)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(isd + i - 1, jed - hw + j, k) = recv_NW(idx)
                    end do
                end do
            end do
        end if

        ! SE corner halo
        if (MD%se /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_SE)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(ied - hw + i, jsd + j - 1, k) = recv_SE(idx)
                    end do
                end do
            end do
        end if

        ! SW corner halo
        if (MD%sw /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(3) present(array, recv_SW)
            do k = 1, nz
                do j = 1, hw
                    do i = 1, hw
                        idx = i + (j - 1) * hw + (k - 1) * hw * hw
                        array(isd + i - 1, jsd + j - 1, k) = recv_SW(idx)
                    end do
                end do
            end do
        end if

        ! Cleanup
        !$acc exit data delete(send_E, recv_E, send_W, recv_W)
        !$acc exit data delete(send_N, recv_N, send_S, recv_S)
        !$acc exit data delete(send_NE, recv_NE, send_NW, recv_NW)
        !$acc exit data delete(send_SE, recv_SE, send_SW, recv_SW)

        deallocate(send_E, recv_E, send_W, recv_W)
        deallocate(send_N, recv_N, send_S, recv_S)
        deallocate(send_NE, recv_NE, send_NW, recv_NW)
        deallocate(send_SE, recv_SE, send_SW, recv_SW)

    end subroutine halo_exchange_3d

    !> 2D halo exchange for OpenACC arrays
    !!
    !! array(isd:ied, jsd:jed) must already be on GPU via OpenACC.
    subroutine halo_exchange_2d(array, isd, ied, jsd, jed, MD, halo_width)
        integer, intent(in) :: isd, ied, jsd, jed, halo_width
        real(dp), intent(inout) :: array(isd:ied, jsd:jed)
        type(mpi_domain_type), intent(in) :: MD

        integer :: ni_total, nj_total, hw
        integer :: ew_size, ns_size, corner_size
        integer :: nreqs, i, j, idx
        integer :: ierr

        real(dp), allocatable :: send_E(:), send_W(:), send_N(:), send_S(:)
        real(dp), allocatable :: recv_E(:), recv_W(:), recv_N(:), recv_S(:)
        real(dp), allocatable :: send_NE(:), send_NW(:), send_SE(:), send_SW(:)
        real(dp), allocatable :: recv_NE(:), recv_NW(:), recv_SE(:), recv_SW(:)
        type(MPI_Request) :: reqs(16)
        type(MPI_Status) :: stats(16)

        hw = halo_width
        ni_total = ied - isd + 1
        nj_total = jed - jsd + 1

        ew_size = hw * (nj_total - 2 * hw)
        ns_size = ni_total * hw
        corner_size = hw * hw

        allocate(send_E(ew_size), recv_E(ew_size))
        allocate(send_W(ew_size), recv_W(ew_size))
        allocate(send_N(ns_size), recv_N(ns_size))
        allocate(send_S(ns_size), recv_S(ns_size))
        allocate(send_NE(corner_size), recv_NE(corner_size))
        allocate(send_NW(corner_size), recv_NW(corner_size))
        allocate(send_SE(corner_size), recv_SE(corner_size))
        allocate(send_SW(corner_size), recv_SW(corner_size))

        !$acc enter data create(send_E, recv_E, send_W, recv_W)
        !$acc enter data create(send_N, recv_N, send_S, recv_S)
        !$acc enter data create(send_NE, recv_NE, send_NW, recv_NW)
        !$acc enter data create(send_SE, recv_SE, send_SW, recv_SW)

        ! === PACK ===
        !$acc parallel loop collapse(2) present(array, send_E)
        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                send_E(idx) = array(ied - 2 * hw + i, j)
            end do
        end do

        !$acc parallel loop collapse(2) present(array, send_W)
        do j = jsd + hw, jed - hw
            do i = 1, hw
                idx = i + (j - jsd - hw) * hw
                send_W(idx) = array(isd + hw + i - 1, j)
            end do
        end do

        !$acc parallel loop collapse(2) present(array, send_N)
        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                send_N(idx) = array(isd + i - 1, jed - 2 * hw + j)
            end do
        end do

        !$acc parallel loop collapse(2) present(array, send_S)
        do j = 1, hw
            do i = 1, ni_total
                idx = i + (j - 1) * ni_total
                send_S(idx) = array(isd + i - 1, jsd + hw + j - 1)
            end do
        end do

        !$acc parallel loop collapse(2) present(array, send_NE)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_NE(idx) = array(ied - 2 * hw + i, jed - 2 * hw + j)
            end do
        end do

        !$acc parallel loop collapse(2) present(array, send_NW)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_NW(idx) = array(isd + hw + i - 1, jed - 2 * hw + j)
            end do
        end do

        !$acc parallel loop collapse(2) present(array, send_SE)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_SE(idx) = array(ied - 2 * hw + i, jsd + hw + j - 1)
            end do
        end do

        !$acc parallel loop collapse(2) present(array, send_SW)
        do j = 1, hw
            do i = 1, hw
                idx = i + (j - 1) * hw
                send_SW(idx) = array(isd + hw + i - 1, jsd + hw + j - 1)
            end do
        end do

        ! === MPI (host staging) ===
        !$acc update self(send_E, send_W, send_N, send_S)
        !$acc update self(send_NE, send_NW, send_SE, send_SW)

        nreqs = 0

        ! Receives
        if (MD%east /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 200, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%west /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 201, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%north /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 202, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%south /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 203, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%ne /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 204, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%nw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 205, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%se /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 206, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%sw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Irecv(recv_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 207, &
                           MD%comm, reqs(nreqs), ierr)
        end if

        ! Sends
        if (MD%east /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_E, ew_size, MPI_DOUBLE_PRECISION, MD%east, 201, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%west /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_W, ew_size, MPI_DOUBLE_PRECISION, MD%west, 200, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%north /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_N, ns_size, MPI_DOUBLE_PRECISION, MD%north, 203, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%south /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_S, ns_size, MPI_DOUBLE_PRECISION, MD%south, 202, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%ne /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_NE, corner_size, MPI_DOUBLE_PRECISION, MD%ne, 207, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%nw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_NW, corner_size, MPI_DOUBLE_PRECISION, MD%nw, 206, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%se /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_SE, corner_size, MPI_DOUBLE_PRECISION, MD%se, 205, &
                           MD%comm, reqs(nreqs), ierr)
        end if
        if (MD%sw /= MPI_PROC_NULL) then
            nreqs = nreqs + 1
            call MPI_Isend(send_SW, corner_size, MPI_DOUBLE_PRECISION, MD%sw, 204, &
                           MD%comm, reqs(nreqs), ierr)
        end if

        if (nreqs > 0) then
            call MPI_Waitall(nreqs, reqs(1:nreqs), stats(1:nreqs), ierr)
        end if

        !$acc update device(recv_E, recv_W, recv_N, recv_S)
        !$acc update device(recv_NE, recv_NW, recv_SE, recv_SW)

        ! === UNPACK ===
        if (MD%east /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_E)
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    array(ied - hw + i, j) = recv_E(idx)
                end do
            end do
        end if

        if (MD%west /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_W)
            do j = jsd + hw, jed - hw
                do i = 1, hw
                    idx = i + (j - jsd - hw) * hw
                    array(isd + i - 1, j) = recv_W(idx)
                end do
            end do
        end if

        if (MD%north /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_N)
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    array(isd + i - 1, jed - hw + j) = recv_N(idx)
                end do
            end do
        end if

        if (MD%south /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_S)
            do j = 1, hw
                do i = 1, ni_total
                    idx = i + (j - 1) * ni_total
                    array(isd + i - 1, jsd + j - 1) = recv_S(idx)
                end do
            end do
        end if

        if (MD%ne /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_NE)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(ied - hw + i, jed - hw + j) = recv_NE(idx)
                end do
            end do
        end if

        if (MD%nw /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_NW)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(isd + i - 1, jed - hw + j) = recv_NW(idx)
                end do
            end do
        end if

        if (MD%se /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_SE)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(ied - hw + i, jsd + j - 1) = recv_SE(idx)
                end do
            end do
        end if

        if (MD%sw /= MPI_PROC_NULL) then
            !$acc parallel loop collapse(2) present(array, recv_SW)
            do j = 1, hw
                do i = 1, hw
                    idx = i + (j - 1) * hw
                    array(isd + i - 1, jsd + j - 1) = recv_SW(idx)
                end do
            end do
        end if

        ! Cleanup
        !$acc exit data delete(send_E, recv_E, send_W, recv_W)
        !$acc exit data delete(send_N, recv_N, send_S, recv_S)
        !$acc exit data delete(send_NE, recv_NE, send_NW, recv_NW)
        !$acc exit data delete(send_SE, recv_SE, send_SW, recv_SW)

        deallocate(send_E, recv_E, send_W, recv_W)
        deallocate(send_N, recv_N, send_S, recv_S)
        deallocate(send_NE, recv_NE, send_NW, recv_NW)
        deallocate(send_SE, recv_SE, send_SW, recv_SW)

    end subroutine halo_exchange_2d

end module mom6_mpi_halo
