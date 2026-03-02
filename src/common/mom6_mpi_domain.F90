!> MPI domain decomposition infrastructure for MOM6 miniapp
!!
!! Provides 2D Cartesian domain decomposition, neighbor lookup,
!! and local grid initialization for multi-GPU/multi-node runs.
!! Physics modules remain completely MPI-unaware.
!!
module mom6_mpi_domain
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, OMEGA
    implicit none
    private

    public :: mpi_domain_type
    public :: mpi_domain_init, mpi_domain_end
    public :: init_ocean_grid_mpi

    !> MPI domain decomposition type
    type :: mpi_domain_type
        ! MPI communicator and topology
        integer :: comm                   ! Cartesian communicator
        integer :: rank = 0              ! Rank in comm
        integer :: npes = 1              ! Total PEs
        integer :: npes_x = 1            ! PEs in x-direction
        integer :: npes_y = 1            ! PEs in y-direction
        integer :: pos_x = 0             ! Position in x (0-based)
        integer :: pos_y = 0             ! Position in y (0-based)

        ! Neighbor ranks (MPI_PROC_NULL for boundaries)
        integer :: north = MPI_PROC_NULL
        integer :: south = MPI_PROC_NULL
        integer :: east  = MPI_PROC_NULL
        integer :: west  = MPI_PROC_NULL
        integer :: ne = MPI_PROC_NULL
        integer :: nw = MPI_PROC_NULL
        integer :: se = MPI_PROC_NULL
        integer :: sw = MPI_PROC_NULL

        ! Global domain size
        integer :: ni_global = 0
        integer :: nj_global = 0

        ! Local domain (compute only, excluding halos)
        integer :: ni_local = 0
        integer :: nj_local = 0

        ! Global offset of local compute (1,1) — 0-based
        integer :: i_offset = 0
        integer :: j_offset = 0

        ! Halo width
        integer :: halo = 3
    end type mpi_domain_type

contains

    !> Initialize MPI domain decomposition with 2D Cartesian topology
    subroutine mpi_domain_init(MD, ni_global, nj_global, npes_x, npes_y)
        type(mpi_domain_type), intent(inout) :: MD
        integer, intent(in) :: ni_global, nj_global
        integer, intent(in) :: npes_x, npes_y

        integer :: ierr, coords(2), nbr_coords(2), nbr_rank
        logical :: periods(2)
        integer :: base_ni, base_nj, rem_i, rem_j

        MD%ni_global = ni_global
        MD%nj_global = nj_global
        MD%npes_x = npes_x
        MD%npes_y = npes_y

        ! Create 2D Cartesian communicator (non-periodic)
        periods = [.false., .false.]
        call MPI_Cart_create(MPI_COMM_WORLD, 2, [npes_x, npes_y], periods, &
                             .true., MD%comm, ierr)

        call MPI_Comm_rank(MD%comm, MD%rank, ierr)
        call MPI_Comm_size(MD%comm, MD%npes, ierr)

        ! Get position in 2D layout
        call MPI_Cart_coords(MD%comm, MD%rank, 2, coords, ierr)
        MD%pos_x = coords(1)
        MD%pos_y = coords(2)

        ! Cardinal neighbors via MPI_Cart_shift
        ! dim=0 is x (east/west), dim=1 is y (north/south)
        call MPI_Cart_shift(MD%comm, 0, 1, MD%west, MD%east, ierr)
        call MPI_Cart_shift(MD%comm, 1, 1, MD%south, MD%north, ierr)

        ! Corner neighbors via MPI_Cart_rank
        ! NE: (pos_x+1, pos_y+1)
        if (MD%pos_x + 1 < npes_x .and. MD%pos_y + 1 < npes_y) then
            nbr_coords = [MD%pos_x + 1, MD%pos_y + 1]
            call MPI_Cart_rank(MD%comm, nbr_coords, nbr_rank, ierr)
            MD%ne = nbr_rank
        end if
        ! NW: (pos_x-1, pos_y+1)
        if (MD%pos_x - 1 >= 0 .and. MD%pos_y + 1 < npes_y) then
            nbr_coords = [MD%pos_x - 1, MD%pos_y + 1]
            call MPI_Cart_rank(MD%comm, nbr_coords, nbr_rank, ierr)
            MD%nw = nbr_rank
        end if
        ! SE: (pos_x+1, pos_y-1)
        if (MD%pos_x + 1 < npes_x .and. MD%pos_y - 1 >= 0) then
            nbr_coords = [MD%pos_x + 1, MD%pos_y - 1]
            call MPI_Cart_rank(MD%comm, nbr_coords, nbr_rank, ierr)
            MD%se = nbr_rank
        end if
        ! SW: (pos_x-1, pos_y-1)
        if (MD%pos_x - 1 >= 0 .and. MD%pos_y - 1 >= 0) then
            nbr_coords = [MD%pos_x - 1, MD%pos_y - 1]
            call MPI_Cart_rank(MD%comm, nbr_coords, nbr_rank, ierr)
            MD%sw = nbr_rank
        end if

        ! Compute local domain size (divide evenly, remainder to last PE)
        base_ni = ni_global / npes_x
        rem_i   = mod(ni_global, npes_x)
        if (MD%pos_x < rem_i) then
            MD%ni_local = base_ni + 1
            MD%i_offset = MD%pos_x * (base_ni + 1)
        else
            MD%ni_local = base_ni
            MD%i_offset = rem_i * (base_ni + 1) + (MD%pos_x - rem_i) * base_ni
        end if

        base_nj = nj_global / npes_y
        rem_j   = mod(nj_global, npes_y)
        if (MD%pos_y < rem_j) then
            MD%nj_local = base_nj + 1
            MD%j_offset = MD%pos_y * (base_nj + 1)
        else
            MD%nj_local = base_nj
            MD%j_offset = rem_j * (base_nj + 1) + (MD%pos_y - rem_j) * base_nj
        end if

    end subroutine mpi_domain_init

    !> Initialize local ocean grid for MPI domain (replaces init_ocean_grid)
    !!
    !! Allocates local-sized arrays with halos and fills metrics using
    !! global offsets. All spatially uniform metrics are identical on every PE.
    !! Position-dependent values (CoriolisBu) use the global j-index.
    subroutine init_ocean_grid_mpi(G, MD, GV, nk, dx_km, lat_deg)
        type(ocean_grid_type), intent(inout) :: G
        type(mpi_domain_type), intent(in) :: MD
        type(verticalGrid_type), intent(inout) :: GV
        integer, intent(in) :: nk
        real(dp), intent(in) :: dx_km
        real(dp), intent(in) :: lat_deg

        real(dp) :: dx_m, f0, beta
        integer :: i, j, j_global, halo

        halo = MD%halo

        ! Store local compute dimensions
        G%ni = MD%ni_local
        G%nj = MD%nj_local
        G%nk = nk

        ! Data domain: 1 to ni_local + 2*halo
        G%isd = 1
        G%ied = MD%ni_local + 2 * halo
        G%jsd = 1
        G%jed = MD%nj_local + 2 * halo

        ! Compute domain: halo+1 to ni_local+halo
        G%isc = halo + 1
        G%iec = MD%ni_local + halo
        G%jsc = halo + 1
        G%jec = MD%nj_local + halo

        ! Store MPI offsets
        G%i_offset = MD%i_offset
        G%j_offset = MD%j_offset

        ! Grid spacing
        dx_m = dx_km * 1000.0_dp
        G%dx = dx_m
        G%dy = dx_m

        ! Vertical grid
        GV%ke = nk
        GV%Angstrom_H = 1.0e-10_dp

        ! Allocate arrays (local size with halos)
        allocate(G%IareaT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%areaT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dy_Cu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IareaBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%areaBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%CoriolisBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IareaCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IareaCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%bathyT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dCv(G%isd:G%ied, G%jsd:G%jed))

        ! Beta-plane Coriolis parameters
        f0 = 2.0_dp * OMEGA * sin(lat_deg * 3.14159265358979_dp / 180.0_dp)
        beta = 2.0_dp * OMEGA * cos(lat_deg * 3.14159265358979_dp / 180.0_dp) / 6.371e6_dp

        ! Initialize grid metrics (uniform except Coriolis)
        do j = G%jsd, G%jed
            ! Global j-index for Coriolis computation
            ! j_local = j - halo gives 1-based local compute index
            ! j_global = (j - halo) + j_offset gives 0-based global index
            j_global = (j - halo) + MD%j_offset

            do i = G%isd, G%ied
                G%areaT(i, j) = dx_m * dx_m
                G%IareaT(i, j) = 1.0_dp / (dx_m * dx_m)
                G%dxT(i, j) = dx_m
                G%dyT(i, j) = dx_m
                G%IdxT(i, j) = 1.0_dp / dx_m
                G%IdyT(i, j) = 1.0_dp / dx_m
                G%dxCu(i, j) = dx_m
                G%dyCu(i, j) = dx_m
                G%dy_Cu(i, j) = dx_m
                G%IdxCu(i, j) = 1.0_dp / dx_m
                G%IdyCu(i, j) = 1.0_dp / dx_m
                G%dxCv(i, j) = dx_m
                G%dyCv(i, j) = dx_m
                G%IdxCv(i, j) = 1.0_dp / dx_m
                G%IdyCv(i, j) = 1.0_dp / dx_m
                G%areaBu(i, j) = dx_m * dx_m
                G%IareaBu(i, j) = 1.0_dp / (dx_m * dx_m)
                G%IareaCu(i, j) = 1.0_dp / (dx_m * dx_m)
                G%IareaCv(i, j) = 1.0_dp / (dx_m * dx_m)
                G%dxBu(i, j) = dx_m
                G%dyBu(i, j) = dx_m
                G%IdxBu(i, j) = 1.0_dp / dx_m
                G%IdyBu(i, j) = 1.0_dp / dx_m
                G%mask2dT(i, j) = 1.0_dp
                G%mask2dBu(i, j) = 1.0_dp
                G%mask2dCu(i, j) = 1.0_dp
                G%mask2dCv(i, j) = 1.0_dp
                G%bathyT(i, j) = 4000.0_dp

                ! Coriolis at q-points using global j-index
                ! Matches: f0 + beta*(real(j_global - 1 - nj_global/2, dp) - 0.5_dp)*dx_m
                ! where j_global is 1-based global index = j_global_0based + 1
                G%CoriolisBu(i, j) = f0 + beta * (real(j_global - MD%nj_global/2, dp) - 0.5_dp) * dx_m
            end do
        end do

        G%first_direction = 0

        ! Compute total bytes: 29 2D arrays of real(dp)
        G%nbytes = 29_int64 * int(G%ied - G%isd + 1, int64) * &
                   int(G%jed - G%jsd + 1, int64) * 8_int64

        ! Copy grid to GPU
        !$acc enter data copyin(G)
        !$acc enter data copyin(G%IareaT, G%areaT, G%dxT, G%dyT, G%IdxT, G%IdyT)
        !$acc enter data copyin(G%dxCu, G%dyCu, G%dy_Cu, G%IdxCu, G%IdyCu)
        !$acc enter data copyin(G%dxCv, G%dyCv, G%IdxCv, G%IdyCv)
        !$acc enter data copyin(G%IareaBu, G%areaBu, G%CoriolisBu)
        !$acc enter data copyin(G%IareaCu, G%IareaCv)
        !$acc enter data copyin(G%dxBu, G%dyBu, G%IdxBu, G%IdyBu)
        !$acc enter data copyin(G%bathyT)
        !$acc enter data copyin(G%mask2dT, G%mask2dBu, G%mask2dCu, G%mask2dCv)

        ! Copy GV to GPU
        !$acc enter data copyin(GV)

    end subroutine init_ocean_grid_mpi

    !> Finalize MPI domain
    subroutine mpi_domain_end(MD)
        type(mpi_domain_type), intent(inout) :: MD
        integer :: ierr

        if (MD%comm /= MPI_COMM_NULL .and. MD%comm /= MPI_COMM_WORLD) then
            call MPI_Comm_free(MD%comm, ierr)
        end if
    end subroutine mpi_domain_end

end module mom6_mpi_domain
