!> Shared types and constants for MOM6 mini-apps
!!
!! This module defines the grid structure and common parameters used by
!! all MOM6 mini-app modules (continuity, Coriolis, barotropic).
!!
module mom6_types
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: ocean_grid_type, verticalGrid_type
    public :: G_EARTH, RHO_0, OMEGA
    public :: init_ocean_grid, end_ocean_grid, init_verticalGrid
    public :: BT_cont_type, alloc_BT_cont_type
    public :: mech_forcing_type, init_mech_forcing, end_mech_forcing
    public :: vertvisc_type, init_vertvisc_visc, end_vertvisc_visc

    !> Ocean grid structure (simplified from MOM6's ocean_grid_type)
    type :: ocean_grid_type
        integer :: ni, nj, nk       ! Grid dimensions (computational domain)
        integer :: isd, ied         ! Data domain i-bounds (with halos)
        integer :: jsd, jed         ! Data domain j-bounds (with halos)
        integer :: isc, iec         ! Computational domain i-bounds
        integer :: jsc, jec         ! Computational domain j-bounds

        ! Grid metrics (2D)
        real(dp), allocatable :: IareaT(:, :)   ! Inverse cell area at h-points [L-2]
        real(dp), allocatable :: areaT(:, :)    ! Cell area at h-points [L2]
        real(dp), allocatable :: dxT(:, :)      ! dx at h-points [L]
        real(dp), allocatable :: dyT(:, :)      ! dy at h-points [L]
        real(dp), allocatable :: IdxT(:, :)     ! Inverse dx at h-points [L-1]
        real(dp), allocatable :: IdyT(:, :)     ! Inverse dy at h-points [L-1]
        real(dp), allocatable :: dxCu(:, :)     ! dx at u-points [L]
        real(dp), allocatable :: dyCu(:, :)     ! dy at u-points [L]
        real(dp), allocatable :: IdxCu(:, :)    ! Inverse dx at u-points [L-1]
        real(dp), allocatable :: IdyCu(:, :)    ! Inverse dy at u-points [L-1]
        real(dp), allocatable :: dy_Cu(:, :)    ! The unblocked lengths of the u-faces of the h-cell
        real(dp), allocatable :: dxCv(:, :)     ! dx at v-points [L]
        real(dp), allocatable :: dyCv(:, :)     ! dy at v-points [L]
        real(dp), allocatable :: IdxCv(:, :)    ! Inverse dx at v-points [L-1]
        real(dp), allocatable :: IdyCv(:, :)    ! Inverse dy at v-points [L-1]
        real(dp), allocatable :: IareaBu(:, :)  ! Inverse area at q-points [L-2]
        real(dp), allocatable :: areaBu(:, :)   ! Area at q-points [L2]
        real(dp), allocatable :: CoriolisBu(:, :)  ! Coriolis parameter at q-points [T-1]

        ! Additional metrics for horizontal viscosity
        real(dp), allocatable :: IareaCu(:, :)  ! Inverse area at u-points [L-2]
        real(dp), allocatable :: IareaCv(:, :)  ! Inverse area at v-points [L-2]
        real(dp), allocatable :: dxBu(:, :)     ! dx at q-points [L]
        real(dp), allocatable :: dyBu(:, :)     ! dy at q-points [L]
        real(dp), allocatable :: IdxBu(:, :)    ! Inverse dx at q-points [L-1]
        real(dp), allocatable :: IdyBu(:, :)    ! Inverse dy at q-points [L-1]
        real(dp), allocatable :: bathyT(:, :)   ! Bathymetry at h-points [Z]
        real(dp), allocatable :: mask2dT(:, :)  ! Mask at h-points (0=land, 1=ocean)
        real(dp), allocatable :: mask2dBu(:, :)  ! Mask at q-points
        real(dp), allocatable :: mask2dCu(:, :)  ! Mask at u-points
        real(dp), allocatable :: mask2dCv(:, :)  ! Mask at v-points

        ! Grid spacing (uniform for simplicity)
        real(dp) :: dx, dy          ! Grid spacing [L]

        ! Direction alternation (for symmetry in split schemes)
        integer :: first_direction  ! 0 or 1, alternates each timestep
    end type ocean_grid_type

    !> Vertical grid structure
    type :: verticalGrid_type
        integer :: ke               ! Number of layers
        real(dp) :: Angstrom_H      ! Minimum layer thickness [H]
    end type verticalGrid_type

    !> Physical constants
    real(dp), parameter :: G_EARTH = 9.80_dp    ! Gravitational acceleration [m s-2]
    real(dp), parameter :: RHO_0 = 1035.0_dp    ! Reference density [kg m-3]
    real(dp), parameter :: OMEGA = 7.2921e-5_dp  ! Earth rotation rate [s-1]

    !> Container for information about the summed layer transports
   !! and how they will vary as the barotropic velocity is changed.
    type :: BT_cont_type
        real(dp), allocatable :: FA_u_EE(:, :) !< The effective open face area for zonal barotropic transport
                                       !! drawing from locations far to the east [H L ~> m2 or kg m-1].
        real(dp), allocatable :: FA_u_E0(:, :) !< The effective open face area for zonal barotropic transport
                                       !! drawing from nearby to the east [H L ~> m2 or kg m-1].
        real(dp), allocatable :: FA_u_W0(:, :) !< The effective open face area for zonal barotropic transport
                                       !! drawing from nearby to the west [H L ~> m2 or kg m-1].
        real(dp), allocatable :: FA_u_WW(:, :) !< The effective open face area for zonal barotropic transport
                                       !! drawing from locations far to the west [H L ~> m2 or kg m-1].
        real(dp), allocatable :: uBT_WW(:, :)  !< uBT_WW is the barotropic velocity [L T-1 ~> m s-1], beyond which the
                                       !! marginal open face area is FA_u_WW.  uBT_WW must be non-negative.
        real(dp), allocatable :: uBT_EE(:, :)  !< uBT_EE is a barotropic velocity [L T-1 ~> m s-1], beyond which the
                                       !! marginal open face area is FA_u_EE. uBT_EE must be non-positive.
        real(dp), allocatable :: h_u(:, :, :)   !< An effective thickness at zonal faces, taking into account the effects
                                       !! of vertical viscosity and fractional open areas [H ~> m or kg m-2].
                                       !! This is primarily used as a non-normalized weight in determining
                                       !! the depth averaged accelerations for the barotropic solver.
        ! would also have equivalent variables for meridional, but those are ignored for this example
    end type BT_cont_type

    !> Mechanical forcing type (wind stress)
    type :: mech_forcing_type
        real(dp), allocatable :: taux(:, :)   ! Zonal wind stress at u-points [Pa]
        real(dp), allocatable :: tauy(:, :)   ! Meridional wind stress at v-points [Pa]
    end type mech_forcing_type

    !> Vertical viscosity input type (Rayleigh drag)
    type :: vertvisc_type
        real(dp), allocatable :: Ray_u(:, :, :)  ! Rayleigh drag at u-points [H T-1 ~> m/s]
        real(dp), allocatable :: Ray_v(:, :, :)  ! Rayleigh drag at v-points [H T-1 ~> m/s]
        logical :: has_Rayleigh = .false.
    end type vertvisc_type

contains

    !> Initialize the ocean grid with uniform spacing
    subroutine init_ocean_grid(G, ni, nj, nk, dx_km, lat_deg)
        type(ocean_grid_type), intent(inout) :: G
        integer, intent(in) :: ni, nj, nk
        real(dp), intent(in) :: dx_km      ! Grid spacing in km
        real(dp), intent(in) :: lat_deg    ! Central latitude for Coriolis

        real(dp) :: dx_m, f0, beta
        integer :: i, j

        ! Store dimensions
        G%ni = ni
        G%nj = nj
        G%nk = nk

        ! With halo of 1: index 1 = halo, 2:n+1 = computational, n+2 = halo
        G%isd = 1; G%ied = ni + 2
        G%jsd = 1; G%jed = nj + 2
        G%isc = G%isd + 3; G%iec = G%ied - 3
        G%jsc = G%jsd + 3; G%jec = G%jed - 3

        ! Grid spacing
        dx_m = dx_km*1000.0_dp
        G%dx = dx_m
        G%dy = dx_m

        ! Allocate arrays
        allocate (G%IareaT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%areaT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dxT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dyT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdxT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdyT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dxCu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dyCu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dy_Cu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdxCu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdyCu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dxCv(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dyCv(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdxCv(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdyCv(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IareaBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%areaBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%CoriolisBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IareaCu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IareaCv(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dxBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%dyBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdxBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%IdyBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%bathyT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%mask2dT(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%mask2dBu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%mask2dCu(G%isd:G%ied, G%jsd:G%jed))
        allocate (G%mask2dCv(G%isd:G%ied, G%jsd:G%jed))

        ! Initialize uniform grid metrics
#ifdef __NVCOMPILER_LLVM__
        !$omp target enter data map(to: G)
        !$omp target enter data map(alloc: G%IareaT, G%areaT, G%dxT, G%dyT, G%IdxT, G%IdyT)
        !$omp target enter data map(alloc: G%dxCu, G%dyCu, G%IdxCu, G%IdyCu)
        !$omp target enter data map(alloc: G%dxCv, G%dyCv, G%IdxCv, G%IdyCv)
        !$omp target enter data map(alloc: G%IareaBu, G%areaBu, G%CoriolisBu)
        !$omp target enter data map(alloc: G%IareaCu, G%IareaCv, G%dxBu, G%dyBu, G%IdxBu, G%IdyBu)
        !$omp target enter data map(alloc: G%bathyT, G%mask2dT, G%mask2dBu, G%mask2dCu, G%mask2dCv)
#endif

        ! Beta-plane Coriolis: f = f0 + beta*y
        f0 = 2.0_dp*OMEGA*sin(lat_deg*3.14159265358979_dp/180.0_dp)
        beta = 2.0_dp*OMEGA*cos(lat_deg*3.14159265358979_dp/180.0_dp)/6.371e6_dp

        ! I still don't understand why I haven't been able to initialize this on the device
        ! Initialize grid metrics on HOST (using regular loops, not do concurrent)
        ! This ensures host has valid data for diagnostics/verification
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                G%areaT(i, j) = dx_m*dx_m
                G%IareaT(i, j) = 1.0_dp/(dx_m*dx_m)
                G%dxT(i, j) = dx_m
                G%dyT(i, j) = dx_m
                G%IdxT(i, j) = 1.0_dp/dx_m
                G%IdyT(i, j) = 1.0_dp/dx_m
                G%dxCu(i, j) = dx_m
                G%dyCu(i, j) = dx_m
                G%IdxCu(i, j) = 1.0_dp/dx_m
                G%IdyCu(i, j) = 1.0_dp/dx_m
                G%dxCv(i, j) = dx_m
                G%dyCv(i, j) = dx_m
                G%IdxCv(i, j) = 1.0_dp/dx_m
                G%IdyCv(i, j) = 1.0_dp/dx_m
                G%areaBu(i, j) = dx_m*dx_m
                G%IareaBu(i, j) = 1.0_dp/(dx_m*dx_m)
                G%IareaCu(i, j) = 1.0_dp/(dx_m*dx_m)
                G%IareaCv(i, j) = 1.0_dp/(dx_m*dx_m)
                G%dxBu(i, j) = dx_m
                G%dyBu(i, j) = dx_m
                G%IdxBu(i, j) = 1.0_dp/dx_m
                G%IdyBu(i, j) = 1.0_dp/dx_m
                G%mask2dT(i, j) = 1.0_dp
                G%mask2dBu(i, j) = 1.0_dp
                G%mask2dCu(i, j) = 1.0_dp
                G%mask2dCv(i, j) = 1.0_dp
                G%bathyT(i, j) = 4000.0_dp  ! 4000m depth

                ! Coriolis at q-points (offset by half grid from center)
                G%CoriolisBu(i, j) = f0 + beta*(real(j - 1 - nj/2, dp) - 0.5_dp)*dx_m
            end do
        end do

        ! Copy grid data from host to device
#ifdef __NVCOMPILER_LLVM__
        !$omp target update to(G%IareaT, G%areaT, G%dxT, G%dyT, G%IdxT, G%IdyT)
        !$omp target update to(G%dxCu, G%dyCu, G%IdxCu, G%IdyCu)
        !$omp target update to(G%dxCv, G%dyCv, G%IdxCv, G%IdyCv)
        !$omp target update to(G%IareaBu, G%areaBu, G%CoriolisBu)
        !$omp target update to(G%IareaCu, G%IareaCv, G%dxBu, G%dyBu, G%IdxBu, G%IdyBu)
        !$omp target update to(G%bathyT, G%mask2dT, G%mask2dBu, G%mask2dCu, G%mask2dCv)
#endif

        G%first_direction = 0

    end subroutine init_ocean_grid

    !> Deallocate grid arrays
    subroutine end_ocean_grid(G)
        type(ocean_grid_type), intent(inout) :: G

#ifdef __NVCOMPILER_LLVM__
        !$omp target exit data map(delete: G%IareaT, G%areaT, G%dxT, G%dyT, G%IdxT, G%IdyT)
        !$omp target exit data map(delete: G%dxCu, G%dyCu, G%IdxCu, G%IdyCu)
        !$omp target exit data map(delete: G%dxCv, G%dyCv, G%IdxCv, G%IdyCv)
        !$omp target exit data map(delete: G%IareaBu, G%areaBu, G%CoriolisBu)
        !$omp target exit data map(delete: G%IareaCu, G%IareaCv, G%dxBu, G%dyBu, G%IdxBu, G%IdyBu)
        !$omp target exit data map(delete: G%bathyT, G%mask2dT, G%mask2dBu, G%mask2dCu, G%mask2dCv)
#endif

        if (allocated(G%IareaT)) deallocate (G%IareaT)
        if (allocated(G%areaT)) deallocate (G%areaT)
        if (allocated(G%dxT)) deallocate (G%dxT)
        if (allocated(G%dyT)) deallocate (G%dyT)
        if (allocated(G%IdxT)) deallocate (G%IdxT)
        if (allocated(G%IdyT)) deallocate (G%IdyT)
        if (allocated(G%dxCu)) deallocate (G%dxCu)
        if (allocated(G%dyCu)) deallocate (G%dyCu)
        if (allocated(G%IdxCu)) deallocate (G%IdxCu)
        if (allocated(G%IdyCu)) deallocate (G%IdyCu)
        if (allocated(G%dxCv)) deallocate (G%dxCv)
        if (allocated(G%dyCv)) deallocate (G%dyCv)
        if (allocated(G%IdxCv)) deallocate (G%IdxCv)
        if (allocated(G%IdyCv)) deallocate (G%IdyCv)
        if (allocated(G%IareaBu)) deallocate (G%IareaBu)
        if (allocated(G%areaBu)) deallocate (G%areaBu)
        if (allocated(G%CoriolisBu)) deallocate (G%CoriolisBu)
        if (allocated(G%IareaCu)) deallocate (G%IareaCu)
        if (allocated(G%IareaCv)) deallocate (G%IareaCv)
        if (allocated(G%dxBu)) deallocate (G%dxBu)
        if (allocated(G%dyBu)) deallocate (G%dyBu)
        if (allocated(G%IdxBu)) deallocate (G%IdxBu)
        if (allocated(G%IdyBu)) deallocate (G%IdyBu)
        if (allocated(G%bathyT)) deallocate (G%bathyT)
        if (allocated(G%mask2dT)) deallocate (G%mask2dT)
        if (allocated(G%mask2dBu)) deallocate (G%mask2dBu)
        if (allocated(G%mask2dCu)) deallocate (G%mask2dCu)
        if (allocated(G%mask2dCv)) deallocate (G%mask2dCv)

    end subroutine end_ocean_grid

    !> Initialize vertical grid
    subroutine init_verticalGrid(GV, nk)
        type(verticalGrid_type), intent(inout) :: GV
        integer, intent(in) :: nk

        GV%ke = nk
        GV%Angstrom_H = 1.0e-10_dp

    end subroutine init_verticalGrid

!> Allocates the arrays contained within a BT_cont_type and initializes them to 0.
    subroutine alloc_BT_cont_type(BT_cont, G, GV)
        type(BT_cont_type), pointer    :: BT_cont !< The BT_cont_type whose elements will be allocated
        type(ocean_grid_type), intent(in) :: G    !< The ocean's grid structure
        type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure.
        integer :: isd, ied, jsd, jed, nz
        isd = G%isd; ied = G%ied; jsd = G%jsd; jed = G%jed; nz = GV%ke

        allocate (BT_cont)
        allocate (BT_cont%FA_u_WW(G%isd:G%ied, G%jsd:G%jed), source=0._dp)
        allocate (BT_cont%FA_u_W0(G%isd:G%ied, G%jsd:G%jed), source=0._dp)
        allocate (BT_cont%FA_u_E0(G%isd:G%ied, G%jsd:G%jed), source=0._dp)
        allocate (BT_cont%FA_u_EE(G%isd:G%ied, G%jsd:G%jed), source=0._dp)
        allocate (BT_cont%uBT_WW(G%isd:G%ied, G%jsd:G%jed), source=0._dp)
        allocate (BT_cont%uBT_EE(G%isd:G%ied, G%jsd:G%jed), source=0._dp)

        allocate (BT_cont%h_u(isd:ied, jsd:jed, 1:nz), source=0._dp)

    end subroutine alloc_BT_cont_type

!> Deallocates the arrays contained within a BT_cont_type.
    subroutine dealloc_BT_cont_type(BT_cont)
        type(BT_cont_type), pointer :: BT_cont !< The BT_cont_type whose elements will be deallocated.

        if (.not. associated(BT_cont)) return

        if (allocated(BT_cont%FA_u_WW)) deallocate (BT_cont%FA_u_WW)
        if (allocated(BT_cont%FA_u_W0)) deallocate (BT_cont%FA_u_W0)
        if (allocated(BT_cont%FA_u_E0)) deallocate (BT_cont%FA_u_E0)
        if (allocated(BT_cont%FA_u_EE)) deallocate (BT_cont%FA_u_EE)
        if (allocated(BT_cont%uBT_WW)) deallocate (BT_cont%uBT_WW)
        if (allocated(BT_cont%uBT_EE)) deallocate (BT_cont%uBT_EE)
        if (allocated(BT_cont%h_u)) deallocate (BT_cont%h_u)

        deallocate (BT_cont)

    end subroutine dealloc_BT_cont_type

    !> Initialize mechanical forcing arrays
    subroutine init_mech_forcing(forces, G)
        type(mech_forcing_type), intent(inout) :: forces
        type(ocean_grid_type), intent(in) :: G

        allocate (forces%taux(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate (forces%tauy(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)

#ifdef __NVCOMPILER_LLVM__
        !$omp target enter data map(alloc: forces%taux, forces%tauy)
#endif

    end subroutine init_mech_forcing

    !> Deallocate mechanical forcing arrays
    subroutine end_mech_forcing(forces)
        type(mech_forcing_type), intent(inout) :: forces

#ifdef __NVCOMPILER_LLVM__
        !$omp target exit data map(delete: forces%taux, forces%tauy)
#endif

        if (allocated(forces%taux)) deallocate (forces%taux)
        if (allocated(forces%tauy)) deallocate (forces%tauy)

    end subroutine end_mech_forcing

    !> Initialize vertvisc_type arrays (Rayleigh drag)
    subroutine init_vertvisc_visc(visc, G, GV, use_rayleigh)
        type(vertvisc_type), intent(inout) :: visc
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        logical, intent(in), optional :: use_rayleigh

        integer :: nz

        nz = GV%ke
        visc%has_Rayleigh = .false.
        if (present(use_rayleigh)) visc%has_Rayleigh = use_rayleigh

        if (visc%has_Rayleigh) then
            allocate (visc%Ray_u(G%isd:G%ied, G%jsd:G%jed, nz), source=0.0_dp)
            allocate (visc%Ray_v(G%isd:G%ied, G%jsd:G%jed, nz), source=0.0_dp)

#ifdef __NVCOMPILER_LLVM__
            !$omp target enter data map(alloc: visc%Ray_u, visc%Ray_v)
#endif
        end if

    end subroutine init_vertvisc_visc

    !> Deallocate vertvisc_type arrays
    subroutine end_vertvisc_visc(visc)
        type(vertvisc_type), intent(inout) :: visc

#ifdef __NVCOMPILER_LLVM__
        if (allocated(visc%Ray_u)) then
            !$omp target exit data map(delete: visc%Ray_u, visc%Ray_v)
        end if
#endif

        if (allocated(visc%Ray_u)) deallocate (visc%Ray_u)
        if (allocated(visc%Ray_v)) deallocate (visc%Ray_v)

    end subroutine end_vertvisc_visc

end module mom6_types
