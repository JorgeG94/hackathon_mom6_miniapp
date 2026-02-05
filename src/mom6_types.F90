!> Shared types and constants for MOM6 mini-apps
!!
!! This module defines the grid structure and common parameters used by
!! all MOM6 mini-app modules (continuity, Coriolis, barotropic).
!!
module mom6_types
  use iso_fortran_env, only: dp => real64
  implicit none
  public

  !> Ocean grid structure (simplified from MOM6's ocean_grid_type)
  type :: ocean_grid_type
    integer :: ni, nj, nk       ! Grid dimensions (computational domain)
    integer :: isd, ied         ! Data domain i-bounds (with halos)
    integer :: jsd, jed         ! Data domain j-bounds (with halos)
    integer :: isc, iec         ! Computational domain i-bounds
    integer :: jsc, jec         ! Computational domain j-bounds

    ! Grid metrics (2D)
    real(dp), allocatable :: IareaT(:,:)   ! Inverse cell area at h-points [L-2]
    real(dp), allocatable :: areaT(:,:)    ! Cell area at h-points [L2]
    real(dp), allocatable :: dxT(:,:)      ! dx at h-points [L]
    real(dp), allocatable :: dyT(:,:)      ! dy at h-points [L]
    real(dp), allocatable :: IdxT(:,:)     ! Inverse dx at h-points [L-1]
    real(dp), allocatable :: IdyT(:,:)     ! Inverse dy at h-points [L-1]
    real(dp), allocatable :: dxCu(:,:)     ! dx at u-points [L]
    real(dp), allocatable :: dyCu(:,:)     ! dy at u-points [L]
    real(dp), allocatable :: IdxCu(:,:)    ! Inverse dx at u-points [L-1]
    real(dp), allocatable :: IdyCu(:,:)    ! Inverse dy at u-points [L-1]
    real(dp), allocatable :: dxCv(:,:)     ! dx at v-points [L]
    real(dp), allocatable :: dyCv(:,:)     ! dy at v-points [L]
    real(dp), allocatable :: IdxCv(:,:)    ! Inverse dx at v-points [L-1]
    real(dp), allocatable :: IdyCv(:,:)    ! Inverse dy at v-points [L-1]
    real(dp), allocatable :: IareaBu(:,:)  ! Inverse area at q-points [L-2]
    real(dp), allocatable :: areaBu(:,:)   ! Area at q-points [L2]
    real(dp), allocatable :: CoriolisBu(:,:) ! Coriolis parameter at q-points [T-1]

    ! Additional metrics for horizontal viscosity
    real(dp), allocatable :: IareaCu(:,:)  ! Inverse area at u-points [L-2]
    real(dp), allocatable :: IareaCv(:,:)  ! Inverse area at v-points [L-2]
    real(dp), allocatable :: dxBu(:,:)     ! dx at q-points [L]
    real(dp), allocatable :: dyBu(:,:)     ! dy at q-points [L]
    real(dp), allocatable :: IdxBu(:,:)    ! Inverse dx at q-points [L-1]
    real(dp), allocatable :: IdyBu(:,:)    ! Inverse dy at q-points [L-1]
    real(dp), allocatable :: bathyT(:,:)   ! Bathymetry at h-points [Z]
    real(dp), allocatable :: mask2dT(:,:)  ! Mask at h-points (0=land, 1=ocean)
    real(dp), allocatable :: mask2dBu(:,:) ! Mask at q-points
    real(dp), allocatable :: mask2dCu(:,:) ! Mask at u-points
    real(dp), allocatable :: mask2dCv(:,:) ! Mask at v-points

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
  real(dp), parameter :: OMEGA = 7.2921e-5_dp ! Earth rotation rate [s-1]

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
    G%isd = 1 ; G%ied = ni + 2
    G%jsd = 1 ; G%jed = nj + 2
    G%isc = 2 ; G%iec = ni + 1
    G%jsc = 2 ; G%jec = nj + 1

    ! Grid spacing
    dx_m = dx_km * 1000.0_dp
    G%dx = dx_m
    G%dy = dx_m

    ! Allocate arrays
    allocate(G%IareaT(G%isd:G%ied, G%jsd:G%jed))
    allocate(G%areaT(G%isd:G%ied, G%jsd:G%jed))
    allocate(G%dxT(G%isd:G%ied, G%jsd:G%jed))
    allocate(G%dyT(G%isd:G%ied, G%jsd:G%jed))
    allocate(G%IdxT(G%isd:G%ied, G%jsd:G%jed))
    allocate(G%IdyT(G%isd:G%ied, G%jsd:G%jed))
    allocate(G%dxCu(G%isd:G%ied, G%jsd:G%jed))
    allocate(G%dyCu(G%isd:G%ied, G%jsd:G%jed))
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

    ! Initialize uniform grid metrics
    !$omp target enter data map(alloc: G%IareaT, G%areaT, G%dxT, G%dyT, G%IdxT, G%IdyT)
    !$omp target enter data map(alloc: G%dxCu, G%dyCu, G%IdxCu, G%IdyCu)
    !$omp target enter data map(alloc: G%dxCv, G%dyCv, G%IdxCv, G%IdyCv)
    !$omp target enter data map(alloc: G%IareaBu, G%areaBu, G%CoriolisBu)
    !$omp target enter data map(alloc: G%IareaCu, G%IareaCv, G%dxBu, G%dyBu, G%IdxBu, G%IdyBu)
    !$omp target enter data map(alloc: G%bathyT, G%mask2dT, G%mask2dBu, G%mask2dCu, G%mask2dCv)

    ! Beta-plane Coriolis: f = f0 + beta*y
    f0 = 2.0_dp * OMEGA * sin(lat_deg * 3.14159265358979_dp / 180.0_dp)
    beta = 2.0_dp * OMEGA * cos(lat_deg * 3.14159265358979_dp / 180.0_dp) / 6.371e6_dp

    ! I still don't understand why I haven't been able to initialize this on the device 
    ! Initialize grid metrics on HOST (using regular loops, not do concurrent)
    ! This ensures host has valid data for diagnostics/verification
    do j = G%jsd, G%jed
      do i = G%isd, G%ied
        G%areaT(i,j) = dx_m * dx_m
        G%IareaT(i,j) = 1.0_dp / (dx_m * dx_m)
        G%dxT(i,j) = dx_m
        G%dyT(i,j) = dx_m
        G%IdxT(i,j) = 1.0_dp / dx_m
        G%IdyT(i,j) = 1.0_dp / dx_m
        G%dxCu(i,j) = dx_m
        G%dyCu(i,j) = dx_m
        G%IdxCu(i,j) = 1.0_dp / dx_m
        G%IdyCu(i,j) = 1.0_dp / dx_m
        G%dxCv(i,j) = dx_m
        G%dyCv(i,j) = dx_m
        G%IdxCv(i,j) = 1.0_dp / dx_m
        G%IdyCv(i,j) = 1.0_dp / dx_m
        G%areaBu(i,j) = dx_m * dx_m
        G%IareaBu(i,j) = 1.0_dp / (dx_m * dx_m)
        G%IareaCu(i,j) = 1.0_dp / (dx_m * dx_m)
        G%IareaCv(i,j) = 1.0_dp / (dx_m * dx_m)
        G%dxBu(i,j) = dx_m
        G%dyBu(i,j) = dx_m
        G%IdxBu(i,j) = 1.0_dp / dx_m
        G%IdyBu(i,j) = 1.0_dp / dx_m
        G%mask2dT(i,j) = 1.0_dp
        G%mask2dBu(i,j) = 1.0_dp
        G%mask2dCu(i,j) = 1.0_dp
        G%mask2dCv(i,j) = 1.0_dp
        G%bathyT(i,j) = 4000.0_dp  ! 4000m depth

        ! Coriolis at q-points (offset by half grid from center)
        G%CoriolisBu(i,j) = f0 + beta * (real(j - 1 - nj/2, dp) - 0.5_dp) * dx_m
      end do
    end do

    ! Copy grid data from host to device
    !$omp target update to(G%IareaT, G%areaT, G%dxT, G%dyT, G%IdxT, G%IdyT)
    !$omp target update to(G%dxCu, G%dyCu, G%IdxCu, G%IdyCu)
    !$omp target update to(G%dxCv, G%dyCv, G%IdxCv, G%IdyCv)
    !$omp target update to(G%IareaBu, G%areaBu, G%CoriolisBu)
    !$omp target update to(G%IareaCu, G%IareaCv, G%dxBu, G%dyBu, G%IdxBu, G%IdyBu)
    !$omp target update to(G%bathyT, G%mask2dT, G%mask2dBu, G%mask2dCu, G%mask2dCv)

    G%first_direction = 0

  end subroutine init_ocean_grid

  !> Deallocate grid arrays
  subroutine end_ocean_grid(G)
    type(ocean_grid_type), intent(inout) :: G

    !$omp target exit data map(delete: G%IareaT, G%areaT, G%dxT, G%dyT, G%IdxT, G%IdyT)
    !$omp target exit data map(delete: G%dxCu, G%dyCu, G%IdxCu, G%IdyCu)
    !$omp target exit data map(delete: G%dxCv, G%dyCv, G%IdxCv, G%IdyCv)
    !$omp target exit data map(delete: G%IareaBu, G%areaBu, G%CoriolisBu)
    !$omp target exit data map(delete: G%IareaCu, G%IareaCv, G%dxBu, G%dyBu, G%IdxBu, G%IdyBu)
    !$omp target exit data map(delete: G%bathyT, G%mask2dT, G%mask2dBu, G%mask2dCu, G%mask2dCv)

    if (allocated(G%IareaT)) deallocate(G%IareaT)
    if (allocated(G%areaT)) deallocate(G%areaT)
    if (allocated(G%dxT)) deallocate(G%dxT)
    if (allocated(G%dyT)) deallocate(G%dyT)
    if (allocated(G%IdxT)) deallocate(G%IdxT)
    if (allocated(G%IdyT)) deallocate(G%IdyT)
    if (allocated(G%dxCu)) deallocate(G%dxCu)
    if (allocated(G%dyCu)) deallocate(G%dyCu)
    if (allocated(G%IdxCu)) deallocate(G%IdxCu)
    if (allocated(G%IdyCu)) deallocate(G%IdyCu)
    if (allocated(G%dxCv)) deallocate(G%dxCv)
    if (allocated(G%dyCv)) deallocate(G%dyCv)
    if (allocated(G%IdxCv)) deallocate(G%IdxCv)
    if (allocated(G%IdyCv)) deallocate(G%IdyCv)
    if (allocated(G%IareaBu)) deallocate(G%IareaBu)
    if (allocated(G%areaBu)) deallocate(G%areaBu)
    if (allocated(G%CoriolisBu)) deallocate(G%CoriolisBu)
    if (allocated(G%IareaCu)) deallocate(G%IareaCu)
    if (allocated(G%IareaCv)) deallocate(G%IareaCv)
    if (allocated(G%dxBu)) deallocate(G%dxBu)
    if (allocated(G%dyBu)) deallocate(G%dyBu)
    if (allocated(G%IdxBu)) deallocate(G%IdxBu)
    if (allocated(G%IdyBu)) deallocate(G%IdyBu)
    if (allocated(G%bathyT)) deallocate(G%bathyT)
    if (allocated(G%mask2dT)) deallocate(G%mask2dT)
    if (allocated(G%mask2dBu)) deallocate(G%mask2dBu)
    if (allocated(G%mask2dCu)) deallocate(G%mask2dCu)
    if (allocated(G%mask2dCv)) deallocate(G%mask2dCv)

  end subroutine end_ocean_grid

  !> Initialize vertical grid
  subroutine init_verticalGrid(GV, nk)
    type(verticalGrid_type), intent(inout) :: GV
    integer, intent(in) :: nk

    GV%ke = nk
    GV%Angstrom_H = 1.0e-10_dp

  end subroutine init_verticalGrid

end module mom6_types
