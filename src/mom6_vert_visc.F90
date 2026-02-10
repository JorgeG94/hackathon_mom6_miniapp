!> MOM6 Vertical Viscosity Module
!!
!! Applies vertical viscosity to momentum using a tridiagonal solver.
!! Realistic version ported from MOM6's MOM_vert_friction.F90 for GPU hackathon.
!!
!! Key features from real MOM6:
!!   - Harmonic-mean thickness with velocity-dependent upwind switching
!!   - Surface wind stress boundary conditions
!!   - Rayleigh drag
!!   - Schopf & Loughe numerically stable tridiagonal formulation
!!   - Bottom stress output
!!   - Kv_extra_bbl path (scalar, botfn-based)
!!
!! This is the parent module. Computational subroutines are implemented
!! in submodules with different loop orderings (jik, ijk, jki, ikj, kji, kij).
!!
module mom6_vert_visc
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, RHO_0, &
                          mech_forcing_type, vertvisc_type
    implicit none
    private

    public :: vert_visc_init, vert_visc_coef, vert_visc_apply, vert_visc_end
    public :: vert_visc_remnant
    public :: vert_visc_CS

    !> Control structure for vertical viscosity solver
    type :: vert_visc_CS
        logical :: initialized = .false.

        ! Viscosity parameters
        real(dp) :: Kv              ! Interior viscosity [m2 s-1]
        real(dp) :: Kv_extra_bbl    ! Extra BBL viscosity [m2 s-1]
        real(dp) :: Hmix            ! Mixed layer depth [m]
        real(dp) :: Hbbl            ! BBL thickness [m]
        real(dp) :: Kv_ml           ! Mixed layer viscosity [m2 s-1]

        ! Thickness computation mode
        logical  :: harmonic_visc = .true.  ! Use harmonic mean thickness

        ! Coupling coefficients at interfaces (nz+1 levels)
        real(dp), allocatable :: a_u(:, :, :)   ! At u-point interfaces [H T-1]
        real(dp), allocatable :: a_v(:, :, :)   ! At v-point interfaces [H T-1]

        ! Effective layer thicknesses at velocity points
        real(dp), allocatable :: h_u(:, :, :)   ! At u-points [H]
        real(dp), allocatable :: h_v(:, :, :)   ! At v-points [H]

        ! Remnant velocity fractions (for barotropic coupling)
        real(dp), allocatable :: visc_rem_u(:, :, :)  ! At u-points [nondim]
        real(dp), allocatable :: visc_rem_v(:, :, :)  ! At v-points [nondim]

        ! Bottom stress output
        real(dp), allocatable :: taux_bot(:, :)  ! Zonal bottom stress [Pa]
        real(dp), allocatable :: tauy_bot(:, :)  ! Meridional bottom stress [Pa]
    end type vert_visc_CS

    !> Interfaces for submodule-implemented computational subroutines
    interface
        !> Compute vertical viscosity coupling coefficients for all columns
        module subroutine vert_visc_coef(u, v, h, CS, G, GV)
            type(ocean_grid_type), intent(in) :: G
            type(verticalGrid_type), intent(in) :: GV
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
            type(vert_visc_CS), intent(inout) :: CS
        end subroutine

        !> Compute the remnant velocity fraction after implicit viscosity
        module subroutine vert_visc_remnant(dt, CS, G, GV, visc)
            type(ocean_grid_type), intent(in) :: G
            type(verticalGrid_type), intent(in) :: GV
            real(dp), intent(in) :: dt
            type(vert_visc_CS), intent(inout) :: CS
            type(vertvisc_type), intent(in), optional :: visc
        end subroutine

        !> Apply vertical viscosity using Schopf & Loughe tridiagonal solver
        module subroutine vert_visc_apply(u, v, h, dt, CS, G, GV, forces, visc)
            type(ocean_grid_type), intent(in) :: G
            type(verticalGrid_type), intent(in) :: GV
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: u, v
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h
            real(dp), intent(in) :: dt
            type(vert_visc_CS), intent(inout) :: CS
            type(mech_forcing_type), intent(in), optional :: forces
            type(vertvisc_type), intent(in), optional :: visc
        end subroutine
    end interface

contains

    !> Initialize the vertical viscosity solver
    subroutine vert_visc_init(CS, G, GV, Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl, harmonic_visc)
        type(vert_visc_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), intent(in), optional :: Kv          ! Interior viscosity [m2/s]
        real(dp), intent(in), optional :: Kv_ml       ! Mixed layer viscosity [m2/s]
        real(dp), intent(in), optional :: Kv_extra_bbl ! Extra BBL viscosity [m2/s]
        real(dp), intent(in), optional :: Hmix        ! Mixed layer depth [m]
        real(dp), intent(in), optional :: Hbbl        ! BBL thickness [m]
        logical, intent(in), optional :: harmonic_visc ! Use harmonic mean thickness

        integer :: nz

        nz = GV%ke

        ! Set viscosity parameters (defaults typical for ocean)
        CS%Kv = 1.0e-4_dp
        if (present(Kv)) CS%Kv = Kv

        CS%Kv_ml = 1.0e-2_dp
        if (present(Kv_ml)) CS%Kv_ml = Kv_ml

        CS%Kv_extra_bbl = 1.0e-2_dp
        if (present(Kv_extra_bbl)) CS%Kv_extra_bbl = Kv_extra_bbl

        CS%Hmix = 50.0_dp
        if (present(Hmix)) CS%Hmix = Hmix

        CS%Hbbl = 10.0_dp
        if (present(Hbbl)) CS%Hbbl = Hbbl

        CS%harmonic_visc = .true.
        if (present(harmonic_visc)) CS%harmonic_visc = harmonic_visc

        ! Allocate coupling coefficients (nz+1 interfaces)
        allocate (CS%a_u(G%isd:G%ied, G%jsd:G%jed, nz + 1))
        allocate (CS%a_v(G%isd:G%ied, G%jsd:G%jed, nz + 1))

        ! Allocate effective thicknesses
        allocate (CS%h_u(G%isd:G%ied, G%jsd:G%jed, nz))
        allocate (CS%h_v(G%isd:G%ied, G%jsd:G%jed, nz))

        ! Allocate remnant fractions
        allocate (CS%visc_rem_u(G%isd:G%ied, G%jsd:G%jed, nz))
        allocate (CS%visc_rem_v(G%isd:G%ied, G%jsd:G%jed, nz))

        ! Allocate bottom stress
        allocate (CS%taux_bot(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate (CS%tauy_bot(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)

#ifdef __NVCOMPILER_LLVM__
        !$omp target enter data map(alloc: CS%a_u, CS%a_v, CS%h_u, CS%h_v)
        !$omp target enter data map(alloc: CS%visc_rem_u, CS%visc_rem_v)
        !$omp target enter data map(alloc: CS%taux_bot, CS%tauy_bot)
#endif

        CS%initialized = .true.

    end subroutine vert_visc_init

    !> Finalize the vertical viscosity solver
    subroutine vert_visc_end(CS)
        type(vert_visc_CS), intent(inout) :: CS

        if (.not. CS%initialized) return

#ifdef __NVCOMPILER_LLVM__
        !$omp target exit data map(delete: CS%a_u, CS%a_v, CS%h_u, CS%h_v)
        !$omp target exit data map(delete: CS%visc_rem_u, CS%visc_rem_v)
        !$omp target exit data map(delete: CS%taux_bot, CS%tauy_bot)
#endif

        if (allocated(CS%a_u)) deallocate (CS%a_u)
        if (allocated(CS%a_v)) deallocate (CS%a_v)
        if (allocated(CS%h_u)) deallocate (CS%h_u)
        if (allocated(CS%h_v)) deallocate (CS%h_v)
        if (allocated(CS%visc_rem_u)) deallocate (CS%visc_rem_u)
        if (allocated(CS%visc_rem_v)) deallocate (CS%visc_rem_v)
        if (allocated(CS%taux_bot)) deallocate (CS%taux_bot)
        if (allocated(CS%tauy_bot)) deallocate (CS%tauy_bot)

        CS%initialized = .false.

    end subroutine vert_visc_end

end module mom6_vert_visc
