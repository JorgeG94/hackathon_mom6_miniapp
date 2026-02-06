!> MOM6 Vertical Viscosity Module
!!
!! Applies vertical viscosity to momentum using a tridiagonal solver.
!! Simplified version of MOM6's MOM_vert_friction.F90 for GPU hackathon.
!!
!! Key functions from MOM6:
!!   - find_coupling_coef: Computes viscous coupling coefficients at interfaces
!!   - vertvisc_remnant: Calculates momentum remnant fraction after implicit viscosity
!!   - vertvisc (here: vert_visc_apply): Applies viscosity via tridiagonal solve
!!
!! GPU Pattern: Parallelizes over horizontal columns (i,j) with collapse(2),
!! sequential tridiagonal solve in k (each column independent).
!!
module mom6_vert_visc
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type
    implicit none
    private

    public :: vert_visc_init, vert_visc_coef, vert_visc_apply, vert_visc_end
    public :: vert_visc_remnant, find_coupling_coef
    public :: vert_visc_CS

    !> Control structure for vertical viscosity solver
    type :: vert_visc_CS
        logical :: initialized = .false.

        ! Viscosity parameters
        real(dp) :: Kv              ! Interior viscosity [m2 s-1]
        real(dp) :: Kv_ml           ! Mixed layer viscosity [m2 s-1]
        real(dp) :: Kv_bbl          ! Bottom boundary layer viscosity [m2 s-1]
        real(dp) :: Hmix            ! Mixed layer depth [m]
        real(dp) :: Hbbl            ! Bottom boundary layer thickness [m]

        ! Coupling coefficients at interfaces (nz+1 levels)
        real(dp), allocatable :: a_u(:, :, :)   ! At u-point interfaces [H T-1]
        real(dp), allocatable :: a_v(:, :, :)   ! At v-point interfaces [H T-1]

        ! Effective layer thicknesses at velocity points
        real(dp), allocatable :: h_u(:, :, :)   ! At u-points [H]
        real(dp), allocatable :: h_v(:, :, :)   ! At v-points [H]

        ! Remnant velocity fractions (for barotropic coupling)
        real(dp), allocatable :: visc_rem_u(:, :, :)  ! At u-points [nondim]
        real(dp), allocatable :: visc_rem_v(:, :, :)  ! At v-points [nondim]
    end type vert_visc_CS

contains

    !> Initialize the vertical viscosity solver
    subroutine vert_visc_init(CS, G, GV, Kv, Kv_ml, Kv_bbl, Hmix, Hbbl)
        type(vert_visc_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), intent(in), optional :: Kv      ! Interior viscosity [m2/s]
        real(dp), intent(in), optional :: Kv_ml   ! Mixed layer viscosity [m2/s]
        real(dp), intent(in), optional :: Kv_bbl  ! BBL viscosity [m2/s]
        real(dp), intent(in), optional :: Hmix    ! Mixed layer depth [m]
        real(dp), intent(in), optional :: Hbbl    ! BBL thickness [m]

        integer :: nz

        nz = GV%ke

        ! Set viscosity parameters (defaults typical for ocean)
        ! These numbers need looking at!!
        CS%Kv = 1.0e-4_dp
        if (present(Kv)) CS%Kv = Kv

        CS%Kv_ml = 1.0e-2_dp
        if (present(Kv_ml)) CS%Kv_ml = Kv_ml

        CS%Kv_bbl = 1.0e-2_dp
        if (present(Kv_bbl)) CS%Kv_bbl = Kv_bbl

        CS%Hmix = 50.0_dp
        if (present(Hmix)) CS%Hmix = Hmix

        CS%Hbbl = 10.0_dp
        if (present(Hbbl)) CS%Hbbl = Hbbl

        ! Allocate coupling coefficients (nz+1 interfaces)
        allocate (CS%a_u(G%isd:G%ied, G%jsd:G%jed, nz + 1))
        allocate (CS%a_v(G%isd:G%ied, G%jsd:G%jed, nz + 1))

        ! Allocate effective thicknesses
        allocate (CS%h_u(G%isd:G%ied, G%jsd:G%jed, nz))
        allocate (CS%h_v(G%isd:G%ied, G%jsd:G%jed, nz))

        ! Allocate remnant fractions
        allocate (CS%visc_rem_u(G%isd:G%ied, G%jsd:G%jed, nz))
        allocate (CS%visc_rem_v(G%isd:G%ied, G%jsd:G%jed, nz))

        !$omp target enter data map(alloc: CS%a_u, CS%a_v, CS%h_u, CS%h_v)
        !$omp target enter data map(alloc: CS%visc_rem_u, CS%visc_rem_v)

        CS%initialized = .true.

    end subroutine vert_visc_init

    !> Finalize the vertical viscosity solver
    subroutine vert_visc_end(CS)
        type(vert_visc_CS), intent(inout) :: CS

        if (.not. CS%initialized) return

        !$omp target exit data map(delete: CS%a_u, CS%a_v, CS%h_u, CS%h_v)
        !$omp target exit data map(delete: CS%visc_rem_u, CS%visc_rem_v)

        if (allocated(CS%a_u)) deallocate (CS%a_u)
        if (allocated(CS%a_v)) deallocate (CS%a_v)
        if (allocated(CS%h_u)) deallocate (CS%h_u)
        if (allocated(CS%h_v)) deallocate (CS%h_v)
        if (allocated(CS%visc_rem_u)) deallocate (CS%visc_rem_u)
        if (allocated(CS%visc_rem_v)) deallocate (CS%visc_rem_v)

        CS%initialized = .false.

    end subroutine vert_visc_end

    !> Compute coupling coefficient for a single interface
  !!
  !! This is a simplified version of MOM6's find_coupling_coef.
  !! Computes the viscous coupling coefficient at interface K based on:
  !!   - Interior viscosity (Kv)
  !!   - Mixed layer enhanced viscosity (Kv_ml) near surface
  !!   - Bottom boundary layer viscosity (Kv_bbl) near bottom
  !!
  !! The coupling coefficient formula is: a = Kv_tot / h_shear
  !! where h_shear is the harmonic mean of adjacent layer thicknesses.
  !!
  !! GPU: This is a pure function suitable for device execution.
  !!
    pure function find_coupling_coef(hvel_k, hvel_km1, z_from_top, z_from_bot, &
                                     Kv, Kv_ml, Kv_bbl, Hmix, Hbbl, h_neglect) result(a_cpl)
        real(dp), intent(in) :: hvel_k      ! Thickness of layer k [m]
        real(dp), intent(in) :: hvel_km1    ! Thickness of layer k-1 [m]
        real(dp), intent(in) :: z_from_top  ! Depth from surface to interface [m]
        real(dp), intent(in) :: z_from_bot  ! Height from bottom to interface [m]
        real(dp), intent(in) :: Kv          ! Interior viscosity [m2/s]
        real(dp), intent(in) :: Kv_ml       ! Mixed layer viscosity [m2/s]
        real(dp), intent(in) :: Kv_bbl      ! BBL viscosity [m2/s]
        real(dp), intent(in) :: Hmix        ! Mixed layer depth [m]
        real(dp), intent(in) :: Hbbl        ! BBL thickness [m]
        real(dp), intent(in) :: h_neglect   ! Small thickness to avoid division by zero [m]
        real(dp) :: a_cpl                   ! Coupling coefficient [m/s]

        real(dp) :: Kv_tot, h_shear, botfn, topfn, z_norm

        ! Start with interior viscosity
        Kv_tot = Kv

        ! Mixed layer enhancement (smooth transition near surface)
        if (z_from_top < Hmix) then
            ! Linear transition in mixed layer
            topfn = 1.0_dp - z_from_top/Hmix
            Kv_tot = Kv_tot + (Kv_ml - Kv)*topfn
        end if

        ! Bottom boundary layer enhancement
        ! Use MOM6's botfn: botfn = 1 / (1 + 0.09 * z_norm^6)
        ! where z_norm = z_from_bot / Hbbl
        if (Hbbl > 0.0_dp .and. z_from_bot < 2.0_dp*Hbbl) then
            z_norm = z_from_bot/Hbbl
            botfn = 1.0_dp/(1.0_dp + 0.09_dp*z_norm**6)
            Kv_tot = Kv_tot + (Kv_bbl - Kv)*botfn
        end if

        ! Harmonic mean of adjacent layer thicknesses (shear length scale)
        h_shear = 2.0_dp*hvel_k*hvel_km1/(hvel_k + hvel_km1 + h_neglect)

        ! Coupling coefficient: Kv / h_shear
        a_cpl = Kv_tot/(h_shear + h_neglect)

    end function find_coupling_coef

    !> Compute vertical viscosity coupling coefficients for all columns
  !!
  !! Computes a_u, a_v at interfaces based on layer thicknesses and viscosity.
  !! Uses find_coupling_coef for each interface.
  !!
    subroutine vert_visc_coef(u, v, h, CS, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
        type(vert_visc_CS), intent(inout) :: CS

        real(dp) :: z_top, z_bot, total_depth, h_neglect
        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        h_neglect = GV%Angstrom_H
        total_depth = G%bathyT(G%isc, G%jsc)  ! Assume uniform depth for simplicity

        ! Compute effective thickness at u-points
        do concurrent(k=1:nz, j=js:je, i=is - 1:ie)
            CS%h_u(i, j, k) = 0.5_dp*(h(i, j, k) + h(i + 1, j, k))
        end do

        ! Compute effective thickness at v-points
        do concurrent(k=1:nz, j=js - 1:je, i=is:ie)
            CS%h_v(i, j, k) = 0.5_dp*(h(i, j, k) + h(i, j + 1, k))
        end do

        ! Compute coupling coefficients at u-point interfaces
        do concurrent(j=js:je, i=is - 1:ie)
            ! Surface (k=1): no flux through top
            CS%a_u(i, j, 1) = 0.0_dp

            ! Interior interfaces (k=2 to nz)
            z_top = 0.0_dp
            do k = 2, nz
                z_top = z_top + CS%h_u(i, j, k - 1)
                z_bot = total_depth - z_top

                CS%a_u(i, j, k) = find_coupling_coef(CS%h_u(i, j, k), CS%h_u(i, j, k - 1), &
                                                     z_top, z_bot, &
                                                     CS%Kv, CS%Kv_ml, CS%Kv_bbl, &
                                                     CS%Hmix, CS%Hbbl, h_neglect)
            end do

            ! Bottom (k=nz+1): no flux through bottom
            CS%a_u(i, j, nz + 1) = 0.0_dp
        end do

        ! Compute coupling coefficients at v-point interfaces
        do concurrent(j=js - 1:je, i=is:ie)
            CS%a_v(i, j, 1) = 0.0_dp

            z_top = 0.0_dp
            do k = 2, nz
                z_top = z_top + CS%h_v(i, j, k - 1)
                z_bot = total_depth - z_top

                CS%a_v(i, j, k) = find_coupling_coef(CS%h_v(i, j, k), CS%h_v(i, j, k - 1), &
                                                     z_top, z_bot, &
                                                     CS%Kv, CS%Kv_ml, CS%Kv_bbl, &
                                                     CS%Hmix, CS%Hbbl, h_neglect)
            end do

            CS%a_v(i, j, nz + 1) = 0.0_dp
        end do

    end subroutine vert_visc_coef

    !> Compute the remnant velocity fraction after implicit viscosity
  !!
  !! This implements MOM6's vertvisc_remnant function.
  !! Calculates what fraction of a barotropic acceleration remains in each layer
  !! after implicit vertical viscosity is applied. This is used for barotropic-
  !! baroclinic coupling.
  !!
  !! The remnant is computed by inverting the tridiagonal viscosity matrix
  !! with a unit forcing (1.0) applied uniformly to all layers.
  !!
  !! Output: visc_rem(k) is the fraction of barotropic acceleration that
  !!         layer k retains after viscosity [nondim, 0 to 1]
  !!
    subroutine vert_visc_remnant(dt, CS, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), intent(in) :: dt
        type(vert_visc_CS), intent(inout) :: CS

        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

        ! Compute remnant for u-points
        call compute_remnant_column(CS%visc_rem_u, CS%h_u, CS%a_u, dt, G, GV, &
                                    is - 1, ie, js, je, nz, G%mask2dCu)

        ! Compute remnant for v-points
        call compute_remnant_column(CS%visc_rem_v, CS%h_v, CS%a_v, dt, G, GV, &
                                    is, ie, js - 1, je, nz, G%mask2dCv)

    end subroutine vert_visc_remnant

    !> Compute remnant fraction for a set of columns
    subroutine compute_remnant_column(visc_rem, hvel, a_cpl, dt, G, GV, &
                                      is, ie, js, je, nz, mask)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: visc_rem
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: hvel
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke + 1), intent(in) :: a_cpl
        real(dp), intent(in) :: dt
        integer, intent(in) :: is, ie, js, je, nz
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: mask

        real(dp) :: b1, d1
        real(dp) :: c1(GV%ke)
        integer :: i, j, k

        do concurrent(j=js:je, i=is:ie)
            if (mask(i, j) > 0.0_dp) then

                ! Forward elimination with unit forcing (d=1 for all k)
                ! First layer (k=1)
                b1 = 1.0_dp/(hvel(i, j, 1) + dt*a_cpl(i, j, 2))
                d1 = (hvel(i, j, 1) + dt*a_cpl(i, j, 2)*1.0_dp)*b1  ! Unit forcing
                c1(1) = dt*a_cpl(i, j, 2)*b1
                visc_rem(i, j, 1) = d1

                ! Interior layers (k=2 to nz)
                do k = 2, nz
                    b1 = 1.0_dp/(hvel(i, j, k) + dt*(a_cpl(i, j, k + 1) + a_cpl(i, j, k)*(1.0_dp - c1(k - 1))))
                    ! d1 includes: layer mass, coupling from below, and unit forcing
                    d1 = (hvel(i, j, k) + dt*a_cpl(i, j, k)*visc_rem(i, j, k - 1))*b1
                    c1(k) = dt*a_cpl(i, j, k + 1)*b1
                    visc_rem(i, j, k) = d1
                end do

                ! Back substitution
                do k = nz - 1, 1, -1
                    visc_rem(i, j, k) = visc_rem(i, j, k) + c1(k)*visc_rem(i, j, k + 1)
                end do

            else
                ! Masked points: no viscosity effect
                do k = 1, nz
                    visc_rem(i, j, k) = 1.0_dp
                end do
            end if
        end do

    end subroutine compute_remnant_column

    !> Apply vertical viscosity using tridiagonal solver
  !!
  !! Solves the implicit vertical diffusion equation:
  !!   h(k) * u_new(k) = h(k) * u_old(k) + dt * a(k+1) * (u_new(k+1) - u_new(k))
  !!                                      - dt * a(k) * (u_new(k) - u_new(k-1))
  !!
  !! This is a tridiagonal system solved by forward elimination and back substitution.
  !! GPU pattern: Each horizontal column (i,j) is independent, parallelize over columns.
  !!
    subroutine vert_visc_apply(u, v, h, dt, CS, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: u, v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h
        real(dp), intent(in) :: dt
        type(vert_visc_CS), intent(in) :: CS

        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

        ! Apply to u-velocity
        call apply_tridiag_u(u, CS%h_u, CS%a_u, dt, G, GV, is - 1, ie, js, je, nz)

        ! Apply to v-velocity
        call apply_tridiag_v(v, CS%h_v, CS%a_v, dt, G, GV, is, ie, js - 1, je, nz)

    end subroutine vert_visc_apply

    !> Tridiagonal solver for u-velocity columns
    subroutine apply_tridiag_u(u, h_u, a_u, dt, G, GV, is, ie, js, je, nz)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: u
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_u
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke + 1), intent(in) :: a_u
        real(dp), intent(in) :: dt
        integer, intent(in) :: is, ie, js, je, nz

        ! Local arrays for tridiagonal solve (per column)
        real(dp) :: b1, d1
        real(dp) :: c1(GV%ke)   ! Super-diagonal coefficients
        integer :: i, j, k

        ! GPU: parallelize over columns (i,j), sequential in k
        do concurrent(j=js:je, i=is:ie)
            if (G%mask2dCu(i, j) > 0.0_dp) then

                ! Forward elimination
                ! First layer (k=1)
                b1 = 1.0_dp/(h_u(i, j, 1) + dt*a_u(i, j, 2))
                d1 = h_u(i, j, 1)*u(i, j, 1)*b1
                c1(1) = dt*a_u(i, j, 2)*b1
                u(i, j, 1) = d1

                ! Interior layers (k=2 to nz)
                do k = 2, nz
                    b1 = 1.0_dp/(h_u(i, j, k) + dt*(a_u(i, j, k + 1) + a_u(i, j, k)*(1.0_dp - c1(k - 1))))
                    d1 = (h_u(i, j, k)*u(i, j, k) + dt*a_u(i, j, k)*u(i, j, k - 1))*b1
                    c1(k) = dt*a_u(i, j, k + 1)*b1
                    u(i, j, k) = d1
                end do

                ! Back substitution (k=nz-1 to 1)
                do k = nz - 1, 1, -1
                    u(i, j, k) = u(i, j, k) + c1(k)*u(i, j, k + 1)
                end do

            end if
        end do

    end subroutine apply_tridiag_u

    !> Tridiagonal solver for v-velocity columns
    subroutine apply_tridiag_v(v, h_v, a_v, dt, G, GV, is, ie, js, je, nz)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke + 1), intent(in) :: a_v
        real(dp), intent(in) :: dt
        integer, intent(in) :: is, ie, js, je, nz

        real(dp) :: b1, d1
        real(dp) :: c1(GV%ke)
        integer :: i, j, k

        do concurrent(j=js:je, i=is:ie)
            if (G%mask2dCv(i, j) > 0.0_dp) then

                ! Forward elimination
                b1 = 1.0_dp/(h_v(i, j, 1) + dt*a_v(i, j, 2))
                d1 = h_v(i, j, 1)*v(i, j, 1)*b1
                c1(1) = dt*a_v(i, j, 2)*b1
                v(i, j, 1) = d1

                do k = 2, nz
                    b1 = 1.0_dp/(h_v(i, j, k) + dt*(a_v(i, j, k + 1) + a_v(i, j, k)*(1.0_dp - c1(k - 1))))
                    d1 = (h_v(i, j, k)*v(i, j, k) + dt*a_v(i, j, k)*v(i, j, k - 1))*b1
                    c1(k) = dt*a_v(i, j, k + 1)*b1
                    v(i, j, k) = d1
                end do

                ! Back substitution
                do k = nz - 1, 1, -1
                    v(i, j, k) = v(i, j, k) + c1(k)*v(i, j, k + 1)
                end do

            end if
        end do

    end subroutine apply_tridiag_v

end module mom6_vert_visc
