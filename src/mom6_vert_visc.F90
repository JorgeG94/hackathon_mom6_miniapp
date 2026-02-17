!> MOM6 Vertical Viscosity Module
!!
!! Applies vertical viscosity to momentum using a tridiagonal solver.
!! Realistic version ported from MOM6's MOM_vert_friction.F90.
!!
!! Key features from real MOM6:
!!   - Harmonic-mean thickness with velocity-dependent upwind switching
!!   - Surface wind stress boundary conditions
!!   - Rayleigh drag
!!   - Schopf & Loughe numerically stable tridiagonal formulation
!!   - Bottom stress output
!!   - Kv_extra_bbl path (scalar, botfn-based)
!!
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

        CS%initialized = .true.

    end subroutine vert_visc_init

    !> Finalize the vertical viscosity solver
    subroutine vert_visc_end(CS)
        type(vert_visc_CS), intent(inout) :: CS

        if (.not. CS%initialized) return

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

    !> Compute vertical viscosity coupling coefficients for all columns
    !!
    !! Uses harmonic-mean thickness with velocity-dependent upwind switching
    !! (following real MOM6 lines 1499-1605), and botfn-based Kv_extra_bbl.
    !!
    subroutine vert_visc_coef(u, v, h, CS, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
        type(vert_visc_CS), intent(inout) :: CS

        real(dp) :: h_neglect, I_Hbbl
        real(dp) :: h_harm, h_arith, h_delta, z2, botfn
        real(dp) :: z_top, Kv_tot, topfn, h_shear
        real(dp) :: hvel(GV%ke), z_i(GV%ke + 1)
        integer :: i, j, k, K2, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        h_neglect = GV%Angstrom_H
        I_Hbbl = 1.0_dp / (CS%Hbbl + h_neglect)

        ! --- U-points ---
        do j=js,je
        do i=is - 1,ie
            if (G%mask2dCu(i, j) > 0.0_dp) then

                ! Bottom-up: compute harmonic mean thickness with upwind switching
                z_i(nz + 1) = 0.0_dp

                do k = nz, 1, -1
                    h_harm = 2.0_dp * h(i, j, k) * h(i + 1, j, k) / &
                             (h(i, j, k) + h(i + 1, j, k) + h_neglect)
                    h_arith = 0.5_dp * (h(i + 1, j, k) + h(i, j, k))
                    h_delta = h(i + 1, j, k) - h(i, j, k)

                    ! Default: harmonic mean
                    hvel(k) = h_harm

                    ! Upwind switching: when velocity opposes thickness gradient, blend toward arithmetic
                    if (u(i, j, k) * h_delta < 0.0_dp) then
                        z2 = z_i(k + 1)
                        botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                        hvel(k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                    end if

                    z_i(k) = z_i(k + 1) + h_harm * I_Hbbl
                end do

                ! Coupling coefficients
                CS%a_u(i, j, 1) = 0.0_dp  ! Surface: no flux (stress BC handled in apply)

                z_top = 0.0_dp
                do K2 = 2, nz
                    z_top = z_top + hvel(K2 - 1)
                    Kv_tot = CS%Kv

                    ! ML enhancement (simple taper)
                    if (z_top < CS%Hmix) then
                        topfn = 1.0_dp - z_top / CS%Hmix
                        Kv_tot = Kv_tot + (CS%Kv_ml - CS%Kv) * topfn
                    end if

                    ! BBL enhancement via Kv_extra_bbl + botfn
                    z2 = z_i(K2)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    Kv_tot = Kv_tot + CS%Kv_extra_bbl * botfn

                    h_shear = 0.5_dp * (hvel(K2) + hvel(K2 - 1) + h_neglect)
                    CS%a_u(i, j, K2) = Kv_tot / h_shear
                end do

                ! Bottom interface (Kv_extra_bbl path)
                CS%a_u(i, j, nz + 1) = (CS%Kv + CS%Kv_extra_bbl) / &
                                        (0.5_dp * hvel(nz) + h_neglect)

                ! Store effective thickness
                do k = 1, nz
                    CS%h_u(i, j, k) = hvel(k) + h_neglect
                end do

            else
                ! Masked points
                do k = 1, nz
                    CS%h_u(i, j, k) = h_neglect
                end do
                do K2 = 1, nz + 1
                    CS%a_u(i, j, K2) = 0.0_dp
                end do
            end if
        end do
        end do

        ! --- V-points ---
        do j=js - 1,je
        do i=is,ie
            if (G%mask2dCv(i, j) > 0.0_dp) then

                z_i(nz + 1) = 0.0_dp

                do k = nz, 1, -1
                    h_harm = 2.0_dp * h(i, j, k) * h(i, j + 1, k) / &
                             (h(i, j, k) + h(i, j + 1, k) + h_neglect)
                    h_arith = 0.5_dp * (h(i, j + 1, k) + h(i, j, k))
                    h_delta = h(i, j + 1, k) - h(i, j, k)

                    hvel(k) = h_harm

                    if (v(i, j, k) * h_delta < 0.0_dp) then
                        z2 = z_i(k + 1)
                        botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                        hvel(k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                    end if

                    z_i(k) = z_i(k + 1) + h_harm * I_Hbbl
                end do

                CS%a_v(i, j, 1) = 0.0_dp

                z_top = 0.0_dp
                do K2 = 2, nz
                    z_top = z_top + hvel(K2 - 1)
                    Kv_tot = CS%Kv

                    if (z_top < CS%Hmix) then
                        topfn = 1.0_dp - z_top / CS%Hmix
                        Kv_tot = Kv_tot + (CS%Kv_ml - CS%Kv) * topfn
                    end if

                    z2 = z_i(K2)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    Kv_tot = Kv_tot + CS%Kv_extra_bbl * botfn

                    h_shear = 0.5_dp * (hvel(K2) + hvel(K2 - 1) + h_neglect)
                    CS%a_v(i, j, K2) = Kv_tot / h_shear
                end do

                CS%a_v(i, j, nz + 1) = (CS%Kv + CS%Kv_extra_bbl) / &
                                        (0.5_dp * hvel(nz) + h_neglect)

                do k = 1, nz
                    CS%h_v(i, j, k) = hvel(k) + h_neglect
                end do

            else
                do k = 1, nz
                    CS%h_v(i, j, k) = h_neglect
                end do
                do K2 = 1, nz + 1
                    CS%a_v(i, j, K2) = 0.0_dp
                end do
            end if
        end do
        end do

    end subroutine vert_visc_coef

    !> Compute the remnant velocity fraction after implicit viscosity
    !!
    !! Uses Schopf & Loughe tridiagonal formulation with optional Rayleigh drag.
    !! Calculates what fraction of a barotropic acceleration remains in each layer
    !! after implicit vertical viscosity is applied.
    !!
    subroutine vert_visc_remnant(dt, CS, G, GV, visc)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), intent(in) :: dt
        type(vert_visc_CS), intent(inout) :: CS
        type(vertvisc_type), intent(in), optional :: visc

        real(dp) :: b1, d1, b_denom_1, Ray
        real(dp) :: c1(GV%ke)
        logical :: have_rayleigh
        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

        have_rayleigh = .false.
        if (present(visc)) have_rayleigh = visc%has_Rayleigh

        ! --- U-points ---
        do j=js,je
        do i=is - 1,ie
            if (G%mask2dCu(i, j) > 0.0_dp) then

                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                ! Layer 1 — Schopf & Loughe
                b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                d1 = b_denom_1 * b1
                CS%visc_rem_u(i, j, 1) = b1 * CS%h_u(i, j, 1)

                ! Interior layers
                do k = 2, nz
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, k)
                    c1(k) = dt * CS%a_u(i, j, k) * b1
                    b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1)
                    b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                    d1 = b_denom_1 * b1
                    CS%visc_rem_u(i, j, k) = (CS%h_u(i, j, k) + &
                        dt * CS%a_u(i, j, k) * CS%visc_rem_u(i, j, k - 1)) * b1
                end do

                ! Back substitution
                do k = nz - 1, 1, -1
                    CS%visc_rem_u(i, j, k) = CS%visc_rem_u(i, j, k) + &
                        c1(k + 1) * CS%visc_rem_u(i, j, k + 1)
                end do

            else
                do k = 1, nz
                    CS%visc_rem_u(i, j, k) = 1.0_dp
                end do
            end if
        end do
        end do

        ! --- V-points ---
        do j=js - 1,je
        do i=is,ie
            if (G%mask2dCv(i, j) > 0.0_dp) then

                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_v(i, j, 1)

                b_denom_1 = CS%h_v(i, j, 1) + dt * (Ray + CS%a_v(i, j, 1))
                b1 = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, 2))
                d1 = b_denom_1 * b1
                CS%visc_rem_v(i, j, 1) = b1 * CS%h_v(i, j, 1)

                do k = 2, nz
                    if (have_rayleigh) Ray = visc%Ray_v(i, j, k)
                    c1(k) = dt * CS%a_v(i, j, k) * b1
                    b_denom_1 = CS%h_v(i, j, k) + dt * (Ray + CS%a_v(i, j, k) * d1)
                    b1 = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, k + 1))
                    d1 = b_denom_1 * b1
                    CS%visc_rem_v(i, j, k) = (CS%h_v(i, j, k) + &
                        dt * CS%a_v(i, j, k) * CS%visc_rem_v(i, j, k - 1)) * b1
                end do

                do k = nz - 1, 1, -1
                    CS%visc_rem_v(i, j, k) = CS%visc_rem_v(i, j, k) + &
                        c1(k + 1) * CS%visc_rem_v(i, j, k + 1)
                end do

            else
                do k = 1, nz
                    CS%visc_rem_v(i, j, k) = 1.0_dp
                end do
            end if
        end do
        end do

    end subroutine vert_visc_remnant

    !> Apply vertical viscosity using Schopf & Loughe tridiagonal solver
    !!
    !! Includes surface wind stress BC, Rayleigh drag, and bottom stress output.
    !! Following real MOM6 lines 744-787.
    !!
    subroutine vert_visc_apply(u, v, h, dt, CS, G, GV, forces, visc)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: u, v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h
        real(dp), intent(in) :: dt
        type(vert_visc_CS), intent(inout) :: CS
        type(mech_forcing_type), intent(in), optional :: forces
        type(vertvisc_type), intent(in), optional :: visc

        real(dp) :: dt_Rho0, sfc_stress, Ray, b1, d1, b_denom_1
        real(dp) :: c1(GV%ke)
        logical :: have_forces, have_rayleigh
        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        dt_Rho0 = dt / RHO_0

        have_forces = present(forces)
        have_rayleigh = .false.
        if (present(visc)) have_rayleigh = visc%has_Rayleigh

        ! --- Apply to u-velocity ---
        do j=js,je
        do i=is - 1,ie
            if (G%mask2dCu(i, j) > 0.0_dp) then

                ! Surface stress BC
                sfc_stress = 0.0_dp
                if (have_forces) sfc_stress = dt_Rho0 * forces%taux(i, j) * G%mask2dCu(i, j)

                ! Rayleigh drag
                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                ! Layer 1 — Schopf & Loughe
                b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                d1 = b_denom_1 * b1
                u(i, j, 1) = b1 * (CS%h_u(i, j, 1) * u(i, j, 1) + sfc_stress)

                ! Interior layers
                do k = 2, nz
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, k)
                    c1(k) = dt * CS%a_u(i, j, k) * b1
                    b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1)
                    b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                    d1 = b_denom_1 * b1
                    u(i, j, k) = (CS%h_u(i, j, k) * u(i, j, k) + &
                        dt * CS%a_u(i, j, k) * u(i, j, k - 1)) * b1
                end do

                ! Back substitution
                do k = nz - 1, 1, -1
                    u(i, j, k) = u(i, j, k) + c1(k + 1) * u(i, j, k + 1)
                end do

                ! Bottom stress output
                CS%taux_bot(i, j) = RHO_0 * u(i, j, nz) * CS%a_u(i, j, nz + 1)
                if (have_rayleigh) then
                    do k = 1, nz
                        CS%taux_bot(i, j) = CS%taux_bot(i, j) + &
                            RHO_0 * visc%Ray_u(i, j, k) * u(i, j, k)
                    end do
                end if

            end if
        end do
        end do

        ! --- Apply to v-velocity ---
        do j=js - 1,je
        do i=is,ie
            if (G%mask2dCv(i, j) > 0.0_dp) then

                sfc_stress = 0.0_dp
                if (have_forces) sfc_stress = dt_Rho0 * forces%tauy(i, j) * G%mask2dCv(i, j)

                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_v(i, j, 1)

                b_denom_1 = CS%h_v(i, j, 1) + dt * (Ray + CS%a_v(i, j, 1))
                b1 = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, 2))
                d1 = b_denom_1 * b1
                v(i, j, 1) = b1 * (CS%h_v(i, j, 1) * v(i, j, 1) + sfc_stress)

                do k = 2, nz
                    if (have_rayleigh) Ray = visc%Ray_v(i, j, k)
                    c1(k) = dt * CS%a_v(i, j, k) * b1
                    b_denom_1 = CS%h_v(i, j, k) + dt * (Ray + CS%a_v(i, j, k) * d1)
                    b1 = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, k + 1))
                    d1 = b_denom_1 * b1
                    v(i, j, k) = (CS%h_v(i, j, k) * v(i, j, k) + &
                        dt * CS%a_v(i, j, k) * v(i, j, k - 1)) * b1
                end do

                do k = nz - 1, 1, -1
                    v(i, j, k) = v(i, j, k) + c1(k + 1) * v(i, j, k + 1)
                end do

                CS%tauy_bot(i, j) = RHO_0 * v(i, j, nz) * CS%a_v(i, j, nz + 1)
                if (have_rayleigh) then
                    do k = 1, nz
                        CS%tauy_bot(i, j) = CS%tauy_bot(i, j) + &
                            RHO_0 * visc%Ray_v(i, j, k) * v(i, j, k)
                    end do
                end if

            end if
        end do
        end do

    end subroutine vert_visc_apply

end module mom6_vert_visc
