!> Vertical viscosity submodule — JKI loop ordering
!!
!! Loop nest: j outermost (parallel), k middle (sequential), i innermost.
!! Row-wise parallelism with contiguous inner sweep.
!!
!! Memory access pattern: a(i, j_fix, k_fix) — i varies contiguously.
!! Best CPU cache locality for Fortran column-major arrays.
!! Workspace: 1D arrays over i, private to each j-thread.
!!
submodule (mom6_vert_visc) mom6_vert_visc_jki
    implicit none

contains

    module subroutine vert_visc_coef(u, v, h, CS, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
        type(vert_visc_CS), intent(inout) :: CS

        real(dp) :: h_neglect, I_Hbbl
        real(dp) :: h_harm, h_arith, h_delta, z2, botfn
        real(dp) :: Kv_tot, topfn, h_shear
        ! 1D workspace arrays over i, private to each j-thread
        real(dp) :: hvel(G%isd:G%ied, GV%ke), z_i(G%isd:G%ied, GV%ke + 1)
        real(dp) :: z_top(G%isd:G%ied)
        integer :: i, j, k, K2, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        h_neglect = GV%Angstrom_H
        I_Hbbl = 1.0_dp / (CS%Hbbl + h_neglect)

        ! --- U-points ---
        !$omp parallel do private(hvel, z_i, z_top, h_harm, h_arith, h_delta, &
        !$omp&  z2, botfn, Kv_tot, topfn, h_shear, i, k, K2)
        do j = js, je
            ! Initialize bottom interface
            do i = is - 1, ie
                z_i(i, nz + 1) = 0.0_dp
            end do

            ! Bottom-up: harmonic mean thickness with upwind switching
            do k = nz, 1, -1
                do i = is - 1, ie
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        h_harm = 2.0_dp * h(i, j, k) * h(i + 1, j, k) / &
                                 (h(i, j, k) + h(i + 1, j, k) + h_neglect)
                        h_arith = 0.5_dp * (h(i + 1, j, k) + h(i, j, k))
                        h_delta = h(i + 1, j, k) - h(i, j, k)

                        hvel(i, k) = h_harm

                        if (u(i, j, k) * h_delta < 0.0_dp) then
                            z2 = z_i(i, k + 1)
                            botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                            hvel(i, k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                        end if

                        z_i(i, k) = z_i(i, k + 1) + h_harm * I_Hbbl
                    end if
                end do
            end do

            ! Coupling coefficients — top-down
            do i = is - 1, ie
                CS%a_u(i, j, 1) = 0.0_dp
                z_top(i) = 0.0_dp
            end do

            do K2 = 2, nz
                do i = is - 1, ie
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        z_top(i) = z_top(i) + hvel(i, K2 - 1)
                        Kv_tot = CS%Kv

                        if (z_top(i) < CS%Hmix) then
                            topfn = 1.0_dp - z_top(i) / CS%Hmix
                            Kv_tot = Kv_tot + (CS%Kv_ml - CS%Kv) * topfn
                        end if

                        z2 = z_i(i, K2)
                        botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                        Kv_tot = Kv_tot + CS%Kv_extra_bbl * botfn

                        h_shear = 0.5_dp * (hvel(i, K2) + hvel(i, K2 - 1) + h_neglect)
                        CS%a_u(i, j, K2) = Kv_tot / h_shear
                    else
                        CS%a_u(i, j, K2) = 0.0_dp
                    end if
                end do
            end do

            ! Bottom interface and effective thickness
            do i = is - 1, ie
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    CS%a_u(i, j, nz + 1) = (CS%Kv + CS%Kv_extra_bbl) / &
                                            (0.5_dp * hvel(i, nz) + h_neglect)
                else
                    CS%a_u(i, j, nz + 1) = 0.0_dp
                end if
            end do

            do k = 1, nz
                do i = is - 1, ie
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        CS%h_u(i, j, k) = hvel(i, k) + h_neglect
                    else
                        CS%h_u(i, j, k) = h_neglect
                    end if
                end do
            end do
        end do

        ! --- V-points ---
        !$omp parallel do private(hvel, z_i, z_top, h_harm, h_arith, h_delta, &
        !$omp&  z2, botfn, Kv_tot, topfn, h_shear, i, k, K2)
        do j = js - 1, je
            do i = is, ie
                z_i(i, nz + 1) = 0.0_dp
            end do

            do k = nz, 1, -1
                do i = is, ie
                    if (G%mask2dCv(i, j) > 0.0_dp) then
                        h_harm = 2.0_dp * h(i, j, k) * h(i, j + 1, k) / &
                                 (h(i, j, k) + h(i, j + 1, k) + h_neglect)
                        h_arith = 0.5_dp * (h(i, j + 1, k) + h(i, j, k))
                        h_delta = h(i, j + 1, k) - h(i, j, k)

                        hvel(i, k) = h_harm

                        if (v(i, j, k) * h_delta < 0.0_dp) then
                            z2 = z_i(i, k + 1)
                            botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                            hvel(i, k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                        end if

                        z_i(i, k) = z_i(i, k + 1) + h_harm * I_Hbbl
                    end if
                end do
            end do

            do i = is, ie
                CS%a_v(i, j, 1) = 0.0_dp
                z_top(i) = 0.0_dp
            end do

            do K2 = 2, nz
                do i = is, ie
                    if (G%mask2dCv(i, j) > 0.0_dp) then
                        z_top(i) = z_top(i) + hvel(i, K2 - 1)
                        Kv_tot = CS%Kv

                        if (z_top(i) < CS%Hmix) then
                            topfn = 1.0_dp - z_top(i) / CS%Hmix
                            Kv_tot = Kv_tot + (CS%Kv_ml - CS%Kv) * topfn
                        end if

                        z2 = z_i(i, K2)
                        botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                        Kv_tot = Kv_tot + CS%Kv_extra_bbl * botfn

                        h_shear = 0.5_dp * (hvel(i, K2) + hvel(i, K2 - 1) + h_neglect)
                        CS%a_v(i, j, K2) = Kv_tot / h_shear
                    else
                        CS%a_v(i, j, K2) = 0.0_dp
                    end if
                end do
            end do

            do i = is, ie
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    CS%a_v(i, j, nz + 1) = (CS%Kv + CS%Kv_extra_bbl) / &
                                            (0.5_dp * hvel(i, nz) + h_neglect)
                else
                    CS%a_v(i, j, nz + 1) = 0.0_dp
                end if
            end do

            do k = 1, nz
                do i = is, ie
                    if (G%mask2dCv(i, j) > 0.0_dp) then
                        CS%h_v(i, j, k) = hvel(i, k) + h_neglect
                    else
                        CS%h_v(i, j, k) = h_neglect
                    end if
                end do
            end do
        end do

    end subroutine vert_visc_coef

    module subroutine vert_visc_remnant(dt, CS, G, GV, visc)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), intent(in) :: dt
        type(vert_visc_CS), intent(inout) :: CS
        type(vertvisc_type), intent(in), optional :: visc

        real(dp) :: b_denom_1, Ray
        ! 1D workspace over i, private to each j-thread
        real(dp) :: b1(G%isd:G%ied), d1(G%isd:G%ied)
        real(dp) :: c1(G%isd:G%ied, GV%ke)
        logical :: have_rayleigh
        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

        have_rayleigh = .false.
        if (present(visc)) have_rayleigh = visc%has_Rayleigh

        ! --- U-points ---
        !$omp parallel do private(b1, d1, c1, b_denom_1, Ray, i, k)
        do j = js, je
            ! Layer 1
            do i = is - 1, ie
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                    b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                    b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                    d1(i) = b_denom_1 * b1(i)
                    CS%visc_rem_u(i, j, 1) = b1(i) * CS%h_u(i, j, 1)
                else
                    CS%visc_rem_u(i, j, 1) = 1.0_dp
                end if
            end do

            ! Forward sweep
            do k = 2, nz
                do i = is - 1, ie
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        Ray = 0.0_dp
                        if (have_rayleigh) Ray = visc%Ray_u(i, j, k)
                        c1(i, k) = dt * CS%a_u(i, j, k) * b1(i)
                        b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1(i))
                        b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                        d1(i) = b_denom_1 * b1(i)
                        CS%visc_rem_u(i, j, k) = (CS%h_u(i, j, k) + &
                            dt * CS%a_u(i, j, k) * CS%visc_rem_u(i, j, k - 1)) * b1(i)
                    else
                        CS%visc_rem_u(i, j, k) = 1.0_dp
                    end if
                end do
            end do

            ! Back substitution
            do k = nz - 1, 1, -1
                do i = is - 1, ie
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        CS%visc_rem_u(i, j, k) = CS%visc_rem_u(i, j, k) + &
                            c1(i, k + 1) * CS%visc_rem_u(i, j, k + 1)
                    end if
                end do
            end do
        end do

        ! --- V-points ---
        !$omp parallel do private(b1, d1, c1, b_denom_1, Ray, i, k)
        do j = js - 1, je
            do i = is, ie
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_v(i, j, 1)

                    b_denom_1 = CS%h_v(i, j, 1) + dt * (Ray + CS%a_v(i, j, 1))
                    b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, 2))
                    d1(i) = b_denom_1 * b1(i)
                    CS%visc_rem_v(i, j, 1) = b1(i) * CS%h_v(i, j, 1)
                else
                    CS%visc_rem_v(i, j, 1) = 1.0_dp
                end if
            end do

            do k = 2, nz
                do i = is, ie
                    if (G%mask2dCv(i, j) > 0.0_dp) then
                        Ray = 0.0_dp
                        if (have_rayleigh) Ray = visc%Ray_v(i, j, k)
                        c1(i, k) = dt * CS%a_v(i, j, k) * b1(i)
                        b_denom_1 = CS%h_v(i, j, k) + dt * (Ray + CS%a_v(i, j, k) * d1(i))
                        b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, k + 1))
                        d1(i) = b_denom_1 * b1(i)
                        CS%visc_rem_v(i, j, k) = (CS%h_v(i, j, k) + &
                            dt * CS%a_v(i, j, k) * CS%visc_rem_v(i, j, k - 1)) * b1(i)
                    else
                        CS%visc_rem_v(i, j, k) = 1.0_dp
                    end if
                end do
            end do

            do k = nz - 1, 1, -1
                do i = is, ie
                    if (G%mask2dCv(i, j) > 0.0_dp) then
                        CS%visc_rem_v(i, j, k) = CS%visc_rem_v(i, j, k) + &
                            c1(i, k + 1) * CS%visc_rem_v(i, j, k + 1)
                    end if
                end do
            end do
        end do

    end subroutine vert_visc_remnant

    module subroutine vert_visc_apply(u, v, h, dt, CS, G, GV, forces, visc)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: u, v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h
        real(dp), intent(in) :: dt
        type(vert_visc_CS), intent(inout) :: CS
        type(mech_forcing_type), intent(in), optional :: forces
        type(vertvisc_type), intent(in), optional :: visc

        real(dp) :: dt_Rho0, sfc_stress, Ray, b_denom_1
        ! 1D workspace over i, private to each j-thread
        real(dp) :: b1(G%isd:G%ied), d1(G%isd:G%ied)
        real(dp) :: c1(G%isd:G%ied, GV%ke)
        logical :: have_forces, have_rayleigh
        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        dt_Rho0 = dt / RHO_0

        have_forces = present(forces)
        have_rayleigh = .false.
        if (present(visc)) have_rayleigh = visc%has_Rayleigh

        ! --- Apply to u-velocity ---
        !$omp parallel do private(b1, d1, c1, sfc_stress, Ray, b_denom_1, i, k)
        do j = js, je
            ! Layer 1
            do i = is - 1, ie
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    sfc_stress = 0.0_dp
                    if (have_forces) sfc_stress = dt_Rho0 * forces%taux(i, j) * G%mask2dCu(i, j)

                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                    b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                    b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                    d1(i) = b_denom_1 * b1(i)
                    u(i, j, 1) = b1(i) * (CS%h_u(i, j, 1) * u(i, j, 1) + sfc_stress)
                end if
            end do

            ! Forward sweep
            do k = 2, nz
                do i = is - 1, ie
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        Ray = 0.0_dp
                        if (have_rayleigh) Ray = visc%Ray_u(i, j, k)
                        c1(i, k) = dt * CS%a_u(i, j, k) * b1(i)
                        b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1(i))
                        b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                        d1(i) = b_denom_1 * b1(i)
                        u(i, j, k) = (CS%h_u(i, j, k) * u(i, j, k) + &
                            dt * CS%a_u(i, j, k) * u(i, j, k - 1)) * b1(i)
                    end if
                end do
            end do

            ! Back substitution
            do k = nz - 1, 1, -1
                do i = is - 1, ie
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        u(i, j, k) = u(i, j, k) + c1(i, k + 1) * u(i, j, k + 1)
                    end if
                end do
            end do

            ! Bottom stress
            do i = is - 1, ie
                if (G%mask2dCu(i, j) > 0.0_dp) then
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
        !$omp parallel do private(b1, d1, c1, sfc_stress, Ray, b_denom_1, i, k)
        do j = js - 1, je
            do i = is, ie
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    sfc_stress = 0.0_dp
                    if (have_forces) sfc_stress = dt_Rho0 * forces%tauy(i, j) * G%mask2dCv(i, j)

                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_v(i, j, 1)

                    b_denom_1 = CS%h_v(i, j, 1) + dt * (Ray + CS%a_v(i, j, 1))
                    b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, 2))
                    d1(i) = b_denom_1 * b1(i)
                    v(i, j, 1) = b1(i) * (CS%h_v(i, j, 1) * v(i, j, 1) + sfc_stress)
                end if
            end do

            do k = 2, nz
                do i = is, ie
                    if (G%mask2dCv(i, j) > 0.0_dp) then
                        Ray = 0.0_dp
                        if (have_rayleigh) Ray = visc%Ray_v(i, j, k)
                        c1(i, k) = dt * CS%a_v(i, j, k) * b1(i)
                        b_denom_1 = CS%h_v(i, j, k) + dt * (Ray + CS%a_v(i, j, k) * d1(i))
                        b1(i) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, k + 1))
                        d1(i) = b_denom_1 * b1(i)
                        v(i, j, k) = (CS%h_v(i, j, k) * v(i, j, k) + &
                            dt * CS%a_v(i, j, k) * v(i, j, k - 1)) * b1(i)
                    end if
                end do
            end do

            do k = nz - 1, 1, -1
                do i = is, ie
                    if (G%mask2dCv(i, j) > 0.0_dp) then
                        v(i, j, k) = v(i, j, k) + c1(i, k + 1) * v(i, j, k + 1)
                    end if
                end do
            end do

            do i = is, ie
                if (G%mask2dCv(i, j) > 0.0_dp) then
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

end submodule mom6_vert_visc_jki
