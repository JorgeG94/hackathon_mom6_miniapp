!> Vertical viscosity submodule — JIK loop ordering
!!
!! Loop nest: do concurrent(j, i) with k innermost (sequential).
!! This is the original/baseline ordering — column-wise parallelism.
!! Each parallel unit handles a full vertical column.
!!
!! Memory access pattern: a(i_fix, j_fix, k) — stride ni*nj per k step.
!! Good for GPU (one thread per column), poor CPU cache locality.
!!
submodule (mom6_vert_visc) mom6_vert_visc_jik
    implicit none

contains

    module subroutine vert_visc_coef(u, v, h, CS, G, GV)
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
        do concurrent(j=js:je, i=is - 1:ie)
            if (G%mask2dCu(i, j) > 0.0_dp) then

                z_i(nz + 1) = 0.0_dp

                do k = nz, 1, -1
                    h_harm = 2.0_dp * h(i, j, k) * h(i + 1, j, k) / &
                             (h(i, j, k) + h(i + 1, j, k) + h_neglect)
                    h_arith = 0.5_dp * (h(i + 1, j, k) + h(i, j, k))
                    h_delta = h(i + 1, j, k) - h(i, j, k)

                    hvel(k) = h_harm

                    if (u(i, j, k) * h_delta < 0.0_dp) then
                        z2 = z_i(k + 1)
                        botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                        hvel(k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                    end if

                    z_i(k) = z_i(k + 1) + h_harm * I_Hbbl
                end do

                CS%a_u(i, j, 1) = 0.0_dp

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
                    CS%a_u(i, j, K2) = Kv_tot / h_shear
                end do

                CS%a_u(i, j, nz + 1) = (CS%Kv + CS%Kv_extra_bbl) / &
                                        (0.5_dp * hvel(nz) + h_neglect)

                do k = 1, nz
                    CS%h_u(i, j, k) = hvel(k) + h_neglect
                end do

            else
                do k = 1, nz
                    CS%h_u(i, j, k) = h_neglect
                end do
                do K2 = 1, nz + 1
                    CS%a_u(i, j, K2) = 0.0_dp
                end do
            end if
        end do

        ! --- V-points ---
        do concurrent(j=js - 1:je, i=is:ie)
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

    end subroutine vert_visc_coef

    module subroutine vert_visc_remnant(dt, CS, G, GV, visc)
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
        do concurrent(j=js:je, i=is - 1:ie)
            if (G%mask2dCu(i, j) > 0.0_dp) then

                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                d1 = b_denom_1 * b1
                CS%visc_rem_u(i, j, 1) = b1 * CS%h_u(i, j, 1)

                do k = 2, nz
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, k)
                    c1(k) = dt * CS%a_u(i, j, k) * b1
                    b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1)
                    b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                    d1 = b_denom_1 * b1
                    CS%visc_rem_u(i, j, k) = (CS%h_u(i, j, k) + &
                        dt * CS%a_u(i, j, k) * CS%visc_rem_u(i, j, k - 1)) * b1
                end do

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

        ! --- V-points ---
        do concurrent(j=js - 1:je, i=is:ie)
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
        do concurrent(j=js:je, i=is - 1:ie)
            if (G%mask2dCu(i, j) > 0.0_dp) then

                sfc_stress = 0.0_dp
                if (have_forces) sfc_stress = dt_Rho0 * forces%taux(i, j) * G%mask2dCu(i, j)

                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                d1 = b_denom_1 * b1
                u(i, j, 1) = b1 * (CS%h_u(i, j, 1) * u(i, j, 1) + sfc_stress)

                do k = 2, nz
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, k)
                    c1(k) = dt * CS%a_u(i, j, k) * b1
                    b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1)
                    b1 = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                    d1 = b_denom_1 * b1
                    u(i, j, k) = (CS%h_u(i, j, k) * u(i, j, k) + &
                        dt * CS%a_u(i, j, k) * u(i, j, k - 1)) * b1
                end do

                do k = nz - 1, 1, -1
                    u(i, j, k) = u(i, j, k) + c1(k + 1) * u(i, j, k + 1)
                end do

                CS%taux_bot(i, j) = RHO_0 * u(i, j, nz) * CS%a_u(i, j, nz + 1)
                if (have_rayleigh) then
                    do k = 1, nz
                        CS%taux_bot(i, j) = CS%taux_bot(i, j) + &
                            RHO_0 * visc%Ray_u(i, j, k) * u(i, j, k)
                    end do
                end if

            end if
        end do

        ! --- Apply to v-velocity ---
        do concurrent(j=js - 1:je, i=is:ie)
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

    end subroutine vert_visc_apply

end submodule mom6_vert_visc_jik
