!> Vertical viscosity submodule — KIJ loop ordering
!!
!! Loop nest: k outermost (sequential), do concurrent(i, j) at each k-level.
!! This is a level-wise parallelism ordering — each k-level is a separate
!! 2D parallel sweep over all (i, j) points.
!!
!! Requires 2D/3D workspace arrays since intermediate values (b1, d1, hvel,
!! z_i) must persist across k-levels for each (i, j) point.
!!
!! Memory access pattern: a(i, j, k_fix) — stride 1 per i step, stride ni per
!! j step. Good cache locality within each 2D slab.
!!
submodule (mom6_vert_visc) mom6_vert_visc_kij
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
        integer :: i, j, k, K2, is, ie, js, je, nz

        ! Workspace arrays
        real(dp), allocatable :: hvel_3d(:, :, :)
        real(dp), allocatable :: z_i_3d(:, :, :)
        real(dp), allocatable :: z_top_2d(:, :)

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        h_neglect = GV%Angstrom_H
        I_Hbbl = 1.0_dp / (CS%Hbbl + h_neglect)

        allocate(hvel_3d(G%isd:G%ied, G%jsd:G%jed, GV%ke))
        allocate(z_i_3d(G%isd:G%ied, G%jsd:G%jed, GV%ke + 1))
        allocate(z_top_2d(G%isd:G%ied, G%jsd:G%jed))

        ! ===================================================================
        ! --- U-points ---
        ! ===================================================================

        ! Initialize bottom interface
        do concurrent(i=is-1:ie, j=js:je)
            z_i_3d(i, j, nz + 1) = 0.0_dp
        end do

        ! Bottom-up: compute hvel and z_i level by level
        do k = nz, 1, -1
            do concurrent(i=is-1:ie, j=js:je)
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    h_harm = 2.0_dp * h(i, j, k) * h(i + 1, j, k) / &
                             (h(i, j, k) + h(i + 1, j, k) + h_neglect)
                    h_arith = 0.5_dp * (h(i + 1, j, k) + h(i, j, k))
                    h_delta = h(i + 1, j, k) - h(i, j, k)

                    hvel_3d(i, j, k) = h_harm

                    if (u(i, j, k) * h_delta < 0.0_dp) then
                        z2 = z_i_3d(i, j, k + 1)
                        botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                        hvel_3d(i, j, k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                    end if

                    z_i_3d(i, j, k) = z_i_3d(i, j, k + 1) + h_harm * I_Hbbl
                else
                    hvel_3d(i, j, k) = 0.0_dp
                end if
            end do
        end do

        ! Top-down: coupling coefficients
        do concurrent(i=is-1:ie, j=js:je)
            CS%a_u(i, j, 1) = 0.0_dp
            z_top_2d(i, j) = 0.0_dp
        end do

        do K2 = 2, nz
            do concurrent(i=is-1:ie, j=js:je)
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    z_top_2d(i, j) = z_top_2d(i, j) + hvel_3d(i, j, K2 - 1)
                    Kv_tot = CS%Kv

                    if (z_top_2d(i, j) < CS%Hmix) then
                        topfn = 1.0_dp - z_top_2d(i, j) / CS%Hmix
                        Kv_tot = Kv_tot + (CS%Kv_ml - CS%Kv) * topfn
                    end if

                    z2 = z_i_3d(i, j, K2)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    Kv_tot = Kv_tot + CS%Kv_extra_bbl * botfn

                    h_shear = 0.5_dp * (hvel_3d(i, j, K2) + hvel_3d(i, j, K2 - 1) + h_neglect)
                    CS%a_u(i, j, K2) = Kv_tot / h_shear
                else
                    CS%a_u(i, j, K2) = 0.0_dp
                end if
            end do
        end do

        ! Bottom interface
        do concurrent(i=is-1:ie, j=js:je)
            if (G%mask2dCu(i, j) > 0.0_dp) then
                CS%a_u(i, j, nz + 1) = (CS%Kv + CS%Kv_extra_bbl) / &
                                        (0.5_dp * hvel_3d(i, j, nz) + h_neglect)
            else
                CS%a_u(i, j, nz + 1) = 0.0_dp
            end if
        end do

        ! Store h_u
        do k = 1, nz
            do concurrent(i=is-1:ie, j=js:je)
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    CS%h_u(i, j, k) = hvel_3d(i, j, k) + h_neglect
                else
                    CS%h_u(i, j, k) = h_neglect
                end if
            end do
        end do

        ! ===================================================================
        ! --- V-points ---
        ! ===================================================================

        ! Initialize bottom interface
        do concurrent(i=is:ie, j=js-1:je)
            z_i_3d(i, j, nz + 1) = 0.0_dp
        end do

        ! Bottom-up: compute hvel and z_i level by level
        do k = nz, 1, -1
            do concurrent(i=is:ie, j=js-1:je)
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    h_harm = 2.0_dp * h(i, j, k) * h(i, j + 1, k) / &
                             (h(i, j, k) + h(i, j + 1, k) + h_neglect)
                    h_arith = 0.5_dp * (h(i, j + 1, k) + h(i, j, k))
                    h_delta = h(i, j + 1, k) - h(i, j, k)

                    hvel_3d(i, j, k) = h_harm

                    if (v(i, j, k) * h_delta < 0.0_dp) then
                        z2 = z_i_3d(i, j, k + 1)
                        botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                        hvel_3d(i, j, k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                    end if

                    z_i_3d(i, j, k) = z_i_3d(i, j, k + 1) + h_harm * I_Hbbl
                else
                    hvel_3d(i, j, k) = 0.0_dp
                end if
            end do
        end do

        ! Top-down: coupling coefficients
        do concurrent(i=is:ie, j=js-1:je)
            CS%a_v(i, j, 1) = 0.0_dp
            z_top_2d(i, j) = 0.0_dp
        end do

        do K2 = 2, nz
            do concurrent(i=is:ie, j=js-1:je)
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    z_top_2d(i, j) = z_top_2d(i, j) + hvel_3d(i, j, K2 - 1)
                    Kv_tot = CS%Kv

                    if (z_top_2d(i, j) < CS%Hmix) then
                        topfn = 1.0_dp - z_top_2d(i, j) / CS%Hmix
                        Kv_tot = Kv_tot + (CS%Kv_ml - CS%Kv) * topfn
                    end if

                    z2 = z_i_3d(i, j, K2)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    Kv_tot = Kv_tot + CS%Kv_extra_bbl * botfn

                    h_shear = 0.5_dp * (hvel_3d(i, j, K2) + hvel_3d(i, j, K2 - 1) + h_neglect)
                    CS%a_v(i, j, K2) = Kv_tot / h_shear
                else
                    CS%a_v(i, j, K2) = 0.0_dp
                end if
            end do
        end do

        ! Bottom interface
        do concurrent(i=is:ie, j=js-1:je)
            if (G%mask2dCv(i, j) > 0.0_dp) then
                CS%a_v(i, j, nz + 1) = (CS%Kv + CS%Kv_extra_bbl) / &
                                        (0.5_dp * hvel_3d(i, j, nz) + h_neglect)
            else
                CS%a_v(i, j, nz + 1) = 0.0_dp
            end if
        end do

        ! Store h_v
        do k = 1, nz
            do concurrent(i=is:ie, j=js-1:je)
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    CS%h_v(i, j, k) = hvel_3d(i, j, k) + h_neglect
                else
                    CS%h_v(i, j, k) = h_neglect
                end if
            end do
        end do

        deallocate(hvel_3d, z_i_3d, z_top_2d)

    end subroutine vert_visc_coef

    module subroutine vert_visc_remnant(dt, CS, G, GV, visc)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), intent(in) :: dt
        type(vert_visc_CS), intent(inout) :: CS
        type(vertvisc_type), intent(in), optional :: visc

        real(dp) :: b_denom_1, Ray
        logical :: have_rayleigh
        integer :: i, j, k, is, ie, js, je, nz

        ! Workspace arrays
        real(dp), allocatable :: b1_2d(:, :)
        real(dp), allocatable :: d1_2d(:, :)
        real(dp), allocatable :: c1_3d(:, :, :)

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

        have_rayleigh = .false.
        if (present(visc)) have_rayleigh = visc%has_Rayleigh

        allocate(b1_2d(G%isd:G%ied, G%jsd:G%jed))
        allocate(d1_2d(G%isd:G%ied, G%jsd:G%jed))
        allocate(c1_3d(G%isd:G%ied, G%jsd:G%jed, GV%ke))

        ! ===================================================================
        ! --- U-points ---
        ! ===================================================================

        ! Layer 1
        do concurrent(i=is-1:ie, j=js:je)
            if (G%mask2dCu(i, j) > 0.0_dp) then
                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                CS%visc_rem_u(i, j, 1) = b1_2d(i, j) * CS%h_u(i, j, 1)
            else
                CS%visc_rem_u(i, j, 1) = 1.0_dp
            end if
        end do

        ! Forward sweep
        do k = 2, nz
            do concurrent(i=is-1:ie, j=js:je)
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, k)

                    c1_3d(i, j, k) = dt * CS%a_u(i, j, k) * b1_2d(i, j)
                    b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1_2d(i, j))
                    b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                    d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                    CS%visc_rem_u(i, j, k) = (CS%h_u(i, j, k) + &
                        dt * CS%a_u(i, j, k) * CS%visc_rem_u(i, j, k - 1)) * b1_2d(i, j)
                else
                    CS%visc_rem_u(i, j, k) = 1.0_dp
                end if
            end do
        end do

        ! Back substitution
        do k = nz - 1, 1, -1
            do concurrent(i=is-1:ie, j=js:je)
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    CS%visc_rem_u(i, j, k) = CS%visc_rem_u(i, j, k) + &
                        c1_3d(i, j, k + 1) * CS%visc_rem_u(i, j, k + 1)
                end if
            end do
        end do

        ! ===================================================================
        ! --- V-points ---
        ! ===================================================================

        ! Layer 1
        do concurrent(i=is:ie, j=js-1:je)
            if (G%mask2dCv(i, j) > 0.0_dp) then
                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_v(i, j, 1)

                b_denom_1 = CS%h_v(i, j, 1) + dt * (Ray + CS%a_v(i, j, 1))
                b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, 2))
                d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                CS%visc_rem_v(i, j, 1) = b1_2d(i, j) * CS%h_v(i, j, 1)
            else
                CS%visc_rem_v(i, j, 1) = 1.0_dp
            end if
        end do

        ! Forward sweep
        do k = 2, nz
            do concurrent(i=is:ie, j=js-1:je)
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_v(i, j, k)

                    c1_3d(i, j, k) = dt * CS%a_v(i, j, k) * b1_2d(i, j)
                    b_denom_1 = CS%h_v(i, j, k) + dt * (Ray + CS%a_v(i, j, k) * d1_2d(i, j))
                    b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, k + 1))
                    d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                    CS%visc_rem_v(i, j, k) = (CS%h_v(i, j, k) + &
                        dt * CS%a_v(i, j, k) * CS%visc_rem_v(i, j, k - 1)) * b1_2d(i, j)
                else
                    CS%visc_rem_v(i, j, k) = 1.0_dp
                end if
            end do
        end do

        ! Back substitution
        do k = nz - 1, 1, -1
            do concurrent(i=is:ie, j=js-1:je)
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    CS%visc_rem_v(i, j, k) = CS%visc_rem_v(i, j, k) + &
                        c1_3d(i, j, k + 1) * CS%visc_rem_v(i, j, k + 1)
                end if
            end do
        end do

        deallocate(b1_2d, d1_2d, c1_3d)

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
        logical :: have_forces, have_rayleigh
        integer :: i, j, k, is, ie, js, je, nz

        ! Workspace arrays
        real(dp), allocatable :: b1_2d(:, :)
        real(dp), allocatable :: d1_2d(:, :)
        real(dp), allocatable :: c1_3d(:, :, :)

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        dt_Rho0 = dt / RHO_0

        have_forces = present(forces)
        have_rayleigh = .false.
        if (present(visc)) have_rayleigh = visc%has_Rayleigh

        allocate(b1_2d(G%isd:G%ied, G%jsd:G%jed))
        allocate(d1_2d(G%isd:G%ied, G%jsd:G%jed))
        allocate(c1_3d(G%isd:G%ied, G%jsd:G%jed, GV%ke))

        ! ===================================================================
        ! --- Apply to u-velocity ---
        ! ===================================================================

        ! Layer 1
        do concurrent(i=is-1:ie, j=js:je)
            if (G%mask2dCu(i, j) > 0.0_dp) then
                sfc_stress = 0.0_dp
                if (have_forces) sfc_stress = dt_Rho0 * forces%taux(i, j) * G%mask2dCu(i, j)

                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_u(i, j, 1)

                b_denom_1 = CS%h_u(i, j, 1) + dt * (Ray + CS%a_u(i, j, 1))
                b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, 2))
                d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                u(i, j, 1) = b1_2d(i, j) * (CS%h_u(i, j, 1) * u(i, j, 1) + sfc_stress)
            end if
        end do

        ! Forward sweep
        do k = 2, nz
            do concurrent(i=is-1:ie, j=js:je)
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_u(i, j, k)

                    c1_3d(i, j, k) = dt * CS%a_u(i, j, k) * b1_2d(i, j)
                    b_denom_1 = CS%h_u(i, j, k) + dt * (Ray + CS%a_u(i, j, k) * d1_2d(i, j))
                    b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_u(i, j, k + 1))
                    d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                    u(i, j, k) = (CS%h_u(i, j, k) * u(i, j, k) + &
                        dt * CS%a_u(i, j, k) * u(i, j, k - 1)) * b1_2d(i, j)
                end if
            end do
        end do

        ! Back substitution
        do k = nz - 1, 1, -1
            do concurrent(i=is-1:ie, j=js:je)
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    u(i, j, k) = u(i, j, k) + c1_3d(i, j, k + 1) * u(i, j, k + 1)
                end if
            end do
        end do

        ! Bottom stress
        do concurrent(i=is-1:ie, j=js:je)
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

        ! ===================================================================
        ! --- Apply to v-velocity ---
        ! ===================================================================

        ! Layer 1
        do concurrent(i=is:ie, j=js-1:je)
            if (G%mask2dCv(i, j) > 0.0_dp) then
                sfc_stress = 0.0_dp
                if (have_forces) sfc_stress = dt_Rho0 * forces%tauy(i, j) * G%mask2dCv(i, j)

                Ray = 0.0_dp
                if (have_rayleigh) Ray = visc%Ray_v(i, j, 1)

                b_denom_1 = CS%h_v(i, j, 1) + dt * (Ray + CS%a_v(i, j, 1))
                b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, 2))
                d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                v(i, j, 1) = b1_2d(i, j) * (CS%h_v(i, j, 1) * v(i, j, 1) + sfc_stress)
            end if
        end do

        ! Forward sweep
        do k = 2, nz
            do concurrent(i=is:ie, j=js-1:je)
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    Ray = 0.0_dp
                    if (have_rayleigh) Ray = visc%Ray_v(i, j, k)

                    c1_3d(i, j, k) = dt * CS%a_v(i, j, k) * b1_2d(i, j)
                    b_denom_1 = CS%h_v(i, j, k) + dt * (Ray + CS%a_v(i, j, k) * d1_2d(i, j))
                    b1_2d(i, j) = 1.0_dp / (b_denom_1 + dt * CS%a_v(i, j, k + 1))
                    d1_2d(i, j) = b_denom_1 * b1_2d(i, j)
                    v(i, j, k) = (CS%h_v(i, j, k) * v(i, j, k) + &
                        dt * CS%a_v(i, j, k) * v(i, j, k - 1)) * b1_2d(i, j)
                end if
            end do
        end do

        ! Back substitution
        do k = nz - 1, 1, -1
            do concurrent(i=is:ie, j=js-1:je)
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    v(i, j, k) = v(i, j, k) + c1_3d(i, j, k + 1) * v(i, j, k + 1)
                end if
            end do
        end do

        ! Bottom stress
        do concurrent(i=is:ie, j=js-1:je)
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

        deallocate(b1_2d, d1_2d, c1_3d)

    end subroutine vert_visc_apply

end submodule mom6_vert_visc_kij
