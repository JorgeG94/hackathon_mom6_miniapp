!> Submodule for zonal_flux_adjust_gpu and set_zonal_BT_cont_gpu — compiled separately at -O1
!! to work around nvfortran >=O2 codegen bug (CUDA_EXCEPTION_14 Warp Illegal Address).
submodule (mom6_continuity_omp) mom6_continuity_adjust_omp
    implicit none
contains

!> GPU kernel: Newton iteration to adjust zonal fluxes to match barotropic transport.
!! collapse(2) over (j, I), each thread runs independent Newton iteration.
    module procedure zonal_flux_adjust_gpu

        ! Scalars for each thread (all private)
        real(dp) :: du_val, du_prev, ddu, uh_err_val, uh_err_best_val
        real(dp) :: duhdu_tot_val, du_max_val, du_min_val
        real(dp) :: tol_eta, tol_vel
        real(dp) :: CFL, curv_3, h_marg, u_adj, uh_k, duhdu_k, visc_rem_val
        logical :: do_more
        integer :: i, j, k, itt, ish, ieh, jsh, jeh, nz
        integer, parameter :: max_itts = 20

        ish = G%isc ; ieh = G%iec ; jsh = G%jsc ; jeh = G%jec ; nz = GV%ke


        !$omp target teams distribute parallel do collapse(2) &
        !$omp   private(du_val, du_prev, ddu, uh_err_val, uh_err_best_val, &
        !$omp           duhdu_tot_val, du_max_val, du_min_val, tol_eta, tol_vel, &
        !$omp           CFL, curv_3, h_marg, u_adj, uh_k, duhdu_k, visc_rem_val, do_more, &
        !$omp           k, itt)
        do j = jsh, jeh
            do I = ish - 1, ieh
                du_val = 0.0_dp
                du_max_val = CS%du_max_CFL(I, j)
                du_min_val = CS%du_min_CFL(I, j)
                uh_err_val = CS%uh_tot_0(I, j) - uhbt(I, j)
                duhdu_tot_val = CS%duhdu_tot_0(I, j)
                uh_err_best_val = abs(uh_err_val)
                do_more = .true.

                do itt = 1, max_itts
                    ! Tolerance selection
                    if (itt <= 1) then
                        tol_eta = 1.0e-6_dp * CS%tol_eta
                    elseif (itt == 2) then
                        tol_eta = 1.0e-4_dp * CS%tol_eta
                    elseif (itt == 3) then
                        tol_eta = 1.0e-2_dp * CS%tol_eta
                    else
                        tol_eta = CS%tol_eta
                    end if
                    tol_vel = CS%tol_vel

                    ! Update bisection bounds
                    if (uh_err_val > 0.0_dp) then
                        du_max_val = du_val
                    elseif (uh_err_val < 0.0_dp) then
                        du_min_val = du_val
                    else
                        do_more = .false.
                    end if

                    if (do_more) then
                        if ((dt * min(G%IareaT(i, j), G%IareaT(i + 1, j)) * abs(uh_err_val) > tol_eta) .or. &
                            (CS%better_iter .and. ((abs(uh_err_val) > tol_vel * duhdu_tot_val) .or. &
                                                   (abs(uh_err_val) > uh_err_best_val)))) then
                            ! Newton step
                            ddu = -uh_err_val / duhdu_tot_val
                            du_prev = du_val
                            du_val = du_val + ddu
                            if (abs(ddu) < 1.0e-15_dp * abs(du_val)) then
                                do_more = .false.
                            elseif (ddu > 0.0_dp) then
                                if (du_val >= du_max_val) then
                                    du_val = 0.5_dp * (du_prev + du_max_val)
                                    if (du_max_val - du_prev < 1.0e-15_dp * abs(du_val)) do_more = .false.
                                end if
                            else
                                if (du_val <= du_min_val) then
                                    du_val = 0.5_dp * (du_prev + du_min_val)
                                    if (du_prev - du_min_val < 1.0e-15_dp * abs(du_val)) do_more = .false.
                                end if
                            end if
                        else
                            do_more = .false.
                        end if
                    end if

                    if (.not. do_more) exit

                    ! Recompute flux with adjusted velocity — sum over k
                    uh_err_val = -uhbt(I, j)
                    duhdu_tot_val = 0.0_dp
                    do k = 1, nz
                        if (use_visc_rem) then
                            visc_rem_val = visc_rem_u(I, j, k)
                        else
                            visc_rem_val = 1.0_dp
                        end if
                        u_adj = u(I, j, k) + du_val * visc_rem_val

                        ! Inline zonal_flux_layer logic
                        if (u_adj > 0.0_dp) then
                            if (CS%vol_CFL) then
                                CFL = (u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i, j))
                            else
                                CFL = u_adj * dt * G%IdxT(i, j)
                            end if
                            curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                            uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                                (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + curv_3 * (CFL - 1.5_dp)))
                            h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                        elseif (u_adj < 0.0_dp) then
                            if (CS%vol_CFL) then
                                CFL = (-u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i + 1, j))
                            else
                                CFL = -u_adj * dt * G%IdxT(i + 1, j)
                            end if
                            curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                            uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                                (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + curv_3 * (CFL - 1.5_dp)))
                            h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                        else
                            uh_k = 0.0_dp
                            h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                        end if
                        duhdu_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * h_marg * visc_rem_val
                        uh_err_val = uh_err_val + uh_k
                        duhdu_tot_val = duhdu_tot_val + duhdu_k
                    end do
                    uh_err_best_val = min(uh_err_best_val, abs(uh_err_val))
                end do ! Newton iterations

                ! Final pass: write converged uh values
                do k = 1, nz
                    if (use_visc_rem) then
                        visc_rem_val = visc_rem_u(I, j, k)
                    else
                        visc_rem_val = 1.0_dp
                    end if
                    u_adj = u(I, j, k) + du_val * visc_rem_val

                    if (u_adj > 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i, j))
                        else
                            CFL = u_adj * dt * G%IdxT(i, j)
                        end if
                        curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                        uh(I, j, k) = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                            (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + curv_3 * (CFL - 1.5_dp)))
                    elseif (u_adj < 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (-u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i + 1, j))
                        else
                            CFL = -u_adj * dt * G%IdxT(i + 1, j)
                        end if
                        curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                        uh(I, j, k) = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                            (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + curv_3 * (CFL - 1.5_dp)))
                    else
                        uh(I, j, k) = 0.0_dp
                    end if
                end do

                ! Write u_cor and du_cor
                CS%du(I, j) = du_val
            end do
        end do

    end procedure zonal_flux_adjust_gpu

!> GPU kernel: Compute BT_cont face areas. collapse(2) over (j, I).
    module procedure set_zonal_BT_cont_gpu

        ! Private scalars for Newton iteration + BT_cont computation
        real(dp) :: du0_val, du_val, du_prev, ddu, uh_err_val, duhdu_tot_val
        real(dp) :: du_max_val, du_min_val, uh_err_best_val
        real(dp) :: tol_eta, tol_vel
        real(dp) :: duL_val, duR_val, du_CFL_val, Idt
        real(dp) :: visc_rem_lim, visc_rem_val
        real(dp) :: CFL, curv_3, h_marg, u_adj, uh_k, duhdu_k
        real(dp) :: FAmt_L_val, FAmt_R_val, FAmt_0_val
        real(dp) :: uhtot_L_val, uhtot_R_val
        real(dp) :: FA_0, FA_avg
        real(dp) :: min_visc_rem, CFL_min_val
        logical :: do_more
        integer :: i, j, k, itt, ish, ieh, jsh, jeh, nz
        integer, parameter :: max_itts = 20

        ish = G%isc ; ieh = G%iec ; jsh = G%jsc ; jeh = G%jec ; nz = GV%ke
        Idt = 1.0_dp / dt
        min_visc_rem = 0.1_dp
        CFL_min_val = 1.0e-6_dp

        !$omp target teams distribute parallel do collapse(2) &
        !$omp   private(du0_val, du_val, du_prev, ddu, uh_err_val, duhdu_tot_val, &
        !$omp           du_max_val, du_min_val, uh_err_best_val, tol_eta, tol_vel, &
        !$omp           duL_val, duR_val, du_CFL_val, visc_rem_lim, visc_rem_val, &
        !$omp           CFL, curv_3, h_marg, u_adj, uh_k, duhdu_k, &
        !$omp           FAmt_L_val, FAmt_R_val, FAmt_0_val, uhtot_L_val, uhtot_R_val, &
        !$omp           FA_0, FA_avg, do_more, k, itt)
        do j = jsh, jeh
            do I = ish - 1, ieh
                ! --- First: Find du0 (zero-transport correction) via Newton ---
                du_val = 0.0_dp
                du_max_val = CS%du_max_CFL(I, j)
                du_min_val = CS%du_min_CFL(I, j)
                uh_err_val = CS%uh_tot_0(I, j) ! target is 0
                duhdu_tot_val = CS%duhdu_tot_0(I, j)
                uh_err_best_val = abs(uh_err_val)
                do_more = .true.

                do itt = 1, max_itts
                    if (itt <= 1) then
                        tol_eta = 1.0e-6_dp * CS%tol_eta
                    elseif (itt == 2) then
                        tol_eta = 1.0e-4_dp * CS%tol_eta
                    elseif (itt == 3) then
                        tol_eta = 1.0e-2_dp * CS%tol_eta
                    else
                        tol_eta = CS%tol_eta
                    end if
                    tol_vel = CS%tol_vel

                    if (uh_err_val > 0.0_dp) then
                        du_max_val = du_val
                    elseif (uh_err_val < 0.0_dp) then
                        du_min_val = du_val
                    else
                        do_more = .false.
                    end if

                    if (do_more) then
                        if ((dt * min(G%IareaT(i, j), G%IareaT(i + 1, j)) * abs(uh_err_val) > tol_eta) .or. &
                            (CS%better_iter .and. ((abs(uh_err_val) > tol_vel * duhdu_tot_val) .or. &
                                                   (abs(uh_err_val) > uh_err_best_val)))) then
                            ddu = -uh_err_val / duhdu_tot_val
                            du_prev = du_val
                            du_val = du_val + ddu
                            if (abs(ddu) < 1.0e-15_dp * abs(du_val)) then
                                do_more = .false.
                            elseif (ddu > 0.0_dp) then
                                if (du_val >= du_max_val) then
                                    du_val = 0.5_dp * (du_prev + du_max_val)
                                    if (du_max_val - du_prev < 1.0e-15_dp * abs(du_val)) do_more = .false.
                                end if
                            else
                                if (du_val <= du_min_val) then
                                    du_val = 0.5_dp * (du_prev + du_min_val)
                                    if (du_prev - du_min_val < 1.0e-15_dp * abs(du_val)) do_more = .false.
                                end if
                            end if
                        else
                            do_more = .false.
                        end if
                    end if

                    if (.not. do_more) exit

                    uh_err_val = 0.0_dp ! target is zero transport
                    duhdu_tot_val = 0.0_dp
                    do k = 1, nz
                        if (use_visc_rem) then
                            visc_rem_val = visc_rem_u(I, j, k)
                        else
                            visc_rem_val = 1.0_dp
                        end if
                        u_adj = u(I, j, k) + du_val * visc_rem_val
                        ! Inline flux
                        if (u_adj > 0.0_dp) then
                            if (CS%vol_CFL) then
                                CFL = (u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i, j))
                            else
                                CFL = u_adj * dt * G%IdxT(i, j)
                            end if
                            curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                            uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                                (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + curv_3 * (CFL - 1.5_dp)))
                            h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                        elseif (u_adj < 0.0_dp) then
                            if (CS%vol_CFL) then
                                CFL = (-u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i + 1, j))
                            else
                                CFL = -u_adj * dt * G%IdxT(i + 1, j)
                            end if
                            curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                            uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                                (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + curv_3 * (CFL - 1.5_dp)))
                            h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                        else
                            uh_k = 0.0_dp
                            h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                        end if
                        duhdu_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * h_marg * visc_rem_val
                        uh_err_val = uh_err_val + uh_k
                        duhdu_tot_val = duhdu_tot_val + duhdu_k
                    end do
                    uh_err_best_val = min(uh_err_best_val, abs(uh_err_val))
                end do ! Newton for du0

                du0_val = du_val

                ! --- Determine test velocities duL, duR ---
                du_CFL_val = (CFL_min_val * Idt) * G%dxCu(I, j)
                duR_val = min(0.0_dp, du0_val - du_CFL_val)
                duL_val = max(0.0_dp, du0_val + du_CFL_val)

                ! Adjust duR, duL so test velocities are truly upwind
                do k = 1, nz
                    if (use_visc_rem) then
                        visc_rem_val = visc_rem_u(I, j, k)
                    else
                        visc_rem_val = 1.0_dp
                    end if
                    visc_rem_lim = max(visc_rem_val, min_visc_rem * CS%visc_rem_max(I, j))
                    if (visc_rem_lim > 0.0_dp) then
                        if (u(I, j, k) + duR_val * visc_rem_lim > -du_CFL_val * visc_rem_val) &
                            duR_val = -(u(I, j, k) + du_CFL_val * visc_rem_val) / visc_rem_lim
                        if (u(I, j, k) + duL_val * visc_rem_lim < du_CFL_val * visc_rem_val) &
                            duL_val = -(u(I, j, k) - du_CFL_val * visc_rem_val) / visc_rem_lim
                    end if
                end do

                ! --- Evaluate fluxes at 3 test velocities (u_0, u_L, u_R) ---
                FAmt_0_val = 0.0_dp ; FAmt_L_val = 0.0_dp ; FAmt_R_val = 0.0_dp
                uhtot_L_val = 0.0_dp ; uhtot_R_val = 0.0_dp

                do k = 1, nz
                    if (use_visc_rem) then
                        visc_rem_val = visc_rem_u(I, j, k)
                    else
                        visc_rem_val = 1.0_dp
                    end if

                    ! --- u_0 test velocity ---
                    u_adj = u(I, j, k) + du0_val * visc_rem_val
                    if (u_adj > 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i, j))
                        else
                            CFL = u_adj * dt * G%IdxT(i, j)
                        end if
                        curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                        h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    elseif (u_adj < 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (-u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i + 1, j))
                        else
                            CFL = -u_adj * dt * G%IdxT(i + 1, j)
                        end if
                        curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                        h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    else
                        h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                    end if
                    FAmt_0_val = FAmt_0_val + (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * h_marg * visc_rem_val

                    ! --- u_L test velocity (westerly, positive) ---
                    u_adj = u(I, j, k) + duL_val * visc_rem_val
                    if (u_adj > 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i, j))
                        else
                            CFL = u_adj * dt * G%IdxT(i, j)
                        end if
                        curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                        uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                            (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + curv_3 * (CFL - 1.5_dp)))
                        h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    elseif (u_adj < 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (-u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i + 1, j))
                        else
                            CFL = -u_adj * dt * G%IdxT(i + 1, j)
                        end if
                        curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                        uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                            (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + curv_3 * (CFL - 1.5_dp)))
                        h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    else
                        uh_k = 0.0_dp
                        h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                    end if
                    FAmt_L_val = FAmt_L_val + (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * h_marg * visc_rem_val
                    uhtot_L_val = uhtot_L_val + uh_k

                    ! --- u_R test velocity (easterly, negative) ---
                    u_adj = u(I, j, k) + duR_val * visc_rem_val
                    if (u_adj > 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i, j))
                        else
                            CFL = u_adj * dt * G%IdxT(i, j)
                        end if
                        curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                        uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                            (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + curv_3 * (CFL - 1.5_dp)))
                        h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    elseif (u_adj < 0.0_dp) then
                        if (CS%vol_CFL) then
                            CFL = (-u_adj * dt) * (G%dy_Cu(I, j) * G%IareaT(i + 1, j))
                        else
                            CFL = -u_adj * dt * G%IdxT(i + 1, j)
                        end if
                        curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                        uh_k = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u_adj * &
                            (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + curv_3 * (CFL - 1.5_dp)))
                        h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    else
                        uh_k = 0.0_dp
                        h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                    end if
                    FAmt_R_val = FAmt_R_val + (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * h_marg * visc_rem_val
                    uhtot_R_val = uhtot_R_val + uh_k
                end do ! k

                ! --- Compute BT_cont fields ---
                ! Westerly (W0, WW)
                FA_0 = FAmt_0_val
                FA_avg = FAmt_0_val
                if ((duL_val - du0_val) /= 0.0_dp) &
                    FA_avg = uhtot_L_val / (duL_val - du0_val)
                if (FA_avg > max(FA_0, FAmt_L_val)) then
                    FA_avg = max(FA_0, FAmt_L_val)
                elseif (FA_avg < min(FA_0, FAmt_L_val)) then
                    FA_0 = FA_avg
                end if
                BT_cont%FA_u_W0(I, j) = FA_0
                BT_cont%FA_u_WW(I, j) = FAmt_L_val
                if (abs(FA_0 - FAmt_L_val) <= 1.0e-12_dp * FA_0) then
                    BT_cont%uBT_WW(I, j) = 0.0_dp
                else
                    BT_cont%uBT_WW(I, j) = (1.5_dp * (duL_val - du0_val)) * &
                        ((FAmt_L_val - FA_avg) / (FAmt_L_val - FA_0))
                end if

                ! Easterly (E0, EE)
                FA_0 = FAmt_0_val
                FA_avg = FAmt_0_val
                if ((duR_val - du0_val) /= 0.0_dp) &
                    FA_avg = uhtot_R_val / (duR_val - du0_val)
                if (FA_avg > max(FA_0, FAmt_R_val)) then
                    FA_avg = max(FA_0, FAmt_R_val)
                elseif (FA_avg < min(FA_0, FAmt_R_val)) then
                    FA_0 = FA_avg
                end if
                BT_cont%FA_u_E0(I, j) = FA_0
                BT_cont%FA_u_EE(I, j) = FAmt_R_val
                if (abs(FAmt_R_val - FA_0) <= 1.0e-12_dp * FA_0) then
                    BT_cont%uBT_EE(I, j) = 0.0_dp
                else
                    BT_cont%uBT_EE(I, j) = (1.5_dp * (duR_val - du0_val)) * &
                        ((FAmt_R_val - FA_avg) / (FAmt_R_val - FA_0))
                end if
            end do
        end do

    end procedure set_zonal_BT_cont_gpu

end submodule mom6_continuity_adjust_omp
