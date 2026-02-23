!> Standalone test driver for pure GPU-callable building blocks.
!! Compares new pure subroutines against dead-code reference implementations.
program test_kernels_driver
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          end_ocean_grid, BT_cont_type, alloc_BT_cont_type
    use mom6_continuity, only: continuity_CS, continuity_init, continuity_end, &
                               zonal_flux_layer, zonal_flux_adjust, set_zonal_BT_cont, &
                               zonal_flux_point, newton_flux_adjust_column, bt_cont_column
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(continuity_CS) :: CS
    type(BT_cont_type), pointer :: BT_cont

    real(dp), allocatable :: h(:,:,:), u(:,:,:), por_face_areaU(:,:,:), visc_rem_u(:,:,:)
    real(dp), allocatable :: uh(:,:,:), uhbt(:,:), u_cor(:,:,:), du_cor(:,:)
    real(dp), allocatable :: h_W(:,:,:), h_E(:,:,:)

    real(dp) :: dt
    integer :: ni, nj, nk, i, j, k
    character(len=32) :: arg
    logical :: all_pass

    ! Default parameters
    ni = 180; nj = 180; nk = 75

    ! Parse command line
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg); read(arg, *) ni
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg); read(arg, *) nj
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, arg); read(arg, *) nk
    end if

    print '(A)', '=================================================='
    print '(A)', 'MOM6 Kernel Test Driver'
    print '(A)', '=================================================='
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A)', '=================================================='

    ! Initialize grid and control structure
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call continuity_init(CS, G, GV, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call alloc_BT_cont_type(BT_cont, G, GV)

    dt = 300.0_dp

    ! Allocate state arrays
    allocate(h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h_W(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h_E(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Initialize state (sinusoidal, matching continuity_driver)
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                h(i,j,k) = 4000.0_dp/real(nk,dp) + 10.0_dp * &
                    sin(real(i-1,dp)/real(ni,dp)*3.14159_dp) * &
                    cos(real(j-1,dp)/real(nj,dp)*3.14159_dp) * &
                    exp(-real(k,dp)/20.0_dp)
                u(i,j,k) = 0.1_dp * sin(real(j-1,dp)/real(nj,dp)*3.14159_dp*2.0_dp) * &
                    exp(-real(k,dp)/30.0_dp)
            end do
        end do
    end do

    ! For upwind_1st: h_W = h_E = h
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                h_W(i,j,k) = h(i,j,k)
                h_E(i,j,k) = h(i,j,k)
            end do
        end do
    end do

    all_pass = .true.

    ! ==========================================
    ! Test 1: zonal_flux_point vs zonal_flux_layer
    ! ==========================================
    call test_zonal_flux_point(all_pass)

    ! ==========================================
    ! Test 2: newton_flux_adjust_column vs zonal_flux_adjust
    ! ==========================================
    call test_newton_flux_adjust(all_pass)

    ! ==========================================
    ! Test 3: bt_cont_column vs set_zonal_BT_cont
    ! ==========================================
    call test_bt_cont(all_pass)

    print '(A)', ''
    print '(A)', '=================================================='
    if (all_pass) then
        print '(A)', 'ALL TESTS PASSED'
    else
        print '(A)', 'SOME TESTS FAILED'
    end if
    print '(A)', '=================================================='

    ! Cleanup
    call continuity_end(CS, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call end_ocean_grid(G)
    deallocate(h, u, uh, h_W, h_E)

contains

    !> Test 1: Compare zonal_flux_point against zonal_flux_layer
    subroutine test_zonal_flux_point(pass)
        logical, intent(inout) :: pass

        ! Reference arrays (1D slices for zonal_flux_layer)
        real(dp), dimension(G%isd:G%ied) :: u_row, h_row, hW_row, hE_row
        real(dp), dimension(G%isd:G%ied) :: uh_ref, duhdu_ref, visc_rem_row, por_row
        logical, dimension(G%isd:G%ied) :: do_I

        real(dp) :: uh_test, duhdu_test, max_uh_err, max_duhdu_err
        real(dp) :: rel_err
        integer :: test_j, I_idx

        print '(A)', ''
        print '(A)', 'Test 1: zonal_flux_point vs zonal_flux_layer'
        print '(A)', '----------------------------------------------'

        test_j = (G%jsc + G%jec) / 2  ! Middle row
        max_uh_err = 0.0_dp
        max_duhdu_err = 0.0_dp

        do k = 1, nk
            ! Extract row data
            do i = G%isd, G%ied
                u_row(i) = u(i, test_j, k)
                h_row(i) = h(i, test_j, k)
                hW_row(i) = h_W(i, test_j, k)
                hE_row(i) = h_E(i, test_j, k)
                visc_rem_row(i) = visc_rem_u(i, test_j, k)
                por_row(i) = por_face_areaU(i, test_j, k)
                do_I(i) = .true.
            end do

            ! Call reference (dead code)
            uh_ref = 0.0_dp
            duhdu_ref = 0.0_dp
            call zonal_flux_layer(u_row, h_row, hW_row, hE_row, uh_ref, duhdu_ref, &
                                  visc_rem_row, dt, G, test_j, G%isc, G%iec, do_I, &
                                  CS%vol_CFL, por_row)

            ! Compare with zonal_flux_point for each face
            do I_idx = G%isc - 1, G%iec
                call zonal_flux_point(u_row(I_idx), &
                    h_row(I_idx), h_row(I_idx + 1), &
                    hW_row(I_idx), hE_row(I_idx), hW_row(I_idx + 1), hE_row(I_idx + 1), &
                    G%dy_Cu(I_idx, test_j), G%IareaT(I_idx, test_j), G%IareaT(I_idx + 1, test_j), &
                    G%IdxT(I_idx, test_j), G%IdxT(I_idx + 1, test_j), &
                    por_row(I_idx), visc_rem_row(I_idx), dt, CS%vol_CFL, &
                    uh_test, duhdu_test)

                if (abs(uh_ref(I_idx)) > 0.0_dp) then
                    rel_err = abs(uh_test - uh_ref(I_idx)) / abs(uh_ref(I_idx))
                    max_uh_err = max(max_uh_err, rel_err)
                else
                    max_uh_err = max(max_uh_err, abs(uh_test - uh_ref(I_idx)))
                end if

                if (abs(duhdu_ref(I_idx)) > 0.0_dp) then
                    rel_err = abs(duhdu_test - duhdu_ref(I_idx)) / abs(duhdu_ref(I_idx))
                    max_duhdu_err = max(max_duhdu_err, rel_err)
                else
                    max_duhdu_err = max(max_duhdu_err, abs(duhdu_test - duhdu_ref(I_idx)))
                end if
            end do
        end do

        print '(A,ES15.8)', '  Max uh relative error:    ', max_uh_err
        print '(A,ES15.8)', '  Max duhdu relative error: ', max_duhdu_err

        if (max_uh_err < 1.0e-14_dp .and. max_duhdu_err < 1.0e-14_dp) then
            print '(A)', '  Status: PASS'
        else
            print '(A)', '  Status: FAIL'
            pass = .false.
        end if
    end subroutine test_zonal_flux_point

    !> Test 2: Compare newton_flux_adjust_column against zonal_flux_adjust
    subroutine test_newton_flux_adjust(pass)
        logical, intent(inout) :: pass

        ! Reference arrays
        real(dp), dimension(G%isd:G%ied) :: uh_ref_row, duhdu_ref_row
        real(dp), dimension(G%isd:G%ied) :: uh_tot_0_row, duhdu_tot_0_row
        real(dp), dimension(G%isd:G%ied) :: du_max_CFL_row, du_min_CFL_row
        real(dp), dimension(G%isd:G%ied) :: uhbt_row, du_ref
        real(dp), dimension(G%isd:G%ied) :: visc_rem_max_row
        real(dp), dimension(G%isd:G%ied, GV%ke) :: visc_rem_2d
        logical, dimension(G%isd:G%ied) :: do_I

        ! Column arrays for new subroutine
        real(dp) :: u_col(nk), h_col(nk), h_ip1_col(nk)
        real(dp) :: hW_col(nk), hE_col(nk), hW_ip1_col(nk), hE_ip1_col(nk)
        real(dp) :: vr_col(nk), pf_col(nk), uh_col_out(nk)
        real(dp) :: du_test, max_du_err, rel_err
        real(dp) :: CFL_dt, I_dt, I_vrm, dx_W_loc, dx_E_loc

        integer :: test_j, I_idx

        print '(A)', ''
        print '(A)', 'Test 2: newton_flux_adjust_column vs zonal_flux_adjust'
        print '(A)', '------------------------------------------------------'

        test_j = (G%jsc + G%jec) / 2
        max_du_err = 0.0_dp

        CFL_dt = CS%CFL_limit_adjust / dt
        I_dt = 1.0_dp / dt

        ! Setup: compute initial fluxes for each face using zonal_flux_layer,
        ! then accumulate pre-Newton quantities
        do i = G%isd, G%ied
            do_I(i) = .true.
        end do

        ! Compute visc_rem_2d and per-face uh_tot_0, duhdu_tot_0, CFL bounds
        do i = G%isd, G%ied
            do k = 1, nk
                visc_rem_2d(i, k) = visc_rem_u(i, test_j, k)
            end do
        end do

        ! Compute initial fluxes per layer, then sum
        uh_tot_0_row = 0.0_dp
        duhdu_tot_0_row = 0.0_dp
        do k = 1, nk
            uh_ref_row = 0.0_dp
            duhdu_ref_row = 0.0_dp
            call zonal_flux_layer(u(:, test_j, k), h(:, test_j, k), &
                h_W(:, test_j, k), h_E(:, test_j, k), &
                uh_ref_row, duhdu_ref_row, visc_rem_2d(:, k), &
                dt, G, test_j, G%isc, G%iec, do_I, CS%vol_CFL, &
                por_face_areaU(:, test_j, k))
            do I_idx = G%isc - 1, G%iec
                uh_tot_0_row(I_idx) = uh_tot_0_row(I_idx) + uh_ref_row(I_idx)
                duhdu_tot_0_row(I_idx) = duhdu_tot_0_row(I_idx) + duhdu_ref_row(I_idx)
            end do
        end do

        ! Compute visc_rem_max and CFL bounds (matching Block 1 logic)
        do I_idx = G%isc - 1, G%iec
            visc_rem_max_row(I_idx) = 0.0_dp
            if (CS%use_visc_rem_max) then
                do k = 1, nk
                    visc_rem_max_row(I_idx) = max(visc_rem_max_row(I_idx), visc_rem_2d(I_idx, k))
                end do
            else
                visc_rem_max_row(I_idx) = 1.0_dp
            end if

            I_vrm = 0.0_dp
            if (visc_rem_max_row(I_idx) > 0.0_dp) I_vrm = 1.0_dp / visc_rem_max_row(I_idx)
            dx_W_loc = G%dxT(I_idx, test_j)
            dx_E_loc = G%dxT(I_idx + 1, test_j)
            du_max_CFL_row(I_idx) = 2.0_dp * (CFL_dt * dx_W_loc) * I_vrm
            du_min_CFL_row(I_idx) = -2.0_dp * (CFL_dt * dx_E_loc) * I_vrm

            ! CFL refinement (non-aggressive, with visc_rem)
            do k = 1, nk
                if (du_max_CFL_row(I_idx) * visc_rem_2d(I_idx, k) > &
                    dx_W_loc * CFL_dt - u(I_idx, test_j, k) * G%mask2dCu(I_idx, test_j)) &
                    du_max_CFL_row(I_idx) = (dx_W_loc * CFL_dt - u(I_idx, test_j, k)) / visc_rem_2d(I_idx, k)
                if (du_min_CFL_row(I_idx) * visc_rem_2d(I_idx, k) < &
                    -dx_E_loc * CFL_dt - u(I_idx, test_j, k) * G%mask2dCu(I_idx, test_j)) &
                    du_min_CFL_row(I_idx) = -(dx_E_loc * CFL_dt + u(I_idx, test_j, k)) / visc_rem_2d(I_idx, k)
            end do
            du_max_CFL_row(I_idx) = max(du_max_CFL_row(I_idx), 0.0_dp)
            du_min_CFL_row(I_idx) = min(du_min_CFL_row(I_idx), 0.0_dp)
        end do

        ! Set uhbt target (use half of uh_tot_0 to exercise Newton)
        do I_idx = G%isc - 1, G%iec
            uhbt_row(I_idx) = 0.5_dp * uh_tot_0_row(I_idx)
        end do

        ! Call reference zonal_flux_adjust
        call zonal_flux_adjust(u, h, h_W, h_E, uhbt_row, uh_tot_0_row, duhdu_tot_0_row, &
            du_ref, du_max_CFL_row, du_min_CFL_row, dt, G, GV, CS, visc_rem_2d, &
            test_j, G%isc, G%iec, do_I, por_face_areaU)

        ! Compare with newton_flux_adjust_column for each face
        do I_idx = G%isc - 1, G%iec
            ! Extract column data
            do k = 1, nk
                u_col(k) = u(I_idx, test_j, k)
                h_col(k) = h(I_idx, test_j, k)
                h_ip1_col(k) = h(I_idx + 1, test_j, k)
                hW_col(k) = h_W(I_idx, test_j, k)
                hE_col(k) = h_E(I_idx, test_j, k)
                hW_ip1_col(k) = h_W(I_idx + 1, test_j, k)
                hE_ip1_col(k) = h_E(I_idx + 1, test_j, k)
                vr_col(k) = visc_rem_2d(I_idx, k)
                pf_col(k) = por_face_areaU(I_idx, test_j, k)
            end do

            call newton_flux_adjust_column(nk, &
                u_col, h_col, h_ip1_col, hW_col, hE_col, hW_ip1_col, hE_ip1_col, &
                vr_col, pf_col, &
                G%dy_Cu(I_idx, test_j), G%IareaT(I_idx, test_j), G%IareaT(I_idx+1, test_j), &
                G%IdxT(I_idx, test_j), G%IdxT(I_idx+1, test_j), &
                dt, CS%vol_CFL, CS%better_iter, CS%tol_eta, CS%tol_vel, &
                uhbt_row(I_idx), uh_tot_0_row(I_idx), duhdu_tot_0_row(I_idx), &
                du_max_CFL_row(I_idx), du_min_CFL_row(I_idx), &
                du_test, uh_col_out)

            if (abs(du_ref(I_idx)) > 0.0_dp) then
                rel_err = abs(du_test - du_ref(I_idx)) / abs(du_ref(I_idx))
            else
                rel_err = abs(du_test - du_ref(I_idx))
            end if
            max_du_err = max(max_du_err, rel_err)
        end do

        print '(A,ES15.8)', '  Max du relative error: ', max_du_err

        if (max_du_err < 1.0e-12_dp) then
            print '(A)', '  Status: PASS'
        else
            print '(A)', '  Status: FAIL'
            pass = .false.
        end if
    end subroutine test_newton_flux_adjust

    !> Test 3: Compare bt_cont_column against set_zonal_BT_cont
    subroutine test_bt_cont(pass)
        logical, intent(inout) :: pass

        ! Reference arrays
        real(dp), dimension(G%isd:G%ied) :: uh_ref_row, duhdu_ref_row
        real(dp), dimension(G%isd:G%ied) :: uh_tot_0_row, duhdu_tot_0_row
        real(dp), dimension(G%isd:G%ied) :: du_max_CFL_row, du_min_CFL_row
        real(dp), dimension(G%isd:G%ied) :: visc_rem_max_row
        real(dp), dimension(G%isd:G%ied, GV%ke) :: visc_rem_2d
        logical, dimension(G%isd:G%ied) :: do_I

        ! Column arrays
        real(dp) :: u_col(nk), h_col(nk), h_ip1_col(nk)
        real(dp) :: hW_col(nk), hE_col(nk), hW_ip1_col(nk), hE_ip1_col(nk)
        real(dp) :: vr_col(nk), pf_col(nk), uh_col_tmp(nk)

        ! For du0 comparison
        real(dp), dimension(G%isd:G%ied) :: zeros_row, du0_ref
        real(dp) :: du0_test, max_du0_err

        ! BT_cont outputs
        real(dp) :: test_W0, test_WW, test_E0, test_EE, test_uBT_WW, test_uBT_EE
        real(dp) :: max_W0_err, max_WW_err, max_E0_err, max_EE_err
        real(dp) :: max_uWW_err, max_uEE_err, rel_err, max_err
        real(dp) :: CFL_dt, I_dt, I_vrm, dx_W_loc, dx_E_loc
        real(dp) :: fa_scale  ! Scale for absolute tolerance on uBT

        integer :: test_j, I_idx

        print '(A)', ''
        print '(A)', 'Test 3: bt_cont_column vs set_zonal_BT_cont'
        print '(A)', '--------------------------------------------'

        test_j = (G%jsc + G%jec) / 2
        max_W0_err = 0.0_dp
        max_WW_err = 0.0_dp
        max_E0_err = 0.0_dp
        max_EE_err = 0.0_dp
        max_uWW_err = 0.0_dp
        max_uEE_err = 0.0_dp

        CFL_dt = CS%CFL_limit_adjust / dt
        I_dt = 1.0_dp / dt

        do i = G%isd, G%ied
            do_I(i) = .true.
        end do

        ! Build visc_rem_2d
        do i = G%isd, G%ied
            do k = 1, nk
                visc_rem_2d(i, k) = visc_rem_u(i, test_j, k)
            end do
        end do

        ! Compute initial fluxes
        uh_tot_0_row = 0.0_dp
        duhdu_tot_0_row = 0.0_dp
        do k = 1, nk
            uh_ref_row = 0.0_dp
            duhdu_ref_row = 0.0_dp
            call zonal_flux_layer(u(:, test_j, k), h(:, test_j, k), &
                h_W(:, test_j, k), h_E(:, test_j, k), &
                uh_ref_row, duhdu_ref_row, visc_rem_2d(:, k), &
                dt, G, test_j, G%isc, G%iec, do_I, CS%vol_CFL, &
                por_face_areaU(:, test_j, k))
            do I_idx = G%isc - 1, G%iec
                uh_tot_0_row(I_idx) = uh_tot_0_row(I_idx) + uh_ref_row(I_idx)
                duhdu_tot_0_row(I_idx) = duhdu_tot_0_row(I_idx) + duhdu_ref_row(I_idx)
            end do
        end do

        ! Compute visc_rem_max and CFL bounds
        do I_idx = G%isc - 1, G%iec
            visc_rem_max_row(I_idx) = 0.0_dp
            if (CS%use_visc_rem_max) then
                do k = 1, nk
                    visc_rem_max_row(I_idx) = max(visc_rem_max_row(I_idx), visc_rem_2d(I_idx, k))
                end do
            else
                visc_rem_max_row(I_idx) = 1.0_dp
            end if

            I_vrm = 0.0_dp
            if (visc_rem_max_row(I_idx) > 0.0_dp) I_vrm = 1.0_dp / visc_rem_max_row(I_idx)
            dx_W_loc = G%dxT(I_idx, test_j)
            dx_E_loc = G%dxT(I_idx + 1, test_j)
            du_max_CFL_row(I_idx) = 2.0_dp * (CFL_dt * dx_W_loc) * I_vrm
            du_min_CFL_row(I_idx) = -2.0_dp * (CFL_dt * dx_E_loc) * I_vrm

            do k = 1, nk
                if (du_max_CFL_row(I_idx) * visc_rem_2d(I_idx, k) > &
                    dx_W_loc * CFL_dt - u(I_idx, test_j, k) * G%mask2dCu(I_idx, test_j)) &
                    du_max_CFL_row(I_idx) = (dx_W_loc * CFL_dt - u(I_idx, test_j, k)) / visc_rem_2d(I_idx, k)
                if (du_min_CFL_row(I_idx) * visc_rem_2d(I_idx, k) < &
                    -dx_E_loc * CFL_dt - u(I_idx, test_j, k) * G%mask2dCu(I_idx, test_j)) &
                    du_min_CFL_row(I_idx) = -(dx_E_loc * CFL_dt + u(I_idx, test_j, k)) / visc_rem_2d(I_idx, k)
            end do
            du_max_CFL_row(I_idx) = max(du_max_CFL_row(I_idx), 0.0_dp)
            du_min_CFL_row(I_idx) = min(du_min_CFL_row(I_idx), 0.0_dp)
        end do

        ! Sub-test 3a: Compare du0 (Newton with target=0)
        zeros_row = 0.0_dp
        call zonal_flux_adjust(u, h, h_W, h_E, zeros_row, uh_tot_0_row, duhdu_tot_0_row, &
            du0_ref, du_max_CFL_row, du_min_CFL_row, dt, G, GV, CS, visc_rem_2d, &
            test_j, G%isc, G%iec, do_I, por_face_areaU)

        max_du0_err = 0.0_dp
        do I_idx = G%isc - 1, G%iec
            do k = 1, nk
                u_col(k) = u(I_idx, test_j, k)
                h_col(k) = h(I_idx, test_j, k)
                h_ip1_col(k) = h(I_idx + 1, test_j, k)
                hW_col(k) = h_W(I_idx, test_j, k)
                hE_col(k) = h_E(I_idx, test_j, k)
                hW_ip1_col(k) = h_W(I_idx + 1, test_j, k)
                hE_ip1_col(k) = h_E(I_idx + 1, test_j, k)
                vr_col(k) = visc_rem_2d(I_idx, k)
                pf_col(k) = por_face_areaU(I_idx, test_j, k)
            end do
            call newton_flux_adjust_column(nk, &
                u_col, h_col, h_ip1_col, hW_col, hE_col, hW_ip1_col, hE_ip1_col, &
                vr_col, pf_col, &
                G%dy_Cu(I_idx, test_j), G%IareaT(I_idx, test_j), G%IareaT(I_idx+1, test_j), &
                G%IdxT(I_idx, test_j), G%IdxT(I_idx+1, test_j), &
                dt, CS%vol_CFL, CS%better_iter, CS%tol_eta, CS%tol_vel, &
                0.0_dp, uh_tot_0_row(I_idx), duhdu_tot_0_row(I_idx), &
                du_max_CFL_row(I_idx), du_min_CFL_row(I_idx), &
                du0_test, uh_col_tmp)
            if (abs(du0_ref(I_idx)) > 0.0_dp) then
                rel_err = abs(du0_test - du0_ref(I_idx)) / abs(du0_ref(I_idx))
            else
                rel_err = abs(du0_test - du0_ref(I_idx))
            end if
            max_du0_err = max(max_du0_err, rel_err)
        end do
        print '(A,ES15.8)', '  du0 (target=0) max rel err: ', max_du0_err

        ! Call reference set_zonal_BT_cont
        call set_zonal_BT_cont(u, h, h_W, h_E, BT_cont, uh_tot_0_row, duhdu_tot_0_row, &
            du_max_CFL_row, du_min_CFL_row, dt, G, GV, CS, visc_rem_2d, &
            visc_rem_max_row, test_j, G%isc, G%iec, do_I, por_face_areaU)

        ! Compare with bt_cont_column for each face
        do I_idx = G%isc - 1, G%iec
            ! Extract column data
            do k = 1, nk
                u_col(k) = u(I_idx, test_j, k)
                h_col(k) = h(I_idx, test_j, k)
                h_ip1_col(k) = h(I_idx + 1, test_j, k)
                hW_col(k) = h_W(I_idx, test_j, k)
                hE_col(k) = h_E(I_idx, test_j, k)
                hW_ip1_col(k) = h_W(I_idx + 1, test_j, k)
                hE_ip1_col(k) = h_E(I_idx + 1, test_j, k)
                vr_col(k) = visc_rem_2d(I_idx, k)
                pf_col(k) = por_face_areaU(I_idx, test_j, k)
            end do

            call bt_cont_column(nk, &
                u_col, h_col, h_ip1_col, hW_col, hE_col, hW_ip1_col, hE_ip1_col, &
                vr_col, pf_col, visc_rem_max_row(I_idx), &
                G%dy_Cu(I_idx, test_j), G%dxCu(I_idx, test_j), &
                G%IareaT(I_idx, test_j), G%IareaT(I_idx+1, test_j), &
                G%IdxT(I_idx, test_j), G%IdxT(I_idx+1, test_j), &
                dt, CS%vol_CFL, CS%better_iter, CS%tol_eta, CS%tol_vel, &
                uh_tot_0_row(I_idx), duhdu_tot_0_row(I_idx), &
                du_max_CFL_row(I_idx), du_min_CFL_row(I_idx), &
                test_W0, test_WW, test_E0, test_EE, test_uBT_WW, test_uBT_EE)

            ! Compare FA_u_W0
            if (abs(BT_cont%FA_u_W0(I_idx, test_j)) > 0.0_dp) then
                rel_err = abs(test_W0 - BT_cont%FA_u_W0(I_idx, test_j)) / &
                          abs(BT_cont%FA_u_W0(I_idx, test_j))
            else
                rel_err = abs(test_W0)
            end if
            max_W0_err = max(max_W0_err, rel_err)

            ! Compare FA_u_WW
            if (abs(BT_cont%FA_u_WW(I_idx, test_j)) > 0.0_dp) then
                rel_err = abs(test_WW - BT_cont%FA_u_WW(I_idx, test_j)) / &
                          abs(BT_cont%FA_u_WW(I_idx, test_j))
            else
                rel_err = abs(test_WW)
            end if
            max_WW_err = max(max_WW_err, rel_err)

            ! Compare FA_u_E0
            if (abs(BT_cont%FA_u_E0(I_idx, test_j)) > 0.0_dp) then
                rel_err = abs(test_E0 - BT_cont%FA_u_E0(I_idx, test_j)) / &
                          abs(BT_cont%FA_u_E0(I_idx, test_j))
            else
                rel_err = abs(test_E0)
            end if
            max_E0_err = max(max_E0_err, rel_err)

            ! Compare FA_u_EE
            if (abs(BT_cont%FA_u_EE(I_idx, test_j)) > 0.0_dp) then
                rel_err = abs(test_EE - BT_cont%FA_u_EE(I_idx, test_j)) / &
                          abs(BT_cont%FA_u_EE(I_idx, test_j))
            else
                rel_err = abs(test_EE)
            end if
            max_EE_err = max(max_EE_err, rel_err)

            ! Compare uBT_WW — use absolute error scaled by FA magnitude
            ! uBT values are ill-conditioned (near-cancellation of large FA values)
            fa_scale = max(abs(test_W0), abs(BT_cont%FA_u_W0(I_idx, test_j)), 1.0_dp)
            rel_err = abs(test_uBT_WW - BT_cont%uBT_WW(I_idx, test_j)) / fa_scale
            max_uWW_err = max(max_uWW_err, rel_err)

            ! Compare uBT_EE
            fa_scale = max(abs(test_E0), abs(BT_cont%FA_u_E0(I_idx, test_j)), 1.0_dp)
            rel_err = abs(test_uBT_EE - BT_cont%uBT_EE(I_idx, test_j)) / fa_scale
            max_uEE_err = max(max_uEE_err, rel_err)
        end do

        print '(A,ES15.8)', '  Max FA_u_W0 rel error: ', max_W0_err
        print '(A,ES15.8)', '  Max FA_u_WW rel error: ', max_WW_err
        print '(A,ES15.8)', '  Max FA_u_E0 rel error: ', max_E0_err
        print '(A,ES15.8)', '  Max FA_u_EE rel error: ', max_EE_err
        print '(A,ES15.8)', '  Max uBT_WW  scaled error: ', max_uWW_err
        print '(A,ES15.8)', '  Max uBT_EE  scaled error: ', max_uEE_err

        ! FA values: 1e-10 tolerance (FMA rounding from row-vs-column flux eval)
        ! uBT values: already scaled by FA magnitude, so same tolerance applies
        max_err = max(max_W0_err, max_WW_err, max_E0_err, max_EE_err, max_uWW_err, max_uEE_err)
        if (max_err < 1.0e-10_dp) then
            print '(A)', '  Status: PASS'
        else
            print '(A)', '  Status: FAIL'
            pass = .false.
        end if
    end subroutine test_bt_cont

    subroutine init_verticalGrid(GV, nk)
        type(verticalGrid_type), intent(inout) :: GV
        integer, intent(in) :: nk
        GV%ke = nk
        GV%Angstrom_H = 1.0e-13_dp
    end subroutine init_verticalGrid

end program test_kernels_driver
