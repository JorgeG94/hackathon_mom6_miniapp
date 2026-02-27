!> Standalone driver for the vertical viscosity miniapp
program vert_visc_driver
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, RHO_0, &
                          mech_forcing_type, vertvisc_type, &
                          init_mech_forcing, end_mech_forcing, &
                          init_vertvisc_visc, end_vertvisc_visc
    use mom6_vert_visc, only: vert_visc_CS, vert_visc_init, vert_visc_coef, &
                              vert_visc_remnant, vert_visc_apply, vert_visc_end, &
                              vert_visc_coef_remnant_apply
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(vert_visc_CS) :: CS
    type(mech_forcing_type) :: forces
    type(vertvisc_type) :: visc

    real(dp), allocatable :: u(:, :, :), v(:, :, :), h(:, :, :)
    real(dp), allocatable :: u_init(:, :, :), v_init(:, :, :)
    real(dp), allocatable :: u_ref(:, :, :), v_ref(:, :, :)

    real(dp) :: dt, t_start, t_end, t_total, t_coef, t_apply, t_remnant
    real(dp) :: t_fused, t_fused_total
    real(dp) :: Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl
    real(dp), parameter :: PI = 3.14159265358979_dp
    integer :: ni, nj, nk, niter, iter, i, j, k
    integer :: clock_start, clock_end, clock_rate
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    Kv = 1.0e-4_dp          ! Interior viscosity [m2/s]
    Kv_ml = 1.0e-2_dp       ! Mixed layer viscosity [m2/s]
    Kv_extra_bbl = 1.0e-2_dp ! Extra BBL viscosity [m2/s]
    Hmix = 50.0_dp           ! Mixed layer depth [m]
    Hbbl = 10.0_dp           ! BBL thickness [m]

    ! Parse command line
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg); read (arg, *) ni
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg); read (arg, *) nj
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, arg); read (arg, *) nk
    end if
    if (command_argument_count() >= 4) then
        call get_command_argument(4, arg); read (arg, *) niter
    end if

    print '(A)', '=================================================='
    print '(A)', 'MOM6 Vertical Viscosity Mini-App'
    print '(A)', '=================================================='
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    print '(A,ES10.3)', 'Interior Kv [m2/s]: ', Kv
    print '(A,ES10.3)', 'Mixed layer Kv [m2/s]: ', Kv_ml
    print '(A,ES10.3)', 'Extra BBL Kv [m2/s]: ', Kv_extra_bbl
    print '(A,F8.1)', 'Mixed layer depth [m]: ', Hmix
    print '(A,F8.1)', 'BBL thickness [m]: ', Hbbl
    print '(A)', '=================================================='

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call vert_visc_init(CS, G, GV, Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl)

    ! Initialize forces (sinusoidal wind stress)
    call init_mech_forcing(forces, G)
    do j = G%jsd, G%jed
        do i = G%isd, G%ied
            forces%taux(i, j) = 0.1_dp * sin(real(j - 1, dp) / real(nj, dp) * PI)
            forces%tauy(i, j) = 0.0_dp
        end do
    end do

    ! Initialize visc (Rayleigh drag in bottom layer)
    call init_vertvisc_visc(visc, G, GV, use_rayleigh=.true.)
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                visc%Ray_u(i, j, k) = 0.0_dp
                visc%Ray_v(i, j, k) = 0.0_dp
            end do
        end do
    end do
    do j = G%jsd, G%jed
        do i = G%isd, G%ied
            visc%Ray_u(i, j, nk) = 1.0e-4_dp
            visc%Ray_v(i, j, nk) = 1.0e-4_dp
        end do
    end do

    dt = 300.0_dp  ! 5 minute timestep

    ! Allocate state arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (u_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (u_ref(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_ref(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Initialize state with vertical shear profile
    ! Surface-intensified flow that should be smoothed by viscosity
    do k=1,nk
      do j=G%jsd,G%jed
        do i=G%isd,G%ied
        ! Layer thickness (uniform)
        h(i, j, k) = 4000.0_dp / real(nk, dp)

        ! Velocity with strong vertical shear (surface jet)
        u_init(i, j, k) = 0.5_dp * exp(-real(k - 1, dp) / 10.0_dp) * &
                          sin(real(j - 1, dp) / real(nj, dp) * PI)
        v_init(i, j, k) = 0.3_dp * exp(-real(k - 1, dp) / 10.0_dp) * &
                          cos(real(i - 1, dp) / real(ni, dp) * PI)

        u(i, j, k) = u_init(i, j, k)
        v(i, j, k) = v_init(i, j, k)
        end do
      end do
    end do

    print '(A)', ''
    print '(A)', 'Running vertical viscosity solver...'

    t_total = 0.0_dp
    t_coef = 0.0_dp
    t_apply = 0.0_dp
    t_remnant = 0.0_dp

    do iter = 1, niter
        ! Reset velocities each iteration for timing consistency
        do k=1,nk
          do j=G%jsd,G%jed
            do i=G%isd,G%ied
            u(i, j, k) = u_init(i, j, k)
            v(i, j, k) = v_init(i, j, k)
            end do
          end do
        end do

        ! Compute coefficients (harmonic mean + upwind switching)
        call system_clock(clock_start, clock_rate)
        call vert_visc_coef(u, v, h, CS, G, GV)
        call system_clock(clock_end)
        t_coef = t_coef + real(clock_end - clock_start, dp) / real(clock_rate, dp)

        ! Compute remnant velocity fractions (with Rayleigh drag)
        call system_clock(clock_start, clock_rate)
        call vert_visc_remnant(dt, CS, G, GV, visc)
        call system_clock(clock_end)
        t_remnant = t_remnant + real(clock_end - clock_start, dp) / real(clock_rate, dp)

        ! Apply viscosity (with surface stress and Rayleigh drag)
        call system_clock(clock_start, clock_rate)
        call vert_visc_apply(u, v, h, dt, CS, G, GV, forces, visc)
        call system_clock(clock_end)
        t_apply = t_apply + real(clock_end - clock_start, dp) / real(clock_rate, dp)
    end do

    t_total = t_coef + t_remnant + t_apply

    print '(A)', ''
    print '(A)', '=================================================='
    print '(A)', 'Unfused Timing Results'
    print '(A)', '=================================================='
    print '(A,F12.6)', 'Total time (s):          ', t_total
    print '(A,F12.6)', '  Coefficient time:      ', t_coef
    print '(A,F12.6)', '  Remnant time:          ', t_remnant
    print '(A,F12.6)', '  Apply time:            ', t_apply
    print '(A,F12.6)', 'Time per iteration:      ', t_total / real(niter, dp)
    print '(A)', '=================================================='

    ! Save unfused results as reference
    do k=1,nk
      do j=G%jsd,G%jed
        do i=G%isd,G%ied
        u_ref(i, j, k) = u(i, j, k)
        v_ref(i, j, k) = v(i, j, k)
        end do
      end do
    end do

    ! =================================================================
    ! Fused benchmark: coef + remnant + apply in one kernel
    ! =================================================================
    print '(A)', ''
    print '(A)', 'Running FUSED vertical viscosity solver...'

    t_fused_total = 0.0_dp

    do iter = 1, niter
        ! Reset velocities each iteration
        do k=1,nk
          do j=G%jsd,G%jed
            do i=G%isd,G%ied
            u(i, j, k) = u_init(i, j, k)
            v(i, j, k) = v_init(i, j, k)
            end do
          end do
        end do

        call system_clock(clock_start, clock_rate)
        call vert_visc_coef_remnant_apply(u, v, h, dt, CS, G, GV, forces, visc)
        call system_clock(clock_end)
        t_fused_total = t_fused_total + real(clock_end - clock_start, dp) / real(clock_rate, dp)
    end do

    print '(A)', ''
    print '(A)', '=================================================='
    print '(A)', 'Fused Timing Results'
    print '(A)', '=================================================='
    print '(A,F12.6)', 'Fused total time (s):    ', t_fused_total
    print '(A,F12.6)', 'Fused time per iter:     ', t_fused_total / real(niter, dp)
    print '(A)', '=================================================='

    ! Compare fused vs unfused results
    call compare_results(u_ref, v_ref, u, v, G, GV)

    ! Summary
    print '(A)', ''
    print '(A)', '=================================================='
    print '(A)', 'Performance Comparison'
    print '(A)', '=================================================='
    print '(A,F12.6)', 'Unfused time/iter (s):   ', t_total / real(niter, dp)
    print '(A,F12.6)', 'Fused time/iter (s):     ', t_fused_total / real(niter, dp)
    if (t_fused_total > 0.0_dp) then
        print '(A,F12.2,A)', 'Speedup:                 ', t_total / t_fused_total, 'x'
    end if
    print '(A)', '=================================================='

    ! Verify that viscosity reduced shear
    call verify_viscosity(u_init, v_init, u, v, G, GV)

    ! Report remnant statistics
    call report_remnant(CS, G, GV)

    ! Report bottom stress
    call report_bottom_stress(CS, G)

    ! Cleanup
    call vert_visc_end(CS)
    call end_mech_forcing(forces)
    call end_vertvisc_visc(visc)
    call end_ocean_grid(G)
    deallocate (u, v, h, u_init, v_init, u_ref, v_ref)

contains

    subroutine verify_viscosity(u_init, v_init, u_final, v_final, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u_init, v_init
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u_final, v_final

        real(dp) :: shear_init, shear_final, shear_reduction
        real(dp) :: max_u_init, max_u_final, max_v_init, max_v_final
        integer :: i, j, k

        shear_init = 0.0_dp; shear_final = 0.0_dp
        max_u_init = 0.0_dp; max_u_final = 0.0_dp
        max_v_init = 0.0_dp; max_v_final = 0.0_dp

        ! Compute vertical shear (du/dz) and max velocities
        do k = 1, GV%ke - 1
            do j = G%jsc, G%jec
                do i = G%isc, G%iec
                    shear_init = shear_init + (u_init(i, j, k) - u_init(i, j, k + 1))**2 + &
                                 (v_init(i, j, k) - v_init(i, j, k + 1))**2
                    shear_final = shear_final + (u_final(i, j, k) - u_final(i, j, k + 1))**2 + &
                                  (v_final(i, j, k) - v_final(i, j, k + 1))**2
                end do
            end do
        end do

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc, G%iec
                    max_u_init = max(max_u_init, abs(u_init(i, j, k)))
                    max_u_final = max(max_u_final, abs(u_final(i, j, k)))
                    max_v_init = max(max_v_init, abs(v_init(i, j, k)))
                    max_v_final = max(max_v_final, abs(v_final(i, j, k)))
                end do
            end do
        end do

        shear_init = sqrt(shear_init)
        shear_final = sqrt(shear_final)
        shear_reduction = (shear_init - shear_final) / shear_init * 100.0_dp

        print '(A)', ''
        print '(A)', 'Viscosity Verification:'
        print '(A)', '--------------------------------------------------'
        print '(A,ES15.8)', '  Initial shear (RMS):  ', shear_init
        print '(A,ES15.8)', '  Final shear (RMS):    ', shear_final
        print '(A,F10.2,A)', '  Shear reduction:      ', shear_reduction, '%'
        print '(A)', ''
        print '(A,ES12.5,A,ES12.5)', '  Max |u|: ', max_u_init, ' -> ', max_u_final
        print '(A,ES12.5,A,ES12.5)', '  Max |v|: ', max_v_init, ' -> ', max_v_final

        if (shear_final < shear_init) then
            print '(A)', ''
            print '(A)', '  Status: PASS (viscosity reduced shear)'
        else
            print '(A)', ''
            print '(A)', '  Status: WARNING (shear not reduced)'
        end if
    end subroutine verify_viscosity

    subroutine report_remnant(CS, G, GV)
        type(vert_visc_CS), intent(in) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV

        real(dp) :: rem_min, rem_max, rem_avg, rem_surf, rem_bot
        integer :: i, j, k, cnt
        integer :: i_mid, j_mid

        ! Sample from middle of domain
        i_mid = (G%isc + G%iec) / 2
        j_mid = (G%jsc + G%jec) / 2

        rem_min = 1.0_dp
        rem_max = 0.0_dp
        rem_avg = 0.0_dp
        cnt = 0

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc - 1, G%iec
                    if (G%mask2dCu(i, j) > 0.0_dp) then
                        rem_min = min(rem_min, CS%visc_rem_u(i, j, k))
                        rem_max = max(rem_max, CS%visc_rem_u(i, j, k))
                        rem_avg = rem_avg + CS%visc_rem_u(i, j, k)
                        cnt = cnt + 1
                    end if
                end do
            end do
        end do

        if (cnt > 0) rem_avg = rem_avg / real(cnt, dp)

        rem_surf = CS%visc_rem_u(i_mid, j_mid, 1)
        rem_bot = CS%visc_rem_u(i_mid, j_mid, GV%ke)

        print '(A)', ''
        print '(A)', 'Viscosity Remnant (visc_rem_u):'
        print '(A)', '--------------------------------------------------'
        print '(A,F10.6)', '  Min remnant:          ', rem_min
        print '(A,F10.6)', '  Max remnant:          ', rem_max
        print '(A,F10.6)', '  Avg remnant:          ', rem_avg
        print '(A,F10.6)', '  Surface (k=1):        ', rem_surf
        print '(A,F10.6)', '  Bottom (k=nz):        ', rem_bot
        print '(A)', ''
        print '(A)', '  (Remnant = fraction of BT acceleration retained)'
        print '(A)', '  (Values near 1.0 = weak viscosity coupling)'
        print '(A)', '  (Values near 0.0 = strong viscosity coupling)'

    end subroutine report_remnant

    subroutine report_bottom_stress(CS, G)
        type(vert_visc_CS), intent(in) :: CS
        type(ocean_grid_type), intent(in) :: G

        real(dp) :: taux_min, taux_max, tauy_min, tauy_max
        integer :: i, j

        taux_min = huge(1.0_dp); taux_max = -huge(1.0_dp)
        tauy_min = huge(1.0_dp); tauy_max = -huge(1.0_dp)

        do j = G%jsc, G%jec
            do i = G%isc - 1, G%iec
                if (G%mask2dCu(i, j) > 0.0_dp) then
                    taux_min = min(taux_min, CS%taux_bot(i, j))
                    taux_max = max(taux_max, CS%taux_bot(i, j))
                end if
            end do
        end do

        do j = G%jsc - 1, G%jec
            do i = G%isc, G%iec
                if (G%mask2dCv(i, j) > 0.0_dp) then
                    tauy_min = min(tauy_min, CS%tauy_bot(i, j))
                    tauy_max = max(tauy_max, CS%tauy_bot(i, j))
                end if
            end do
        end do

        print '(A)', ''
        print '(A)', 'Bottom Stress [Pa]:'
        print '(A)', '--------------------------------------------------'
        print '(A,ES12.5,A,ES12.5)', '  taux_bot: min = ', taux_min, '  max = ', taux_max
        print '(A,ES12.5,A,ES12.5)', '  tauy_bot: min = ', tauy_min, '  max = ', tauy_max

    end subroutine report_bottom_stress

    subroutine compare_results(u_ref, v_ref, u_fused, v_fused, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u_ref, v_ref
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u_fused, v_fused

        real(dp) :: max_u_err, max_v_err, max_u_rel, max_v_rel
        real(dp) :: diff, ref_val
        integer :: i, j, k

        max_u_err = 0.0_dp; max_v_err = 0.0_dp
        max_u_rel = 0.0_dp; max_v_rel = 0.0_dp

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc - 1, G%iec
                    diff = abs(u_fused(i, j, k) - u_ref(i, j, k))
                    max_u_err = max(max_u_err, diff)
                    ref_val = abs(u_ref(i, j, k))
                    if (ref_val > 1.0e-30_dp) then
                        max_u_rel = max(max_u_rel, diff / ref_val)
                    end if
                end do
            end do
        end do

        do k = 1, GV%ke
            do j = G%jsc - 1, G%jec
                do i = G%isc, G%iec
                    diff = abs(v_fused(i, j, k) - v_ref(i, j, k))
                    max_v_err = max(max_v_err, diff)
                    ref_val = abs(v_ref(i, j, k))
                    if (ref_val > 1.0e-30_dp) then
                        max_v_rel = max(max_v_rel, diff / ref_val)
                    end if
                end do
            end do
        end do

        print '(A)', ''
        print '(A)', 'Fused vs Unfused Comparison:'
        print '(A)', '--------------------------------------------------'
        print '(A,ES15.8)', '  Max |u_fused - u_ref|:     ', max_u_err
        print '(A,ES15.8)', '  Max |v_fused - v_ref|:     ', max_v_err
        print '(A,ES15.8)', '  Max rel error (u):         ', max_u_rel
        print '(A,ES15.8)', '  Max rel error (v):         ', max_v_rel

        if (max_u_rel < 1.0e-14_dp .and. max_v_rel < 1.0e-14_dp) then
            print '(A)', '  Status: PASS (bit-identical or < 1e-14 relative error)'
        else if (max_u_rel < 1.0e-10_dp .and. max_v_rel < 1.0e-10_dp) then
            print '(A)', '  Status: ACCEPTABLE (< 1e-10 relative error)'
        else
            print '(A)', '  Status: WARNING (significant difference detected)'
        end if

    end subroutine compare_results

end program vert_visc_driver
