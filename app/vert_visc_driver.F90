!> Standalone driver for the vertical viscosity miniapp
program vert_visc_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid
    use mom6_vert_visc, only: vert_visc_CS, vert_visc_init, vert_visc_coef, &
                              vert_visc_remnant, vert_visc_apply, vert_visc_end
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(vert_visc_CS) :: CS

    real(dp), allocatable :: u(:, :, :), v(:, :, :), h(:, :, :)
    real(dp), allocatable :: u_init(:, :, :), v_init(:, :, :)

    real(dp) :: dt, t_start, t_end, t_total, t_coef, t_apply, t_remnant
    real(dp) :: Kv, Kv_ml, Kv_bbl, Hmix, Hbbl
    integer :: ni, nj, nk, niter, iter, i, j, k
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    Kv = 1.0e-4_dp      ! Interior viscosity [m2/s]
    Kv_ml = 1.0e-2_dp   ! Mixed layer viscosity [m2/s]
    Kv_bbl = 1.0e-2_dp  ! Bottom boundary layer viscosity [m2/s]
    Hmix = 50.0_dp      ! Mixed layer depth [m]
    Hbbl = 10.0_dp      ! BBL thickness [m]

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
    print '(A,ES10.3)', 'BBL Kv [m2/s]: ', Kv_bbl
    print '(A,F8.1)', 'Mixed layer depth [m]: ', Hmix
    print '(A,F8.1)', 'BBL thickness [m]: ', Hbbl
    print '(A)', '=================================================='

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call vert_visc_init(CS, G, GV, Kv, Kv_ml, Kv_bbl, Hmix, Hbbl)

    dt = 300.0_dp  ! 5 minute timestep

    ! Allocate state arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (u_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_init(G%isd:G%ied, G%jsd:G%jed, nk))

    !$omp target enter data map(alloc: u, v, h, u_init, v_init)

    ! Initialize state with vertical shear profile
    ! Surface-intensified flow that should be smoothed by viscosity
    do concurrent(k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
        ! Layer thickness (uniform)
        h(i, j, k) = 4000.0_dp/real(nk, dp)

        ! Velocity with strong vertical shear (surface jet)
        u_init(i, j, k) = 0.5_dp*exp(-real(k - 1, dp)/10.0_dp)* &
                          sin(real(j - 1, dp)/real(nj, dp)*3.14159_dp)
        v_init(i, j, k) = 0.3_dp*exp(-real(k - 1, dp)/10.0_dp)* &
                          cos(real(i - 1, dp)/real(ni, dp)*3.14159_dp)

        u(i, j, k) = u_init(i, j, k)
        v(i, j, k) = v_init(i, j, k)
    end do

    !$omp target update to(u, v, h)

    print '(A)', ''
    print '(A)', 'Running vertical viscosity solver...'

    t_total = 0.0_dp
    t_coef = 0.0_dp
    t_apply = 0.0_dp
    t_remnant = 0.0_dp

    do iter = 1, niter
        ! Reset velocities each iteration for timing consistency
        do concurrent(k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
            u(i, j, k) = u_init(i, j, k)
            v(i, j, k) = v_init(i, j, k)
        end do
        !$omp target update to(u, v)

        ! Compute coefficients (uses find_coupling_coef internally)
        t_start = omp_get_wtime()
        call vert_visc_coef(u, v, h, CS, G, GV)
        t_end = omp_get_wtime()
        t_coef = t_coef + (t_end - t_start)

        ! Compute remnant velocity fractions (vertvisc_remnant)
        t_start = omp_get_wtime()
        call vert_visc_remnant(dt, CS, G, GV)
        t_end = omp_get_wtime()
        t_remnant = t_remnant + (t_end - t_start)

        ! Apply viscosity
        t_start = omp_get_wtime()
        call vert_visc_apply(u, v, h, dt, CS, G, GV)
        t_end = omp_get_wtime()
        t_apply = t_apply + (t_end - t_start)
    end do

    t_total = t_coef + t_remnant + t_apply

    !$omp target exit data map(from: u, v)
    !$omp target exit data map(delete: h, u_init, v_init)

    print '(A)', ''
    print '(A)', '=================================================='
    print '(A)', 'Timing Results'
    print '(A)', '=================================================='
    print '(A,F12.6)', 'Total time (s):          ', t_total
    print '(A,F12.6)', '  Coefficient time:      ', t_coef
    print '(A,F12.6)', '  Remnant time:          ', t_remnant
    print '(A,F12.6)', '  Apply time:            ', t_apply
    print '(A,F12.6)', 'Time per iteration:      ', t_total/real(niter, dp)
    print '(A)', '=================================================='

    ! Verify that viscosity reduced shear
    call verify_viscosity(u_init, v_init, u, v, G, GV)

    ! Report remnant statistics
    call report_remnant(CS, G, GV)

    ! Cleanup
    call vert_visc_end(CS)
    call end_ocean_grid(G)
    deallocate (u, v, h, u_init, v_init)

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
        shear_reduction = (shear_init - shear_final)/shear_init*100.0_dp

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
        i_mid = (G%isc + G%iec)/2
        j_mid = (G%jsc + G%jec)/2

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

        if (cnt > 0) rem_avg = rem_avg/real(cnt, dp)

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

end program vert_visc_driver
