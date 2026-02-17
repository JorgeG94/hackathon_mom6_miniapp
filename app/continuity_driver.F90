!> Standalone driver for the continuity miniapp
program continuity_driver
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, BT_cont_type, alloc_BT_cont_type
    use mom6_continuity, only: continuity_CS, continuity_init, continuity_PPM, continuity_end
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(continuity_CS) :: CS
    type(BT_cont_type), pointer :: BT_cont

    real(dp), allocatable :: h(:, :, :), hin(:, :, :), u(:, :, :), por_face_areaU(:, :, :), visc_rem_u(:, :, :)
    real(dp), allocatable :: uh(:, :, :), uhbt(:, :), u_cor(:, :, :), du_cor(:, :)

    real(dp) :: dt, t_start, t_end, t_total
    integer :: ni, nj, nk, niter, iter, i, j, k
    integer :: clock_start, clock_end, clock_rate
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10

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
    print '(A)', 'MOM6 Continuity Mini-App'
    print '(A)', '=================================================='
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    print '(A)', '=================================================='

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call continuity_init(CS, G, GV, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call alloc_BT_cont_type(BT_cont, G, GV)

    dt = 300.0_dp  ! 5 minute timestep

    ! Allocate state arrays
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (hin(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Initialize state
    do k=1,nk
      do j=G%jsd,G%jed
        do i=G%isd,G%ied
        hin(i, j, k) = 4000.0_dp/real(nk, dp) + 10.0_dp*sin(real(i - 1, dp)/real(ni, dp)*3.14159_dp)* &
                       cos(real(j - 1, dp)/real(nj, dp)*3.14159_dp)*exp(-real(k, dp)/20.0_dp)
        h(i, j, k) = hin(i, j, k)
        u(i, j, k) = 0.1_dp*sin(real(j - 1, dp)/real(nj, dp)*3.14159_dp*2.0_dp)*exp(-real(k, dp)/30.0_dp)
        end do
      end do
    end do

    print *, G%jsc, G%jec, G%isc, G%iec
    print *, sum(hin(G%isc:G%iec, G%jsc:G%jec, :)), sum(h(G%isc:G%iec, G%jsc:G%jec, :))

    print '(A)', ''
    print '(A)', 'Running continuity solver...'

    t_total = 0.0_dp

    do iter = 1, niter
        ! Reset state
        do k=1,nk
          do j=G%jsd,G%jed
            do i=G%isd,G%ied
            h(i, j, k) = hin(i, j, k)
            end do
          end do
        end do

        call system_clock(clock_start, clock_rate)
        call continuity_PPM(u, hin, h, uh, dt, G, GV, CS, por_face_areaU, uhbt, visc_rem_u, u_cor, BT_cont, du_cor)
        call system_clock(clock_end)

        t_total = t_total + real(clock_end - clock_start, dp) / real(clock_rate, dp)
    end do

    print '(A)', ''
    print '(A)', '=================================================='
    print '(A,F12.6)', 'Total time (s):      ', t_total
    print '(A,F12.6)', 'Time per iteration:  ', t_total/real(niter, dp)
    print '(A)', '=================================================='

    ! Verify
    call verify_mass(hin, h, G, GV)

    ! Cleanup
    call continuity_end(CS, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call end_ocean_grid(G)
    deallocate (h, hin, u, uh)

contains

    subroutine verify_mass(h_init, h_final, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_init, h_final

        real(dp) :: mass_init, mass_final, rel_error
        integer :: i, j, k

        mass_init = 0.0_dp; mass_final = 0.0_dp

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc, G%iec
                    mass_init = mass_init + h_init(i, j, k)
                    mass_final = mass_final + h_final(i, j, k)
                end do
            end do
        end do

        rel_error = abs(mass_final - mass_init)/mass_init

        print '(A)', ''
        print '(A)', 'Mass Conservation Check:'
        print '(A,ES15.8)', '  Initial mass:  ', mass_init
        print '(A,ES15.8)', '  Final mass:    ', mass_final
        print '(A,ES15.8)', '  Relative error:', rel_error

        if (rel_error < 1.0e-10_dp) then
            print '(A)', '  Status: PASS'
        else
            print '(A)', '  Status: WARNING'
        end if
    end subroutine verify_mass

end program continuity_driver
