!> Standalone driver for the Coriolis/momentum advection miniapp
program coriolis_driver
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid
    use mom6_coriolis, only: coriolis_CS, coriolis_init, CorAdCalc, coriolis_end, &
                             SADOURNY75_ENERGY, ARAKAWA_HSU90, ARAKAWA_LAMB81
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(coriolis_CS) :: CS

    real(dp), allocatable :: u(:, :, :), v(:, :, :), h(:, :, :)
    real(dp), allocatable :: uh(:, :, :), vh(:, :, :)
    real(dp), allocatable :: CAu(:, :, :), CAv(:, :, :)

    real(dp) :: t_start, t_end, t_total
    integer :: ni, nj, nk, niter, iter, i, j, k, scheme
    integer :: clock_start, clock_end, clock_rate
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    scheme = SADOURNY75_ENERGY

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
    if (command_argument_count() >= 5) then
        call get_command_argument(5, arg)
        select case (trim(arg))
        case ('sadourny', 'SADOURNY', '1')
            scheme = SADOURNY75_ENERGY
        case ('hsu', 'HSU', '2')
            scheme = ARAKAWA_HSU90
        case ('lamb', 'LAMB', '3')
            scheme = ARAKAWA_LAMB81
        end select
    end if

    print '(A)', '=================================================='
    print '(A)', 'MOM6 Coriolis Mini-App'
    print '(A)', '=================================================='
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    select case (scheme)
    case (SADOURNY75_ENERGY)
        print '(A)', 'Scheme: Sadourny (1975) Energy-conserving'
    case (ARAKAWA_HSU90)
        print '(A)', 'Scheme: Arakawa-Hsu (1990)'
    case (ARAKAWA_LAMB81)
        print '(A)', 'Scheme: Arakawa-Lamb (1981)'
    end select
    print '(A)', '=================================================='

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call coriolis_init(CS, G, scheme)

    ! Allocate state arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Initialize state with realistic patterns
    do k=1,nk
      do j=G%jsd,G%jed
        do i=G%isd,G%ied
        h(i, j, k) = 4000.0_dp/real(nk, dp)
        u(i, j, k) = 0.1_dp*sin(real(j - 1, dp)/real(nj, dp)*3.14159_dp*2.0_dp)* &
                     exp(-real(k, dp)/30.0_dp)
        v(i, j, k) = 0.1_dp*cos(real(i - 1, dp)/real(ni, dp)*3.14159_dp*2.0_dp)* &
                     exp(-real(k, dp)/30.0_dp)
        uh(i, j, k) = u(i, j, k)*h(i, j, k)*G%dyCu(i, j)
        vh(i, j, k) = v(i, j, k)*h(i, j, k)*G%dxCv(i, j)
        end do
      end do
    end do

    print '(A)', ''
    print '(A)', 'Running Coriolis solver...'

    t_total = 0.0_dp

    do iter = 1, niter
        call system_clock(clock_start, clock_rate)
        call CorAdCalc(u, v, h, uh, vh, CAu, CAv, G, GV, CS)
        call system_clock(clock_end)

        t_total = t_total + real(clock_end - clock_start, dp) / real(clock_rate, dp)
    end do

    print '(A)', ''
    print '(A)', '=================================================='
    print '(A,F12.6)', 'Total time (s):      ', t_total
    print '(A,F12.6)', 'Time per iteration:  ', t_total/real(niter, dp)
    print '(A)', '=================================================='

    ! Verify results
    call verify_acceleration(CAu, CAv, G, GV)

    ! Cleanup
    call coriolis_end(CS)
    call end_ocean_grid(G)
    deallocate (u, v, h, uh, vh, CAu, CAv)

contains

    subroutine verify_acceleration(CAu, CAv, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: CAu, CAv

        real(dp) :: max_CAu, max_CAv, avg_CAu, avg_CAv
        real(dp) :: sum_CAu, sum_CAv
        integer :: i, j, k, count

        max_CAu = 0.0_dp; max_CAv = 0.0_dp
        sum_CAu = 0.0_dp; sum_CAv = 0.0_dp
        count = 0

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc, G%iec - 1
                    max_CAu = max(max_CAu, abs(CAu(i, j, k)))
                    sum_CAu = sum_CAu + abs(CAu(i, j, k))
                end do
            end do
            do j = G%jsc, G%jec - 1
                do i = G%isc, G%iec
                    max_CAv = max(max_CAv, abs(CAv(i, j, k)))
                    sum_CAv = sum_CAv + abs(CAv(i, j, k))
                    count = count + 1
                end do
            end do
        end do

        avg_CAu = sum_CAu/real(count, dp)
        avg_CAv = sum_CAv/real(count, dp)

        print '(A)', ''
        print '(A)', 'Acceleration Statistics:'
        print '(A,ES15.8)', '  Max |CAu|: ', max_CAu
        print '(A,ES15.8)', '  Max |CAv|: ', max_CAv
        print '(A,ES15.8)', '  Avg |CAu|: ', avg_CAu
        print '(A,ES15.8)', '  Avg |CAv|: ', avg_CAv

        if (max_CAu > 0.0_dp .and. max_CAv > 0.0_dp) then
            print '(A)', '  Status: PASS (non-zero accelerations computed)'
        else
            print '(A)', '  Status: WARNING (zero accelerations)'
        end if
    end subroutine verify_acceleration

end program coriolis_driver
