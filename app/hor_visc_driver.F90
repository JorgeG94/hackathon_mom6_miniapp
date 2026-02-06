!> Standalone driver for the horizontal viscosity miniapp
!!
!! This driver demonstrates realistic MOM6 horizontal viscosity computation
!! with multiple physics schemes and mixed GPU/CPU execution patterns.
!!
!! Usage:
!!   ./hor_visc_driver [ni] [nj] [nk] [niter] [options]
!!
!! Options:
!!   --laplacian    Enable Laplacian viscosity only (default)
!!   --biharmonic   Enable biharmonic (4th order) viscosity
!!   --smagorinsky  Enable Smagorinsky dynamic viscosity
!!   --leith        Enable Leith vorticity-gradient viscosity (CPU/GPU mixed)
!!   --no-slip      Use no-slip boundary conditions (vs free-slip)
!!   --better-bound Use thickness-aware stability bounding
!!   --full         Enable all schemes (biharmonic + smagorinsky + leith + better-bound)
!!   --help         Show this help message
!!
!! Examples:
!!   ./hor_visc_driver 180 180 75 10              # Simple Laplacian
!!   ./hor_visc_driver 180 180 75 10 --biharmonic --smagorinsky
!!   ./hor_visc_driver 180 180 75 10 --leith      # Demonstrates CPU/GPU mixed execution
!!   ./hor_visc_driver 180 180 75 10 --full       # All features enabled
!!
program hor_visc_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid
    use mom6_hor_visc, only: hor_visc_CS, hor_visc_init, hor_visc, hor_visc_end
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(hor_visc_CS) :: CS

    real(dp), allocatable :: u(:, :, :), v(:, :, :), h(:, :, :)
    real(dp), allocatable :: u_init(:, :, :), v_init(:, :, :)
    real(dp), allocatable :: diffu(:, :, :), diffv(:, :, :)

    real(dp) :: t_start, t_end, t_total
    real(dp) :: Kh, Ah
    integer :: ni, nj, nk, niter, iter, i, j, k, arg_idx
    character(len=64) :: arg
    character(len=256) :: schemes_str

    ! Scheme flags
    logical :: use_laplacian, use_biharmonic, use_smagorinsky, use_leith
    logical :: use_no_slip, use_better_bound

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    Kh = 100.0_dp     ! Laplacian viscosity [m2/s]
    Ah = 1.0e10_dp    ! Biharmonic viscosity [m4/s]

    ! Default scheme selection
    use_laplacian = .true.
    use_biharmonic = .false.
    use_smagorinsky = .false.
    use_leith = .false.
    use_no_slip = .false.
    use_better_bound = .false.

    ! Parse command line arguments
    arg_idx = 1
    do while (arg_idx <= command_argument_count())
        call get_command_argument(arg_idx, arg)

        select case (trim(arg))
        case ('--help', '-h')
            call print_help()
            stop

        case ('--laplacian')
            use_laplacian = .true.

        case ('--biharmonic')
            use_biharmonic = .true.

        case ('--smagorinsky')
            use_smagorinsky = .true.

        case ('--leith')
            use_leith = .true.

        case ('--no-slip')
            use_no_slip = .true.

        case ('--better-bound')
            use_better_bound = .true.

        case ('--full')
            use_biharmonic = .true.
            use_smagorinsky = .true.
            use_leith = .true.
            use_better_bound = .true.

        case default
            ! Assume it's a positional argument (ni, nj, nk, niter)
            if (arg(1:1) == '-') then
                print '(A,A)', 'Unknown option: ', trim(arg)
                call print_help()
                stop 1
            end if

            ! Parse positional arguments
            select case (arg_idx)
            case (1)
                read (arg, *) ni
            case (2)
                read (arg, *) nj
            case (3)
                read (arg, *) nk
            case (4)
                read (arg, *) niter
            case default
                ! Skip extra positional arguments
            end select
        end select

        arg_idx = arg_idx + 1
    end do

    ! Build schemes string for display
    schemes_str = ''
    if (use_laplacian) schemes_str = trim(schemes_str)//' Laplacian'
    if (use_biharmonic) schemes_str = trim(schemes_str)//' Biharmonic'
    if (use_smagorinsky) schemes_str = trim(schemes_str)//' Smagorinsky'
    if (use_leith) schemes_str = trim(schemes_str)//' Leith'
    if (use_no_slip) schemes_str = trim(schemes_str)//' No-Slip'
    if (use_better_bound) schemes_str = trim(schemes_str)//' BetterBound'
    if (len_trim(schemes_str) == 0) schemes_str = ' Laplacian'

    print '(A)', '=================================================================='
    print '(A)', 'MOM6 Horizontal Viscosity Mini-App (Enhanced)'
    print '(A)', '=================================================================='
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    print '(A,A)', 'Schemes:', trim(schemes_str)
    if (use_laplacian) print '(A,ES12.4)', '  Kh (Laplacian) [m2/s]:  ', Kh
    if (use_biharmonic) print '(A,ES12.4)', '  Ah (Biharmonic) [m4/s]: ', Ah
    print '(A)', '=================================================================='

    if (use_leith) then
        print '(A)', ''
        print '(A)', 'NOTE: Leith viscosity uses CPU-only vorticity gradient calculation'
        print '(A)', '      with explicit GPU<->CPU data transfers (!$omp target update).'
        print '(A)', '      This demonstrates the real MOM6 GPU porting challenge.'
        print '(A)', ''
    end if

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! Initialize horizontal viscosity solver with selected schemes
    call hor_visc_init(CS, G, GV, Kh=Kh, Ah=Ah, &
                       Laplacian=use_laplacian, &
                       biharmonic=use_biharmonic, &
                       Smagorinsky=use_smagorinsky, &
                       Leith=use_leith, &
                       no_slip=use_no_slip, &
                       better_bound=use_better_bound)

    ! Allocate state arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (u_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffu(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffv(G%isd:G%ied, G%jsd:G%jed, nk))

    !$omp target enter data map(alloc: u, v, h, u_init, v_init, diffu, diffv)

    ! Initialize state with horizontal structure that should be smoothed
    ! Use different patterns to exercise different viscosity schemes
    do concurrent(k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
        ! Layer thickness with some variation (for better_bound testing)
        h(i, j, k) = 4000.0_dp/real(nk, dp)*(1.0_dp + &
                                             0.1_dp*sin(real(i - 1, dp)/real(ni, dp)*6.28318_dp)* &
                                             sin(real(j - 1, dp)/real(nj, dp)*6.28318_dp))

        ! Velocity with jets (large scale) + eddies (small scale)
        ! Jets: for Laplacian/Smagorinsky testing
        ! Eddies: for biharmonic small-scale dissipation testing
        ! Vorticity structure: for Leith testing
        u_init(i, j, k) = 0.5_dp*sin(real(j - 1, dp)/real(nj, dp)*6.28318_dp)* &
                          exp(-real(k - 1, dp)/30.0_dp) + &
                          0.2_dp*sin(real(i - 1, dp)/real(ni, dp)*12.56637_dp)* &
                          sin(real(j - 1, dp)/real(nj, dp)*12.56637_dp) + &
                          0.1_dp*sin(real(i - 1, dp)/real(ni, dp)*25.13274_dp)* &
                          sin(real(j - 1, dp)/real(nj, dp)*25.13274_dp)

        v_init(i, j, k) = 0.3_dp*cos(real(i - 1, dp)/real(ni, dp)*6.28318_dp)* &
                          exp(-real(k - 1, dp)/30.0_dp) + &
                          0.2_dp*cos(real(i - 1, dp)/real(ni, dp)*12.56637_dp)* &
                          cos(real(j - 1, dp)/real(nj, dp)*12.56637_dp) + &
                          0.1_dp*cos(real(i - 1, dp)/real(ni, dp)*25.13274_dp)* &
                          cos(real(j - 1, dp)/real(nj, dp)*25.13274_dp)

        u(i, j, k) = u_init(i, j, k)
        v(i, j, k) = v_init(i, j, k)
    end do

    !$omp target update to(u, v, h)

    print '(A)', ''
    print '(A)', 'Running horizontal viscosity solver...'

    t_total = 0.0_dp

    do iter = 1, niter
        ! Reset velocities each iteration for timing consistency
        do concurrent(k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
            u(i, j, k) = u_init(i, j, k)
            v(i, j, k) = v_init(i, j, k)
        end do
        !$omp target update to(u, v)

        ! Compute horizontal viscous accelerations
        t_start = omp_get_wtime()
        call hor_visc(u, v, h, diffu, diffv, G, GV, CS)
        t_end = omp_get_wtime()
        t_total = t_total + (t_end - t_start)
    end do

    !$omp target exit data map(from: u, v, diffu, diffv)
    !$omp target exit data map(delete: h, u_init, v_init)

    print '(A)', ''
    print '(A)', '=================================================================='
    print '(A)', 'Timing Results'
    print '(A)', '=================================================================='
    print '(A,F12.6)', 'Total time (s):          ', t_total
    print '(A,F12.6)', 'Time per iteration:      ', t_total/real(niter, dp)
    print '(A)', '=================================================================='

    ! Verify viscous accelerations
    call verify_hor_visc(u_init, v_init, diffu, diffv, G, GV, CS)

    ! Cleanup
    call hor_visc_end(CS)
    call end_ocean_grid(G)
    deallocate (u, v, h, u_init, v_init, diffu, diffv)

contains

    subroutine print_help()
        print '(A)', 'MOM6 Horizontal Viscosity Mini-App'
        print '(A)', ''
        print '(A)', 'Usage: ./hor_visc_driver [ni] [nj] [nk] [niter] [options]'
        print '(A)', ''
        print '(A)', 'Positional arguments:'
        print '(A)', '  ni        Grid size in x-direction (default: 180)'
        print '(A)', '  nj        Grid size in y-direction (default: 180)'
        print '(A)', '  nk        Number of vertical layers (default: 75)'
        print '(A)', '  niter     Number of iterations (default: 10)'
        print '(A)', ''
        print '(A)', 'Options:'
        print '(A)', '  --laplacian     Enable Laplacian viscosity only (default)'
        print '(A)', '  --biharmonic    Enable biharmonic (4th order) viscosity'
        print '(A)', '  --smagorinsky   Enable Smagorinsky dynamic viscosity'
        print '(A)', '  --leith         Enable Leith vorticity-gradient viscosity'
        print '(A)', '  --no-slip       Use no-slip boundary conditions'
        print '(A)', '  --better-bound  Use thickness-aware stability bounding'
        print '(A)', '  --full          Enable all schemes'
        print '(A)', '  --help, -h      Show this help message'
        print '(A)', ''
        print '(A)', 'Examples:'
        print '(A)', '  ./hor_visc_driver 180 180 75 10'
        print '(A)', '  ./hor_visc_driver 180 180 75 10 --biharmonic --smagorinsky'
        print '(A)', '  ./hor_visc_driver 180 180 75 10 --leith'
        print '(A)', '  ./hor_visc_driver 180 180 75 10 --full'
        print '(A)', ''
        print '(A)', 'The --leith option demonstrates mixed GPU/CPU execution with'
        print '(A)', 'explicit data transfers, representing real MOM6 porting challenges.'
    end subroutine print_help

    subroutine verify_hor_visc(u, v, diffu, diffv, G, GV, CS)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        type(hor_visc_CS), intent(in) :: CS
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: diffu, diffv

        real(dp) :: max_u, max_v, max_diffu, max_diffv
        real(dp) :: avg_diffu, avg_diffv, rms_diffu, rms_diffv
        integer :: i, j, k, cnt

        max_u = 0.0_dp; max_v = 0.0_dp
        max_diffu = 0.0_dp; max_diffv = 0.0_dp
        avg_diffu = 0.0_dp; avg_diffv = 0.0_dp
        rms_diffu = 0.0_dp; rms_diffv = 0.0_dp
        cnt = 0

        ! Compute statistics
        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc, G%iec - 1
                    max_u = max(max_u, abs(u(i, j, k)))
                    max_diffu = max(max_diffu, abs(diffu(i, j, k)))
                    avg_diffu = avg_diffu + diffu(i, j, k)
                    rms_diffu = rms_diffu + diffu(i, j, k)**2
                end do
            end do
            do j = G%jsc, G%jec - 1
                do i = G%isc, G%iec
                    max_v = max(max_v, abs(v(i, j, k)))
                    max_diffv = max(max_diffv, abs(diffv(i, j, k)))
                    avg_diffv = avg_diffv + diffv(i, j, k)
                    rms_diffv = rms_diffv + diffv(i, j, k)**2
                    cnt = cnt + 1
                end do
            end do
        end do

        if (cnt > 0) then
            avg_diffu = avg_diffu/real(cnt, dp)
            avg_diffv = avg_diffv/real(cnt, dp)
            rms_diffu = sqrt(rms_diffu/real(cnt, dp))
            rms_diffv = sqrt(rms_diffv/real(cnt, dp))
        end if

        print '(A)', ''
        print '(A)', 'Horizontal Viscosity Verification:'
        print '(A)', '------------------------------------------------------------------'
        print '(A)', 'Active schemes:'
        if (CS%Laplacian) print '(A)', '  - Laplacian (Kh)'
        if (CS%biharmonic) print '(A)', '  - Biharmonic (Ah)'
        if (CS%Smagorinsky_Kh) print '(A)', '  - Smagorinsky for Kh'
        if (CS%Smagorinsky_Ah) print '(A)', '  - Smagorinsky for Ah'
        if (CS%Leith_Kh) print '(A)', '  - Leith vorticity gradient (CPU-only)'
        if (CS%no_slip) print '(A)', '  - No-slip boundary conditions'
        if (CS%better_bound_Kh) print '(A)', '  - Thickness-aware Kh bounding'
        if (CS%better_bound_Ah) print '(A)', '  - Thickness-aware Ah bounding'
        print '(A)', ''
        print '(A)', 'Input velocities:'
        print '(A,ES15.8)', '  Max |u|:              ', max_u
        print '(A,ES15.8)', '  Max |v|:              ', max_v
        print '(A)', ''
        print '(A)', 'Viscous accelerations:'
        print '(A,ES15.8)', '  Max |diffu|:          ', max_diffu
        print '(A,ES15.8)', '  Max |diffv|:          ', max_diffv
        print '(A,ES15.8)', '  RMS diffu:            ', rms_diffu
        print '(A,ES15.8)', '  RMS diffv:            ', rms_diffv
        print '(A)', ''

        ! Check that accelerations are reasonable (not NaN or huge)
        if (max_diffu < 1.0e10_dp .and. max_diffv < 1.0e10_dp .and. &
            max_diffu == max_diffu .and. max_diffv == max_diffv) then
            print '(A)', '  Status: PASS (accelerations are finite and reasonable)'
        else
            print '(A)', '  Status: FAIL (accelerations are NaN or too large)'
        end if

        ! Physical expectations
        print '(A)', ''
        print '(A)', 'Physical expectations:'
        if (CS%biharmonic) then
            print '(A)', '  - Biharmonic should more strongly damp small scales'
        end if
        if (CS%Smagorinsky_Kh .or. CS%Smagorinsky_Ah) then
            print '(A)', '  - Smagorinsky should increase viscosity near strong shear'
        end if
        if (CS%Leith_Kh) then
            print '(A)', '  - Leith should increase viscosity near vorticity gradients'
        end if
        if (CS%better_bound_Kh .or. CS%better_bound_Ah) then
            print '(A)', '  - Stability bounds should limit viscosity in thin layers'
        end if

    end subroutine verify_hor_visc

end program hor_visc_driver
