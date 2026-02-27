!> Comparison driver: OpenACC vs CUDA barotropic solver
!!   A. OpenACC  -- btstep with acc parallel loop
!!   B. CUDA     -- btstep_cuda with explicit CUDA Fortran kernels
program barotropic_compare_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, init_ocean_grid, end_ocean_grid, G_EARTH
    use mom6_barotropic, only: barotropic_CS, barotropic_init, btstep, barotropic_end
    use cudafor
    use mom6_barotropic_cuda, only: barotropic_CS_cuda, barotropic_init_cuda, &
                                     btstep_cuda, barotropic_end_cuda
    implicit none

    type(ocean_grid_type) :: G
    type(barotropic_CS) :: CS
    type(barotropic_CS_cuda) :: CS_cuda

    ! Host state arrays for OpenACC variant
    real(dp), allocatable :: eta_in(:,:), ubt_in(:,:), vbt_in(:,:)
    real(dp), allocatable :: u_av(:,:), v_av(:,:), eta_av(:,:)
    real(dp), allocatable :: eta_init(:,:)

    ! Host arrays for CUDA results (copy-back)
    real(dp), allocatable :: u_av_cuda(:,:), v_av_cuda(:,:), eta_av_cuda(:,:)

    ! Device arrays for CUDA variant
    real(dp), device, allocatable :: eta_in_d(:,:), ubt_in_d(:,:), vbt_in_d(:,:)
    real(dp), device, allocatable :: u_av_d(:,:), v_av_d(:,:), eta_av_d(:,:)

    real(dp) :: t_start, t_end, t_acc, t_cuda
    real(dp) :: dt
    integer :: ni, nj, nsteps, niter, iter, i, j
    integer :: first_direction
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nsteps = 15; niter = 10
    dt = 300.0_dp           ! 5-minute baroclinic timestep
    first_direction = 0

    ! Parse command line: ni nj nsteps niter
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg); read (arg, *) ni
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg); read (arg, *) nj
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, arg); read (arg, *) nsteps
    end if
    if (command_argument_count() >= 4) then
        call get_command_argument(4, arg); read (arg, *) niter
    end if

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Barotropic A/B Comparison'
    print '(A)', '  A: OpenACC  (acc parallel loop)'
    print '(A)', '  B: CUDA     (explicit CUDA Fortran kernels)'
    print '(A)', '================================================================'
    print '(A,I5,A,I5)', 'Grid: ', ni, ' x ', nj
    print '(A,I4)',       'Barotropic substeps: ', nsteps
    print '(A,I4)',       'Iterations: ', niter
    print '(A,F8.1)',     'dt: ', dt
    print '(A,F6.2)',     'bebt: ', 0.2_dp
    print '(A)', '================================================================'

    ! ----------------------------------------------------------------
    ! Initialize grid (2D barotropic, nk=1)
    ! ----------------------------------------------------------------
    call init_ocean_grid(G, ni, nj, 1, 10.0_dp, 45.0_dp)

    ! ----------------------------------------------------------------
    ! Initialize OpenACC control structure
    ! ----------------------------------------------------------------
    call barotropic_init(CS, G, dt, nsteps)

    ! ----------------------------------------------------------------
    ! Initialize CUDA control structure from the OpenACC CS data
    ! ----------------------------------------------------------------
    call barotropic_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
        G%isc, G%iec, G%jsc, G%jec, &
        nsteps, dt, CS%bebt, first_direction, &
        CS%Datu, CS%Datv, CS%gtot_E, CS%gtot_W, CS%gtot_N, CS%gtot_S, &
        CS%f_4_u, CS%f_4_v, CS%bt_rem_u, CS%bt_rem_v, &
        G%IareaT, G%IdxCu, G%IdyCv)

    ! ----------------------------------------------------------------
    ! Allocate host state arrays
    ! ----------------------------------------------------------------
    allocate (eta_in  (G%isd:G%ied, G%jsd:G%jed))
    allocate (ubt_in  (G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt_in  (G%isd:G%ied, G%jsd:G%jed))
    allocate (u_av    (G%isd:G%ied, G%jsd:G%jed))
    allocate (v_av    (G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_av  (G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_init(G%isd:G%ied, G%jsd:G%jed))

    ! Host buffers for CUDA result copy-back
    allocate (u_av_cuda  (G%isd:G%ied, G%jsd:G%jed))
    allocate (v_av_cuda  (G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_av_cuda(G%isd:G%ied, G%jsd:G%jed))

    ! Device arrays for CUDA variant
    allocate (eta_in_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (ubt_in_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt_in_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (u_av_d  (G%isd:G%ied, G%jsd:G%jed))
    allocate (v_av_d  (G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_av_d(G%isd:G%ied, G%jsd:G%jed))

    ! ----------------------------------------------------------------
    ! Initialize state with realistic patterns
    ! ----------------------------------------------------------------
    do j = G%jsd, G%jed
        do i = G%isd, G%ied
            eta_in(i, j) = 0.5_dp * sin(real(i - 1, dp) / real(ni, dp) * 3.14159_dp * 2.0_dp) * &
                           cos(real(j - 1, dp) / real(nj, dp) * 3.14159_dp * 2.0_dp)
            eta_init(i, j) = eta_in(i, j)
            ubt_in(i, j) = 0.05_dp * sin(real(j - 1, dp) / real(nj, dp) * 3.14159_dp)
            vbt_in(i, j) = 0.05_dp * cos(real(i - 1, dp) / real(ni, dp) * 3.14159_dp)
        end do
    end do

    ! Copy initial velocities to device (these are not reset per iteration)
    ubt_in_d = ubt_in
    vbt_in_d = vbt_in

    ! ----------------------------------------------------------------
    ! Warmup both implementations
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', 'Warming up both variants...'

    ! Warmup A: OpenACC
    eta_in = eta_init
    call btstep(eta_in, ubt_in, vbt_in, u_av, v_av, eta_av, G, CS)

    ! Warmup B: CUDA
    eta_in_d = eta_init
    call btstep_cuda(eta_in_d, ubt_in_d, vbt_in_d, u_av_d, v_av_d, eta_av_d, CS_cuda)

    ! ----------------------------------------------------------------
    ! Benchmark A: OpenACC
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', 'Benchmarking A: OpenACC (acc parallel loop)...'
    t_acc = 0.0_dp
    do iter = 1, niter
        ! Reset eta_in from initial condition each iteration
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                eta_in(i, j) = eta_init(i, j)
            end do
        end do

        t_start = omp_get_wtime()
        call btstep(eta_in, ubt_in, vbt_in, u_av, v_av, eta_av, G, CS)
        t_end = omp_get_wtime()
        t_acc = t_acc + (t_end - t_start)
    end do

    ! ----------------------------------------------------------------
    ! Benchmark B: CUDA
    ! ----------------------------------------------------------------
    print '(A)', 'Benchmarking B: CUDA (explicit kernels)...'
    t_cuda = 0.0_dp
    do iter = 1, niter
        ! Reset eta_in on device from host initial condition each iteration
        eta_in_d = eta_init

        t_start = omp_get_wtime()
        call btstep_cuda(eta_in_d, ubt_in_d, vbt_in_d, u_av_d, v_av_d, eta_av_d, CS_cuda)
        t_end = omp_get_wtime()
        t_cuda = t_cuda + (t_end - t_start)
    end do

    ! ----------------------------------------------------------------
    ! Copy CUDA results back to host for comparison
    ! ----------------------------------------------------------------
    u_av_cuda   = u_av_d
    v_av_cuda   = v_av_d
    eta_av_cuda = eta_av_d

    ! ----------------------------------------------------------------
    ! Timing report
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  A: OpenACC:             ', t_acc,  ' s'
    print '(A,F12.6,A)', '  B: CUDA:                ', t_cuda, ' s'
    print '(A)', '  -----------------------------------------------'
    print '(A,F12.6,A)', '  Per-iter A:             ', t_acc  / real(niter, dp), ' s'
    print '(A,F12.6,A)', '  Per-iter B:             ', t_cuda / real(niter, dp), ' s'
    print '(A)', '  -----------------------------------------------'
    if (t_cuda > 0.0_dp) then
        print '(A,F8.2,A)', '  Speedup B vs A:         ', t_acc / t_cuda, 'x'
    end if

    ! ----------------------------------------------------------------
    ! Correctness: B (CUDA) vs A (OpenACC)
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'CORRECTNESS: B (CUDA) vs A (OpenACC)'
    print '(A)', '================================================================'
    call check_diff_2d(u_av_cuda, u_av, G, 'u_av')
    call check_diff_2d(v_av_cuda, v_av, G, 'v_av')
    call check_diff_2d(eta_av_cuda, eta_av, G, 'eta_av')

    print '(A)', '================================================================'

    ! ----------------------------------------------------------------
    ! Cleanup
    ! ----------------------------------------------------------------
    call barotropic_end(CS)
    call barotropic_end_cuda(CS_cuda)
    call end_ocean_grid(G)

    deallocate (eta_in, ubt_in, vbt_in, u_av, v_av, eta_av, eta_init)
    deallocate (u_av_cuda, v_av_cuda, eta_av_cuda)
    deallocate (eta_in_d, ubt_in_d, vbt_in_d, u_av_d, v_av_d, eta_av_d)

contains

    !> Compare a 2D test array against a reference array over the compute domain.
    !! Reports max absolute and relative differences and a PASS/WARN/FAIL status.
    subroutine check_diff_2d(test, ref, G, label)
        type(ocean_grid_type), intent(in) :: G
        real(dp), intent(in) :: test(G%isd:G%ied, G%jsd:G%jed)
        real(dp), intent(in) :: ref (G%isd:G%ied, G%jsd:G%jed)
        character(len=*), intent(in) :: label

        real(dp) :: max_abs_diff, max_rel_diff, max_val
        integer :: ii, jj

        max_abs_diff = 0.0_dp
        max_val      = 0.0_dp

        do jj = G%jsc, G%jec
            do ii = G%isc, G%iec
                max_abs_diff = max(max_abs_diff, abs(test(ii, jj) - ref(ii, jj)))
                max_val      = max(max_val, abs(ref(ii, jj)))
            end do
        end do

        if (max_val > 0.0_dp) then
            max_rel_diff = max_abs_diff / max_val
        else
            max_rel_diff = 0.0_dp
        end if

        print '(A,A)', '  Field: ', label
        print '(A,ES15.8)', '    Max abs diff: ', max_abs_diff
        print '(A,ES15.8)', '    Max rel diff: ', max_rel_diff
        if (max_rel_diff < 1.0e-10_dp) then
            print '(A)', '    Status: PASS (results match within roundoff)'
        else if (max_rel_diff < 1.0e-6_dp) then
            print '(A)', '    Status: WARN (small differences, likely FP reordering)'
        else
            print '(A)', '    Status: FAIL (significant differences!)'
        end if
    end subroutine check_diff_2d

end program barotropic_compare_driver
