!> Standalone CUDA Fortran driver for barotropic solver
!!
!! Benchmarks the explicit CUDA Fortran barotropic solver (btstep_cuda).
!! Uses the OpenACC barotropic_init only to extract precomputed metrics,
!! then runs exclusively with CUDA kernels.
program barotropic_cuda_driver
    use cudafor
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, init_ocean_grid, end_ocean_grid, PI
    use mom6_barotropic, only: barotropic_CS, barotropic_init, barotropic_end
    use mom6_barotropic_cuda, only: barotropic_CS_cuda, barotropic_init_cuda, &
                                     btstep_cuda, barotropic_end_cuda
    implicit none

    type(ocean_grid_type) :: G
    type(barotropic_CS) :: CS
    type(barotropic_CS_cuda) :: CS_cuda

    ! Host state arrays
    real(dp), allocatable :: eta_in(:,:), ubt_in(:,:), vbt_in(:,:)
    real(dp), allocatable :: u_av_h(:,:), v_av_h(:,:), eta_av_h(:,:)
    real(dp), allocatable :: eta_init(:,:)

    ! Device arrays for CUDA variant
    real(dp), device, allocatable :: eta_in_d(:,:), ubt_in_d(:,:), vbt_in_d(:,:)
    real(dp), device, allocatable :: u_av_d(:,:), v_av_d(:,:), eta_av_d(:,:)

    real(dp) :: t_start, t_end, t_total
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
    print '(A)', 'MOM6 Barotropic CUDA Standalone Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5)', 'Grid: ', ni, ' x ', nj
    print '(A,I4)',       'Barotropic substeps: ', nsteps
    print '(A,I4)',       'Iterations: ', niter
    print '(A,F8.1)',     'dt: ', dt
    print '(A)', '================================================================'

    ! ----------------------------------------------------------------
    ! Initialize grid (2D barotropic, nk=1)
    ! ----------------------------------------------------------------
    call init_ocean_grid(G, ni, nj, 1, 10.0_dp, 45.0_dp)

    ! ----------------------------------------------------------------
    ! Initialize OpenACC control structure (for metric extraction only)
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
    allocate (u_av_h  (G%isd:G%ied, G%jsd:G%jed))
    allocate (v_av_h  (G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_av_h(G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_init(G%isd:G%ied, G%jsd:G%jed))

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
            eta_in(i, j) = 0.5_dp * sin(real(i - 1, dp) / real(ni, dp) * PI * 2.0_dp) * &
                           cos(real(j - 1, dp) / real(nj, dp) * PI * 2.0_dp)
            eta_init(i, j) = eta_in(i, j)
            ubt_in(i, j) = 0.05_dp * sin(real(j - 1, dp) / real(nj, dp) * PI)
            vbt_in(i, j) = 0.05_dp * cos(real(i - 1, dp) / real(ni, dp) * PI)
        end do
    end do

    ! Copy initial velocities to device (these are not reset per iteration)
    ubt_in_d = ubt_in
    vbt_in_d = vbt_in

    ! ----------------------------------------------------------------
    ! Warmup
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', 'Warming up CUDA variant...'

    eta_in_d = eta_init
    call btstep_cuda(eta_in_d, ubt_in_d, vbt_in_d, u_av_d, v_av_d, eta_av_d, CS_cuda)

    ! ----------------------------------------------------------------
    ! Benchmark CUDA
    ! ----------------------------------------------------------------
    print '(A)', 'Benchmarking CUDA (explicit kernels)...'
    t_total = 0.0_dp
    do iter = 1, niter
        ! Reset eta_in on device from host initial condition each iteration
        eta_in_d = eta_init

        t_start = omp_get_wtime()
        call btstep_cuda(eta_in_d, ubt_in_d, vbt_in_d, u_av_d, v_av_d, eta_av_d, CS_cuda)
        t_end = omp_get_wtime()
        t_total = t_total + (t_end - t_start)
    end do

    ! ----------------------------------------------------------------
    ! Copy results back to host for sanity check
    ! ----------------------------------------------------------------
    u_av_h   = u_av_d
    v_av_h   = v_av_d
    eta_av_h = eta_av_d

    ! ----------------------------------------------------------------
    ! Sanity check: print max absolute values
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'SANITY CHECK'
    print '(A)', '================================================================'
    print '(A,ES15.8)', '  max |u_av|:   ', maxval(abs(u_av_h(G%isc:G%iec, G%jsc:G%jec)))
    print '(A,ES15.8)', '  max |v_av|:   ', maxval(abs(v_av_h(G%isc:G%iec, G%jsc:G%jec)))
    print '(A,ES15.8)', '  max |eta_av|: ', maxval(abs(eta_av_h(G%isc:G%iec, G%jsc:G%jec)))

    ! ----------------------------------------------------------------
    ! Timing report
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  Total time:    ', t_total, ' s'
    print '(A,F12.6,A)', '  Per-iteration: ', t_total / real(niter, dp), ' s'
    print '(A)', '================================================================'

    ! ----------------------------------------------------------------
    ! Cleanup
    ! ----------------------------------------------------------------
    call barotropic_end(CS)
    call barotropic_end_cuda(CS_cuda)
    call end_ocean_grid(G)

    deallocate (eta_in, ubt_in, vbt_in, u_av_h, v_av_h, eta_av_h, eta_init)
    deallocate (eta_in_d, ubt_in_d, vbt_in_d, u_av_d, v_av_d, eta_av_d)

end program barotropic_cuda_driver
