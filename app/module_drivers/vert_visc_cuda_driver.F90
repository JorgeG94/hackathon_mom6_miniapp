!> Standalone CUDA Fortran driver for vertical viscosity solver
program vert_visc_cuda_driver
    use cudafor
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, PI
    use mom6_vert_visc_cuda, only: vert_visc_CS_cuda, vert_visc_init_cuda, &
                                    vert_visc_cra_cuda, vert_visc_end_cuda
    implicit none

    type(ocean_grid_type)    :: G
    type(verticalGrid_type)  :: GV
    type(vert_visc_CS_cuda)  :: CS_cuda

    ! Host state arrays
    real(dp), allocatable :: u_init(:,:,:), v_init(:,:,:), h(:,:,:)
    real(dp), allocatable :: u_h(:,:,:), v_h(:,:,:)

    ! Device arrays
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:), h_d(:,:,:)
    real(dp), device, allocatable :: taux_d(:,:), tauy_d(:,:)

    real(dp) :: t_start, t_end, t_total
    real(dp) :: dt, Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl
    real(dp) :: max_u, max_v
    integer  :: ni, nj, nk, niter, iter, i, j, k
    character(len=32) :: arg

    ! ----------------------------------------------------------------
    ! Default parameters
    ! ----------------------------------------------------------------
    ni = 180; nj = 180; nk = 75; niter = 10
    Kv          = 1.0e-4_dp
    Kv_ml       = 1.0e-2_dp
    Kv_extra_bbl = 1.0e-2_dp
    Hmix        = 50.0_dp
    Hbbl        = 10.0_dp
    dt          = 300.0_dp

    ! ----------------------------------------------------------------
    ! Parse command line: ni nj nk niter
    ! ----------------------------------------------------------------
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

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Vertical Viscosity — CUDA Fortran Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)',            'Iterations: ', niter
    print '(A,ES10.3)',        'Interior Kv [m2/s]: ', Kv
    print '(A,ES10.3)',        'Mixed layer Kv [m2/s]: ', Kv_ml
    print '(A,ES10.3)',        'Extra BBL Kv [m2/s]: ', Kv_extra_bbl
    print '(A,F8.1)',          'Mixed layer depth [m]: ', Hmix
    print '(A,F8.1)',          'BBL thickness [m]: ', Hbbl
    print '(A,F8.1)',          'Timestep dt [s]: ', dt
    print '(A)', '================================================================'

    ! ----------------------------------------------------------------
    ! Initialize grid and vertical grid
    ! ----------------------------------------------------------------
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! ----------------------------------------------------------------
    ! Initialize CUDA control structure
    ! ----------------------------------------------------------------
    call vert_visc_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                             G%isc, G%iec, G%jsc, G%jec, nk, &
                             G%mask2dCu, G%mask2dCv, &
                             Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl)

    ! ----------------------------------------------------------------
    ! Allocate host arrays
    ! ----------------------------------------------------------------
    allocate (u_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (u_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! ----------------------------------------------------------------
    ! Allocate device arrays
    ! ----------------------------------------------------------------
    allocate (u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (taux_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (tauy_d(G%isd:G%ied, G%jsd:G%jed))

    ! ----------------------------------------------------------------
    ! Initialize state with vertical shear profile
    ! ----------------------------------------------------------------
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                h(i, j, k) = 4000.0_dp / real(nk, dp)
                u_init(i, j, k) = 0.5_dp * exp(-real(k - 1, dp) / 10.0_dp) * &
                                  sin(real(j - 1, dp) / real(nj, dp) * PI)
                v_init(i, j, k) = 0.3_dp * exp(-real(k - 1, dp) / 10.0_dp) * &
                                  cos(real(i - 1, dp) / real(ni, dp) * PI)
            end do
        end do
    end do

    ! Copy thickness and surface stress to device (constant across iterations)
    h_d = h
    do j = G%jsd, G%jed
        do i = G%isd, G%ied
            taux_d(i, j) = 0.1_dp * sin(real(j - 1, dp) / real(nj, dp) * PI)
            tauy_d(i, j) = 0.0_dp
        end do
    end do

    ! ----------------------------------------------------------------
    ! Warmup
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', 'Warming up CUDA variant...'
    u_d = u_init; v_d = v_init
    call vert_visc_cra_cuda(u_d, v_d, h_d, dt, CS_cuda, taux_d, tauy_d)

    ! ----------------------------------------------------------------
    ! Benchmark
    ! ----------------------------------------------------------------
    print '(A)', 'Benchmarking CUDA (fused coef+remnant+apply)...'
    t_total = 0.0_dp
    do iter = 1, niter
        ! Reset u, v on device from initial conditions
        u_d = u_init; v_d = v_init

        t_start = omp_get_wtime()
        call vert_visc_cra_cuda(u_d, v_d, h_d, dt, CS_cuda, taux_d, tauy_d)
        t_end = omp_get_wtime()
        t_total = t_total + (t_end - t_start)
    end do

    ! ----------------------------------------------------------------
    ! Copy results back to host
    ! ----------------------------------------------------------------
    u_h = u_d
    v_h = v_d

    ! ----------------------------------------------------------------
    ! Sanity check: print max |u|, |v|
    ! ----------------------------------------------------------------
    max_u = 0.0_dp
    max_v = 0.0_dp
    do k = 1, nk
        do j = G%jsc, G%jec
            do i = G%isc, G%iec
                max_u = max(max_u, abs(u_h(i, j, k)))
                max_v = max(max_v, abs(v_h(i, j, k)))
            end do
        end do
    end do

    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'SANITY CHECK'
    print '(A)', '================================================================'
    print '(A,ES15.8)', '  Max |u|: ', max_u
    print '(A,ES15.8)', '  Max |v|: ', max_v

    ! ----------------------------------------------------------------
    ! Timing results
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  Total time:             ', t_total, ' s'
    print '(A,F12.6,A)', '  Per-iteration:          ', t_total / real(niter, dp), ' s'
    print '(A)', '================================================================'

    ! ----------------------------------------------------------------
    ! Cleanup
    ! ----------------------------------------------------------------
    call vert_visc_end_cuda(CS_cuda)
    call end_ocean_grid(G)
    deallocate (u_init, v_init, h, u_h, v_h)
    deallocate (u_d, v_d, h_d, taux_d, tauy_d)

end program vert_visc_cuda_driver
