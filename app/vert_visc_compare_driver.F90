!> Comparison driver: OpenACC fused vs CUDA vertical viscosity
!!   A: OpenACC fused (coef+remnant+apply, do concurrent via acc)
!!   B: CUDA Fortran  (fused coef+remnant+apply, explicit kernels)
program vert_visc_compare_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, RHO_0, &
                          mech_forcing_type, vertvisc_type, &
                          init_mech_forcing, end_mech_forcing, &
                          init_vertvisc_visc, end_vertvisc_visc
    use mom6_vert_visc, only: vert_visc_CS, vert_visc_init, &
                              vert_visc_coef_remnant_apply, vert_visc_end
    use cudafor
    use mom6_vert_visc_cuda, only: vert_visc_CS_cuda, vert_visc_init_cuda, &
                                    vert_visc_cra_cuda, vert_visc_end_cuda
    implicit none

    type(ocean_grid_type)    :: G
    type(verticalGrid_type)  :: GV
    type(vert_visc_CS)       :: CS
    type(vert_visc_CS_cuda)  :: CS_cuda
    type(mech_forcing_type)  :: forces
    type(vertvisc_type)      :: visc

    ! Host state arrays
    real(dp), allocatable :: u(:,:,:), v(:,:,:), h(:,:,:)
    real(dp), allocatable :: u_init(:,:,:), v_init(:,:,:)

    ! Host arrays for CUDA results (copy-back)
    real(dp), allocatable :: u_cuda_h(:,:,:), v_cuda_h(:,:,:)
    real(dp), allocatable :: visc_rem_u_h(:,:,:), visc_rem_v_h(:,:,:)

    ! Device arrays for CUDA variant
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:), h_d(:,:,:)
    real(dp), device, allocatable :: taux_d(:,:), tauy_d(:,:)

    real(dp) :: t_start, t_end, t_acc, t_cuda
    real(dp) :: dt, Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl
    real(dp), parameter :: PI = 3.14159265358979_dp
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
    print '(A)', 'MOM6 Vertical Viscosity A/B Comparison'
    print '(A)', '  A: OpenACC fused (coef+remnant+apply)'
    print '(A)', '  B: CUDA Fortran  (fused coef+remnant+apply)'
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
    ! Initialize OpenACC control structure
    ! ----------------------------------------------------------------
    call vert_visc_init(CS, G, GV, Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl)

    ! ----------------------------------------------------------------
    ! Initialize CUDA control structure
    ! ----------------------------------------------------------------
    call vert_visc_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                             G%isc, G%iec, G%jsc, G%jec, nk, &
                             G%mask2dCu, G%mask2dCv, &
                             Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl)

    ! ----------------------------------------------------------------
    ! Initialize forces (sinusoidal taux, zero tauy)
    ! ----------------------------------------------------------------
    call init_mech_forcing(forces, G)
    do j = G%jsd, G%jed
        do i = G%isd, G%ied
            forces%taux(i, j) = 0.1_dp * sin(real(j - 1, dp) / real(nj, dp) * PI)
            forces%tauy(i, j) = 0.0_dp
        end do
    end do
    !$acc update device(forces%taux, forces%tauy)

    ! ----------------------------------------------------------------
    ! Initialize visc (NO Rayleigh drag -- CUDA has no Rayleigh support)
    ! ----------------------------------------------------------------
    call init_vertvisc_visc(visc, G, GV, use_rayleigh=.false.)

    ! ----------------------------------------------------------------
    ! Allocate host state arrays
    ! ----------------------------------------------------------------
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (u_init(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_init(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Host arrays for CUDA result copy-back
    allocate (u_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (visc_rem_u_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (visc_rem_v_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! ----------------------------------------------------------------
    ! Allocate device arrays for CUDA variant
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
    taux_d = forces%taux
    tauy_d = forces%tauy

    ! ----------------------------------------------------------------
    ! Warmup both implementations
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', 'Warming up both variants...'

    ! Warmup A: OpenACC fused
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                u(i, j, k) = u_init(i, j, k)
                v(i, j, k) = v_init(i, j, k)
            end do
        end do
    end do
    call vert_visc_coef_remnant_apply(u, v, h, dt, CS, G, GV, forces, visc)

    ! Warmup B: CUDA
    u_d = u_init; v_d = v_init
    call vert_visc_cra_cuda(u_d, v_d, h_d, dt, CS_cuda, taux_d, tauy_d)

    ! ----------------------------------------------------------------
    ! Benchmark A: OpenACC fused (coef+remnant+apply)
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', 'Benchmarking A: OpenACC fused (coef+remnant+apply)...'
    t_acc = 0.0_dp
    do iter = 1, niter
        ! Reset u, v from initial conditions each iteration
        do k = 1, nk
            do j = G%jsd, G%jed
                do i = G%isd, G%ied
                    u(i, j, k) = u_init(i, j, k)
                    v(i, j, k) = v_init(i, j, k)
                end do
            end do
        end do

        t_start = omp_get_wtime()
        call vert_visc_coef_remnant_apply(u, v, h, dt, CS, G, GV, forces, visc)
        t_end = omp_get_wtime()
        t_acc = t_acc + (t_end - t_start)
    end do

    ! Copy OpenACC results from device to host for comparison
    !$acc update self(CS%visc_rem_u, CS%visc_rem_v, CS%taux_bot, CS%tauy_bot)

    ! ----------------------------------------------------------------
    ! Benchmark B: CUDA (fused coef+remnant+apply)
    ! ----------------------------------------------------------------
    print '(A)', 'Benchmarking B: CUDA (fused coef+remnant+apply)...'
    t_cuda = 0.0_dp
    do iter = 1, niter
        ! Reset u, v on device from initial conditions
        u_d = u_init; v_d = v_init

        t_start = omp_get_wtime()
        call vert_visc_cra_cuda(u_d, v_d, h_d, dt, CS_cuda, taux_d, tauy_d)
        t_end = omp_get_wtime()
        t_cuda = t_cuda + (t_end - t_start)
    end do

    ! ----------------------------------------------------------------
    ! Copy CUDA results back to host
    ! ----------------------------------------------------------------
    u_cuda_h = u_d
    v_cuda_h = v_d
    visc_rem_u_h(:,:,:) = CS_cuda%visc_rem_u(:,:,:)
    visc_rem_v_h(:,:,:) = CS_cuda%visc_rem_v(:,:,:)

    ! ----------------------------------------------------------------
    ! Timing results
    ! ----------------------------------------------------------------
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  A: OpenACC fused:       ', t_acc,  ' s'
    print '(A,F12.6,A)', '  B: CUDA fused:          ', t_cuda, ' s'
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
    call check_diff_3d(u_cuda_h, u, G, nk, 'u')
    call check_diff_3d(v_cuda_h, v, G, nk, 'v')
    call check_diff_3d(visc_rem_u_h, CS%visc_rem_u, G, nk, 'visc_rem_u')
    call check_diff_3d(visc_rem_v_h, CS%visc_rem_v, G, nk, 'visc_rem_v')

    print '(A)', '================================================================'

    ! ----------------------------------------------------------------
    ! Cleanup
    ! ----------------------------------------------------------------
    call vert_visc_end(CS)
    call vert_visc_end_cuda(CS_cuda)
    call end_mech_forcing(forces)
    call end_vertvisc_visc(visc)
    call end_ocean_grid(G)
    deallocate (u, v, h, u_init, v_init)
    deallocate (u_cuda_h, v_cuda_h, visc_rem_u_h, visc_rem_v_h)
    deallocate (u_d, v_d, h_d, taux_d, tauy_d)

contains

    subroutine check_diff_3d(arr_test, arr_ref, G, nk, label)
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in) :: nk
        real(dp), intent(in) :: arr_test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: arr_ref (G%isd:G%ied, G%jsd:G%jed, nk)
        character(len=*), intent(in) :: label

        real(dp) :: max_diff, rel_diff, max_val
        integer  :: ii, jj, kk

        max_diff = 0.0_dp
        max_val  = 0.0_dp

        do kk = 1, nk
            do jj = G%jsc, G%jec
                do ii = G%isc, G%iec
                    max_diff = max(max_diff, abs(arr_test(ii, jj, kk) - arr_ref(ii, jj, kk)))
                    max_val  = max(max_val,  abs(arr_ref(ii, jj, kk)))
                end do
            end do
        end do

        if (max_val > 0.0_dp) then
            rel_diff = max_diff / max_val
        else
            rel_diff = 0.0_dp
        end if

        print '(A,A)', '  ', label
        print '(A,ES15.8)', '    Max abs diff: ', max_diff
        print '(A,ES15.8)', '    Max rel diff: ', rel_diff
        if (rel_diff < 1.0e-10_dp) then
            print '(A)', '    Status: PASS (results match within roundoff)'
        else if (rel_diff < 1.0e-6_dp) then
            print '(A)', '    Status: WARN (small differences, likely FP reordering)'
        else
            print '(A)', '    Status: FAIL (significant differences!)'
        end if

    end subroutine check_diff_3d

end program vert_visc_compare_driver
