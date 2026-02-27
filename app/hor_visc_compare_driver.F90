!> Comparison driver: OpenACC vs CUDA horizontal viscosity
!!   A: OpenACC  -- Laplacian path via mom6_hor_visc
!!   B: CUDA     -- Laplacian path via mom6_hor_visc_cuda
!!
!! Usage:
!!   ./hor_visc_compare_driver [ni] [nj] [nk] [niter]
!!   Defaults: 180 180 75 10
!!
program hor_visc_compare_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid
    use mom6_hor_visc, only: hor_visc_CS, hor_visc_init, hor_visc, hor_visc_end
    use cudafor
    use mom6_hor_visc_cuda, only: hor_visc_CS_cuda, hor_visc_init_cuda, &
                                   hor_visc_cuda, hor_visc_end_cuda
    implicit none

    real(dp), parameter :: PI = 3.14159265358979323846_dp

    type(ocean_grid_type)  :: G
    type(verticalGrid_type) :: GV
    type(hor_visc_CS)       :: CS
    type(hor_visc_CS_cuda)  :: CS_cuda

    ! Host state arrays
    real(dp), allocatable :: u(:,:,:), v(:,:,:), h(:,:,:)
    real(dp), allocatable :: diffu(:,:,:), diffv(:,:,:)
    real(dp), allocatable :: diffu_cuda_h(:,:,:), diffv_cuda_h(:,:,:)

    ! Device arrays for CUDA variant
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:), h_d(:,:,:)
    real(dp), device, allocatable :: diffu_d(:,:,:), diffv_d(:,:,:)

    real(dp) :: t_start, t_end, t_acc, t_cuda
    real(dp) :: Kh_val, h_neglect_val
    integer  :: ni, nj, nk, niter, iter, i, j, k
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    Kh_val = 100.0_dp
    h_neglect_val = 1.0e-10_dp

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

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Horizontal Viscosity A/B Comparison'
    print '(A)', '  A: OpenACC  (Laplacian, do concurrent / acc)'
    print '(A)', '  B: CUDA     (Laplacian, explicit CUDA kernels)'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    print '(A,ES12.4)', 'Kh [m2/s]:  ', Kh_val
    print '(A)', '================================================================'

    ! Initialize grid and vertical grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! Initialize OpenACC control structure (Laplacian only)
    call hor_visc_init(CS, G, GV, Kh=Kh_val)

    ! Initialize CUDA control structure, extracting metrics from CS and G
    call hor_visc_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, &
                            Kh_val, CS%h_neglect, &
                            CS%DY_dxT, CS%DX_dyT, CS%DY_dxBu, CS%DX_dyBu, &
                            G%IdyCu, G%IdxCu, G%IdyCv, G%IdxCv, &
                            G%IareaCu, G%IareaCv, &
                            G%mask2dT, G%mask2dBu, &
                            CS%reduction_xx, CS%reduction_xy, &
                            CS%dy2h, CS%dx2h, CS%dy2q, CS%dx2q)

    ! Allocate host state arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffu(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffv(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffu_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffv_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Allocate device arrays for CUDA variant
    allocate (u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffu_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffv_d(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Initialize state with multi-scale velocity pattern
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                h(i, j, k) = 4000.0_dp / real(nk, dp)
                u(i, j, k) = 0.5_dp * sin(real(j-1,dp)/real(nj,dp)*PI*2.0_dp) * &
                             cos(real(i-1,dp)/real(ni,dp)*PI) * exp(-real(k-1,dp)/20.0_dp) + &
                             0.2_dp * sin(real(j-1,dp)/real(nj,dp)*PI*6.0_dp)
                v(i, j, k) = 0.5_dp * cos(real(j-1,dp)/real(nj,dp)*PI) * &
                             sin(real(i-1,dp)/real(ni,dp)*PI*2.0_dp) * exp(-real(k-1,dp)/20.0_dp) + &
                             0.2_dp * cos(real(i-1,dp)/real(ni,dp)*PI*6.0_dp)
            end do
        end do
    end do

    ! Copy inputs to CUDA device arrays
    u_d = u; v_d = v; h_d = h

    ! ---- Warmup both implementations ----
    print '(A)', ''
    print '(A)', 'Warming up both variants...'
    call hor_visc(u, v, h, diffu, diffv, G, GV, CS)
    call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, CS_cuda, nk)

    ! ---- Benchmark A: OpenACC (Laplacian) ----
    print '(A)', ''
    print '(A)', 'Benchmarking A: OpenACC (Laplacian)...'
    t_acc = 0.0_dp
    do iter = 1, niter
        t_start = omp_get_wtime()
        call hor_visc(u, v, h, diffu, diffv, G, GV, CS)
        t_end = omp_get_wtime()
        t_acc = t_acc + (t_end - t_start)
    end do

    ! ---- Benchmark B: CUDA (Laplacian) ----
    print '(A)', 'Benchmarking B: CUDA (Laplacian)...'
    t_cuda = 0.0_dp
    do iter = 1, niter
        t_start = omp_get_wtime()
        call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, CS_cuda, nk)
        t_end = omp_get_wtime()
        t_cuda = t_cuda + (t_end - t_start)
    end do

    ! ---- Copy CUDA results back for comparison ----
    diffu_cuda_h = diffu_d
    diffv_cuda_h = diffv_d

    ! ---- Timing Report ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  A: OpenACC (Laplacian): ', t_acc, ' s'
    print '(A,F12.6,A)', '  B: CUDA    (Laplacian): ', t_cuda, ' s'
    print '(A)', '  -----------------------------------------------'
    print '(A,F12.6,A)', '  Per-iter A:             ', t_acc/real(niter, dp), ' s'
    print '(A,F12.6,A)', '  Per-iter B:             ', t_cuda/real(niter, dp), ' s'
    print '(A)', '  -----------------------------------------------'
    if (t_cuda > 0.0_dp) then
        print '(A,F8.2,A)', '  Speedup B vs A:         ', t_acc/t_cuda, 'x'
    end if

    ! ---- Correctness: B (CUDA) vs A (OpenACC) ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'CORRECTNESS: B (CUDA) vs A (OpenACC)'
    print '(A)', '================================================================'
    call check_diff(diffu_cuda_h, diffu, diffv_cuda_h, diffv, G, nk, 'CUDA vs OpenACC')

    print '(A)', '================================================================'

    ! Cleanup
    call hor_visc_end(CS)
    call hor_visc_end_cuda(CS_cuda)
    call end_ocean_grid(G)
    deallocate (u, v, h, diffu, diffv)
    deallocate (diffu_cuda_h, diffv_cuda_h)
    deallocate (u_d, v_d, h_d, diffu_d, diffv_d)

contains

    subroutine check_diff(diffu_test, diffu_ref, diffv_test, diffv_ref, G, nk, label)
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in) :: nk
        real(dp), intent(in) :: diffu_test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: diffu_ref(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: diffv_test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: diffv_ref(G%isd:G%ied, G%jsd:G%jed, nk)
        character(len=*), intent(in) :: label

        real(dp) :: max_diff_u, max_diff_v, rel_diff_u, rel_diff_v
        real(dp) :: max_val_u, max_val_v
        integer :: ii, jj, kk

        max_diff_u = 0.0_dp; max_diff_v = 0.0_dp
        max_val_u = 0.0_dp; max_val_v = 0.0_dp

        ! diffu comparison: i in [isc-1:iec], j in [jsc:jec], k in [1:nk]
        do kk = 1, nk
            do jj = G%jsc, G%jec
                do ii = G%isc - 1, G%iec
                    max_diff_u = max(max_diff_u, abs(diffu_test(ii, jj, kk) - diffu_ref(ii, jj, kk)))
                    max_val_u = max(max_val_u, abs(diffu_ref(ii, jj, kk)))
                end do
            end do
        end do

        ! diffv comparison: i in [isc:iec], j in [jsc-1:jec], k in [1:nk]
        do kk = 1, nk
            do jj = G%jsc - 1, G%jec
                do ii = G%isc, G%iec
                    max_diff_v = max(max_diff_v, abs(diffv_test(ii, jj, kk) - diffv_ref(ii, jj, kk)))
                    max_val_v = max(max_val_v, abs(diffv_ref(ii, jj, kk)))
                end do
            end do
        end do

        if (max_val_u > 0.0_dp) then
            rel_diff_u = max_diff_u / max_val_u
        else
            rel_diff_u = 0.0_dp
        end if
        if (max_val_v > 0.0_dp) then
            rel_diff_v = max_diff_v / max_val_v
        else
            rel_diff_v = 0.0_dp
        end if

        print '(A,A)', '  ', label
        print '(A,ES15.8)', '  Max abs diff diffu: ', max_diff_u
        print '(A,ES15.8)', '  Max abs diff diffv: ', max_diff_v
        print '(A,ES15.8)', '  Max rel diff diffu: ', rel_diff_u
        print '(A,ES15.8)', '  Max rel diff diffv: ', rel_diff_v
        if (rel_diff_u < 1.0e-10_dp .and. rel_diff_v < 1.0e-10_dp) then
            print '(A)', '  Status: PASS (results match within roundoff)'
        else if (rel_diff_u < 1.0e-6_dp .and. rel_diff_v < 1.0e-6_dp) then
            print '(A)', '  Status: WARN (small differences, likely FP reordering)'
        else
            print '(A)', '  Status: FAIL (significant differences!)'
        end if
    end subroutine check_diff

end program hor_visc_compare_driver
