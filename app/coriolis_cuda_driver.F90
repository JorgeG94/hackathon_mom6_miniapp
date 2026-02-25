!> Standalone driver for the CUDA Fortran Coriolis solver
!!
!! Usage: ./coriolis_cuda_driver [ni] [nj] [nk] [niter] [scheme] [bx] [by]
!!
!! Runs the explicit CUDA kernel Coriolis solver and optionally sweeps
!! over block-size configurations for tuning.
!! Examples:
!!   ./coriolis_cuda_driver 2048 2048 75 10 sadourny         # sweep mode
!!   ./coriolis_cuda_driver 2048 2048 75 10 sadourny 32 4    # single config
program coriolis_cuda_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    !use cudafor
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid
    use mom6_coriolis, only: coriolis_CS, coriolis_init, CorAdCalc, coriolis_end, &
                            SADOURNY75_ENERGY, ARAKAWA_HSU90, ARAKAWA_LAMB81
    use mom6_coriolis_cuda, only: coriolis_CS_cuda, coriolis_init_cuda, CorAdCalc_cuda, &
                                  coriolis_end_cuda, SADOURNY75_ENERGY_CUDA, &
                                  ARAKAWA_HSU90_CUDA, ARAKAWA_LAMB81_CUDA
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(coriolis_CS) :: CS_ref
    type(coriolis_CS_cuda) :: CS_cuda

    real(dp), allocatable :: u(:,:,:), v(:,:,:), h(:,:,:)
    real(dp), allocatable :: uh(:,:,:), vh(:,:,:)
    real(dp), allocatable :: CAu_ref(:,:,:), CAv_ref(:,:,:)
    real(dp), allocatable :: CAu_h(:,:,:), CAv_h(:,:,:)

    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:), h_d(:,:,:)
    real(dp), device, allocatable :: uh_d(:,:,:), vh_d(:,:,:)
    real(dp), device, allocatable :: CAu_d(:,:,:), CAv_d(:,:,:)

    real(dp) :: t_start, t_end, t_total, t_best, t_ref
    integer :: ni, nj, nk, niter, iter, i, j, k
    integer :: scheme, scheme_ref, bx, by, sweep
    integer :: bx_list(12), by_list(12), nb, ib
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    scheme = SADOURNY75_ENERGY_CUDA
    scheme_ref = SADOURNY75_ENERGY
    bx = 0; by = 0  ! 0 = sweep mode
    sweep = 1

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
            scheme = SADOURNY75_ENERGY_CUDA; scheme_ref = SADOURNY75_ENERGY
        case ('hsu', 'HSU', '2')
            scheme = ARAKAWA_HSU90_CUDA; scheme_ref = ARAKAWA_HSU90
        case ('lamb', 'LAMB', '3')
            scheme = ARAKAWA_LAMB81_CUDA; scheme_ref = ARAKAWA_LAMB81
        end select
    end if
    if (command_argument_count() >= 6) then
        call get_command_argument(6, arg)
        read (arg, *) bx
        if (bx > 0) sweep = 0
    end if
    if (command_argument_count() >= 7) then
        call get_command_argument(7, arg)
        read (arg, *) by
    else if (bx > 0) then
        by = 128 / bx  ! default: keep 128 threads total
        if (by < 1) by = 1
    end if

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Coriolis CUDA Fortran Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    select case (scheme)
    case (SADOURNY75_ENERGY_CUDA)
        print '(A)', 'Scheme: Sadourny (1975) Energy-conserving'
    case (ARAKAWA_HSU90_CUDA)
        print '(A)', 'Scheme: Arakawa-Hsu (1990)'
    case (ARAKAWA_LAMB81_CUDA)
        print '(A)', 'Scheme: Arakawa-Lamb (1981)'
    end select
    if (sweep == 1) then
        print '(A)', 'Block size: SWEEP (testing multiple configurations)'
    else
        print '(A,I4,A,I4)', 'Block size: ', bx, ' x ', by
    end if
    print '(A)', '================================================================'

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call coriolis_init(CS_ref, G, scheme_ref)
    call coriolis_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, nk, &
                            G%areaT, G%IareaBu, G%CoriolisBu, G%mask2dBu, &
                            G%dyCv, G%dxCu, G%dyCu, G%dxCv, G%IdxCu, G%IdyCv, &
                            scheme)

    ! Allocate host arrays
    allocate(u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(vh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAu_ref(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAv_ref(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAu_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAv_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Allocate device arrays
    allocate(u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(v_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(vh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAu_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAv_d(G%isd:G%ied, G%jsd:G%jed, nk))

#ifdef __NVCOMPILER_LLVM__
    !$omp target enter data map(alloc: u, v, h, uh, vh, CAu_ref, CAv_ref)
#endif

    ! Initialize state
    do concurrent(k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
        h(i, j, k) = 4000.0_dp / real(nk, dp)
        u(i, j, k) = 0.1_dp * sin(real(j - 1, dp) / real(nj, dp) * 3.14159_dp * 2.0_dp) * &
                     exp(-real(k, dp) / 30.0_dp)
        v(i, j, k) = 0.1_dp * cos(real(i - 1, dp) / real(ni, dp) * 3.14159_dp * 2.0_dp) * &
                     exp(-real(k, dp) / 30.0_dp)
        uh(i, j, k) = u(i, j, k) * h(i, j, k) * G%dyCu(i, j)
        vh(i, j, k) = v(i, j, k) * h(i, j, k) * G%dxCv(i, j)
    end do

#ifdef __NVCOMPILER_LLVM__
    !$omp target update to(u, v, h, uh, vh)
#endif

    ! Copy to device
    u_d = u; v_d = v; h_d = h; uh_d = uh; vh_d = vh

    ! Reference result (original CPU/do-concurrent) — also time it
    print '(A)', ''
    print '(A)', 'Computing reference (3D do-concurrent)...'
    ! Warmup
    call CorAdCalc(u, v, h, uh, vh, CAu_ref, CAv_ref, G, GV, CS_ref)
    t_ref = 0.0_dp
    do iter = 1, niter
        t_start = omp_get_wtime()
        call CorAdCalc(u, v, h, uh, vh, CAu_ref, CAv_ref, G, GV, CS_ref)
        t_end = omp_get_wtime()
        t_ref = t_ref + (t_end - t_start)
    end do
    print '(A,F12.4,A)', '  Reference per-iter: ', 1000.0_dp * t_ref / real(niter, dp), ' ms'

    ! Warmup CUDA
    call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS_cuda)

    if (sweep == 1) then
        ! ---- Block-size sweep ----
        ! Configurations to test: (bx, by) with bx*by = 64, 128, 256, 512
        nb = 0
        ! 64 threads
        nb = nb + 1; bx_list(nb) = 32; by_list(nb) = 2
        nb = nb + 1; bx_list(nb) = 16; by_list(nb) = 4
        nb = nb + 1; bx_list(nb) = 8;  by_list(nb) = 8
        ! 128 threads
        nb = nb + 1; bx_list(nb) = 32; by_list(nb) = 4
        nb = nb + 1; bx_list(nb) = 16; by_list(nb) = 8
        ! 256 threads
        nb = nb + 1; bx_list(nb) = 32; by_list(nb) = 8
        nb = nb + 1; bx_list(nb) = 16; by_list(nb) = 16
        ! 512 threads
        nb = nb + 1; bx_list(nb) = 32; by_list(nb) = 16
        nb = nb + 1; bx_list(nb) = 16; by_list(nb) = 32
        ! 1024 threads
        nb = nb + 1; bx_list(nb) = 32; by_list(nb) = 32

        print '(A)', ''
        print '(A)', '================================================================'
        print '(A)', 'BLOCK-SIZE SWEEP'
        print '(A)', '================================================================'
        print '(A)', '  bx   by   threads   total(s)    per-iter(ms)    vs-best   vs-ref'
        print '(A)', '  ----+----+--------+-----------+--------------+---------+--------'

        t_best = huge(1.0_dp)

        do ib = 1, nb
            bx = bx_list(ib); by = by_list(ib)

            ! Warmup this config
            call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS_cuda, bx, by)

            t_total = 0.0_dp
            do iter = 1, niter
                t_start = omp_get_wtime()
                call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS_cuda, bx, by)
                t_end = omp_get_wtime()
                t_total = t_total + (t_end - t_start)
            end do

            if (t_total < t_best) t_best = t_total

            print '(2X,I4,1X,I4,2X,I6,4X,F9.6,6X,F10.4,6X,F6.2,A,2X,F7.2,A)', &
                bx, by, bx*by, t_total, &
                1000.0_dp * t_total / real(niter, dp), &
                t_best / t_total, 'x', &
                t_ref / t_total, 'x'
        end do

        ! Verify best (last run)
        CAu_h = CAu_d; CAv_h = CAv_d
#ifdef __NVCOMPILER_LLVM__
        !$omp target update from(CAu_ref, CAv_ref)
#endif
        print '(A)', ''
        call check_diff(CAu_h, CAu_ref, CAv_h, CAv_ref, G, nk)

    else
        ! ---- Single block-size run ----
        print '(A)', ''
        print '(A,I4,A,I4,A,I6,A)', 'Running with block (', bx, ' x ', by, &
            ') = ', bx*by, ' threads/block...'

        t_total = 0.0_dp
        do iter = 1, niter
            t_start = omp_get_wtime()
            call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS_cuda, bx, by)
            t_end = omp_get_wtime()
            t_total = t_total + (t_end - t_start)
        end do

        print '(A)', ''
        print '(A)', '================================================================'
        print '(A)', 'TIMING RESULTS'
        print '(A)', '================================================================'
        print '(A,F12.6,A)', '  Total time:    ', t_total, ' s'
        print '(A,F12.4,A)', '  Per iteration: ', 1000.0_dp * t_total / real(niter, dp), ' ms'
        print '(A,F10.2,A)', '  Speedup vs do-concurrent: ', t_ref / t_total, 'x'
        print '(A)', '================================================================'

        ! Correctness check
        CAu_h = CAu_d; CAv_h = CAv_d
#ifdef __NVCOMPILER_LLVM__
        !$omp target update from(CAu_ref, CAv_ref)
#endif
        call check_diff(CAu_h, CAu_ref, CAv_h, CAv_ref, G, nk)
    end if

    ! Cleanup
#ifdef __NVCOMPILER_LLVM__
    !$omp target exit data map(delete: u, v, h, uh, vh, CAu_ref, CAv_ref)
#endif
    call coriolis_end(CS_ref)
    call coriolis_end_cuda(CS_cuda)
    call end_ocean_grid(G)
    deallocate(u, v, h, uh, vh, CAu_ref, CAv_ref, CAu_h, CAv_h)
    deallocate(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d)

contains

    subroutine check_diff(CAu_test, CAu_ref, CAv_test, CAv_ref, G, nk)
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in) :: nk
        real(dp), intent(in) :: CAu_test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: CAu_ref(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: CAv_test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: CAv_ref(G%isd:G%ied, G%jsd:G%jed, nk)

        real(dp) :: max_diff_u, max_diff_v, rel_diff_u, rel_diff_v
        real(dp) :: max_val_u, max_val_v
        integer :: ii, jj, kk

        max_diff_u = 0.0_dp; max_diff_v = 0.0_dp
        max_val_u = 0.0_dp; max_val_v = 0.0_dp
        do kk = 1, nk
            do jj = G%jsc, G%jec
                do ii = G%isc, G%iec - 1
                    max_diff_u = max(max_diff_u, abs(CAu_test(ii, jj, kk) - CAu_ref(ii, jj, kk)))
                    max_val_u = max(max_val_u, abs(CAu_ref(ii, jj, kk)))
                end do
            end do
            do jj = G%jsc, G%jec - 1
                do ii = G%isc, G%iec
                    max_diff_v = max(max_diff_v, abs(CAv_test(ii, jj, kk) - CAv_ref(ii, jj, kk)))
                    max_val_v = max(max_val_v, abs(CAv_ref(ii, jj, kk)))
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

        print '(A)', 'CORRECTNESS (CUDA vs 3D do-concurrent):'
        print '(A,ES15.8)', '  Max abs diff CAu: ', max_diff_u
        print '(A,ES15.8)', '  Max abs diff CAv: ', max_diff_v
        print '(A,ES15.8)', '  Max rel diff CAu: ', rel_diff_u
        print '(A,ES15.8)', '  Max rel diff CAv: ', rel_diff_v
        if (rel_diff_u < 1.0e-10_dp .and. rel_diff_v < 1.0e-10_dp) then
            print '(A)', '  Status: PASS (results match within roundoff)'
        else if (rel_diff_u < 1.0e-6_dp .and. rel_diff_v < 1.0e-6_dp) then
            print '(A)', '  Status: WARN (small differences, likely FP reordering)'
        else
            print '(A)', '  Status: FAIL (significant differences!)'
        end if
    end subroutine check_diff

end program coriolis_cuda_driver
