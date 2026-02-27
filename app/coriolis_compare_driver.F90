!> Comparison driver: OpenACC vs CUDA Coriolis
!!   A: OpenACC  (3D arrays, do concurrent k-j-i)
!!   B: CUDA     (3D arrays, explicit CUDA Fortran kernels)
program coriolis_compare_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid
    use mom6_coriolis, only: coriolis_CS, coriolis_init, CorAdCalc, coriolis_end, &
                             SADOURNY75_ENERGY, ARAKAWA_HSU90, ARAKAWA_LAMB81
    use cudafor
    use mom6_coriolis_cuda, only: coriolis_CS_cuda, coriolis_init_cuda, CorAdCalc_cuda, &
                                  coriolis_end_cuda, SADOURNY75_ENERGY_CUDA, &
                                  ARAKAWA_HSU90_CUDA, ARAKAWA_LAMB81_CUDA
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(coriolis_CS) :: CS_acc
    type(coriolis_CS_cuda) :: CS_cuda

    real(dp), allocatable :: u(:, :, :), v(:, :, :), h(:, :, :)
    real(dp), allocatable :: uh(:, :, :), vh(:, :, :)
    real(dp), allocatable :: CAu_acc(:, :, :), CAv_acc(:, :, :)
    real(dp), allocatable :: CAu_cuda_h(:, :, :), CAv_cuda_h(:, :, :)

    ! Device arrays for CUDA variant
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:), h_d(:,:,:)
    real(dp), device, allocatable :: uh_d(:,:,:), vh_d(:,:,:)
    real(dp), device, allocatable :: CAu_cuda_d(:,:,:), CAv_cuda_d(:,:,:)

    real(dp) :: t_start, t_end, t_acc, t_cuda
    integer :: ni, nj, nk, niter, iter, i, j, k, scheme, scheme_cuda
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

    scheme_cuda = scheme

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Coriolis A/B Comparison'
    print '(A)', '  A: OpenACC  (3D arrays, do concurrent k-j-i)'
    print '(A)', '  B: CUDA     (3D arrays, explicit CUDA kernels)'
    print '(A)', '================================================================'
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
    print '(A)', '================================================================'

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call coriolis_init(CS_acc, G, GV, scheme)
    call coriolis_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, nk, &
                            G%areaT, G%IareaBu, G%CoriolisBu, G%mask2dBu, &
                            G%dyCv, G%dxCu, G%dyCu, G%dxCv, G%IdxCu, G%IdyCv, &
                            scheme_cuda)

    ! Allocate host state arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu_acc(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv_acc(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Allocate device arrays for CUDA variant
    allocate (u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu_cuda_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv_cuda_d(G%isd:G%ied, G%jsd:G%jed, nk))

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

    ! Copy inputs to CUDA device arrays
    u_d = u; v_d = v; h_d = h; uh_d = uh; vh_d = vh

    ! ---- Warmup both implementations ----
    print '(A)', ''
    print '(A)', 'Warming up both variants...'
    call CorAdCalc(u, v, h, uh, vh, CAu_acc, CAv_acc, G, GV, CS_acc)
    call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_cuda_d, CAv_cuda_d, CS_cuda)

    ! ---- Benchmark A: OpenACC (3D k-parallel, do concurrent) ----
    print '(A)', ''
    print '(A)', 'Benchmarking A: OpenACC (do concurrent k-j-i)...'
    t_acc = 0.0_dp
    do iter = 1, niter
        t_start = omp_get_wtime()
        call CorAdCalc(u, v, h, uh, vh, CAu_acc, CAv_acc, G, GV, CS_acc)
        t_end = omp_get_wtime()
        t_acc = t_acc + (t_end - t_start)
    end do

    ! ---- Benchmark B: CUDA explicit kernels ----
    print '(A)', 'Benchmarking B: CUDA (3D arrays, explicit kernels)...'
    t_cuda = 0.0_dp
    do iter = 1, niter
        t_start = omp_get_wtime()
        call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_cuda_d, CAv_cuda_d, CS_cuda)
        t_end = omp_get_wtime()
        t_cuda = t_cuda + (t_end - t_start)
    end do

    ! ---- Copy CUDA results back for comparison ----
    CAu_cuda_h = CAu_cuda_d
    CAv_cuda_h = CAv_cuda_d

    ! ---- Report ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  A: OpenACC (do concurrent):  ', t_acc, ' s'
    print '(A,F12.6,A)', '  B: CUDA (explicit):          ', t_cuda, ' s'
    print '(A)', '  -----------------------------------------------'
    print '(A,F12.6,A)', '  Per-iter A:                  ', t_acc/real(niter, dp), ' s'
    print '(A,F12.6,A)', '  Per-iter B:                  ', t_cuda/real(niter, dp), ' s'
    print '(A)', '  -----------------------------------------------'
    if (t_cuda > 0.0_dp .and. t_acc > 0.0_dp) then
        print '(A,F8.2,A)', '  Speedup B vs A:              ', t_acc/t_cuda, 'x'
    end if

    ! ---- Correctness: CUDA vs OpenACC ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'CORRECTNESS: B (CUDA) vs A (OpenACC)'
    print '(A)', '================================================================'
    call check_diff(CAu_cuda_h, CAu_acc, CAv_cuda_h, CAv_acc, G, nk, 'CUDA vs OpenACC')

    print '(A)', '================================================================'

    ! Cleanup
    call coriolis_end(CS_acc)
    call coriolis_end_cuda(CS_cuda)
    call end_ocean_grid(G)
    deallocate (u, v, h, uh, vh, CAu_acc, CAv_acc)
    deallocate (CAu_cuda_h, CAv_cuda_h)
    deallocate (u_d, v_d, h_d, uh_d, vh_d, CAu_cuda_d, CAv_cuda_d)

contains

    subroutine check_diff(CAu_test, CAu_ref, CAv_test, CAv_ref, G, nk, label)
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in) :: nk
        real(dp), intent(in) :: CAu_test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: CAu_ref(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: CAv_test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: CAv_ref(G%isd:G%ied, G%jsd:G%jed, nk)
        character(len=*), intent(in) :: label

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

        print '(A,A)', '  ', label
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

end program coriolis_compare_driver
