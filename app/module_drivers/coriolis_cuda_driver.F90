!> Standalone CUDA Fortran driver for Coriolis solver
program coriolis_cuda_driver
    use cudafor
    use mom6_coriolis_cuda, only: coriolis_CS_cuda, coriolis_init_cuda, CorAdCalc_cuda, &
                                   coriolis_end_cuda, SADOURNY75_ENERGY_CUDA, &
                                   ARAKAWA_HSU90_CUDA, ARAKAWA_LAMB81_CUDA
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, PI
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    implicit none

    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(coriolis_CS_cuda) :: CS_cuda

    ! Host arrays
    real(dp), allocatable :: u(:,:,:), v(:,:,:), h(:,:,:)
    real(dp), allocatable :: uh(:,:,:), vh(:,:,:)
    real(dp), allocatable :: CAu_h(:,:,:), CAv_h(:,:,:)

    ! Device arrays
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:), h_d(:,:,:)
    real(dp), device, allocatable :: uh_d(:,:,:), vh_d(:,:,:)
    real(dp), device, allocatable :: CAu_d(:,:,:), CAv_d(:,:,:)

    real(dp) :: t_start, t_end, t_total
    integer :: ni, nj, nk, niter, iter, i, j, k, scheme_cuda
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    scheme_cuda = SADOURNY75_ENERGY_CUDA

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
            scheme_cuda = SADOURNY75_ENERGY_CUDA
        case ('hsu', 'HSU', '2')
            scheme_cuda = ARAKAWA_HSU90_CUDA
        case ('lamb', 'LAMB', '3')
            scheme_cuda = ARAKAWA_LAMB81_CUDA
        end select
    end if

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Coriolis — CUDA Fortran Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    select case (scheme_cuda)
    case (SADOURNY75_ENERGY_CUDA)
        print '(A)', 'Scheme: Sadourny (1975) Energy-conserving'
    case (ARAKAWA_HSU90_CUDA)
        print '(A)', 'Scheme: Arakawa-Hsu (1990)'
    case (ARAKAWA_LAMB81_CUDA)
        print '(A)', 'Scheme: Arakawa-Lamb (1981)'
    end select
    print '(A)', '================================================================'

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! Initialize CUDA Coriolis solver
    call coriolis_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, nk, &
                            G%areaT, G%IareaBu, G%CoriolisBu, G%mask2dBu, &
                            G%dyCv, G%dxCu, G%dyCu, G%dxCv, G%IdxCu, G%IdyCv, &
                            scheme_cuda)

    ! Allocate host arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Allocate device arrays
    allocate (u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv_d(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Initialize state with realistic patterns
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                h(i, j, k) = 4000.0_dp / real(nk, dp)
                u(i, j, k) = 0.1_dp * sin(real(j - 1, dp) / real(nj, dp) * PI * 2.0_dp) * &
                              exp(-real(k, dp) / 30.0_dp)
                v(i, j, k) = 0.1_dp * cos(real(i - 1, dp) / real(ni, dp) * PI * 2.0_dp) * &
                              exp(-real(k, dp) / 30.0_dp)
                uh(i, j, k) = u(i, j, k) * h(i, j, k) * G%dyCu(i, j)
                vh(i, j, k) = v(i, j, k) * h(i, j, k) * G%dxCv(i, j)
            end do
        end do
    end do

    ! Copy host → device
    u_d = u; v_d = v; h_d = h; uh_d = uh; vh_d = vh

    ! Warmup
    print '(A)', ''
    print '(A)', 'Warming up CUDA kernel...'
    call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS_cuda)

    ! Benchmark
    print '(A)', 'Benchmarking CUDA (explicit kernels)...'
    t_total = 0.0_dp
    do iter = 1, niter
        t_start = omp_get_wtime()
        call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS_cuda)
        t_end = omp_get_wtime()
        t_total = t_total + (t_end - t_start)
    end do

    ! Copy results back to host
    CAu_h = CAu_d
    CAv_h = CAv_d

    ! Sanity check: print max values
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'SANITY CHECK'
    print '(A)', '================================================================'
    print '(A,ES15.8)', '  Max |CAu|: ', maxval(abs(CAu_h(G%isc:G%iec-1, G%jsc:G%jec, :)))
    print '(A,ES15.8)', '  Max |CAv|: ', maxval(abs(CAv_h(G%isc:G%iec, G%jsc:G%jec-1, :)))

    ! Timing report
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  Total:         ', t_total, ' s'
    print '(A,F12.6,A)', '  Per-iteration: ', t_total / real(niter, dp), ' s'
    print '(A)', '================================================================'

    ! Cleanup
    call coriolis_end_cuda(CS_cuda)
    call end_ocean_grid(G)
    deallocate (u, v, h, uh, vh, CAu_h, CAv_h)
    deallocate (u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d)

end program coriolis_cuda_driver
