!> Standalone CUDA Fortran driver for continuity PPM solver
!!
!! Tests only the CUDA kernel path (no OpenACC).
!! Command-line: ni nj nk niter (defaults 180 180 75 10)
program continuity_cuda_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use cudafor
    use mom6_types, only: ocean_grid_type, verticalGrid_type, &
                          init_ocean_grid, init_verticalGrid, end_ocean_grid
    use mom6_continuity_cuda, only: continuity_CS_cuda, continuity_init_cuda, &
                                     continuity_PPM_cuda, continuity_end_cuda
    implicit none

    type(ocean_grid_type)    :: G
    type(verticalGrid_type)  :: GV
    type(continuity_CS_cuda) :: CS_cuda

    ! Host state arrays
    real(dp), allocatable :: hin(:,:,:), h(:,:,:), u(:,:,:), uh(:,:,:)

    ! Device arrays
    real(dp), device, allocatable :: u_d(:,:,:), hin_d(:,:,:), h_d(:,:,:), uh_d(:,:,:)

    real(dp) :: dt, t_start, t_end, t_cuda
    integer  :: ni, nj, nk, niter, iter, i, j, k
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    dt = 300.0_dp

    ! Parse command line
    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg); read(arg, *) ni
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg); read(arg, *) nj
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, arg); read(arg, *) nk
    end if
    if (command_argument_count() >= 4) then
        call get_command_argument(4, arg); read(arg, *) niter
    end if

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Continuity PPM — CUDA Fortran Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    print '(A,F8.1,A)', 'dt = ', dt, ' s'
    print '(A)', '================================================================'

    ! ---- Initialize grid and vertical grid ----
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! ---- Initialize CUDA continuity solver ----
    call continuity_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                              G%isc, G%iec, G%jsc, G%jec, nk, &
                              G%IareaT, G%IdxT, G%dy_Cu, G%mask2dT, .true.)

    ! ---- Allocate host state arrays ----
    allocate(hin(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh(G%isd:G%ied, G%jsd:G%jed, nk))

    ! ---- Allocate device arrays ----
    allocate(u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(hin_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh_d(G%isd:G%ied, G%jsd:G%jed, nk))

    ! ---- Initialize state with realistic patterns ----
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                hin(i, j, k) = 4000.0_dp/real(nk, dp) + &
                    10.0_dp * sin(real(i - 1, dp)/real(ni, dp) * 3.14159_dp) * &
                    cos(real(j - 1, dp)/real(nj, dp) * 3.14159_dp) * exp(-real(k, dp)/20.0_dp)
                h(i, j, k) = hin(i, j, k)
                u(i, j, k) = 0.1_dp * sin(real(j - 1, dp)/real(nj, dp) * 3.14159_dp * 2.0_dp) * &
                    exp(-real(k, dp)/30.0_dp)
            end do
        end do
    end do

    ! Copy host arrays to device
    u_d = u
    hin_d = hin
    h_d = hin

    ! ---- Warmup ----
    print '(A)', ''
    print '(A)', 'Warming up CUDA kernels...'
    call continuity_PPM_cuda(u_d, hin_d, h_d, uh_d, dt, CS_cuda)

    ! ---- Benchmark ----
    print '(A)', 'Benchmarking CUDA continuity PPM...'
    t_cuda = 0.0_dp
    do iter = 1, niter
        ! Reset device arrays from host hin each iteration
        hin_d = hin
        h_d = hin

        t_start = omp_get_wtime()
        call continuity_PPM_cuda(u_d, hin_d, h_d, uh_d, dt, CS_cuda)
        t_end = omp_get_wtime()
        t_cuda = t_cuda + (t_end - t_start)
    end do

    ! ---- Copy results back to host ----
    h = h_d
    uh = uh_d

    ! ---- Timing Report ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  CUDA total:          ', t_cuda, ' s'
    print '(A,F12.6,A)', '  CUDA per-iteration:  ', t_cuda / real(niter, dp), ' s'
    print '(A)', '================================================================'

    ! ---- Mass conservation check ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'MASS CONSERVATION (CUDA)'
    print '(A)', '================================================================'
    call verify_mass(hin, h, G, GV)
    print '(A)', '================================================================'

    ! ---- Cleanup ----
    call continuity_end_cuda(CS_cuda)
    call end_ocean_grid(G)
    deallocate(hin, h, u, uh)
    deallocate(u_d, hin_d, h_d, uh_d)

contains

    subroutine verify_mass(h_init, h_final, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_init, h_final

        real(dp) :: mass_init, mass_final, rel_error
        integer :: i, j, k

        mass_init  = 0.0_dp
        mass_final = 0.0_dp

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc, G%iec
                    mass_init  = mass_init  + h_init(i, j, k)
                    mass_final = mass_final + h_final(i, j, k)
                end do
            end do
        end do

        rel_error = abs(mass_final - mass_init) / mass_init

        print '(A,ES15.8)', '  Initial mass:  ', mass_init
        print '(A,ES15.8)', '  Final mass:    ', mass_final
        print '(A,ES15.8)', '  Relative error:', rel_error

        if (rel_error < 1.0e-10_dp) then
            print '(A)', '  Status: PASS'
        else
            print '(A)', '  Status: WARNING'
        end if
    end subroutine verify_mass

end program continuity_cuda_driver
