!> Comparison driver: OpenACC vs CUDA continuity PPM solver
!!   A: OpenACC  -- do concurrent / OpenACC offload
!!   B: CUDA     -- explicit CUDA Fortran kernels
program continuity_compare_driver
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, BT_cont_type, &
                          alloc_BT_cont_type, dealloc_BT_cont_type
    use mom6_continuity, only: continuity_CS, continuity_init, continuity_PPM, continuity_end
    use cudafor
    use mom6_continuity_cuda, only: continuity_CS_cuda, continuity_init_cuda, &
                                     continuity_PPM_cuda, continuity_end_cuda
    implicit none

    type(ocean_grid_type)   :: G
    type(verticalGrid_type) :: GV
    type(continuity_CS)     :: CS
    type(continuity_CS_cuda):: CS_cuda
    type(BT_cont_type), pointer :: BT_cont => null()

    ! Host state arrays
    real(dp), allocatable :: hin(:,:,:), h(:,:,:), u(:,:,:), uh(:,:,:)
    real(dp), allocatable :: h_cuda_h(:,:,:), uh_cuda_h(:,:,:)

    ! Arrays allocated by continuity_init (OpenACC)
    real(dp), allocatable :: uhbt(:,:), u_cor(:,:,:), du_cor(:,:)
    real(dp), allocatable :: por_face_areaU(:,:,:), visc_rem_u(:,:,:)

    ! Device arrays for CUDA variant
    real(dp), device, allocatable :: u_d(:,:,:), hin_d(:,:,:), h_d(:,:,:), uh_d(:,:,:)

    real(dp) :: dt, t_start, t_end, t_acc, t_cuda
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
    print '(A)', 'MOM6 Continuity PPM A/B Comparison'
    print '(A)', '  A: OpenACC  (do concurrent / OpenACC offload)'
    print '(A)', '  B: CUDA     (explicit CUDA Fortran kernels)'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    print '(A,F8.1,A)', 'dt = ', dt, ' s'
    print '(A)', '================================================================'

    ! ---- Initialize grid and vertical grid ----
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! ---- Initialize OpenACC continuity solver ----
    call continuity_init(CS, G, GV, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call alloc_BT_cont_type(BT_cont, G, GV)

    ! ---- Initialize CUDA continuity solver ----
    ! G%IdxT already exists in the grid type
    call continuity_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                              G%isc, G%iec, G%jsc, G%jec, nk, &
                              G%IareaT, G%IdxT, G%dy_Cu, G%mask2dT, .true.)

    ! ---- Allocate host state arrays ----
    allocate(hin(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh_cuda_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! ---- Allocate device arrays for CUDA ----
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

    ! Copy velocity to device (does not change between iterations)
    u_d = u

    ! ---- Warmup both implementations ----
    print '(A)', ''
    print '(A)', 'Warming up both variants...'

    ! State arrays must be on device for OpenACC default(present) kernels
    !$acc enter data copyin(u, hin) create(h, uh)

    ! Warmup OpenACC
    !$acc parallel loop collapse(3) default(present)
    do k = 1, nk
        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                h(i, j, k) = hin(i, j, k)
            end do
        end do
    end do
    call continuity_PPM(u, hin, h, uh, dt, G, GV, CS, por_face_areaU, uhbt, &
                        visc_rem_u, u_cor, BT_cont, du_cor)

    ! Warmup CUDA
    hin_d = hin
    h_d = hin
    call continuity_PPM_cuda(u_d, hin_d, h_d, uh_d, dt, CS_cuda)

    ! ---- Benchmark A: OpenACC ----
    print '(A)', ''
    print '(A)', 'Benchmarking A: OpenACC (do concurrent / acc offload)...'
    t_acc = 0.0_dp
    do iter = 1, niter
        ! Reset h from hin each iteration (on device)
        !$acc parallel loop collapse(3) default(present)
        do k = 1, nk
            do j = G%jsd, G%jed
                do i = G%isd, G%ied
                    h(i, j, k) = hin(i, j, k)
                end do
            end do
        end do

        t_start = omp_get_wtime()
        call continuity_PPM(u, hin, h, uh, dt, G, GV, CS, por_face_areaU, uhbt, &
                            visc_rem_u, u_cor, BT_cont, du_cor)
        t_end = omp_get_wtime()
        t_acc = t_acc + (t_end - t_start)
    end do

    ! Copy OpenACC results back to host and release device data
    !$acc update self(h, uh)
    !$acc exit data delete(u, hin, h, uh)

    ! ---- Benchmark B: CUDA ----
    print '(A)', 'Benchmarking B: CUDA (explicit CUDA kernels)...'
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

    ! ---- Copy CUDA results back to host ----
    h_cuda_h = h_d
    uh_cuda_h = uh_d

    ! ---- Timing Report ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  A: OpenACC total:       ', t_acc, ' s'
    print '(A,F12.6,A)', '  B: CUDA total:          ', t_cuda, ' s'
    print '(A)', '  -----------------------------------------------'
    print '(A,F12.6,A)', '  Per-iter A (OpenACC):   ', t_acc / real(niter, dp), ' s'
    print '(A,F12.6,A)', '  Per-iter B (CUDA):      ', t_cuda / real(niter, dp), ' s'
    print '(A)', '  -----------------------------------------------'
    if (t_cuda > 0.0_dp) then
        print '(A,F8.2,A)', '  Speedup B vs A:         ', t_acc / t_cuda, 'x'
    end if

    ! ---- Correctness: h (thickness) ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'CORRECTNESS: B (CUDA) vs A (OpenACC)'
    print '(A)', '================================================================'
    call check_diff_3d(h_cuda_h, h, G, nk, G%isc, G%iec, G%jsc, G%jec, 'h (thickness)')
    call check_diff_3d(uh_cuda_h, uh, G, nk, G%isc - 1, G%iec, G%jsc, G%jec, 'uh (zonal flux)')

    ! ---- Mass conservation check (OpenACC result) ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'MASS CONSERVATION (OpenACC)'
    print '(A)', '================================================================'
    call verify_mass(hin, h, G, GV)

    ! ---- Mass conservation check (CUDA result) ----
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'MASS CONSERVATION (CUDA)'
    print '(A)', '================================================================'
    call verify_mass(hin, h_cuda_h, G, GV)

    print '(A)', '================================================================'

    ! ---- Cleanup ----
    call continuity_end(CS, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call continuity_end_cuda(CS_cuda)
    call dealloc_BT_cont_type(BT_cont)
    call end_ocean_grid(G)
    deallocate(hin, h, u, uh)
    deallocate(h_cuda_h, uh_cuda_h)
    deallocate(u_d, hin_d, h_d, uh_d)

contains

    subroutine check_diff_3d(test, ref, G, nk, is, ie, js, je, label)
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in) :: nk, is, ie, js, je
        real(dp), intent(in) :: test(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), intent(in) :: ref(G%isd:G%ied, G%jsd:G%jed, nk)
        character(len=*), intent(in) :: label

        real(dp) :: max_diff, max_val, rel_diff
        integer :: ii, jj, kk

        max_diff = 0.0_dp
        max_val  = 0.0_dp

        do kk = 1, nk
            do jj = js, je
                do ii = is, ie
                    max_diff = max(max_diff, abs(test(ii, jj, kk) - ref(ii, jj, kk)))
                    max_val  = max(max_val, abs(ref(ii, jj, kk)))
                end do
            end do
        end do

        if (max_val > 0.0_dp) then
            rel_diff = max_diff / max_val
        else
            rel_diff = 0.0_dp
        end if

        print '(A,A)', '  Field: ', label
        print '(A,ES15.8)', '  Max abs diff:  ', max_diff
        print '(A,ES15.8)', '  Max ref value: ', max_val
        print '(A,ES15.8)', '  Max rel diff:  ', rel_diff
        if (rel_diff < 1.0e-10_dp) then
            print '(A)', '  Status: PASS (results match within roundoff)'
        else if (rel_diff < 1.0e-6_dp) then
            print '(A)', '  Status: WARN (small differences, likely FP reordering)'
        else
            print '(A)', '  Status: FAIL (significant differences!)'
        end if
        print '(A)', ''
    end subroutine check_diff_3d

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

end program continuity_compare_driver
