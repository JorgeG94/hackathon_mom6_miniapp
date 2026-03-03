!> Standalone CUDA Fortran driver for horizontal viscosity solver
!!
!! Usage:
!!   ./hor_visc_cuda_driver [ni] [nj] [nk] [niter]
!!   Defaults: 180 180 75 10
!!
program hor_visc_cuda_driver
    use cudafor
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, PI
    use mom6_hor_visc, only: hor_visc_CS, hor_visc_init, hor_visc_end
    use mom6_hor_visc_cuda, only: hor_visc_CS_cuda, hor_visc_init_cuda, &
                                   hor_visc_cuda, hor_visc_end_cuda
    use cuda_workspace, only: cuda_workspace_type, workspace_init, workspace_end
    implicit none


    type(ocean_grid_type)   :: G
    type(verticalGrid_type) :: GV
    type(hor_visc_CS)       :: CS
    type(hor_visc_CS_cuda)  :: CS_cuda
    type(cuda_workspace_type) :: ws

    ! Host arrays
    real(dp), allocatable :: u(:,:,:), v(:,:,:), h(:,:,:)
    real(dp), allocatable :: diffu_h(:,:,:), diffv_h(:,:,:)

    ! Device arrays
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:), h_d(:,:,:)
    real(dp), device, allocatable :: diffu_d(:,:,:), diffv_d(:,:,:)

    real(dp) :: t_start, t_end, t_cuda
    real(dp) :: Kh_val
    real(dp) :: max_diffu, max_diffv
    integer  :: ni, nj, nk, niter, iter, i, j, k
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10
    Kh_val = 100.0_dp

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
    print '(A)', 'MOM6 Horizontal Viscosity — CUDA Fortran Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'Iterations: ', niter
    print '(A,ES12.4)', 'Kh [m2/s]:  ', Kh_val
    print '(A)', '================================================================'

    ! Initialize grid and vertical grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! Initialize OpenACC control structure (metrics extraction only)
    call hor_visc_init(CS, G, GV, Kh=Kh_val)

    ! Initialize CUDA control structure from precomputed metrics
    call hor_visc_init_cuda(CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, nk, &
                            Kh_val, CS%h_neglect, &
                            CS%DY_dxT, CS%DX_dyT, CS%DY_dxBu, CS%DX_dyBu, &
                            G%IdyCu, G%IdxCu, G%IdyCv, G%IdxCv, &
                            G%IareaCu, G%IareaCv, &
                            G%mask2dT, G%mask2dBu, &
                            CS%reduction_xx, CS%reduction_xy, &
                            CS%dy2h, CS%dx2h, CS%dy2q, CS%dx2q)

    ! Allocate host arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffu_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffv_h(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Allocate device arrays
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

    ! Copy inputs to device once
    u_d = u; v_d = v; h_d = h

    ! Initialize workspace pool (2 3D slots for hor_visc scratch)
    call workspace_init(ws, G%isd, G%ied, G%jsd, G%jed, nk, 2, 0)

    ! Warmup
    print '(A)', ''
    print '(A)', 'Warming up CUDA kernel...'
    call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, CS_cuda, nk, ws)

    ! Benchmark
    print '(A)', 'Benchmarking CUDA (Laplacian)...'
    t_cuda = 0.0_dp
    do iter = 1, niter
        t_start = omp_get_wtime()
        call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, CS_cuda, nk, ws)
        t_end = omp_get_wtime()
        t_cuda = t_cuda + (t_end - t_start)
    end do

    ! Copy results back to host
    diffu_h = diffu_d
    diffv_h = diffv_d

    ! Sanity check: print max absolute values
    max_diffu = 0.0_dp
    max_diffv = 0.0_dp
    do k = 1, nk
        do j = G%jsc, G%jec
            do i = G%isc - 1, G%iec
                max_diffu = max(max_diffu, abs(diffu_h(i, j, k)))
            end do
        end do
    end do
    do k = 1, nk
        do j = G%jsc - 1, G%jec
            do i = G%isc, G%iec
                max_diffv = max(max_diffv, abs(diffv_h(i, j, k)))
            end do
        end do
    end do

    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'SANITY CHECK'
    print '(A)', '================================================================'
    print '(A,ES15.8)', '  Max |diffu|: ', max_diffu
    print '(A,ES15.8)', '  Max |diffv|: ', max_diffv

    ! Timing report
    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'TIMING RESULTS'
    print '(A)', '================================================================'
    print '(A,F12.6,A)', '  CUDA total:    ', t_cuda, ' s'
    print '(A,F12.6,A)', '  CUDA per-iter: ', t_cuda/real(niter, dp), ' s'
    print '(A)', '================================================================'

    ! Cleanup
    call workspace_end(ws)
    call hor_visc_end(CS)
    call hor_visc_end_cuda(CS_cuda)
    call end_ocean_grid(G)
    deallocate (u, v, h, diffu_h, diffv_h)
    deallocate (u_d, v_d, h_d, diffu_d, diffv_d)

end program hor_visc_cuda_driver
