!> MPI-aware RK2 driver using CUDA Fortran kernels
!!
!! Based on rk2_cuda_driver.F90 with MPI domain decomposition and halo exchanges.
!! Physics modules remain completely unmodified.
!!
!! Usage: mpirun -np N rk2_mpi_cuda_driver ni nj nk niter bt_nsteps [npes_x npes_y]
!!
program rk2_mpi_cuda_driver
    use cudafor
    use mpi
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64, int64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, &
                          G_EARTH, HALO_WIDTH, PI, OMEGA, EARTH_RADIUS
    use mom6_mpi_domain, only: mpi_domain_type, mpi_domain_init, &
                               mpi_domain_end
    use mom6_mpi_halo_cuda, only: halo_exchange_3d_cuda, halo_exchange_2d_cuda, halo_cleanup_cuda
    use mom6_profiler, only: profiler_init, profiler_end, profiler_start, profiler_stop, &
                             profiler_report

    ! CUDA kernel modules
    use mom6_continuity_cuda, only: continuity_CS_cuda, continuity_init_cuda, &
                                     continuity_PPM_cuda, continuity_end_cuda, &
                                     BT_cont_type_cuda, alloc_BT_cont_type_cuda, &
                                     dealloc_BT_cont_type_cuda
    use mom6_coriolis_cuda, only: coriolis_CS_cuda, coriolis_init_cuda, &
                                   CorAdCalc_cuda, coriolis_end_cuda, &
                                   SADOURNY75_ENERGY_CUDA
    use mom6_barotropic_cuda, only: barotropic_CS_cuda, barotropic_init_cuda, &
                                     btstep_cuda, barotropic_end_cuda, &
                                     btstep_cuda_init_state, btstep_cuda_do_step, &
                                     btstep_cuda_get_output
    use mom6_vert_visc_cuda, only: vert_visc_CS_cuda, vert_visc_init_cuda, &
                                    vert_visc_cra_cuda, vert_visc_end_cuda
    use mom6_hor_visc_cuda, only: hor_visc_CS_cuda, hor_visc_init_cuda, &
                                   hor_visc_cuda, hor_visc_end_cuda
    use cuda_workspace, only: cuda_workspace_type, workspace_init, workspace_end, workspace_copy_3d

    implicit none


    ! Grid structures
    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(mpi_domain_type) :: MD

    ! CUDA control structures
    type(continuity_CS_cuda) :: cont_CS
    type(coriolis_CS_cuda)   :: cor_CS
    type(barotropic_CS_cuda) :: bt_CS_cuda
    type(vert_visc_CS_cuda)  :: visc_CS
    type(hor_visc_CS_cuda)   :: hvisc_CS

    ! Shared GPU workspace pool
    type(cuda_workspace_type) :: ws

    ! Host arrays
    real(dp), allocatable :: u_h(:,:,:), v_h(:,:,:)
    real(dp), allocatable :: h_h(:,:,:), h0_h(:,:,:)
    real(dp), allocatable :: uh_h(:,:,:), vh_h(:,:,:)
    real(dp), allocatable :: eta_h(:,:), ubt_h(:,:), vbt_h(:,:)

    ! Device 3D arrays
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:)
    real(dp), device, allocatable :: h_d(:,:,:), h0_d(:,:,:)
    real(dp), device, allocatable :: uh_d(:,:,:), vh_d(:,:,:)
    real(dp), device, allocatable :: CAu_d(:,:,:), CAv_d(:,:,:)
    real(dp), device, allocatable :: up_d(:,:,:), vp_d(:,:,:)
    real(dp), device, allocatable :: diffu_d(:,:,:), diffv_d(:,:,:)

    ! Device 2D arrays
    real(dp), device, allocatable :: eta_d(:,:), ubt_d(:,:), vbt_d(:,:)
    real(dp), device, allocatable :: ubt_av_d(:,:), vbt_av_d(:,:), eta_av_d(:,:)
    real(dp), device, allocatable :: taux_d(:,:), tauy_d(:,:)
    real(dp), device, allocatable :: dyCu_d(:,:), dxCv_d(:,:)

    ! Device arrays for full continuity solver
    real(dp), device, allocatable :: uhbt_cont_d(:,:)
    type(BT_cont_type_cuda) :: BT_cont_cuda

    ! Timing
    real(dp) :: t_start, t_end, t_init_start, t_init_end
    real(dp) :: t_total, t_coriolis, t_barotropic, t_continuity
    real(dp) :: t_vert_visc, t_hor_visc, t_compute, t_halo
    real(dp) :: t_iter_start, t_iter_end

    ! Parameters
    integer :: ni, nj, nk, niter, bt_nsteps
    integer :: npes_x, npes_y
    integer :: i, j, k, iter, istat, bt_n
    integer :: ierr, nprocs, local_rank, dims(2)
    integer :: node_comm, node_rank
    real(dp) :: dt
    character(len=32) :: arg

    ! MPI initialization
    call MPI_Init(ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, local_rank, ierr)

    ! Get node-local rank for GPU device assignment
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                             MPI_INFO_NULL, node_comm, ierr)
    call MPI_Comm_rank(node_comm, node_rank, ierr)
    istat = cudaSetDevice(node_rank)
    call MPI_Comm_free(node_comm, ierr)

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10; bt_nsteps = 30
    npes_x = 0; npes_y = 0
    dt = 300.0_dp

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
        call get_command_argument(5, arg); read (arg, *) bt_nsteps
    end if
    if (command_argument_count() >= 6) then
        call get_command_argument(6, arg); read (arg, *) npes_x
    end if
    if (command_argument_count() >= 7) then
        call get_command_argument(7, arg); read (arg, *) npes_y
    end if

    ! Auto-decompose if not specified
    if (npes_x == 0 .or. npes_y == 0) then
        dims = [0, 0]
        call MPI_Dims_create(nprocs, 2, dims, ierr)
        npes_x = dims(1)
        npes_y = dims(2)
    end if

    ! Initialize MPI domain decomposition
    call mpi_domain_init(MD, ni, nj, npes_x, npes_y)

    call profiler_init()
    call profiler_start("Total")

    if (MD%rank == 0) then
        print '(A)', '================================================================'
        print '(A)', 'MOM6 Split RK2 — CUDA Fortran MPI Driver'
        print '(A)', '================================================================'
        print '(A,I5,A,I5,A,I4)', 'Global grid: ', ni, ' x ', nj, ' x ', nk
        print '(A,I4,A,I4,A,I4)', 'PE layout: ', npes_x, ' x ', npes_y, ' = ', nprocs
        print '(A,I4)', 'RK2 iterations: ', niter
        print '(A,I4)', 'Barotropic substeps: ', bt_nsteps
        print '(A,F8.1,A)', 'dt = ', dt, ' s'
        call check_bt_cfl(dt, bt_nsteps, 10.0_dp, 4000.0_dp)
        print '(A)', '================================================================'
    end if

    !=========================================================================
    ! INITIALIZATION
    !=========================================================================
    call profiler_start("Initialization")
    t_init_start = omp_get_wtime()

    call init_ocean_grid_mpi_cuda(G, MD, GV, nk, 10.0_dp, 45.0_dp)

    ! Continuity CUDA init
    call continuity_init_cuda(cont_CS, G%isd, G%ied, G%jsd, G%jed, &
                              G%isc, G%iec, G%jsc, G%jec, nk, &
                              G%IareaT, G%IdxT, G%dy_Cu, G%mask2dT, .true., &
                              G%dxT, G%areaT, G%dxCu, G%mask2dCu)

    ! Coriolis CUDA init
    call coriolis_init_cuda(cor_CS, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, nk, &
                            G%areaT, G%IareaBu, G%CoriolisBu, G%mask2dBu, &
                            G%dyCv, G%dxCu, G%dyCu, G%dxCv, G%IdxCu, G%IdyCv, &
                            SADOURNY75_ENERGY_CUDA)

    ! Barotropic CUDA init (direct from grid metrics, no OpenACC)
    call barotropic_init_cuda_from_grid(bt_CS_cuda, G, dt, bt_nsteps)

    ! Vertical viscosity CUDA init
    call vert_visc_init_cuda(visc_CS, G%isd, G%ied, G%jsd, G%jed, &
                             G%isc, G%iec, G%jsc, G%jec, nk, &
                             G%mask2dCu, G%mask2dCv, &
                             1.0e-4_dp, 1.0e-2_dp, 1.0e-2_dp, 50.0_dp, 10.0_dp)

    ! Horizontal viscosity CUDA init (direct from grid metrics, no OpenACC)
    call hor_visc_init_cuda_from_grid(hvisc_CS, G, nk, 100.0_dp)

    ! Allocate host arrays
    allocate(u_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(v_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h0_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(vh_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(eta_h(G%isd:G%ied, G%jsd:G%jed))
    allocate(ubt_h(G%isd:G%ied, G%jsd:G%jed))
    allocate(vbt_h(G%isd:G%ied, G%jsd:G%jed))

    ! Allocate device arrays
    allocate(u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(v_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h0_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(vh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAu_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAv_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(up_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(vp_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(diffu_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(diffv_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(eta_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(ubt_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(vbt_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(ubt_av_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(vbt_av_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(eta_av_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(taux_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(tauy_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(dyCu_d(G%isd:G%ied, G%jsd:G%jed))
    allocate(dxCv_d(G%isd:G%ied, G%jsd:G%jed))
    dyCu_d = G%dyCu
    dxCv_d = G%dxCv
    ! Full continuity solver device arrays
    allocate(uhbt_cont_d(G%isd:G%ied, G%jsd:G%jed))
    uhbt_cont_d = 0.0_dp
    call alloc_BT_cont_type_cuda(BT_cont_cuda, G%isd, G%ied, G%jsd, G%jed, nk)

    ! Initialize shared GPU workspace pool (9 3D + 6 2D slots)
    call workspace_init(ws, G%isd, G%ied, G%jsd, G%jed, nk, 9, 6)

    ! Initialize host state using global coordinates
    call initialize_state_mpi(u_h, v_h, h_h, h0_h, eta_h, ubt_h, vbt_h, G, GV, MD)

    ! Copy to device
    u_d = u_h; v_d = v_h; h_d = h_h; h0_d = h0_h
    eta_d = eta_h; ubt_d = ubt_h; vbt_d = vbt_h

    ! Initialize surface stress
    call initialize_stress_mpi(taux_d, tauy_d, G, MD)

    ! Initial halo exchanges
    call halo_exchange_3d_cuda(u_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d_cuda(v_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d_cuda(h_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d_cuda(h0_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_2d_cuda(eta_d, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
    call halo_exchange_2d_cuda(ubt_d, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
    call halo_exchange_2d_cuda(vbt_d, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)

    t_init_end = omp_get_wtime()
    call profiler_stop("Initialization")

    if (MD%rank == 0) then
        print '(A)', ''
        print '(A,I5,A,I5)', 'Local compute domain: ', MD%ni_local, ' x ', MD%nj_local
        print '(A,F12.6,A)', 'Initialization time: ', t_init_end - t_init_start, ' s'
        print '(A)', ''
    end if

    !=========================================================================
    ! WARMUP
    !=========================================================================
    call profiler_start("Warmup")
    if (MD%rank == 0) print '(A)', 'Warming up CUDA kernels...'
    call compute_transports_cuda(u_d, v_d, h_d, uh_d, vh_d, dyCu_d, dxCv_d, G, GV)
    call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, hvisc_CS, nk, ws)
    call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, cor_CS, ws)
    call vert_visc_cra_cuda(u_d, v_d, h_d, dt, visc_CS, taux_d, tauy_d, ws)
    call btstep_cuda(eta_d, ubt_d, vbt_d, ubt_av_d, vbt_av_d, eta_av_d, bt_CS_cuda)
    call continuity_PPM_cuda(u_d, h0_d, h_d, uh_d, dt, cont_CS, ws, &
                             uhbt_cont_d, BT_cont_cuda)

    ! Reset after warmup
    u_d = u_h; v_d = v_h; h_d = h_h
    eta_d = eta_h; ubt_d = ubt_h; vbt_d = vbt_h

    ! Re-exchange halos after reset
    call halo_exchange_3d_cuda(u_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d_cuda(v_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d_cuda(h_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call profiler_stop("Warmup")

    !=========================================================================
    ! RK2 TIME-STEPPING LOOP
    !=========================================================================
    if (MD%rank == 0) then
        print '(A)', 'Running split RK2 time-stepping (CUDA + MPI)...'
        print '(A)', ''
    end if

    t_coriolis = 0.0_dp; t_barotropic = 0.0_dp; t_continuity = 0.0_dp
    t_vert_visc = 0.0_dp; t_hor_visc = 0.0_dp; t_compute = 0.0_dp; t_halo = 0.0_dp

    call profiler_start("RK2_step", nvtx_only=.true.)
    do iter = 1, niter
        t_iter_start = omp_get_wtime()

        ! Reset thickness
        h_d = h0_d

        !=====================================================================
        ! PREDICTOR PHASE
        !=====================================================================

        ! 1. Compute layer transports
        call profiler_start("Transports")
        call compute_transports_cuda(u_d, v_d, h_d, uh_d, vh_d, dyCu_d, dxCv_d, G, GV)
        call profiler_stop("Transports")

        ! Halo exchange: uh, vh
        t_start = omp_get_wtime()
        istat = cudaDeviceSynchronize()
        call halo_exchange_3d_cuda(uh_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        call halo_exchange_3d_cuda(vh_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        t_end = omp_get_wtime()
        t_halo = t_halo + (t_end - t_start)

        ! 2. Horizontal viscosity
        call profiler_start("HorVisc")
        t_start = omp_get_wtime()
        call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, hvisc_CS, nk, ws)
        t_end = omp_get_wtime()
        t_hor_visc = t_hor_visc + (t_end - t_start)
        call profiler_stop("HorVisc")

        ! 3. Coriolis
        call profiler_start("Coriolis")
        t_start = omp_get_wtime()
        call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, cor_CS, ws)
        t_end = omp_get_wtime()
        t_coriolis = t_coriolis + (t_end - t_start)
        call profiler_stop("Coriolis")

        ! 4. Predictor velocity update
        call profiler_start("VelUpdate_pred")
        call velocity_update_cuda(up_d, u_d, CAu_d, diffu_d, dt, &
                                  G%isc, G%iec-1, G%jsc, G%jec, nk)
        call velocity_update_cuda(vp_d, v_d, CAv_d, diffv_d, dt, &
                                  G%isc, G%iec, G%jsc, G%jec-1, nk)
        call profiler_stop("VelUpdate_pred")

        ! 5. Vertical viscosity
        call profiler_start("VertVisc")
        t_start = omp_get_wtime()
        call vert_visc_cra_cuda(up_d, vp_d, h_d, dt, visc_CS, taux_d, tauy_d, ws)
        t_end = omp_get_wtime()
        t_vert_visc = t_vert_visc + (t_end - t_start)
        call profiler_stop("VertVisc")

        ! 6. Barotropic predictor (split API with halo exchanges every substep)
        call profiler_start("Barotropic")
        t_start = omp_get_wtime()
        call btstep_cuda_init_state(bt_CS_cuda, eta_d, ubt_d, vbt_d)
        do bt_n = 1, bt_CS_cuda%nstep
            call btstep_cuda_do_step(bt_CS_cuda, bt_n)
            if (bt_n < bt_CS_cuda%nstep) then
                istat = cudaDeviceSynchronize()
                call halo_exchange_2d_cuda(bt_CS_cuda%ubt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                call halo_exchange_2d_cuda(bt_CS_cuda%vbt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                call halo_exchange_2d_cuda(bt_CS_cuda%eta, G%isd, G%ied, G%jsd, G%jed, MD, 1)
            end if
        end do
        call btstep_cuda_get_output(bt_CS_cuda, ubt_av_d, vbt_av_d, eta_av_d)
        t_end = omp_get_wtime()
        t_barotropic = t_barotropic + (t_end - t_start)
        call profiler_stop("Barotropic")

        ! Halo exchange: up, vp
        t_start = omp_get_wtime()
        istat = cudaDeviceSynchronize()
        call halo_exchange_3d_cuda(up_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        call halo_exchange_3d_cuda(vp_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        t_end = omp_get_wtime()
        t_halo = t_halo + (t_end - t_start)

        ! 7. Continuity
        call profiler_start("Continuity")
        t_start = omp_get_wtime()
        call continuity_PPM_cuda(up_d, h0_d, h_d, uh_d, dt, cont_CS, ws, &
                                 uhbt_cont_d, BT_cont_cuda)
        t_end = omp_get_wtime()
        t_continuity = t_continuity + (t_end - t_start)
        call profiler_stop("Continuity")

        ! Halo exchange: h
        t_start = omp_get_wtime()
        istat = cudaDeviceSynchronize()
        call halo_exchange_3d_cuda(h_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        t_end = omp_get_wtime()
        t_halo = t_halo + (t_end - t_start)

        !=====================================================================
        ! CORRECTOR PHASE
        !=====================================================================

        ! 8. Recompute transports
        call profiler_start("Transports")
        call compute_transports_cuda(up_d, vp_d, h_d, uh_d, vh_d, dyCu_d, dxCv_d, G, GV)
        call profiler_stop("Transports")

        t_start = omp_get_wtime()
        istat = cudaDeviceSynchronize()
        call halo_exchange_3d_cuda(uh_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        call halo_exchange_3d_cuda(vh_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        t_end = omp_get_wtime()
        t_halo = t_halo + (t_end - t_start)

        ! 9. Horizontal viscosity
        call profiler_start("HorVisc")
        t_start = omp_get_wtime()
        call hor_visc_cuda(up_d, vp_d, h_d, diffu_d, diffv_d, hvisc_CS, nk, ws)
        t_end = omp_get_wtime()
        t_hor_visc = t_hor_visc + (t_end - t_start)
        call profiler_stop("HorVisc")

        ! 10. Coriolis
        call profiler_start("Coriolis")
        t_start = omp_get_wtime()
        call CorAdCalc_cuda(up_d, vp_d, h_d, uh_d, vh_d, CAu_d, CAv_d, cor_CS, ws)
        t_end = omp_get_wtime()
        t_coriolis = t_coriolis + (t_end - t_start)
        call profiler_stop("Coriolis")

        ! 11. Corrector velocity update
        call profiler_start("VelUpdate_corr")
        call velocity_update_cuda(u_d, u_d, CAu_d, diffu_d, 0.5_dp*dt, &
                                  G%isc, G%iec-1, G%jsc, G%jec, nk)
        call velocity_update_cuda(v_d, v_d, CAv_d, diffv_d, 0.5_dp*dt, &
                                  G%isc, G%iec, G%jsc, G%jec-1, nk)
        call profiler_stop("VelUpdate_corr")

        ! 12. Vertical viscosity
        call profiler_start("VertVisc")
        t_start = omp_get_wtime()
        call vert_visc_cra_cuda(u_d, v_d, h_d, 0.5_dp*dt, visc_CS, taux_d, tauy_d, ws)
        t_end = omp_get_wtime()
        t_vert_visc = t_vert_visc + (t_end - t_start)
        call profiler_stop("VertVisc")

        ! 13. Barotropic corrector (split API with halo exchanges every substep)
        call profiler_start("Barotropic")
        t_start = omp_get_wtime()
        call btstep_cuda_init_state(bt_CS_cuda, eta_av_d, ubt_av_d, vbt_av_d)
        do bt_n = 1, bt_CS_cuda%nstep
            call btstep_cuda_do_step(bt_CS_cuda, bt_n)
            if (bt_n < bt_CS_cuda%nstep) then
                istat = cudaDeviceSynchronize()
                call halo_exchange_2d_cuda(bt_CS_cuda%ubt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                call halo_exchange_2d_cuda(bt_CS_cuda%vbt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                call halo_exchange_2d_cuda(bt_CS_cuda%eta, G%isd, G%ied, G%jsd, G%jed, MD, 1)
            end if
        end do
        call btstep_cuda_get_output(bt_CS_cuda, ubt_av_d, vbt_av_d, eta_d)
        t_end = omp_get_wtime()
        t_barotropic = t_barotropic + (t_end - t_start)
        call profiler_stop("Barotropic")

        ! 14. Final continuity
        call profiler_start("Continuity")
        t_start = omp_get_wtime()
        call workspace_copy_3d(ws%s3d(:,:,1:nk,4), h_d, G%ied-G%isd+1, G%jed-G%jsd+1, nk)
        call continuity_PPM_cuda(u_d, ws%s3d(:,:,1:nk,4), h_d, uh_d, 0.5_dp*dt, cont_CS, ws, &
                                 uhbt_cont_d, BT_cont_cuda)
        t_end = omp_get_wtime()
        t_continuity = t_continuity + (t_end - t_start)
        call profiler_stop("Continuity")

        ! End-of-step halo exchanges
        t_start = omp_get_wtime()
        istat = cudaDeviceSynchronize()
        call halo_exchange_3d_cuda(u_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        call halo_exchange_3d_cuda(v_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        call halo_exchange_3d_cuda(h_d, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
        call halo_exchange_2d_cuda(eta_d, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
        call halo_exchange_2d_cuda(ubt_d, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
        call halo_exchange_2d_cuda(vbt_d, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
        t_end = omp_get_wtime()
        t_halo = t_halo + (t_end - t_start)

        t_iter_end = omp_get_wtime()
        t_compute = t_compute + (t_iter_end - t_iter_start)
        if (MD%rank == 0) then
            print '(A,I4,A,F10.6,A)', '  RK2 iteration ', iter, ':  ', &
                t_iter_end - t_iter_start, ' s'
        end if
    end do
    call profiler_stop("RK2_step")

    !=========================================================================
    ! TIMING REPORT
    !=========================================================================
    t_total = t_coriolis + t_barotropic + t_continuity + t_vert_visc + t_hor_visc

    if (MD%rank == 0) then
        print '(A)', ''
        print '(A)', '================================================================'
        print '(A)', 'Timing Results'
        print '(A)', '================================================================'
        print '(A,F12.6)', 'Compute (wall clock):  ', t_compute
        print '(A,F12.6)', 'Compute (sum of parts):', t_total
        print '(A,F12.6)', 'Halo exchange:         ', t_halo
        print '(A)', '----------------------------------------------------------------'
        print '(A,F12.6,A,F5.1,A)', '  Coriolis:            ', t_coriolis, &
            '  (', 100.0_dp*t_coriolis/t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Hor viscosity:       ', t_hor_visc, &
            '  (', 100.0_dp*t_hor_visc/t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Vert viscosity:      ', t_vert_visc, &
            '  (', 100.0_dp*t_vert_visc/t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Barotropic:          ', t_barotropic, &
            '  (', 100.0_dp*t_barotropic/t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Continuity:          ', t_continuity, &
            '  (', 100.0_dp*t_continuity/t_total, '%)'
        print '(A)', '----------------------------------------------------------------'
        print '(A,F12.6)', 'Time per RK2 step:     ', t_compute/real(niter, dp)
        print '(A)', '================================================================'
    end if

    !=========================================================================
    ! VERIFICATION
    !=========================================================================
    u_h = u_d; v_h = v_d; h_h = h_d; eta_h = eta_d
    call verify_state_mpi(h0_h, h_h, u_h, v_h, eta_h, G, GV, MD)

    !=========================================================================
    ! CLEANUP
    !=========================================================================
    call profiler_start("Finalize")
    call workspace_end(ws)
    call dealloc_BT_cont_type_cuda(BT_cont_cuda)
    call continuity_end_cuda(cont_CS)
    call coriolis_end_cuda(cor_CS)
    call barotropic_end_cuda(bt_CS_cuda)
    call vert_visc_end_cuda(visc_CS)
    call hor_visc_end_cuda(hvisc_CS)
    call end_ocean_grid_cuda(G)

    deallocate(u_h, v_h, h_h, h0_h, uh_h, vh_h)
    deallocate(eta_h, ubt_h, vbt_h)
    deallocate(u_d, v_d, h_d, h0_d, uh_d, vh_d)
    deallocate(CAu_d, CAv_d, up_d, vp_d, diffu_d, diffv_d)
    deallocate(eta_d, ubt_d, vbt_d, ubt_av_d, vbt_av_d, eta_av_d)
    deallocate(taux_d, tauy_d, dyCu_d, dxCv_d)
    deallocate(uhbt_cont_d)
    call profiler_stop("Finalize")

    call profiler_stop("Total")
    if (MD%rank == 0) then
        call profiler_report("RK2 CUDA MPI Driver", root_region="Total")
    end if
    call profiler_end()

    call halo_cleanup_cuda()
    call mpi_domain_end(MD)
    call MPI_Finalize(ierr)

contains

    subroutine initialize_state_mpi(u, v, h, h0, eta, ubt, vbt, G, GV, MD)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        type(mpi_domain_type), intent(in) :: MD
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: u, v, h, h0
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: eta, ubt, vbt

        integer :: i, j, k, i_global, j_global
        real(dp) :: total_depth

        total_depth = 4000.0_dp

        do k = 1, GV%ke
            do j = G%jsd, G%jed
                do i = G%isd, G%ied
                    i_global = i + MD%i_offset
                    j_global = j + MD%j_offset

                    h0(i, j, k) = total_depth / real(GV%ke, dp) + &
                                  10.0_dp * sin(real(i_global - 1, dp) / real(MD%ni_global, dp) * PI) * &
                                  cos(real(j_global - 1, dp) / real(MD%nj_global, dp) * PI) * &
                                  exp(-real(k, dp) / 20.0_dp)
                    h(i, j, k) = h0(i, j, k)
                    u(i, j, k) = 0.1_dp * sin(real(j_global - 1, dp) / real(MD%nj_global, dp) * PI * 2.0_dp) * &
                                 exp(-real(k, dp) / 30.0_dp)
                    v(i, j, k) = 0.1_dp * cos(real(i_global - 1, dp) / real(MD%ni_global, dp) * PI * 2.0_dp) * &
                                 exp(-real(k, dp) / 30.0_dp)
                end do
            end do
        end do

        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                i_global = i + MD%i_offset
                j_global = j + MD%j_offset
                eta(i, j) = 0.5_dp * sin(real(i_global - 1, dp) / real(MD%ni_global, dp) * PI * 2.0_dp) * &
                            cos(real(j_global - 1, dp) / real(MD%nj_global, dp) * PI * 2.0_dp)
                ubt(i, j) = 0.05_dp * sin(real(j_global - 1, dp) / real(MD%nj_global, dp) * PI)
                vbt(i, j) = 0.05_dp * cos(real(i_global - 1, dp) / real(MD%ni_global, dp) * PI)
            end do
        end do
    end subroutine initialize_state_mpi

    subroutine initialize_stress_mpi(taux_d, tauy_d, G, MD)
        type(ocean_grid_type), intent(in) :: G
        type(mpi_domain_type), intent(in) :: MD
        real(dp), device, intent(out) :: taux_d(G%isd:G%ied, G%jsd:G%jed)
        real(dp), device, intent(out) :: tauy_d(G%isd:G%ied, G%jsd:G%jed)

        real(dp), allocatable :: taux_h(:,:), tauy_h(:,:)
        integer :: i, j, j_global

        allocate(taux_h(G%isd:G%ied, G%jsd:G%jed))
        allocate(tauy_h(G%isd:G%ied, G%jsd:G%jed))

        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                j_global = j + MD%j_offset
                taux_h(i, j) = 0.1_dp * sin(real(j_global - 1, dp) / real(MD%nj_global, dp) * PI)
                tauy_h(i, j) = 0.0_dp
            end do
        end do

        taux_d = taux_h
        tauy_d = tauy_h
        deallocate(taux_h, tauy_h)
    end subroutine initialize_stress_mpi

    subroutine compute_transports_cuda(u_d, v_d, h_d, uh_d, vh_d, dyCu_d, dxCv_d, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), device, intent(in)  :: u_d(G%isd:G%ied, G%jsd:G%jed, GV%ke)
        real(dp), device, intent(in)  :: v_d(G%isd:G%ied, G%jsd:G%jed, GV%ke)
        real(dp), device, intent(in)  :: h_d(G%isd:G%ied, G%jsd:G%jed, GV%ke)
        real(dp), device, intent(out) :: uh_d(G%isd:G%ied, G%jsd:G%jed, GV%ke)
        real(dp), device, intent(out) :: vh_d(G%isd:G%ied, G%jsd:G%jed, GV%ke)
        real(dp), device, intent(in)  :: dyCu_d(G%isd:G%ied, G%jsd:G%jed)
        real(dp), device, intent(in)  :: dxCv_d(G%isd:G%ied, G%jsd:G%jed)

        integer :: i, j, k, isd, ied, jsd, jed

        isd = G%isd; ied = G%ied; jsd = G%jsd; jed = G%jed

        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, GV%ke
            do j = jsd, jed
                do i = isd, ied
                    uh_d(i, j, k) = u_d(i, j, k) * 0.5_dp * &
                        (h_d(i, j, k) + h_d(min(i + 1, ied), j, k)) * dyCu_d(i, j)
                    vh_d(i, j, k) = v_d(i, j, k) * 0.5_dp * &
                        (h_d(i, j, k) + h_d(i, min(j + 1, jed), k)) * dxCv_d(i, j)
                end do
            end do
        end do
    end subroutine compute_transports_cuda

    subroutine velocity_update_cuda(vel_out_d, vel_in_d, accel_d, diff_d, dt_scale, &
                                    is, ie, js, je, nk)
        integer, intent(in) :: is, ie, js, je, nk
        real(dp), intent(in) :: dt_scale
        real(dp), device, intent(inout) :: vel_out_d(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), device, intent(in)    :: vel_in_d(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), device, intent(in)    :: accel_d(G%isd:G%ied, G%jsd:G%jed, nk)
        real(dp), device, intent(in)    :: diff_d(G%isd:G%ied, G%jsd:G%jed, nk)

        integer :: i, j, k

        !$cuf kernel do(3) <<< *, * >>>
        do k = 1, nk
            do j = js, je
                do i = is, ie
                    vel_out_d(i, j, k) = vel_in_d(i, j, k) + &
                        dt_scale * (accel_d(i, j, k) + diff_d(i, j, k))
                end do
            end do
        end do
    end subroutine velocity_update_cuda

    subroutine verify_state_mpi(h_init, h_final, u, v, eta, G, GV, MD)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        type(mpi_domain_type), intent(in) :: MD
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_init, h_final
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: eta

        real(dp) :: mass_init_local, mass_final_local, rel_error
        real(dp) :: max_u_local, max_v_local, max_eta_local, ke_local
        real(dp) :: mass_init, mass_final, max_u, max_v, max_eta, ke_total
        integer :: i, j, k, ierr

        mass_init_local = 0.0_dp; mass_final_local = 0.0_dp
        max_u_local = 0.0_dp; max_v_local = 0.0_dp; max_eta_local = 0.0_dp
        ke_local = 0.0_dp

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc, G%iec
                    mass_init_local = mass_init_local + h_init(i, j, k) * G%areaT(i, j)
                    mass_final_local = mass_final_local + h_final(i, j, k) * G%areaT(i, j)
                    max_u_local = max(max_u_local, abs(u(i, j, k)))
                    max_v_local = max(max_v_local, abs(v(i, j, k)))
                    ke_local = ke_local + 0.5_dp * (u(i, j, k)**2 + v(i, j, k)**2) * &
                               h_final(i, j, k) * G%areaT(i, j)
                end do
            end do
        end do

        do j = G%jsc, G%jec
            do i = G%isc, G%iec
                max_eta_local = max(max_eta_local, abs(eta(i, j)))
            end do
        end do

        call MPI_Allreduce(mass_init_local, mass_init, 1, MPI_DOUBLE_PRECISION, &
                           MPI_SUM, MD%comm, ierr)
        call MPI_Allreduce(mass_final_local, mass_final, 1, MPI_DOUBLE_PRECISION, &
                           MPI_SUM, MD%comm, ierr)
        call MPI_Allreduce(max_u_local, max_u, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MD%comm, ierr)
        call MPI_Allreduce(max_v_local, max_v, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MD%comm, ierr)
        call MPI_Allreduce(max_eta_local, max_eta, 1, MPI_DOUBLE_PRECISION, &
                           MPI_MAX, MD%comm, ierr)
        call MPI_Allreduce(ke_local, ke_total, 1, MPI_DOUBLE_PRECISION, &
                           MPI_SUM, MD%comm, ierr)

        rel_error = abs(mass_final - mass_init) / mass_init

        if (MD%rank == 0) then
            print '(A)', ''
            print '(A)', 'State Verification:'
            print '(A)', '----------------------------------------------------------------'
            print '(A)', 'Mass Conservation:'
            print '(A,ES15.8)', '  Initial mass:  ', mass_init
            print '(A,ES15.8)', '  Final mass:    ', mass_final
            print '(A,ES15.8)', '  Rel. error:    ', rel_error

            print '(A)', ''
            print '(A)', 'State Statistics:'
            print '(A,ES15.8)', '  Max |u|:       ', max_u
            print '(A,ES15.8)', '  Max |v|:       ', max_v
            print '(A,ES15.8)', '  Max |eta|:     ', max_eta
            print '(A,ES15.8)', '  Total KE:      ', ke_total

            print '(A)', ''
            if (rel_error < 1.0e-10_dp) then
                print '(A)', 'Status: PASS'
            else if (rel_error < 1.0e-6_dp) then
                print '(A)', 'Status: ACCEPTABLE'
            else
                print '(A)', 'Status: WARNING - mass conservation issue'
            end if
        end if
    end subroutine verify_state_mpi

    subroutine check_bt_cfl(dt, nsteps, dx_km, depth)
        real(dp), intent(in) :: dt
        integer, intent(in) :: nsteps
        real(dp), intent(in) :: dx_km, depth

        real(dp) :: dtbt, dx_m, c_grav, cfl, cfl_2d
        integer :: min_nsteps

        dx_m = dx_km * 1000.0_dp
        dtbt = dt / real(nsteps, dp)
        c_grav = sqrt(G_EARTH * depth)
        cfl = c_grav * dtbt / dx_m
        cfl_2d = cfl * sqrt(2.0_dp)
        min_nsteps = ceiling(dt * c_grav / dx_m * 2.5_dp)

        print '(A)', ''
        print '(A)', 'Barotropic CFL Check:'
        print '(A,F8.1,A)', '  Gravity wave speed: ', c_grav, ' m/s'
        print '(A,F8.3,A)', '  BT timestep:        ', dtbt, ' s'
        print '(A,F8.4)', '  CFL (1D):           ', cfl
        print '(A,F8.4)', '  CFL (2D diagonal):  ', cfl_2d

        if (cfl_2d > 0.9_dp) then
            print '(A)', '  *** WARNING: CFL > 0.9 - UNSTABLE! ***'
            print '(A,I4,A)', '  Recommend at least ', min_nsteps, ' substeps'
        else if (cfl_2d > 0.7_dp) then
            print '(A)', '  * CAUTION: CFL > 0.7 - marginally stable'
        else
            print '(A)', '  CFL OK'
        end if
    end subroutine check_bt_cfl

    !> Initialize CUDA hor_visc directly from grid metrics (no OpenACC)
    subroutine hor_visc_init_cuda_from_grid(CS, G, nk, Kh)
        type(hor_visc_CS_cuda), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in) :: nk
        real(dp), intent(in) :: Kh

        real(dp), allocatable :: DY_dxT(:,:), DX_dyT(:,:)
        real(dp), allocatable :: DY_dxBu(:,:), DX_dyBu(:,:)
        real(dp), allocatable :: reduction_xx(:,:), reduction_xy(:,:)
        real(dp), allocatable :: dy2h(:,:), dx2h(:,:), dy2q(:,:), dx2q(:,:)
        real(dp) :: h_neglect
        integer :: i, j, isd, ied, jsd, jed

        isd = G%isd; ied = G%ied; jsd = G%jsd; jed = G%jed

        ! Allocate temporary host arrays for metric computation
        allocate(DY_dxT(isd:ied, jsd:jed), DX_dyT(isd:ied, jsd:jed))
        allocate(DY_dxBu(isd:ied, jsd:jed), DX_dyBu(isd:ied, jsd:jed))
        allocate(reduction_xx(isd:ied, jsd:jed), reduction_xy(isd:ied, jsd:jed))
        allocate(dy2h(isd:ied, jsd:jed), dx2h(isd:ied, jsd:jed))
        allocate(dy2q(isd:ied, jsd:jed), dx2q(isd:ied, jsd:jed))

        ! Compute metrics directly from grid
        do j = jsd, jed
            do i = isd, ied
                ! Metric ratios
                DX_dyT(i,j) = G%dxT(i,j) * G%IdyT(i,j)
                DY_dxT(i,j) = G%dyT(i,j) * G%IdxT(i,j)
                DX_dyBu(i,j) = G%dxBu(i,j) * G%IdyBu(i,j)
                DY_dxBu(i,j) = G%dyBu(i,j) * G%IdxBu(i,j)

                ! Grid spacing squared
                dx2h(i,j) = G%dxT(i,j) * G%dxT(i,j)
                dy2h(i,j) = G%dyT(i,j) * G%dyT(i,j)
                dx2q(i,j) = G%dxBu(i,j) * G%dxBu(i,j)
                dy2q(i,j) = G%dyBu(i,j) * G%dyBu(i,j)

                ! Reduction factors (1.0 for uniform grid)
                reduction_xx(i,j) = 1.0_dp
                reduction_xy(i,j) = 1.0_dp
            end do
        end do

        ! h_neglect for numerical stability (same as max(Angstrom_H, 1e-3))
        h_neglect = 1.0e-3_dp

        ! Call the CUDA init with computed metrics
        call hor_visc_init_cuda(CS, isd, ied, jsd, jed, &
                                G%isc, G%iec, G%jsc, G%jec, nk, &
                                Kh, h_neglect, &
                                DY_dxT, DX_dyT, DY_dxBu, DX_dyBu, &
                                G%IdyCu, G%IdxCu, G%IdyCv, G%IdxCv, &
                                G%IareaCu, G%IareaCv, &
                                G%mask2dT, G%mask2dBu, &
                                reduction_xx, reduction_xy, &
                                dy2h, dx2h, dy2q, dx2q)

        ! Free temporary host arrays
        deallocate(DY_dxT, DX_dyT, DY_dxBu, DX_dyBu)
        deallocate(reduction_xx, reduction_xy)
        deallocate(dy2h, dx2h, dy2q, dx2q)

    end subroutine hor_visc_init_cuda_from_grid

    !> Initialize CUDA barotropic directly from grid metrics (no OpenACC)
    subroutine barotropic_init_cuda_from_grid(CS, G, dt, nstep)
        type(barotropic_CS_cuda), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        real(dp), intent(in) :: dt
        integer, intent(in) :: nstep

        real(dp), allocatable :: Datu(:,:), Datv(:,:)
        real(dp), allocatable :: gtot_E(:,:), gtot_W(:,:), gtot_N(:,:), gtot_S(:,:)
        real(dp), allocatable :: f_4_u(:,:,:), f_4_v(:,:,:)
        real(dp), allocatable :: bt_rem_u(:,:), bt_rem_v(:,:)
        real(dp) :: depth, bebt, f0
        integer :: i, j, isd, ied, jsd, jed

        isd = G%isd; ied = G%ied; jsd = G%jsd; jed = G%jed
        depth = 4000.0_dp
        bebt = 0.2_dp

        ! Allocate temporary host arrays
        allocate(Datu(isd:ied, jsd:jed), Datv(isd:ied, jsd:jed))
        allocate(gtot_E(isd:ied, jsd:jed), gtot_W(isd:ied, jsd:jed))
        allocate(gtot_N(isd:ied, jsd:jed), gtot_S(isd:ied, jsd:jed))
        allocate(f_4_u(4, isd:ied, jsd:jed), f_4_v(4, isd:ied, jsd:jed))
        allocate(bt_rem_u(isd:ied, jsd:jed), bt_rem_v(isd:ied, jsd:jed))

        ! Compute metrics directly from grid
        do j = jsd, jed
            do i = isd, ied
                Datu(i,j) = depth * G%dyCu(i,j)
                Datv(i,j) = depth * G%dxCv(i,j)
                gtot_E(i,j) = G_EARTH
                gtot_W(i,j) = G_EARTH
                gtot_N(i,j) = G_EARTH
                gtot_S(i,j) = G_EARTH
                bt_rem_u(i,j) = 0.999_dp
                bt_rem_v(i,j) = 0.999_dp

                ! Coriolis coefficients (f/4 at each corner)
                f0 = G%CoriolisBu(i,j)
                f_4_u(1,i,j) = 0.25_dp * f0
                f_4_u(2,i,j) = 0.25_dp * f0
                f_4_u(3,i,j) = 0.25_dp * f0
                f_4_u(4,i,j) = 0.25_dp * f0
                f_4_v(1,i,j) = 0.25_dp * f0
                f_4_v(2,i,j) = 0.25_dp * f0
                f_4_v(3,i,j) = 0.25_dp * f0
                f_4_v(4,i,j) = 0.25_dp * f0
            end do
        end do

        ! Call the CUDA init with computed metrics
        call barotropic_init_cuda(CS, isd, ied, jsd, jed, &
                                  G%isc, G%iec, G%jsc, G%jec, &
                                  nstep, dt, bebt, 0, &
                                  Datu, Datv, &
                                  gtot_E, gtot_W, gtot_N, gtot_S, &
                                  f_4_u, f_4_v, bt_rem_u, bt_rem_v, &
                                  G%IareaT, G%IdxCu, G%IdyCv)

        ! Free temporary host arrays
        deallocate(Datu, Datv)
        deallocate(gtot_E, gtot_W, gtot_N, gtot_S)
        deallocate(f_4_u, f_4_v, bt_rem_u, bt_rem_v)

    end subroutine barotropic_init_cuda_from_grid

    !> Initialize ocean grid for MPI CUDA driver (no OpenACC directives)
    subroutine init_ocean_grid_mpi_cuda(G, MD, GV, nk, dx_km, lat_deg)
        type(ocean_grid_type), intent(inout) :: G
        type(mpi_domain_type), intent(in) :: MD
        type(verticalGrid_type), intent(inout) :: GV
        integer, intent(in) :: nk
        real(dp), intent(in) :: dx_km
        real(dp), intent(in) :: lat_deg

        real(dp) :: dx_m, f0, beta
        integer :: i, j, j_global, halo

        halo = MD%halo

        ! Store local compute dimensions
        G%ni = MD%ni_local
        G%nj = MD%nj_local
        G%nk = nk

        ! Data domain: 1 to ni_local + 2*halo
        G%isd = 1
        G%ied = MD%ni_local + 2 * halo
        G%jsd = 1
        G%jed = MD%nj_local + 2 * halo

        ! Compute domain: halo+1 to ni_local+halo
        G%isc = halo + 1
        G%iec = MD%ni_local + halo
        G%jsc = halo + 1
        G%jec = MD%nj_local + halo

        ! Store MPI offsets
        G%i_offset = MD%i_offset
        G%j_offset = MD%j_offset

        ! Grid spacing
        dx_m = dx_km * 1000.0_dp
        G%dx = dx_m
        G%dy = dx_m

        ! Vertical grid (no OpenACC)
        GV%ke = nk
        GV%Angstrom_H = 1.0e-10_dp

        ! Allocate arrays (local size with halos)
        allocate(G%IareaT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%areaT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dy_Cu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IareaBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%areaBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%CoriolisBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IareaCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IareaCv(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dxBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%dyBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdxBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%IdyBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%bathyT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dT(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dBu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dCu(G%isd:G%ied, G%jsd:G%jed))
        allocate(G%mask2dCv(G%isd:G%ied, G%jsd:G%jed))

        ! Beta-plane Coriolis parameters
        f0 = 2.0_dp * OMEGA * sin(lat_deg * PI / 180.0_dp)
        beta = 2.0_dp * OMEGA * cos(lat_deg * PI / 180.0_dp) / EARTH_RADIUS

        ! Initialize grid metrics (uniform except Coriolis)
        do j = G%jsd, G%jed
            j_global = (j - halo) + MD%j_offset

            do i = G%isd, G%ied
                G%areaT(i, j) = dx_m * dx_m
                G%IareaT(i, j) = 1.0_dp / (dx_m * dx_m)
                G%dxT(i, j) = dx_m
                G%dyT(i, j) = dx_m
                G%IdxT(i, j) = 1.0_dp / dx_m
                G%IdyT(i, j) = 1.0_dp / dx_m
                G%dxCu(i, j) = dx_m
                G%dyCu(i, j) = dx_m
                G%dy_Cu(i, j) = dx_m
                G%IdxCu(i, j) = 1.0_dp / dx_m
                G%IdyCu(i, j) = 1.0_dp / dx_m
                G%dxCv(i, j) = dx_m
                G%dyCv(i, j) = dx_m
                G%IdxCv(i, j) = 1.0_dp / dx_m
                G%IdyCv(i, j) = 1.0_dp / dx_m
                G%areaBu(i, j) = dx_m * dx_m
                G%IareaBu(i, j) = 1.0_dp / (dx_m * dx_m)
                G%IareaCu(i, j) = 1.0_dp / (dx_m * dx_m)
                G%IareaCv(i, j) = 1.0_dp / (dx_m * dx_m)
                G%dxBu(i, j) = dx_m
                G%dyBu(i, j) = dx_m
                G%IdxBu(i, j) = 1.0_dp / dx_m
                G%IdyBu(i, j) = 1.0_dp / dx_m
                G%mask2dT(i, j) = 1.0_dp
                G%mask2dBu(i, j) = 1.0_dp
                G%mask2dCu(i, j) = 1.0_dp
                G%mask2dCv(i, j) = 1.0_dp
                G%bathyT(i, j) = 4000.0_dp
                G%CoriolisBu(i, j) = f0 + beta * (real(j_global - MD%nj_global/2, dp) - 0.5_dp) * dx_m
            end do
        end do

        G%first_direction = 0
        G%nbytes = 29_int64 * int(G%ied - G%isd + 1, int64) * &
                   int(G%jed - G%jsd + 1, int64) * 8_int64

        ! NO OpenACC data transfers - grid stays on host only
        ! CUDA kernels will access grid data via explicit copies in their init routines

    end subroutine init_ocean_grid_mpi_cuda

    !> Deallocate ocean grid without OpenACC (for MPI CUDA driver)
    subroutine end_ocean_grid_cuda(G)
        type(ocean_grid_type), intent(inout) :: G

        ! NO OpenACC exit data - grid was never on OpenACC device
        if (allocated(G%IareaT)) deallocate(G%IareaT)
        if (allocated(G%areaT)) deallocate(G%areaT)
        if (allocated(G%dxT)) deallocate(G%dxT)
        if (allocated(G%dyT)) deallocate(G%dyT)
        if (allocated(G%IdxT)) deallocate(G%IdxT)
        if (allocated(G%IdyT)) deallocate(G%IdyT)
        if (allocated(G%dxCu)) deallocate(G%dxCu)
        if (allocated(G%dyCu)) deallocate(G%dyCu)
        if (allocated(G%dy_Cu)) deallocate(G%dy_Cu)
        if (allocated(G%IdxCu)) deallocate(G%IdxCu)
        if (allocated(G%IdyCu)) deallocate(G%IdyCu)
        if (allocated(G%dxCv)) deallocate(G%dxCv)
        if (allocated(G%dyCv)) deallocate(G%dyCv)
        if (allocated(G%IdxCv)) deallocate(G%IdxCv)
        if (allocated(G%IdyCv)) deallocate(G%IdyCv)
        if (allocated(G%IareaBu)) deallocate(G%IareaBu)
        if (allocated(G%areaBu)) deallocate(G%areaBu)
        if (allocated(G%CoriolisBu)) deallocate(G%CoriolisBu)
        if (allocated(G%IareaCu)) deallocate(G%IareaCu)
        if (allocated(G%IareaCv)) deallocate(G%IareaCv)
        if (allocated(G%dxBu)) deallocate(G%dxBu)
        if (allocated(G%dyBu)) deallocate(G%dyBu)
        if (allocated(G%IdxBu)) deallocate(G%IdxBu)
        if (allocated(G%IdyBu)) deallocate(G%IdyBu)
        if (allocated(G%bathyT)) deallocate(G%bathyT)
        if (allocated(G%mask2dT)) deallocate(G%mask2dT)
        if (allocated(G%mask2dBu)) deallocate(G%mask2dBu)
        if (allocated(G%mask2dCu)) deallocate(G%mask2dCu)
        if (allocated(G%mask2dCv)) deallocate(G%mask2dCv)

    end subroutine end_ocean_grid_cuda

end program rk2_mpi_cuda_driver
