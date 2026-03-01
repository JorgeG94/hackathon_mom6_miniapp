!> Unified RK2 driver using explicit CUDA Fortran kernels
!!
!! This driver orchestrates all CUDA solvers (continuity, Coriolis, barotropic,
!! vert_visc, hor_visc) in a simplified split RK2 time-stepping scheme similar
!! to MOM6's step_MOM_dyn_split_RK2.
!!
!! All computation runs on the GPU via explicit CUDA kernels (no OpenACC compute).
!! OpenACC-based _init routines are used only for metric extraction where needed
!! (barotropic, hor_visc).
!!
!! Usage: rk2_cuda_driver [ni] [nj] [nk] [niter] [bt_nsteps]
!!        Defaults: 180 180 75 10 30
program rk2_cuda_driver
    use cudafor
    use omp_lib, only: omp_get_wtime
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, G_EARTH
    use mom6_profiler, only: profiler_init, profiler_end, profiler_start, profiler_stop, &
                             profiler_report

    ! CUDA kernel modules
    use mom6_continuity_cuda, only: continuity_CS_cuda, continuity_init_cuda, &
                                     continuity_PPM_cuda, continuity_end_cuda
    use mom6_coriolis_cuda, only: coriolis_CS_cuda, coriolis_init_cuda, &
                                   CorAdCalc_cuda, coriolis_end_cuda, &
                                   SADOURNY75_ENERGY_CUDA
    use mom6_barotropic_cuda, only: barotropic_CS_cuda, barotropic_init_cuda, &
                                     btstep_cuda, barotropic_end_cuda
    use mom6_vert_visc_cuda, only: vert_visc_CS_cuda, vert_visc_init_cuda, &
                                    vert_visc_cra_cuda, vert_visc_end_cuda
    use mom6_hor_visc_cuda, only: hor_visc_CS_cuda, hor_visc_init_cuda, &
                                   hor_visc_cuda, hor_visc_end_cuda

    ! OpenACC modules used ONLY for metric extraction
    use mom6_barotropic, only: barotropic_CS, barotropic_init, barotropic_end
    use mom6_hor_visc, only: hor_visc_CS, hor_visc_init, hor_visc_end
    implicit none

    real(dp), parameter :: PI = 3.14159265358979_dp

    ! Grid structures
    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV

    ! CUDA control structures
    type(continuity_CS_cuda) :: cont_CS
    type(coriolis_CS_cuda)   :: cor_CS
    type(barotropic_CS_cuda) :: bt_CS_cuda
    type(vert_visc_CS_cuda)  :: visc_CS
    type(hor_visc_CS_cuda)   :: hvisc_CS

    ! OpenACC control structures (metric extraction only)
    type(barotropic_CS) :: bt_CS_acc
    type(hor_visc_CS)   :: hvisc_CS_acc

    ! Host 3D state arrays
    real(dp), allocatable :: u_h(:,:,:), v_h(:,:,:)
    real(dp), allocatable :: h_h(:,:,:), h0_h(:,:,:)
    real(dp), allocatable :: uh_h(:,:,:), vh_h(:,:,:)

    ! Host 2D state arrays
    real(dp), allocatable :: eta_h(:,:), ubt_h(:,:), vbt_h(:,:)

    ! Device 3D arrays
    real(dp), device, allocatable :: u_d(:,:,:), v_d(:,:,:)
    real(dp), device, allocatable :: h_d(:,:,:), h0_d(:,:,:), htmp_d(:,:,:)
    real(dp), device, allocatable :: uh_d(:,:,:), vh_d(:,:,:)
    real(dp), device, allocatable :: CAu_d(:,:,:), CAv_d(:,:,:)
    real(dp), device, allocatable :: up_d(:,:,:), vp_d(:,:,:)
    real(dp), device, allocatable :: diffu_d(:,:,:), diffv_d(:,:,:)

    ! Device 2D arrays
    real(dp), device, allocatable :: eta_d(:,:), ubt_d(:,:), vbt_d(:,:)
    real(dp), device, allocatable :: ubt_av_d(:,:), vbt_av_d(:,:), eta_av_d(:,:)
    real(dp), device, allocatable :: taux_d(:,:), tauy_d(:,:)

    ! Device grid metric arrays for transport computation
    real(dp), device, allocatable :: dyCu_d(:,:), dxCv_d(:,:)

    ! Timing
    real(dp) :: t_start, t_end, t_init_start, t_init_end
    real(dp) :: t_total, t_coriolis, t_barotropic, t_continuity
    real(dp) :: t_vert_visc, t_hor_visc, t_compute
    real(dp) :: t_iter_start, t_iter_end

    ! Parameters
    integer :: ni, nj, nk, niter, bt_nsteps
    integer :: i, j, k, iter, istat
    real(dp) :: dt
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10; bt_nsteps = 30
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

    call profiler_init()
    call profiler_start("Total")

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Split RK2 — CUDA Fortran Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'RK2 iterations: ', niter
    print '(A,I4)', 'Barotropic substeps: ', bt_nsteps
    print '(A,F8.1,A)', 'dt = ', dt, ' s'

    ! CFL check
    call check_bt_cfl(dt, bt_nsteps, 10.0_dp, 4000.0_dp)
    print '(A)', '================================================================'

    !=========================================================================
    ! INITIALIZATION
    !=========================================================================
    call profiler_start("Initialization")
    t_init_start = omp_get_wtime()

    ! Initialize grid
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! --- Continuity CUDA init (direct from grid metrics) ---
    call continuity_init_cuda(cont_CS, G%isd, G%ied, G%jsd, G%jed, &
                              G%isc, G%iec, G%jsc, G%jec, nk, &
                              G%IareaT, G%IdxT, G%dy_Cu, G%mask2dT, .true.)

    ! --- Coriolis CUDA init (direct from grid metrics) ---
    call coriolis_init_cuda(cor_CS, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, nk, &
                            G%areaT, G%IareaBu, G%CoriolisBu, G%mask2dBu, &
                            G%dyCv, G%dxCu, G%dyCu, G%dxCv, G%IdxCu, G%IdyCv, &
                            SADOURNY75_ENERGY_CUDA)

    ! --- Barotropic: use OpenACC init for metric extraction, then CUDA init ---
    ! Free OpenACC device memory immediately after CUDA init copies what it needs
    call barotropic_init(bt_CS_acc, G, dt, bt_nsteps)
    call barotropic_init_cuda(bt_CS_cuda, G%isd, G%ied, G%jsd, G%jed, &
        G%isc, G%iec, G%jsc, G%jec, &
        bt_nsteps, dt, bt_CS_acc%bebt, 0, &
        bt_CS_acc%Datu, bt_CS_acc%Datv, &
        bt_CS_acc%gtot_E, bt_CS_acc%gtot_W, bt_CS_acc%gtot_N, bt_CS_acc%gtot_S, &
        bt_CS_acc%f_4_u, bt_CS_acc%f_4_v, bt_CS_acc%bt_rem_u, bt_CS_acc%bt_rem_v, &
        G%IareaT, G%IdxCu, G%IdyCv)
    call barotropic_end(bt_CS_acc)

    ! --- Vertical viscosity CUDA init (direct from grid metrics) ---
    call vert_visc_init_cuda(visc_CS, G%isd, G%ied, G%jsd, G%jed, &
                             G%isc, G%iec, G%jsc, G%jec, nk, &
                             G%mask2dCu, G%mask2dCv, &
                             1.0e-4_dp, 1.0e-2_dp, 1.0e-2_dp, 50.0_dp, 10.0_dp)

    ! --- Horizontal viscosity: use OpenACC init for metric extraction, then CUDA init ---
    ! Free OpenACC device memory immediately after CUDA init copies what it needs
    call hor_visc_init(hvisc_CS_acc, G, GV, Kh=100.0_dp)
    call hor_visc_init_cuda(hvisc_CS, G%isd, G%ied, G%jsd, G%jed, &
                            G%isc, G%iec, G%jsc, G%jec, &
                            100.0_dp, hvisc_CS_acc%h_neglect, &
                            hvisc_CS_acc%DY_dxT, hvisc_CS_acc%DX_dyT, &
                            hvisc_CS_acc%DY_dxBu, hvisc_CS_acc%DX_dyBu, &
                            G%IdyCu, G%IdxCu, G%IdyCv, G%IdxCv, &
                            G%IareaCu, G%IareaCv, &
                            G%mask2dT, G%mask2dBu, &
                            hvisc_CS_acc%reduction_xx, hvisc_CS_acc%reduction_xy, &
                            hvisc_CS_acc%dy2h, hvisc_CS_acc%dx2h, &
                            hvisc_CS_acc%dy2q, hvisc_CS_acc%dx2q)
    call hor_visc_end(hvisc_CS_acc)

    ! --- Allocate host arrays ---
    allocate (u_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h0_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh_h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (eta_h(G%isd:G%ied, G%jsd:G%jed))
    allocate (ubt_h(G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt_h(G%isd:G%ied, G%jsd:G%jed))

    ! --- Allocate device arrays ---
    ! 3D
    allocate (u_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h0_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (htmp_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (up_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vp_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffu_d(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffv_d(G%isd:G%ied, G%jsd:G%jed, nk))
    ! 2D
    allocate (eta_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (ubt_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (ubt_av_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt_av_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_av_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (taux_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (tauy_d(G%isd:G%ied, G%jsd:G%jed))
    ! Grid metrics for transport computation
    allocate (dyCu_d(G%isd:G%ied, G%jsd:G%jed))
    allocate (dxCv_d(G%isd:G%ied, G%jsd:G%jed))
    dyCu_d = G%dyCu
    dxCv_d = G%dxCv

    ! --- Initialize host state ---
    call initialize_state(u_h, v_h, h_h, h0_h, eta_h, ubt_h, vbt_h, G, GV)

    ! --- Copy host → device ---
    u_d = u_h; v_d = v_h; h_d = h_h; h0_d = h0_h
    eta_d = eta_h; ubt_d = ubt_h; vbt_d = vbt_h

    ! Initialize surface stress on device
    call initialize_stress(taux_d, tauy_d, G)

    t_init_end = omp_get_wtime()
    call profiler_stop("Initialization")

    print '(A)', ''
    print '(A,F12.6,A)', 'Initialization time: ', t_init_end - t_init_start, ' s'
    print '(A)', ''

    !=========================================================================
    ! WARMUP
    !=========================================================================
    call profiler_start("Warmup")
    print '(A)', 'Warming up CUDA kernels...'
    call compute_transports_cuda(u_d, v_d, h_d, uh_d, vh_d, dyCu_d, dxCv_d, G, GV)
    call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, hvisc_CS, nk)
    call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, cor_CS)
    call vert_visc_cra_cuda(u_d, v_d, h_d, dt, visc_CS, taux_d, tauy_d)
    call btstep_cuda(eta_d, ubt_d, vbt_d, ubt_av_d, vbt_av_d, eta_av_d, bt_CS_cuda)
    call continuity_PPM_cuda(u_d, h0_d, h_d, uh_d, dt, cont_CS)

    ! Reset state after warmup
    u_d = u_h; v_d = v_h; h_d = h_h
    eta_d = eta_h; ubt_d = ubt_h; vbt_d = vbt_h
    call profiler_stop("Warmup")

    !=========================================================================
    ! RK2 TIME-STEPPING LOOP
    !=========================================================================
    print '(A)', 'Running split RK2 time-stepping (CUDA)...'
    print '(A)', ''

    t_coriolis = 0.0_dp
    t_barotropic = 0.0_dp
    t_continuity = 0.0_dp
    t_vert_visc = 0.0_dp
    t_hor_visc = 0.0_dp
    t_compute = 0.0_dp

    call profiler_start("RK2_step", nvtx_only=.true.)
    do iter = 1, niter
        t_iter_start = omp_get_wtime()

        ! Reset thickness to initial state for timing consistency
        h_d = h0_d

        !=====================================================================
        ! PREDICTOR PHASE
        !=====================================================================

        ! 1. Compute layer transports
        call profiler_start("Transports")
        call compute_transports_cuda(u_d, v_d, h_d, uh_d, vh_d, dyCu_d, dxCv_d, G, GV)
        call profiler_stop("Transports")

        ! 2. Horizontal viscosity
        call profiler_start("HorVisc")
        t_start = omp_get_wtime()
        call hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, hvisc_CS, nk)
        t_end = omp_get_wtime()
        t_hor_visc = t_hor_visc + (t_end - t_start)
        call profiler_stop("HorVisc")

        ! 3. Coriolis and momentum advection
        call profiler_start("Coriolis")
        t_start = omp_get_wtime()
        call CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, cor_CS)
        t_end = omp_get_wtime()
        t_coriolis = t_coriolis + (t_end - t_start)
        call profiler_stop("Coriolis")

        ! 4. Predictor velocity update: up = u + dt*(CAu + diffu)
        call profiler_start("VelUpdate_pred")
        call velocity_update_cuda(up_d, u_d, CAu_d, diffu_d, dt, &
                                  G%isc, G%iec-1, G%jsc, G%jec, nk)
        call velocity_update_cuda(vp_d, v_d, CAv_d, diffv_d, dt, &
                                  G%isc, G%iec, G%jsc, G%jec-1, nk)
        call profiler_stop("VelUpdate_pred")

        ! 5. Vertical viscosity on predictor velocities
        call profiler_start("VertVisc")
        t_start = omp_get_wtime()
        call vert_visc_cra_cuda(up_d, vp_d, h_d, dt, visc_CS, taux_d, tauy_d)
        t_end = omp_get_wtime()
        t_vert_visc = t_vert_visc + (t_end - t_start)
        call profiler_stop("VertVisc")

        ! 6. Barotropic predictor step
        call profiler_start("Barotropic")
        t_start = omp_get_wtime()
        call btstep_cuda(eta_d, ubt_d, vbt_d, ubt_av_d, vbt_av_d, eta_av_d, bt_CS_cuda)
        t_end = omp_get_wtime()
        t_barotropic = t_barotropic + (t_end - t_start)
        call profiler_stop("Barotropic")

        ! 7. Continuity (update thicknesses)
        call profiler_start("Continuity")
        t_start = omp_get_wtime()
        call continuity_PPM_cuda(up_d, h0_d, h_d, uh_d, dt, cont_CS)
        t_end = omp_get_wtime()
        t_continuity = t_continuity + (t_end - t_start)
        call profiler_stop("Continuity")

        !=====================================================================
        ! CORRECTOR PHASE
        !=====================================================================

        ! 8. Recompute transports with updated thickness
        call profiler_start("Transports")
        call compute_transports_cuda(up_d, vp_d, h_d, uh_d, vh_d, dyCu_d, dxCv_d, G, GV)
        call profiler_stop("Transports")

        ! 9. Horizontal viscosity with updated state
        call profiler_start("HorVisc")
        t_start = omp_get_wtime()
        call hor_visc_cuda(up_d, vp_d, h_d, diffu_d, diffv_d, hvisc_CS, nk)
        t_end = omp_get_wtime()
        t_hor_visc = t_hor_visc + (t_end - t_start)
        call profiler_stop("HorVisc")

        ! 10. Coriolis with updated state
        call profiler_start("Coriolis")
        t_start = omp_get_wtime()
        call CorAdCalc_cuda(up_d, vp_d, h_d, uh_d, vh_d, CAu_d, CAv_d, cor_CS)
        t_end = omp_get_wtime()
        t_coriolis = t_coriolis + (t_end - t_start)
        call profiler_stop("Coriolis")

        ! 11. Corrector velocity update: u = u + 0.5*dt*(CAu + diffu)
        call profiler_start("VelUpdate_corr")
        call velocity_update_cuda(u_d, u_d, CAu_d, diffu_d, 0.5_dp*dt, &
                                  G%isc, G%iec-1, G%jsc, G%jec, nk)
        call velocity_update_cuda(v_d, v_d, CAv_d, diffv_d, 0.5_dp*dt, &
                                  G%isc, G%iec, G%jsc, G%jec-1, nk)
        call profiler_stop("VelUpdate_corr")

        ! 12. Vertical viscosity on corrector velocities
        call profiler_start("VertVisc")
        t_start = omp_get_wtime()
        call vert_visc_cra_cuda(u_d, v_d, h_d, 0.5_dp*dt, visc_CS, taux_d, tauy_d)
        t_end = omp_get_wtime()
        t_vert_visc = t_vert_visc + (t_end - t_start)
        call profiler_stop("VertVisc")

        ! 13. Barotropic corrector
        call profiler_start("Barotropic")
        t_start = omp_get_wtime()
        call btstep_cuda(eta_av_d, ubt_av_d, vbt_av_d, ubt_av_d, vbt_av_d, eta_d, bt_CS_cuda)
        t_end = omp_get_wtime()
        t_barotropic = t_barotropic + (t_end - t_start)
        call profiler_stop("Barotropic")

        ! 14. Final continuity
        call profiler_start("Continuity")
        t_start = omp_get_wtime()
        htmp_d = h_d
        call continuity_PPM_cuda(u_d, htmp_d, h_d, uh_d, 0.5_dp*dt, cont_CS)
        t_end = omp_get_wtime()
        t_continuity = t_continuity + (t_end - t_start)
        call profiler_stop("Continuity")

        t_iter_end = omp_get_wtime()
        t_compute = t_compute + (t_iter_end - t_iter_start)
        print '(A,I4,A,F10.6,A)', '  RK2 iteration ', iter, ':  ', &
            t_iter_end - t_iter_start, ' s'
    end do
    call profiler_stop("RK2_step")

    !=========================================================================
    ! TIMING REPORT
    !=========================================================================
    t_total = t_coriolis + t_barotropic + t_continuity + t_vert_visc + t_hor_visc

    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'Timing Results'
    print '(A)', '================================================================'
    print '(A,F12.6)', 'Compute (wall clock):  ', t_compute
    print '(A,F12.6)', 'Compute (sum of parts):', t_total
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

    !=========================================================================
    ! VERIFICATION — copy final state back to host
    !=========================================================================
    u_h = u_d; v_h = v_d; h_h = h_d; eta_h = eta_d
    call verify_state(h0_h, h_h, u_h, v_h, eta_h, G, GV)

    !=========================================================================
    ! CLEANUP
    !=========================================================================
    call profiler_start("Finalize")
    call continuity_end_cuda(cont_CS)
    call coriolis_end_cuda(cor_CS)
    call barotropic_end_cuda(bt_CS_cuda)
    call vert_visc_end_cuda(visc_CS)
    call hor_visc_end_cuda(hvisc_CS)
    call end_ocean_grid(G)

    deallocate (u_h, v_h, h_h, h0_h, uh_h, vh_h)
    deallocate (eta_h, ubt_h, vbt_h)
    deallocate (u_d, v_d, h_d, h0_d, htmp_d, uh_d, vh_d)
    deallocate (CAu_d, CAv_d, up_d, vp_d, diffu_d, diffv_d)
    deallocate (eta_d, ubt_d, vbt_d, ubt_av_d, vbt_av_d, eta_av_d)
    deallocate (taux_d, tauy_d)
    deallocate (dyCu_d, dxCv_d)
    call profiler_stop("Finalize")

    call profiler_stop("Total")
    call profiler_report("RK2 CUDA Driver", root_region="Total")
    call profiler_end()

contains

    subroutine initialize_state(u, v, h, h0, eta, ubt, vbt, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: u, v, h, h0
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: eta, ubt, vbt

        integer :: i, j, k
        real(dp) :: total_depth

        total_depth = 4000.0_dp

        do k = 1, GV%ke
            do j = G%jsd, G%jed
                do i = G%isd, G%ied
                    h0(i, j, k) = total_depth / real(GV%ke, dp) + &
                                  10.0_dp * sin(real(i - 1, dp) / real(G%ni, dp) * PI) * &
                                  cos(real(j - 1, dp) / real(G%nj, dp) * PI) * &
                                  exp(-real(k, dp) / 20.0_dp)
                    h(i, j, k) = h0(i, j, k)
                    u(i, j, k) = 0.1_dp * sin(real(j - 1, dp) / real(G%nj, dp) * PI * 2.0_dp) * &
                                 exp(-real(k, dp) / 30.0_dp)
                    v(i, j, k) = 0.1_dp * cos(real(i - 1, dp) / real(G%ni, dp) * PI * 2.0_dp) * &
                                 exp(-real(k, dp) / 30.0_dp)
                end do
            end do
        end do

        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                eta(i, j) = 0.5_dp * sin(real(i - 1, dp) / real(G%ni, dp) * PI * 2.0_dp) * &
                            cos(real(j - 1, dp) / real(G%nj, dp) * PI * 2.0_dp)
                ubt(i, j) = 0.05_dp * sin(real(j - 1, dp) / real(G%nj, dp) * PI)
                vbt(i, j) = 0.05_dp * cos(real(i - 1, dp) / real(G%ni, dp) * PI)
            end do
        end do
    end subroutine initialize_state

    subroutine initialize_stress(taux_d, tauy_d, G)
        type(ocean_grid_type), intent(in) :: G
        real(dp), device, intent(out) :: taux_d(G%isd:G%ied, G%jsd:G%jed)
        real(dp), device, intent(out) :: tauy_d(G%isd:G%ied, G%jsd:G%jed)

        real(dp), allocatable :: taux_h(:,:), tauy_h(:,:)
        integer :: i, j

        allocate (taux_h(G%isd:G%ied, G%jsd:G%jed))
        allocate (tauy_h(G%isd:G%ied, G%jsd:G%jed))

        do j = G%jsd, G%jed
            do i = G%isd, G%ied
                taux_h(i, j) = 0.1_dp * sin(real(j - 1, dp) / real(G%nj, dp) * PI)
                tauy_h(i, j) = 0.0_dp
            end do
        end do

        taux_d = taux_h
        tauy_d = tauy_h
        deallocate (taux_h, tauy_h)
    end subroutine initialize_stress

    !> Compute layer transports on the GPU using a CUF kernel
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

    !> Update velocity on the GPU: vel_out = vel_in + dt_scale * (accel + diff)
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

    subroutine verify_state(h_init, h_final, u, v, eta, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_init, h_final
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: eta

        real(dp) :: mass_init, mass_final, rel_error
        real(dp) :: max_u, max_v, max_eta, ke_total
        integer :: i, j, k

        mass_init = 0.0_dp; mass_final = 0.0_dp
        max_u = 0.0_dp; max_v = 0.0_dp; max_eta = 0.0_dp
        ke_total = 0.0_dp

        do k = 1, GV%ke
            do j = G%jsc, G%jec
                do i = G%isc, G%iec
                    mass_init = mass_init + h_init(i, j, k) * G%areaT(i, j)
                    mass_final = mass_final + h_final(i, j, k) * G%areaT(i, j)
                    max_u = max(max_u, abs(u(i, j, k)))
                    max_v = max(max_v, abs(v(i, j, k)))
                    ke_total = ke_total + 0.5_dp * (u(i, j, k)**2 + v(i, j, k)**2) * &
                               h_final(i, j, k) * G%areaT(i, j)
                end do
            end do
        end do

        do j = G%jsc, G%jec
            do i = G%isc, G%iec
                max_eta = max(max_eta, abs(eta(i, j)))
            end do
        end do

        rel_error = abs(mass_final - mass_init) / mass_init

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
    end subroutine verify_state

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
            print '(A)', ''
            print '(A)', '  *** WARNING: CFL > 0.9 - UNSTABLE! ***'
            print '(A,I4,A)', '  Recommend at least ', min_nsteps, ' substeps'
        else if (cfl_2d > 0.7_dp) then
            print '(A)', ''
            print '(A)', '  * CAUTION: CFL > 0.7 - marginally stable'
            print '(A,I4,A)', '  Recommend at least ', min_nsteps, ' substeps'
        else
            print '(A)', '  CFL OK'
        end if
    end subroutine check_bt_cfl

end program rk2_cuda_driver
