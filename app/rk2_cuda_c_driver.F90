!> RK2 driver using CUDA C kernels for ALL physics modules.
!!
!! OpenMP target offloading handles data management (target enter/exit data)
!! and simple inline kernels (transports, velocity updates).
!! All 5 physics modules use CUDA C kernels via iso_c_binding.
!! No cudafor dependency — compilable with any OpenMP-offload-capable compiler
!! that can link against nvcc-compiled objects.
!!
!! Usage: rk2_cuda_c_driver [ni] [nj] [nk] [niter] [bt_nsteps] [diag_mode]
!!        Defaults: 180 180 75 10 30 0
program rk2_cuda_c_driver
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding, only: c_ptr, c_loc, c_null_ptr
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          end_ocean_grid, G_EARTH, PI, OMEGA, EARTH_RADIUS, &
                          mech_forcing_type, vertvisc_type, &
                          init_mech_forcing, end_mech_forcing, &
                          init_vertvisc_visc, end_vertvisc_visc
    ! CUDA C physics modules (portable — no cudafor dependency)
    use mom6_continuity_cuda_c, only: continuity_CS_cuda_c, continuity_init_cuda_c, &
                                       continuity_PPM_cuda_c, continuity_end_cuda_c, &
                                       BT_cont_type_cuda_c, alloc_BT_cont_type_cuda_c, &
                                       dealloc_BT_cont_type_cuda_c
    use mom6_coriolis_cuda_c, only: coriolis_CS_cuda_c, coriolis_init_cuda_c, &
                                     CorAdCalc_cuda_c, coriolis_end_cuda_c, &
                                     SADOURNY75_ENERGY_CUDA_C
    use mom6_barotropic_cuda_c, only: barotropic_CS_cuda_c, barotropic_init_cuda_c, &
                                       btstep_cuda_c, barotropic_end_cuda_c
    use mom6_vert_visc_cuda_c, only: vert_visc_CS_cuda_c, vert_visc_init_cuda_c, &
                                      vert_visc_cra_cuda_c, vert_visc_end_cuda_c
    use mom6_hor_visc_cuda_c, only: hor_visc_CS_cuda_c, hor_visc_init_cuda_c, &
                                     hor_visc_cuda_c, hor_visc_end_cuda_c
    use mom6_diag, only: diag_ctrl, diag_init, diag_end, register_diag_field, DIAG_STATS, &
                         post_data_3d, post_data_2d, post_product_sum_u, post_product_sum_v, &
                         diag_report_timing
    use mom6_profiler, only: profiler_init, profiler_end, profiler_start, profiler_stop, &
                             profiler_report
    implicit none

    ! Grid structures
    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV

    ! CUDA C control structures
    type(continuity_CS_cuda_c) :: cont_CS
    type(coriolis_CS_cuda_c) :: cor_CS
    type(barotropic_CS_cuda_c) :: bt_CS
    type(vert_visc_CS_cuda_c) :: visc_CS
    type(hor_visc_CS_cuda_c) :: hvisc_CS
    type(diag_ctrl) :: diag_CS
    type(BT_cont_type_cuda_c) :: BT_cont_cc
    type(mech_forcing_type) :: forces
    type(vertvisc_type) :: visc

    ! Diagnostic IDs
    integer :: id_KE, id_diffu_sum, id_diffv_sum, id_mass

    ! 3D state variables (target attribute needed for c_loc in use_device_addr regions)
    real(dp), allocatable, target :: u(:, :, :), v(:, :, :)
    real(dp), allocatable, target :: h(:, :, :), h0(:, :, :)
    real(dp), allocatable, target :: uh(:, :, :), vh(:, :, :)
    real(dp), allocatable, target :: CAu(:, :, :), CAv(:, :, :)
    real(dp), allocatable, target :: up(:, :, :), vp(:, :, :)
    real(dp), allocatable, target :: diffu(:, :, :), diffv(:, :, :)
    real(dp), allocatable, target :: htmp(:, :, :)

    ! 2D barotropic variables (target for c_loc)
    real(dp), allocatable, target :: eta(:, :)
    real(dp), allocatable, target :: ubt(:, :), vbt(:, :)
    real(dp), allocatable, target :: uhbt(:, :)
    real(dp), allocatable, target :: ubt_av(:, :), vbt_av(:, :)
    real(dp), allocatable, target :: eta_av(:, :)

    ! Memory tracking
    integer(int64) :: total_bytes, state_bytes

    ! Timing
    real(dp) :: dt, t_start, t_end, t_total
    real(dp) :: t_coriolis, t_barotropic, t_continuity, t_vert_visc, t_hor_visc
    real(dp) :: t_diag, t_init, t_compute
    integer :: clock_start, clock_end, clock_rate
    integer :: init_clock_start, init_clock_end, init_clock_rate
    integer :: compute_clock_start, compute_clock_end, compute_clock_rate

    ! Parameters
    integer :: ni, nj, nk, niter, bt_nsteps
    integer :: iter, i, j, k
    integer :: diag_mode
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10; bt_nsteps = 30
    diag_mode = 0
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
        call get_command_argument(6, arg); read (arg, *) diag_mode
    end if
    call profiler_init()
    call profiler_start("Total")

    print '(A)', '================================================================'
    print '(A)', 'MOM6 Split RK2 — Full CUDA C Kernels Driver'
    print '(A)', '================================================================'
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'RK2 iterations: ', niter
    print '(A,I4)', 'Barotropic substeps: ', bt_nsteps
    if (diag_mode > 0) then
        print '(A)', 'Diagnostics: ENABLED'
    else
        print '(A)', 'Diagnostics: disabled'
    end if

    call check_bt_cfl(dt, bt_nsteps, 10.0_dp, 4000.0_dp)

    print '(A)', '================================================================'

    !=========================================================================
    ! INITIALIZATION
    !=========================================================================
    call profiler_start("Initialization")
    call system_clock(init_clock_start, init_clock_rate)
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)

    ! --- CUDA C Continuity init ---
    call continuity_init_cuda_c(cont_CS, G%isd, G%ied, G%jsd, G%jed, &
                                 G%isc, G%iec, G%jsc, G%jec, nk, &
                                 G%IareaT, G%IdxT, G%dy_Cu, G%mask2dT, .true., &
                                 G%dxT, G%areaT, G%dxCu, G%mask2dCu)

    ! --- CUDA C BT_cont ---
    call alloc_BT_cont_type_cuda_c(BT_cont_cc, G%isd, G%ied, G%jsd, G%jed, nk)

    ! --- CUDA C Coriolis init ---
    call coriolis_init_cuda_c(cor_CS, G%isd, G%ied, G%jsd, G%jed, &
                              G%isc, G%iec, G%jsc, G%jec, nk, &
                              G%areaT, G%IareaBu, G%CoriolisBu, G%mask2dBu, &
                              G%dyCv, G%dxCu, G%dyCu, G%dxCv, G%IdxCu, G%IdyCv, &
                              SADOURNY75_ENERGY_CUDA_C)

    ! --- CUDA C Barotropic init (builds metrics from grid) ---
    call barotropic_init_cuda_c_from_grid(bt_CS, G, dt, bt_nsteps)

    ! --- CUDA C Vertical viscosity init ---
    call vert_visc_init_cuda_c(visc_CS, G%isd, G%ied, G%jsd, G%jed, &
                                G%isc, G%iec, G%jsc, G%jec, nk, &
                                G%mask2dCu, G%mask2dCv, &
                                1.0e-4_dp, 1.0e-2_dp, 1.0e-2_dp, 50.0_dp, 10.0_dp)

    ! --- CUDA C Horizontal viscosity init (builds metrics from grid) ---
    call hor_visc_init_cuda_c_from_grid(hvisc_CS, G, nk, 100.0_dp)

    ! Forces and visc (still use OMP target for taux/tauy data management)
    call init_mech_forcing(forces, G)
    call init_vertvisc_visc(visc, G, GV, use_rayleigh=.true.)

    call diag_init(diag_CS, G, GV, output_dir='./', output_freq=1)

    if (diag_mode > 0) then
        id_KE = register_diag_field(diag_CS, 'KE', 'Kinetic Energy', 'm2/s2', &
                                    3, 'h', DIAG_STATS)
        id_diffu_sum = register_diag_field(diag_CS, 'diffu_sum', &
                                           'Vertically summed u-diffusion', 'm/s2', &
                                           2, 'u', DIAG_STATS)
        id_diffv_sum = register_diag_field(diag_CS, 'diffv_sum', &
                                           'Vertically summed v-diffusion', 'm/s2', &
                                           2, 'v', DIAG_STATS)
        id_mass = register_diag_field(diag_CS, 'mass', &
                                      'Vertically summed thickness', 'm', &
                                      2, 'h', DIAG_STATS)
    else
        id_KE = -1; id_diffu_sum = -1; id_diffv_sum = -1; id_mass = -1
    end if

    ! Allocate 3D arrays
    allocate (u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (h0(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (htmp(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAu(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (CAv(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (up(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (vp(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffu(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate (diffv(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Allocate 2D arrays
    allocate (eta(G%isd:G%ied, G%jsd:G%jed))
    allocate (ubt(G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt(G%isd:G%ied, G%jsd:G%jed))
    allocate (uhbt(G%isd:G%ied, G%jsd:G%jed))
    allocate (ubt_av(G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt_av(G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_av(G%isd:G%ied, G%jsd:G%jed))

    call initialize_forces_visc(forces, visc, G, GV)
    call initialize_state(u, v, h, h0, eta, ubt, vbt, G, GV)
    uhbt = 0.0_dp

    ! Copy all state arrays to GPU
    !$omp target enter data map(to: u, v, h, h0, uh, vh, eta, ubt, vbt, uhbt)
    !$omp target enter data map(alloc: CAu, CAv, up, vp, diffu, diffv, htmp)
    !$omp target enter data map(alloc: ubt_av, vbt_av, eta_av)
    !$omp target update to(forces%taux, forces%tauy)
    !$omp target update to(visc%Ray_u, visc%Ray_v)
    call profiler_stop("Initialization")

    call system_clock(init_clock_end)
    t_init = real(init_clock_end - init_clock_start, dp) / real(init_clock_rate, dp)

    ! Memory usage report
    state_bytes = int(G%ied - G%isd + 1, int64) * int(G%jed - G%jsd + 1, int64) &
        * (13_int64 * int(nk, int64) + 7_int64) * 8_int64
    total_bytes = G%nbytes + forces%nbytes + visc%nbytes + state_bytes
    print '(A)', ''
    print '(A)', 'GPU Memory Usage:'
    print '(A, F10.2, A)', '  state arrays:     ', real(state_bytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  ocean_grid_type:  ', real(G%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, A)',        '  All physics (CUDA C): managed by CUDA runtime'
    print '(A, F10.2, A)', '  mech_forcing:     ', real(forces%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  vertvisc_type:    ', real(visc%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A)', '  ----------------------------------'
    print '(A, F10.2, A)', '  Total (OMP-managed):', real(total_bytes, dp) / (1024.0_dp**2), ' MB'

    print '(A)', ''
    print '(A,F12.6,A)', 'Initialization time: ', t_init, ' s'
    print '(A)', ''
    print '(A)', 'Running split RK2 time-stepping (Full CUDA C)...'
    print '(A)', ''

    t_total = 0.0_dp
    t_coriolis = 0.0_dp
    t_barotropic = 0.0_dp
    t_continuity = 0.0_dp
    t_vert_visc = 0.0_dp
    t_hor_visc = 0.0_dp
    t_diag = 0.0_dp
    call system_clock(compute_clock_start, compute_clock_rate)
    block
        real(dp) :: iter_start, iter_end
        integer :: iter_clock_start, iter_clock_end, iter_clock_rate
        call profiler_start("RK2_step", nvtx_only=.true.)
        do iter = 1, niter
            call system_clock(iter_clock_start, iter_clock_rate)

            !$omp target teams distribute parallel do collapse(3)
            do k=1,nk
              do j=G%jsd,G%jed
                do i=G%isd,G%ied
                h(i, j, k) = h0(i, j, k)
                end do
              end do
            end do

            !=========================================================================
            ! PREDICTOR PHASE
            !=========================================================================

            call profiler_start("Transports")
            call compute_transports(u, v, h, uh, vh, G, GV)
            call profiler_stop("Transports")

            ! Horizontal viscosity — CUDA C
            call profiler_start("HorVisc")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(u, v, h, diffu, diffv)
            call hor_visc_cuda_c( &
                c_loc(u(G%isd, G%jsd, 1)), c_loc(v(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), &
                c_loc(diffu(G%isd, G%jsd, 1)), c_loc(diffv(G%isd, G%jsd, 1)), &
                hvisc_CS)
            !$omp end target data
            call system_clock(clock_end)
            t_hor_visc = t_hor_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("HorVisc")

            ! Coriolis — CUDA C
            call profiler_start("Coriolis")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(u, v, h, uh, vh, CAu, CAv)
            call CorAdCalc_cuda_c( &
                c_loc(u(G%isd, G%jsd, 1)), c_loc(v(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), c_loc(uh(G%isd, G%jsd, 1)), &
                c_loc(vh(G%isd, G%jsd, 1)), &
                c_loc(CAu(G%isd, G%jsd, 1)), c_loc(CAv(G%isd, G%jsd, 1)), &
                cor_CS)
            !$omp end target data
            call system_clock(clock_end)
            t_coriolis = t_coriolis + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Coriolis")

            call profiler_start("VelUpdate_pred")
            !$omp target teams distribute parallel do collapse(3)
            do k=1,nk
              do j=G%jsc,G%jec
                do i=G%isc,G%iec - 1
                up(i, j, k) = u(i, j, k) + dt*(CAu(i, j, k) + diffu(i, j, k))
                end do
              end do
            end do
            !$omp target teams distribute parallel do collapse(3)
            do k=1,nk
              do j=G%jsc,G%jec - 1
                do i=G%isc,G%iec
                vp(i, j, k) = v(i, j, k) + dt*(CAv(i, j, k) + diffv(i, j, k))
                end do
              end do
            end do
            call profiler_stop("VelUpdate_pred")

            ! Vertical viscosity — CUDA C
            call profiler_start("VertVisc")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(up, vp, h, forces%taux, forces%tauy)
            call vert_visc_cra_cuda_c( &
                c_loc(up(G%isd, G%jsd, 1)), c_loc(vp(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), dt, visc_CS, &
                c_loc(forces%taux(G%isd, G%jsd)), c_loc(forces%tauy(G%isd, G%jsd)))
            !$omp end target data
            call system_clock(clock_end)
            t_vert_visc = t_vert_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("VertVisc")

            ! Barotropic — CUDA C
            call profiler_start("Barotropic")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(eta, ubt, vbt, ubt_av, vbt_av, eta_av)
            call btstep_cuda_c( &
                c_loc(eta(G%isd, G%jsd)), c_loc(ubt(G%isd, G%jsd)), &
                c_loc(vbt(G%isd, G%jsd)), &
                c_loc(ubt_av(G%isd, G%jsd)), c_loc(vbt_av(G%isd, G%jsd)), &
                c_loc(eta_av(G%isd, G%jsd)), bt_CS)
            !$omp end target data
            call system_clock(clock_end)
            t_barotropic = t_barotropic + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Barotropic")

            ! Continuity — CUDA C
            call profiler_start("Continuity")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(up, h0, h, uh, uhbt)
            call continuity_PPM_cuda_c( &
                c_loc(up(G%isd, G%jsd, 1)), c_loc(h0(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), c_loc(uh(G%isd, G%jsd, 1)), &
                dt, cont_CS, &
                c_loc(uhbt(G%isd, G%jsd)), BT_cont_cc)
            !$omp end target data
            call system_clock(clock_end)
            t_continuity = t_continuity + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Continuity")

            if (id_diffu_sum > 0 .or. id_diffv_sum > 0) then
                call profiler_start("Diagnostics")
                call system_clock(clock_start, clock_rate)
                call post_product_sum_u(id_diffu_sum, diffu, h, G, nk, diag_CS)
                call post_product_sum_v(id_diffv_sum, diffv, h, G, nk, diag_CS)
                call system_clock(clock_end)
                t_diag = t_diag + real(clock_end - clock_start, dp) / real(clock_rate, dp)
                call profiler_stop("Diagnostics")
            end if

            !=========================================================================
            ! CORRECTOR PHASE
            !=========================================================================

            call profiler_start("Transports")
            call compute_transports(up, vp, h, uh, vh, G, GV)
            call profiler_stop("Transports")

            ! Horizontal viscosity corrector — CUDA C
            call profiler_start("HorVisc")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(up, vp, h, diffu, diffv)
            call hor_visc_cuda_c( &
                c_loc(up(G%isd, G%jsd, 1)), c_loc(vp(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), &
                c_loc(diffu(G%isd, G%jsd, 1)), c_loc(diffv(G%isd, G%jsd, 1)), &
                hvisc_CS)
            !$omp end target data
            call system_clock(clock_end)
            t_hor_visc = t_hor_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("HorVisc")

            ! Coriolis corrector — CUDA C
            call profiler_start("Coriolis")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(up, vp, h, uh, vh, CAu, CAv)
            call CorAdCalc_cuda_c( &
                c_loc(up(G%isd, G%jsd, 1)), c_loc(vp(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), c_loc(uh(G%isd, G%jsd, 1)), &
                c_loc(vh(G%isd, G%jsd, 1)), &
                c_loc(CAu(G%isd, G%jsd, 1)), c_loc(CAv(G%isd, G%jsd, 1)), &
                cor_CS)
            !$omp end target data
            call system_clock(clock_end)
            t_coriolis = t_coriolis + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Coriolis")

            call profiler_start("VelUpdate_corr")
            !$omp target teams distribute parallel do collapse(3)
            do k=1,nk
              do j=G%jsc,G%jec
                do i=G%isc,G%iec - 1
                u(i, j, k) = u(i, j, k) + 0.5_dp*dt*(CAu(i, j, k) + diffu(i, j, k))
                end do
              end do
            end do
            !$omp target teams distribute parallel do collapse(3)
            do k=1,nk
              do j=G%jsc,G%jec - 1
                do i=G%isc,G%iec
                v(i, j, k) = v(i, j, k) + 0.5_dp*dt*(CAv(i, j, k) + diffv(i, j, k))
                end do
              end do
            end do
            call profiler_stop("VelUpdate_corr")

            ! Vertical viscosity corrector — CUDA C
            call profiler_start("VertVisc")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(u, v, h, forces%taux, forces%tauy)
            call vert_visc_cra_cuda_c( &
                c_loc(u(G%isd, G%jsd, 1)), c_loc(v(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), 0.5_dp*dt, visc_CS, &
                c_loc(forces%taux(G%isd, G%jsd)), c_loc(forces%tauy(G%isd, G%jsd)))
            !$omp end target data
            call system_clock(clock_end)
            t_vert_visc = t_vert_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("VertVisc")

            ! Barotropic corrector — CUDA C
            call profiler_start("Barotropic")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(eta_av, ubt_av, vbt_av, eta)
            call btstep_cuda_c( &
                c_loc(eta_av(G%isd, G%jsd)), c_loc(ubt_av(G%isd, G%jsd)), &
                c_loc(vbt_av(G%isd, G%jsd)), &
                c_loc(ubt_av(G%isd, G%jsd)), c_loc(vbt_av(G%isd, G%jsd)), &
                c_loc(eta(G%isd, G%jsd)), bt_CS)
            !$omp end target data
            call system_clock(clock_end)
            t_barotropic = t_barotropic + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Barotropic")

            ! Continuity corrector — CUDA C
            call profiler_start("Continuity")
            call system_clock(clock_start, clock_rate)
            !$omp target teams distribute parallel do collapse(3)
            do k=1,GV%ke
              do j=G%jsd,G%jed
                do i=G%isd,G%ied
                htmp(i, j, k) = h(i, j, k)
                end do
              end do
            end do
            !$omp target data use_device_addr(u, htmp, h, uh, uhbt)
            call continuity_PPM_cuda_c( &
                c_loc(u(G%isd, G%jsd, 1)), c_loc(htmp(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), c_loc(uh(G%isd, G%jsd, 1)), &
                0.5_dp*dt, cont_CS, &
                c_loc(uhbt(G%isd, G%jsd)), BT_cont_cc)
            !$omp end target data
            call system_clock(clock_end)
            t_continuity = t_continuity + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Continuity")

            if (iter == niter .and. (id_KE > 0 .or. id_mass > 0)) then
                call profiler_start("Diagnostics")
                call system_clock(clock_start, clock_rate)
                call compute_and_post_KE(id_KE, u, v, h, G, GV, diag_CS)
                call compute_and_post_mass(id_mass, h, G, GV, diag_CS)
                call system_clock(clock_end)
                t_diag = t_diag + real(clock_end - clock_start, dp) / real(clock_rate, dp)
                call profiler_stop("Diagnostics")
            end if
            call system_clock(iter_clock_end)
            print '(A,I4,A,F10.6,A)', '  RK2 iteration ', iter, ':  ', &
                real(iter_clock_end - iter_clock_start, dp) / real(iter_clock_rate, dp), ' s'

        end do
    end block
    call profiler_stop("RK2_step")
    call system_clock(compute_clock_end)
    t_compute = real(compute_clock_end - compute_clock_start, dp) / real(compute_clock_rate, dp)

    t_total = t_coriolis + t_barotropic + t_continuity + t_vert_visc + t_hor_visc + t_diag

    print '(A)', ''
    print '(A)', '================================================================'
    print '(A)', 'Timing Results'
    print '(A)', '================================================================'
    print '(A,F12.6)', 'Init (alloc+GPU xfer): ', t_init
    print '(A,F12.6)', 'Compute (wall clock):  ', t_compute
    print '(A,F12.6)', 'Compute (sum of parts):', t_total
    print '(A)', '----------------------------------------------------------------'
    print '(A,F12.6,A,F5.1,A)', '  Coriolis (CUDA C):   ', t_coriolis, &
        '  (', 100.0_dp*t_coriolis/t_total, '%)'
    print '(A,F12.6,A,F5.1,A)', '  Hor viscosity:       ', t_hor_visc, &
        '  (', 100.0_dp*t_hor_visc/t_total, '%)'
    print '(A,F12.6,A,F5.1,A)', '  Vert viscosity:      ', t_vert_visc, &
        '  (', 100.0_dp*t_vert_visc/t_total, '%)'
    print '(A,F12.6,A,F5.1,A)', '  Barotropic:          ', t_barotropic, &
        '  (', 100.0_dp*t_barotropic/t_total, '%)'
    print '(A,F12.6,A,F5.1,A)', '  Continuity:          ', t_continuity, &
        '  (', 100.0_dp*t_continuity/t_total, '%)'
    if (t_diag > 0.0_dp) then
        print '(A,F12.6,A,F5.1,A)', '  Diagnostics:         ', t_diag, &
            '  (', 100.0_dp*t_diag/t_total, '%)'
    end if
    print '(A)', '----------------------------------------------------------------'
    print '(A,F12.6)', 'Time per RK2 step:     ', t_compute/real(niter, dp)
    print '(A)', '================================================================'
    call diag_report_timing(diag_CS)

    ! Bring final state back to host for verification
    !$omp target update from(u, v, h, eta)

    call verify_state(h0, h, u, v, eta, G, GV)

    ! Cleanup
    call profiler_start("Finalize")
    call continuity_end_cuda_c(cont_CS)
    call dealloc_BT_cont_type_cuda_c(BT_cont_cc)
    call coriolis_end_cuda_c(cor_CS)
    call barotropic_end_cuda_c(bt_CS)
    call end_mech_forcing(forces)
    call end_vertvisc_visc(visc)
    call vert_visc_end_cuda_c(visc_CS)
    call hor_visc_end_cuda_c(hvisc_CS)
    call diag_end(diag_CS)
    call end_ocean_grid(G)
    call profiler_stop("Finalize")

    call profiler_stop("Total")
    call profiler_report("RK2 Full CUDA C Driver", root_region="Total")
    call profiler_end()

    ! Release GPU memory for state arrays
    !$omp target exit data map(delete: u, v, h, h0, uh, vh, eta, ubt, vbt, uhbt)
    !$omp target exit data map(delete: CAu, CAv, up, vp, diffu, diffv, htmp)
    !$omp target exit data map(delete: ubt_av, vbt_av, eta_av)

    deallocate (u, v, h, h0, uh, vh, CAu, CAv, up, vp)
    deallocate (diffu, diffv, htmp)
    deallocate (eta, ubt, vbt, uhbt, ubt_av, vbt_av, eta_av)

contains

    subroutine initialize_state(u, v, h, h0, eta, ubt, vbt, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: u, v, h, h0
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: eta, ubt, vbt
        integer :: i, j, k
        real(dp) :: total_depth
        total_depth = 4000.0_dp
        !$omp parallel do collapse(3) private(i,j,k)
        do k=1,GV%ke
          do j=G%jsd,G%jed
            do i=G%isd,G%ied
            h0(i, j, k) = total_depth/real(GV%ke, dp) + &
                          10.0_dp*sin(real(i - 1, dp)/real(G%ni, dp)*PI)* &
                          cos(real(j - 1, dp)/real(G%nj, dp)*PI)* &
                          exp(-real(k, dp)/20.0_dp)
            h(i, j, k) = h0(i, j, k)
            u(i, j, k) = 0.1_dp*sin(real(j - 1, dp)/real(G%nj, dp)*PI*2.0_dp)* &
                         exp(-real(k, dp)/30.0_dp)
            v(i, j, k) = 0.1_dp*cos(real(i - 1, dp)/real(G%ni, dp)*PI*2.0_dp)* &
                         exp(-real(k, dp)/30.0_dp)
            end do
          end do
        end do
        !$omp parallel do collapse(2) private(i,j)
        do j=G%jsd,G%jed
          do i=G%isd,G%ied
            eta(i, j) = 0.5_dp*sin(real(i - 1, dp)/real(G%ni, dp)*PI*2.0_dp)* &
                        cos(real(j - 1, dp)/real(G%nj, dp)*PI*2.0_dp)
            ubt(i, j) = 0.05_dp*sin(real(j - 1, dp)/real(G%nj, dp)*PI)
            vbt(i, j) = 0.05_dp*cos(real(i - 1, dp)/real(G%ni, dp)*PI)
          end do
        end do
    end subroutine initialize_state

    subroutine initialize_forces_visc(forces, visc, G, GV)
        type(mech_forcing_type), intent(inout) :: forces
        type(vertvisc_type), intent(inout) :: visc
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        integer :: i, j, k
        !$omp parallel do collapse(2) private(i,j)
        do j=G%jsd,G%jed
          do i=G%isd,G%ied
            forces%taux(i, j) = 0.1_dp * sin(real(j - 1, dp) / real(G%nj, dp) * PI)
            forces%tauy(i, j) = 0.0_dp
          end do
        end do
        if (visc%has_Rayleigh) then
            !$omp parallel do collapse(3) private(i,j,k)
            do k=1,GV%ke
              do j=G%jsd,G%jed
                do i=G%isd,G%ied
                visc%Ray_u(i, j, k) = 0.0_dp
                visc%Ray_v(i, j, k) = 0.0_dp
                end do
              end do
            end do
            !$omp parallel do collapse(2) private(i,j)
            do j=G%jsd,G%jed
              do i=G%isd,G%ied
                visc%Ray_u(i, j, GV%ke) = 1.0e-4_dp
                visc%Ray_v(i, j, GV%ke) = 1.0e-4_dp
              end do
            end do
        end if
    end subroutine initialize_forces_visc

    subroutine compute_transports(u, v, h, uh, vh, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh, vh
        integer :: i, j, k
        !$omp target teams distribute parallel do collapse(3)
        do k=1,GV%ke
          do j=G%jsd,G%jed
            do i=G%isd,G%ied
            uh(i, j, k) = u(i, j, k)*0.5_dp*(h(i, j, k) + h(min(i + 1, G%ied), j, k))*G%dyCu(i, j)
            vh(i, j, k) = v(i, j, k)*0.5_dp*(h(i, j, k) + h(i, min(j + 1, G%jed), k))*G%dxCv(i, j)
            end do
          end do
        end do
    end subroutine compute_transports

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
                    mass_init = mass_init + h_init(i, j, k)*G%areaT(i, j)
                    mass_final = mass_final + h_final(i, j, k)*G%areaT(i, j)
                    max_u = max(max_u, abs(u(i, j, k)))
                    max_v = max(max_v, abs(v(i, j, k)))
                    ke_total = ke_total + 0.5_dp*(u(i, j, k)**2 + v(i, j, k)**2)* &
                               h_final(i, j, k)*G%areaT(i, j)
                end do
            end do
        end do
        do j = G%jsc, G%jec
            do i = G%isc, G%iec
                max_eta = max(max_eta, abs(eta(i, j)))
            end do
        end do
        rel_error = abs(mass_final - mass_init)/mass_init
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
        dx_m = dx_km*1000.0_dp
        dtbt = dt/real(nsteps, dp)
        c_grav = sqrt(G_EARTH*depth)
        cfl = c_grav*dtbt/dx_m
        cfl_2d = cfl*sqrt(2.0_dp)
        min_nsteps = ceiling(dt*c_grav/dx_m*2.5_dp)
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
            print '(A,I4,A)', '  Recommend at least ', min_nsteps, ' substeps'
        else
            print '(A)', '  CFL OK'
        end if
    end subroutine check_bt_cfl

    !> Initialize CUDA C hor_visc from grid metrics (computes derived metrics)
    subroutine hor_visc_init_cuda_c_from_grid(CS, G, nk, Kh)
        type(hor_visc_CS_cuda_c), intent(inout) :: CS
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

        allocate(DY_dxT(isd:ied, jsd:jed), DX_dyT(isd:ied, jsd:jed))
        allocate(DY_dxBu(isd:ied, jsd:jed), DX_dyBu(isd:ied, jsd:jed))
        allocate(reduction_xx(isd:ied, jsd:jed), reduction_xy(isd:ied, jsd:jed))
        allocate(dy2h(isd:ied, jsd:jed), dx2h(isd:ied, jsd:jed))
        allocate(dy2q(isd:ied, jsd:jed), dx2q(isd:ied, jsd:jed))

        do j = jsd, jed
            do i = isd, ied
                DX_dyT(i,j) = G%dxT(i,j) * G%IdyT(i,j)
                DY_dxT(i,j) = G%dyT(i,j) * G%IdxT(i,j)
                DX_dyBu(i,j) = G%dxBu(i,j) * G%IdyBu(i,j)
                DY_dxBu(i,j) = G%dyBu(i,j) * G%IdxBu(i,j)
                dx2h(i,j) = G%dxT(i,j) * G%dxT(i,j)
                dy2h(i,j) = G%dyT(i,j) * G%dyT(i,j)
                dx2q(i,j) = G%dxBu(i,j) * G%dxBu(i,j)
                dy2q(i,j) = G%dyBu(i,j) * G%dyBu(i,j)
                reduction_xx(i,j) = 1.0_dp
                reduction_xy(i,j) = 1.0_dp
            end do
        end do

        h_neglect = 1.0e-3_dp

        call hor_visc_init_cuda_c(CS, isd, ied, jsd, jed, &
                                   G%isc, G%iec, G%jsc, G%jec, nk, &
                                   Kh, h_neglect, &
                                   DY_dxT, DX_dyT, DY_dxBu, DX_dyBu, &
                                   G%IdyCu, G%IdxCu, G%IdyCv, G%IdxCv, &
                                   G%IareaCu, G%IareaCv, &
                                   G%mask2dT, G%mask2dBu, &
                                   reduction_xx, reduction_xy, &
                                   dy2h, dx2h, dy2q, dx2q)

        deallocate(DY_dxT, DX_dyT, DY_dxBu, DX_dyBu)
        deallocate(reduction_xx, reduction_xy)
        deallocate(dy2h, dx2h, dy2q, dx2q)

    end subroutine hor_visc_init_cuda_c_from_grid

    !> Initialize CUDA C barotropic from grid metrics (computes derived metrics)
    subroutine barotropic_init_cuda_c_from_grid(CS, G, dt, nstep)
        type(barotropic_CS_cuda_c), intent(inout) :: CS
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

        allocate(Datu(isd:ied, jsd:jed), Datv(isd:ied, jsd:jed))
        allocate(gtot_E(isd:ied, jsd:jed), gtot_W(isd:ied, jsd:jed))
        allocate(gtot_N(isd:ied, jsd:jed), gtot_S(isd:ied, jsd:jed))
        allocate(f_4_u(4, isd:ied, jsd:jed), f_4_v(4, isd:ied, jsd:jed))
        allocate(bt_rem_u(isd:ied, jsd:jed), bt_rem_v(isd:ied, jsd:jed))

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

        call barotropic_init_cuda_c(CS, isd, ied, jsd, jed, &
                                     G%isc, G%iec, G%jsc, G%jec, &
                                     nstep, dt, bebt, 0, &
                                     Datu, Datv, &
                                     gtot_E, gtot_W, gtot_N, gtot_S, &
                                     f_4_u, f_4_v, bt_rem_u, bt_rem_v, &
                                     G%IareaT, G%IdxCu, G%IdyCv)

        deallocate(Datu, Datv)
        deallocate(gtot_E, gtot_W, gtot_N, gtot_S)
        deallocate(f_4_u, f_4_v, bt_rem_u, bt_rem_v)

    end subroutine barotropic_init_cuda_c_from_grid

    subroutine compute_and_post_KE(id, u, v, h, G, GV, CS)
        integer, intent(in) :: id
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
        type(diag_ctrl), intent(inout) :: CS
        real(dp), allocatable :: KE(:, :, :)
        integer :: i, j, k
        if (id <= 0) return
        allocate (KE(G%isd:G%ied, G%jsd:G%jed, GV%ke))
        do k=1,GV%ke
          do j=G%jsc,G%jec
            do i=G%isc,G%iec
            KE(i, j, k) = 0.5_dp*( &
                          0.5_dp*(u(i, j, k)**2 + u(i - 1, j, k)**2) + &
                          0.5_dp*(v(i, j, k)**2 + v(i, j - 1, k)**2))
            end do
          end do
        end do
        call post_data_3d(id, KE, G, GV, CS)
        deallocate (KE)
    end subroutine compute_and_post_KE

    subroutine compute_and_post_mass(id, h, G, GV, CS)
        integer, intent(in) :: id
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h
        type(diag_ctrl), intent(inout) :: CS
        real(dp), allocatable :: mass(:, :)
        integer :: i, j, k
        if (id <= 0) return
        allocate (mass(G%isd:G%ied, G%jsd:G%jed))
        mass = 0.0_dp
        do k = 1, GV%ke
            do j=G%jsc,G%jec
              do i=G%isc,G%iec
                mass(i, j) = mass(i, j) + h(i, j, k)
              end do
            end do
        end do
        call post_data_2d(id, mass, G, CS)
        deallocate (mass)
    end subroutine compute_and_post_mass

    subroutine init_verticalGrid(GV, nk)
        type(verticalGrid_type), intent(inout) :: GV
        integer, intent(in) :: nk
        GV%ke = nk
        GV%Angstrom_H = 1.0e-10_dp
    end subroutine init_verticalGrid

end program rk2_cuda_c_driver
