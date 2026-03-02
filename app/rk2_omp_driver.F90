!> Unified RK2 driver for MOM6 miniapps (OpenMP target offloading)
!!
!! This driver orchestrates all solvers (continuity, Coriolis, barotropic, vert_visc)
!! in a simplified split RK2 time-stepping scheme similar to MOM6's step_MOM_dyn_split_RK2.
!!
!! Translated from OpenACC (rk2_driver) to OpenMP target offloading.
!!
!! The split RK2 scheme:
!!   1. Predictor phase:
!!      - Compute Coriolis/momentum advection (CAu, CAv)
!!      - Apply vertical viscosity
!!      - Advance barotropic mode (fast 2D dynamics)
!!      - Update layer thicknesses via continuity
!!   2. Corrector phase:
!!      - Recompute tendencies with updated state
!!      - Apply vertical viscosity
!!      - Final barotropic step
!!      - Final continuity update
!!
program rk2_omp_driver
    use iso_fortran_env, only: dp => real64, int64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                          init_verticalGrid, end_ocean_grid, G_EARTH, BT_cont_type, &
                          alloc_BT_cont_type, &
                          mech_forcing_type, vertvisc_type, &
                          init_mech_forcing, end_mech_forcing, &
                          init_vertvisc_visc, end_vertvisc_visc
    use mom6_continuity_omp, only: continuity_CS, continuity_init, continuity_PPM, continuity_end
    use mom6_coriolis_omp, only: coriolis_CS, coriolis_init, CorAdCalc, coriolis_end, &
                             SADOURNY75_ENERGY
    use mom6_barotropic_omp, only: barotropic_CS, barotropic_init, btstep, barotropic_end
    use mom6_vert_visc_omp, only: vert_visc_CS, vert_visc_init, vert_visc_coef, &
                              vert_visc_apply, vert_visc_end, vert_visc_remnant, &
                              vert_visc_coef_apply
    use mom6_hor_visc_omp, only: hor_visc_CS, hor_visc_init, hor_visc, hor_visc_end
    use mom6_diag, only: diag_ctrl, diag_init, diag_end, register_diag_field, DIAG_STATS, &
                         post_data_3d, post_data_2d, post_product_sum_u, post_product_sum_v, &
                         diag_report_timing
    use mom6_profiler, only: profiler_init, profiler_end, profiler_start, profiler_stop, &
                             profiler_report
    implicit none

    ! Grid structures
    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV

    ! Control structures
    type(continuity_CS) :: cont_CS
    type(coriolis_CS) :: cor_CS
    type(barotropic_CS) :: bt_CS
    type(vert_visc_CS) :: visc_CS
    type(hor_visc_CS) :: hvisc_CS
    type(diag_ctrl) :: diag_CS
    type(BT_cont_type), pointer :: BT_cont
    type(mech_forcing_type) :: forces
    type(vertvisc_type) :: visc

    ! Diagnostic IDs
    integer :: id_KE, id_diffu_sum, id_diffv_sum, id_mass

    ! 3D state variables
    real(dp), allocatable :: u(:, :, :), v(:, :, :)       ! Velocities
    real(dp), allocatable :: u_cor(:, :, :)
    real(dp), allocatable :: h(:, :, :), h0(:, :, :)      ! Layer thickness (current, initial)
    real(dp), allocatable :: uh(:, :, :), vh(:, :, :)     ! Layer transports
    real(dp), allocatable :: CAu(:, :, :), CAv(:, :, :)   ! Coriolis accelerations
    real(dp), allocatable :: up(:, :, :), vp(:, :, :)     ! Predictor velocities
    real(dp), allocatable :: diffu(:, :, :), diffv(:, :, :)  ! Horizontal viscous accelerations
    real(dp), allocatable :: por_face_areaU(:, :, :)
    real(dp), allocatable :: visc_rem_u(:, :, :)
    real(dp), allocatable :: htmp(:, :, :) ! temporary thickness

    ! 2D barotropic variables
    real(dp), allocatable :: eta(:, :)                 ! Sea surface height
    real(dp), allocatable :: ubt(:, :), vbt(:, :)       ! Barotropic velocities
    real(dp), allocatable :: uhbt(:, :)
    real(dp), allocatable :: ubt_av(:, :), vbt_av(:, :)  ! Time-averaged BT velocities
    real(dp), allocatable :: eta_av(:, :)              ! Time-averaged SSH
    real(dp), allocatable :: du_cor(:, :)

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
    integer :: diag_mode  ! 0 = disabled, 1 = enabled
    character(len=32) :: arg

    ! Default parameters
    ni = 180; nj = 180; nk = 75; niter = 10; bt_nsteps = 30
    diag_mode = 0  ! Diagnostics disabled by default for benchmarking
    dt = 300.0_dp  ! 5 minute baroclinic timestep

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

    print '(A)', '=================================================='
    print '(A)', 'MOM6 Split RK2 Driver (OpenMP target)'
    print '(A)', '=================================================='
    print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
    print '(A,I4)', 'RK2 iterations: ', niter
    print '(A,I4)', 'Barotropic substeps: ', bt_nsteps
    if (diag_mode > 0) then
        print '(A)', 'Diagnostics: ENABLED'
    else
        print '(A)', 'Diagnostics: disabled'
    end if

    ! CFL check for barotropic solver
    call check_bt_cfl(dt, bt_nsteps, 10.0_dp, 4000.0_dp)

    print '(A)', '=================================================='

    ! Initialize grids and control structures
    call profiler_start("Initializatoin")
    call system_clock(init_clock_start, init_clock_rate)
    call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
    call init_verticalGrid(GV, nk)
    call continuity_init(cont_CS, G, GV, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call alloc_BT_cont_type(BT_cont, G, GV)
    call coriolis_init(cor_CS, G, GV, SADOURNY75_ENERGY)
    call barotropic_init(bt_CS, G, dt, bt_nsteps)
    call vert_visc_init(visc_CS, G, GV, Kv=1.0e-4_dp, Kv_ml=1.0e-2_dp, &
                        Kv_extra_bbl=1.0e-2_dp, Hmix=50.0_dp)
    call hor_visc_init(hvisc_CS, G, GV, Kh=100.0_dp)

    ! Initialize forces and visc for vertical viscosity
    call init_mech_forcing(forces, G)
    call init_vertvisc_visc(visc, G, GV, use_rayleigh=.true.)

    ! Initialize diagnostics
    call diag_init(diag_CS, G, GV, output_dir='./', output_freq=1)

    ! Register diagnostics (controlled by diag_mode: 0=off, 1=on)
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
        id_KE = -1
        id_diffu_sum = -1
        id_diffv_sum = -1
        id_mass = -1
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
    allocate (ubt_av(G%isd:G%ied, G%jsd:G%jed))
    allocate (vbt_av(G%isd:G%ied, G%jsd:G%jed))
    allocate (eta_av(G%isd:G%ied, G%jsd:G%jed))

    ! Initialize forces/visc data
    call initialize_forces_visc(forces, visc, G, GV)

    ! Initialize state
    call initialize_state(u, v, h, h0, eta, ubt, vbt, G, GV)

    ! Copy all state arrays to GPU for persistent residency
    !$omp target enter data map(to: u, v, h, h0, uh, vh, eta, ubt, vbt)
    !$omp target enter data map(alloc: CAu, CAv, up, vp, diffu, diffv, htmp)
    !$omp target enter data map(alloc: ubt_av, vbt_av, eta_av)
    ! Update forces/visc data that was filled after their init (init did map(to:) with zeros)
    !$omp target update to(forces%taux, forces%tauy)
    !$omp target update to(visc%Ray_u, visc%Ray_v)
    call profiler_stop("Initializatoin")

    call system_clock(init_clock_end)
    t_init = real(init_clock_end - init_clock_start, dp) / real(init_clock_rate, dp)

    ! Memory usage report
    ! Driver state arrays: 13 3D + 6 2D
    state_bytes = int(G%ied - G%isd + 1, int64) * int(G%jed - G%jsd + 1, int64) &
        * (13_int64 * int(nk, int64) + 6_int64) * 8_int64
    total_bytes = G%nbytes + cont_CS%nbytes + BT_cont%nbytes + cor_CS%nbytes &
        + bt_CS%nbytes + visc_CS%nbytes + hvisc_CS%nbytes + forces%nbytes + visc%nbytes &
        + state_bytes
    print '(A)', ''
    print '(A)', 'GPU Memory Usage:'
    print '(A, F10.2, A)', '  state arrays:     ', real(state_bytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  ocean_grid_type:  ', real(G%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  continuity_CS:    ', real(cont_CS%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  BT_cont_type:     ', real(BT_cont%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  coriolis_CS:      ', real(cor_CS%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  barotropic_CS:    ', real(bt_CS%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  vert_visc_CS:     ', real(visc_CS%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  hor_visc_CS:      ', real(hvisc_CS%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  mech_forcing:     ', real(forces%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A, F10.2, A)', '  vertvisc_type:    ', real(visc%nbytes, dp) / (1024.0_dp**2), ' MB'
    print '(A)', '  ----------------------------------'
    print '(A, F10.2, A)', '  Total:            ', real(total_bytes, dp) / (1024.0_dp**2), ' MB'

    print '(A)', ''
    print '(A,F12.6,A)', 'Initialization time: ', t_init, ' s'
    print '(A)', ''
    print '(A)', 'Running split RK2 time-stepping...'
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

            ! Reset to initial state for timing consistency
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

            ! 1. Compute layer transports from velocities
            call profiler_start("Transports")
            call compute_transports(u, v, h, uh, vh, G, GV)
            call profiler_stop("Transports")

            ! 2. Horizontal viscosity (before Coriolis, per MOM6 step_MOM_dyn_split_RK2)
            call profiler_start("HorVisc")
            call system_clock(clock_start, clock_rate)
            call hor_visc(u, v, h, diffu, diffv, G, GV, hvisc_CS, uh, vh)
            call system_clock(clock_end)
            t_hor_visc = t_hor_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("HorVisc")

            ! 3. Coriolis and momentum advection
            call profiler_start("Coriolis")
            call system_clock(clock_start, clock_rate)
            call CorAdCalc(u, v, h, uh, vh, CAu, CAv, G, GV, cor_CS)
            call system_clock(clock_end)
            t_coriolis = t_coriolis + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Coriolis")

            ! 4. Predictor velocity update (add Coriolis + horizontal viscosity accelerations)
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

            ! 5. Apply vertical viscosity to predictor velocities
            call profiler_start("VertVisc")
            call system_clock(clock_start, clock_rate)
            call vert_visc_coef_apply(up, vp, h, dt, visc_CS, G, GV, forces, visc)
            call system_clock(clock_end)
            t_vert_visc = t_vert_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("VertVisc")

            ! 6. Barotropic predictor step
            call profiler_start("Barotropic")
            call system_clock(clock_start, clock_rate)
            call btstep(eta, ubt, vbt, ubt_av, vbt_av, eta_av, G, bt_CS)
            call system_clock(clock_end)
            t_barotropic = t_barotropic + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Barotropic")

            ! 7. Continuity (update thicknesses) — fully GPU, no memcpys
            call profiler_start("Continuity")
            call system_clock(clock_start, clock_rate)
            call continuity_PPM(up, h0, h, uh, dt, G, GV, cont_CS, por_face_areaU, uhbt, &
                                visc_rem_u, u_cor, BT_cont, du_cor)
            call system_clock(clock_end)
            t_continuity = t_continuity + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Continuity")

            ! Post predictor diagnostics (skip transfers if disabled)
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

            ! 8. Recompute transports with updated thickness
            call profiler_start("Transports")
            call compute_transports(up, vp, h, uh, vh, G, GV)
            call profiler_stop("Transports")

            ! 9. Horizontal viscosity with updated state
            call profiler_start("HorVisc")
            call system_clock(clock_start, clock_rate)
            call hor_visc(up, vp, h, diffu, diffv, G, GV, hvisc_CS, uh, vh)
            call system_clock(clock_end)
            t_hor_visc = t_hor_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("HorVisc")

            ! 10. Coriolis with updated state
            call profiler_start("Coriolis")
            call system_clock(clock_start, clock_rate)
            call CorAdCalc(up, vp, h, uh, vh, CAu, CAv, G, GV, cor_CS)
            call system_clock(clock_end)
            t_coriolis = t_coriolis + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Coriolis")

            ! 11. Final velocity update (RK2 average with Coriolis + horizontal viscosity)
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

            ! 12. Apply vertical viscosity to corrector velocities
            call profiler_start("VertVisc")
            call system_clock(clock_start, clock_rate)
            call vert_visc_coef_apply(u, v, h, 0.5_dp*dt, visc_CS, G, GV, forces, visc)
            call system_clock(clock_end)
            t_vert_visc = t_vert_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("VertVisc")

            ! 13. Barotropic corrector
            call profiler_start("Barotropic")
            call system_clock(clock_start, clock_rate)
            call btstep(eta_av, ubt_av, vbt_av, ubt_av, vbt_av, eta, G, bt_CS)
            call system_clock(clock_end)
            t_barotropic = t_barotropic + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Barotropic")

            ! 14. Final continuity
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
            call continuity_PPM(u, htmp, h, uh, 0.5_dp*dt, G, GV, cont_CS, por_face_areaU, uhbt, visc_rem_u, u_cor, BT_cont, du_cor)
            call system_clock(clock_end)
            t_continuity = t_continuity + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Continuity")

            ! Post end-of-step diagnostics (only on last iteration, skip if disabled)
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

    print '(A)', '=================================================='
    print '(A)', 'Timing Results'
    print '(A)', '=================================================='
    print '(A,F12.6)', 'Init (alloc+GPU xfer): ', t_init
    print '(A,F12.6)', 'Compute (wall clock):  ', t_compute
    print '(A,F12.6)', 'Compute (sum of parts):', t_total
    print '(A)', '--------------------------------------------------'
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
    if (t_diag > 0.0_dp) then
        print '(A,F12.6,A,F5.1,A)', '  Diagnostics:         ', t_diag, &
            '  (', 100.0_dp*t_diag/t_total, '%)'
    end if
    print '(A)', '--------------------------------------------------'
    print '(A,F12.6)', 'Time per RK2 step:     ', t_compute/real(niter, dp)
    print '(A)', '=================================================='
    call diag_report_timing(diag_CS)

    ! Bring final state back to host for verification
    !$omp target update from(u, v, h, eta)

    ! Verify results
    call verify_state(h0, h, u, v, eta, G, GV)

    ! Cleanup - each _end() call deallocates device memory
    call profiler_start("Finalize_continuity")
    call continuity_end(cont_CS, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
    call profiler_stop("Finalize_continuity")

    call profiler_start("Finalize_coriolis")
    call coriolis_end(cor_CS)
    call profiler_stop("Finalize_coriolis")

    call profiler_start("Finalize_barotropic")
    call barotropic_end(bt_CS)
    call profiler_stop("Finalize_barotropic")

    call profiler_start("Finalize_vert_visc")
    call end_mech_forcing(forces)
    call end_vertvisc_visc(visc)
    call vert_visc_end(visc_CS)
    call profiler_stop("Finalize_vert_visc")

    call profiler_start("Finalize_hor_visc")
    call hor_visc_end(hvisc_CS)
    call profiler_stop("Finalize_hor_visc")

    call profiler_start("Finalize_diag")
    call diag_end(diag_CS)
    call profiler_stop("Finalize_diag")

    call profiler_start("Finalize_grid")
    call end_ocean_grid(G)
    call profiler_stop("Finalize_grid")

    call profiler_stop("Total")

    ! Print profiler report after all regions are recorded
    call profiler_report("RK2 OMP Driver", root_region="Total")
    call profiler_end()

    ! Release GPU memory for state arrays
    !$omp target exit data map(delete: u, v, h, h0, uh, vh, eta, ubt, vbt)
    !$omp target exit data map(delete: CAu, CAv, up, vp, diffu, diffv, htmp)
    !$omp target exit data map(delete: ubt_av, vbt_av, eta_av)

    deallocate (u, v, h, h0, uh, vh, CAu, CAv, up, vp)
    deallocate (diffu, diffv)
    deallocate (eta, ubt, vbt, ubt_av, vbt_av, eta_av)

contains

    subroutine initialize_state(u, v, h, h0, eta, ubt, vbt, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: u, v, h, h0
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: eta, ubt, vbt

        integer :: i, j, k
        real(dp) :: total_depth

        total_depth = 4000.0_dp

        do k=1,GV%ke
          do j=G%jsd,G%jed
            do i=G%isd,G%ied
            ! Layer thickness with baroclinic structure
            h0(i, j, k) = total_depth/real(GV%ke, dp) + &
                          10.0_dp*sin(real(i - 1, dp)/real(G%ni, dp)*3.14159_dp)* &
                          cos(real(j - 1, dp)/real(G%nj, dp)*3.14159_dp)* &
                          exp(-real(k, dp)/20.0_dp)
            h(i, j, k) = h0(i, j, k)

            ! Baroclinic velocities (surface-intensified)
            u(i, j, k) = 0.1_dp*sin(real(j - 1, dp)/real(G%nj, dp)*3.14159_dp*2.0_dp)* &
                         exp(-real(k, dp)/30.0_dp)
            v(i, j, k) = 0.1_dp*cos(real(i - 1, dp)/real(G%ni, dp)*3.14159_dp*2.0_dp)* &
                         exp(-real(k, dp)/30.0_dp)
            end do
          end do
        end do

        ! Sea surface height and barotropic velocities
        do j=G%jsd,G%jed
          do i=G%isd,G%ied
            eta(i, j) = 0.5_dp*sin(real(i - 1, dp)/real(G%ni, dp)*3.14159_dp*2.0_dp)* &
                        cos(real(j - 1, dp)/real(G%nj, dp)*3.14159_dp*2.0_dp)
            ubt(i, j) = 0.05_dp*sin(real(j - 1, dp)/real(G%nj, dp)*3.14159_dp)
            vbt(i, j) = 0.05_dp*cos(real(i - 1, dp)/real(G%ni, dp)*3.14159_dp)
          end do
        end do

    end subroutine initialize_state

    subroutine initialize_forces_visc(forces, visc, G, GV)
        type(mech_forcing_type), intent(inout) :: forces
        type(vertvisc_type), intent(inout) :: visc
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV

        integer :: i, j, k

        ! Sinusoidal zonal wind stress (~0.1 Pa)
        do j=G%jsd,G%jed
          do i=G%isd,G%ied
            forces%taux(i, j) = 0.1_dp * sin(real(j - 1, dp) / real(G%nj, dp) * 3.14159_dp)
            forces%tauy(i, j) = 0.0_dp
          end do
        end do

        ! Rayleigh drag in bottom layer only
        if (visc%has_Rayleigh) then
            do k=1,GV%ke
              do j=G%jsd,G%jed
                do i=G%isd,G%ied
                visc%Ray_u(i, j, k) = 0.0_dp
                visc%Ray_v(i, j, k) = 0.0_dp
                end do
              end do
            end do
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
        print '(A)', '--------------------------------------------------'
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
        real(dp), intent(in) :: dt       ! Baroclinic timestep [s]
        integer, intent(in) :: nsteps    ! Number of barotropic substeps
        real(dp), intent(in) :: dx_km    ! Grid spacing [km]
        real(dp), intent(in) :: depth    ! Ocean depth [m]

        real(dp) :: dtbt, dx_m, c_grav, cfl, cfl_2d
        integer :: min_nsteps

        dx_m = dx_km*1000.0_dp
        dtbt = dt/real(nsteps, dp)
        c_grav = sqrt(G_EARTH*depth)  ! Gravity wave speed [m/s]
        cfl = c_grav*dtbt/dx_m
        cfl_2d = cfl*sqrt(2.0_dp)     ! 2D diagonal CFL

        ! Minimum substeps for CFL < 0.5 (with safety margin)
        min_nsteps = ceiling(dt*c_grav/dx_m*2.5_dp)

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

        ! Compute KE = 0.5 * (u^2 + v^2) at h-points (simplified averaging)
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

        ! Sum layer thicknesses
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

end program rk2_omp_driver
