!> MPI-aware RK2 driver using CUDA C kernels for all physics modules.
!!
!! Based on rk2_mpi_omp_driver.F90 (MPI structure) + rk2_cuda_c_driver.F90 (CUDA C physics).
!! OpenMP target offloading handles data management (target enter/exit data)
!! and simple inline kernels (transports, velocity updates).
!! All 5 physics modules use CUDA C kernels via iso_c_binding.
!! Halo exchange uses mom6_mpi_halo_omp (OpenMP target compatible).
!!
!! Usage: mpirun -np N rk2_mpi_cuda_c_driver ni nj nk niter bt_nsteps [npes_x npes_y]
!!        npes_x * npes_y must equal N
!!        If npes_x/npes_y omitted, auto-decomposed via MPI_Dims_create
!!
program rk2_mpi_cuda_c_driver
    use mpi
    use iso_fortran_env, only: dp => real64, int64
    use iso_c_binding, only: c_ptr, c_loc, c_null_ptr
    use omp_lib, only: omp_set_default_device
    use mom6_types, only: ocean_grid_type, verticalGrid_type, &
                          end_ocean_grid, G_EARTH, HALO_WIDTH, PI, &
                          mech_forcing_type, vertvisc_type, &
                          init_mech_forcing, end_mech_forcing, &
                          init_vertvisc_visc, end_vertvisc_visc
    use mom6_mpi_domain, only: mpi_domain_type, mpi_domain_init, &
                               mpi_domain_end, init_ocean_grid_mpi
    use mom6_mpi_halo_omp, only: halo_exchange_3d, halo_exchange_2d, halo_cleanup
    ! CUDA C physics modules (portable — no cudafor dependency)
    use mom6_continuity_cuda_c, only: continuity_CS_cuda_c, continuity_init_cuda_c, &
                                       continuity_PPM_cuda_c, continuity_end_cuda_c, &
                                       BT_cont_type_cuda_c, alloc_BT_cont_type_cuda_c, &
                                       dealloc_BT_cont_type_cuda_c
    use mom6_coriolis_cuda_c, only: coriolis_CS_cuda_c, coriolis_init_cuda_c, &
                                     CorAdCalc_cuda_c, coriolis_end_cuda_c, &
                                     SADOURNY75_ENERGY_CUDA_C
    use mom6_barotropic_cuda_c, only: barotropic_CS_cuda_c, barotropic_init_cuda_c, &
                                       barotropic_end_cuda_c, &
                                       btstep_cuda_c_init_state, btstep_cuda_c_do_step, &
                                       btstep_cuda_c_get_output, &
                                       btstep_cuda_c_export_state, btstep_cuda_c_import_state
    use mom6_vert_visc_cuda_c, only: vert_visc_CS_cuda_c, vert_visc_init_cuda_c, &
                                      vert_visc_cra_cuda_c, vert_visc_end_cuda_c
    use mom6_hor_visc_cuda_c, only: hor_visc_CS_cuda_c, hor_visc_init_cuda_c, &
                                     hor_visc_cuda_c, hor_visc_end_cuda_c
    use mom6_cuda_c_common, only: cuda_set_device_c
    use mom6_profiler, only: profiler_init, profiler_end, profiler_start, profiler_stop, &
                             profiler_report
    implicit none

    ! Grid structures
    type(ocean_grid_type) :: G
    type(verticalGrid_type) :: GV
    type(mpi_domain_type) :: MD

    ! CUDA C control structures
    type(continuity_CS_cuda_c) :: cont_CS
    type(coriolis_CS_cuda_c) :: cor_CS
    type(barotropic_CS_cuda_c) :: bt_CS
    type(vert_visc_CS_cuda_c) :: visc_CS
    type(hor_visc_CS_cuda_c) :: hvisc_CS
    type(BT_cont_type_cuda_c) :: BT_cont_cc
    type(mech_forcing_type) :: forces
    type(vertvisc_type) :: visc

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

    ! Scratch arrays for barotropic substep halo exchange
    real(dp), allocatable, target :: bt_ubt(:, :), bt_vbt(:, :), bt_eta(:, :)

    ! Timing
    real(dp) :: dt, t_total
    real(dp) :: t_coriolis, t_barotropic, t_continuity, t_vert_visc, t_hor_visc
    real(dp) :: t_init, t_compute, t_halo
    integer :: clock_start, clock_end, clock_rate
    integer :: init_clock_start, init_clock_end, init_clock_rate
    integer :: compute_clock_start, compute_clock_end, compute_clock_rate
    integer :: halo_clock_start, halo_clock_end, halo_clock_rate

    ! Parameters
    integer :: ni, nj, nk, niter, bt_nsteps
    integer :: npes_x, npes_y
    integer :: iter, i, j, k, bt_n
    integer :: ierr, nprocs, dims(2)
    integer :: local_rank
    integer :: node_comm, node_rank, istat
    character(len=32) :: arg

    ! MPI initialization
    call MPI_Init(ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, local_rank, ierr)

    ! Get node-local rank for GPU device assignment
    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                             MPI_INFO_NULL, node_comm, ierr)
    call MPI_Comm_rank(node_comm, node_rank, ierr)
    !$ call omp_set_default_device(node_rank)
    istat = cuda_set_device_c(node_rank)  ! Pin CUDA runtime to same GPU
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
        print '(A)', '=================================================='
        print '(A)', 'MOM6 Split RK2 MPI Driver (CUDA C kernels)'
        print '(A)', '=================================================='
        print '(A,I5,A,I5,A,I4)', 'Global grid: ', ni, ' x ', nj, ' x ', nk
        print '(A,I4,A,I4,A,I4)', 'PE layout: ', npes_x, ' x ', npes_y, ' = ', nprocs
        print '(A,I4)', 'RK2 iterations: ', niter
        print '(A,I4)', 'Barotropic substeps: ', bt_nsteps
        call check_bt_cfl(dt, bt_nsteps, 10.0_dp, 4000.0_dp)
        print '(A)', '=================================================='
    end if

    ! Initialize grid and control structures
    call profiler_start("Initialization")
    call system_clock(init_clock_start, init_clock_rate)

    call init_ocean_grid_mpi(G, MD, GV, nk, 10.0_dp, 45.0_dp)

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

    ! Forces and visc (OMP target for taux/tauy data management)
    call init_mech_forcing(forces, G)
    call init_vertvisc_visc(visc, G, GV, use_rayleigh=.true.)

    ! Allocate 3D arrays
    allocate(u(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(v(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(h0(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(htmp(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(uh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(vh(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAu(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(CAv(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(up(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(vp(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(diffu(G%isd:G%ied, G%jsd:G%jed, nk))
    allocate(diffv(G%isd:G%ied, G%jsd:G%jed, nk))

    ! Allocate 2D arrays
    allocate(eta(G%isd:G%ied, G%jsd:G%jed))
    allocate(ubt(G%isd:G%ied, G%jsd:G%jed))
    allocate(vbt(G%isd:G%ied, G%jsd:G%jed))
    allocate(uhbt(G%isd:G%ied, G%jsd:G%jed))
    allocate(ubt_av(G%isd:G%ied, G%jsd:G%jed))
    allocate(vbt_av(G%isd:G%ied, G%jsd:G%jed))
    allocate(eta_av(G%isd:G%ied, G%jsd:G%jed))

    ! Scratch for barotropic substep halo exchange
    allocate(bt_ubt(G%isd:G%ied, G%jsd:G%jed))
    allocate(bt_vbt(G%isd:G%ied, G%jsd:G%jed))
    allocate(bt_eta(G%isd:G%ied, G%jsd:G%jed))

    ! Initialize state using global coordinates
    call initialize_state_mpi(u, v, h, h0, eta, ubt, vbt, G, GV, MD)
    call initialize_forces_visc_mpi(forces, visc, G, GV, MD)
    uhbt = 0.0_dp

    ! Copy all state arrays to GPU
    !$omp target enter data map(to: u, v, h, h0, uh, vh, eta, ubt, vbt, uhbt)
    !$omp target enter data map(alloc: CAu, CAv, up, vp, diffu, diffv, htmp)
    !$omp target enter data map(alloc: ubt_av, vbt_av, eta_av)
    !$omp target enter data map(alloc: bt_ubt, bt_vbt, bt_eta)
    !$omp target update to(forces%taux, forces%tauy)
    !$omp target update to(visc%Ray_u, visc%Ray_v)

    ! Initial halo exchange to fill halos before first iteration
    call halo_exchange_3d(u, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d(v, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d(h, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_3d(h0, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
    call halo_exchange_2d(eta, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
    call halo_exchange_2d(ubt, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
    call halo_exchange_2d(vbt, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)

    call profiler_stop("Initialization")
    call system_clock(init_clock_end)
    t_init = real(init_clock_end - init_clock_start, dp) / real(init_clock_rate, dp)

    if (MD%rank == 0) then
        print '(A)', ''
        print '(A,I5,A,I5)', 'Local compute domain: ', MD%ni_local, ' x ', MD%nj_local
        print '(A,F12.6,A)', 'Initialization time: ', t_init, ' s'
        print '(A)', ''
        print '(A)', 'Running split RK2 time-stepping (MPI + CUDA C)...'
        print '(A)', ''
    end if

    t_total = 0.0_dp
    t_coriolis = 0.0_dp
    t_barotropic = 0.0_dp
    t_continuity = 0.0_dp
    t_vert_visc = 0.0_dp
    t_hor_visc = 0.0_dp
    t_halo = 0.0_dp

    call system_clock(compute_clock_start, compute_clock_rate)
    block
        integer :: iter_clock_start, iter_clock_end, iter_clock_rate
        call profiler_start("RK2_step", nvtx_only=.true.)
        do iter = 1, niter
            call system_clock(iter_clock_start, iter_clock_rate)

            ! Reset to initial state for timing consistency
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nk
              do j = G%jsd, G%jed
                do i = G%isd, G%ied
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

            ! Halo exchange: uh, vh needed by Coriolis
            call system_clock(halo_clock_start, halo_clock_rate)
            call halo_exchange_3d(uh, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call halo_exchange_3d(vh, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call system_clock(halo_clock_end)
            t_halo = t_halo + real(halo_clock_end - halo_clock_start, dp) / real(halo_clock_rate, dp)

            ! 2. Horizontal viscosity — CUDA C
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

            ! 3. Coriolis and momentum advection — CUDA C
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

            ! 4. Predictor velocity update
            call profiler_start("VelUpdate_pred")
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nk
              do j = G%jsc, G%jec
                do i = G%isc, G%iec - 1
                  up(i, j, k) = u(i, j, k) + dt * (CAu(i, j, k) + diffu(i, j, k))
                end do
              end do
            end do
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nk
              do j = G%jsc, G%jec - 1
                do i = G%isc, G%iec
                  vp(i, j, k) = v(i, j, k) + dt * (CAv(i, j, k) + diffv(i, j, k))
                end do
              end do
            end do
            call profiler_stop("VelUpdate_pred")

            ! 5. Vertical viscosity on predictor velocities — CUDA C
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

            ! 6. Barotropic predictor step (split API with halo exchanges every substep)
            call profiler_start("Barotropic")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(eta, ubt, vbt)
            call btstep_cuda_c_init_state(bt_CS, c_loc(eta(G%isd, G%jsd)), &
                                           c_loc(ubt(G%isd, G%jsd)), &
                                           c_loc(vbt(G%isd, G%jsd)))
            !$omp end target data
            do bt_n = 1, bt_CS%nstep
                call btstep_cuda_c_do_step(bt_CS, bt_n)
                if (bt_n < bt_CS%nstep) then
                    ! Export internal state to OMP-managed arrays for halo exchange
                    !$omp target data use_device_addr(bt_ubt, bt_vbt, bt_eta)
                    call btstep_cuda_c_export_state(bt_CS, c_loc(bt_ubt(G%isd, G%jsd)), &
                                                    c_loc(bt_vbt(G%isd, G%jsd)), &
                                                    c_loc(bt_eta(G%isd, G%jsd)))
                    !$omp end target data
                    call halo_exchange_2d(bt_ubt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                    call halo_exchange_2d(bt_vbt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                    call halo_exchange_2d(bt_eta, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                    ! Import back after halo exchange
                    !$omp target data use_device_addr(bt_ubt, bt_vbt, bt_eta)
                    call btstep_cuda_c_import_state(bt_CS, c_loc(bt_ubt(G%isd, G%jsd)), &
                                                    c_loc(bt_vbt(G%isd, G%jsd)), &
                                                    c_loc(bt_eta(G%isd, G%jsd)))
                    !$omp end target data
                end if
            end do
            !$omp target data use_device_addr(ubt_av, vbt_av, eta_av)
            call btstep_cuda_c_get_output(bt_CS, c_loc(ubt_av(G%isd, G%jsd)), &
                                           c_loc(vbt_av(G%isd, G%jsd)), &
                                           c_loc(eta_av(G%isd, G%jsd)))
            !$omp end target data
            call system_clock(clock_end)
            t_barotropic = t_barotropic + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Barotropic")

            ! Halo exchange: up, vp needed by corrector continuity
            call system_clock(halo_clock_start, halo_clock_rate)
            call halo_exchange_3d(up, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call halo_exchange_3d(vp, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call system_clock(halo_clock_end)
            t_halo = t_halo + real(halo_clock_end - halo_clock_start, dp) / real(halo_clock_rate, dp)

            ! 7. Continuity (update thicknesses) — CUDA C
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

            ! Halo exchange after predictor: h needed by corrector
            call system_clock(halo_clock_start, halo_clock_rate)
            call halo_exchange_3d(h, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call system_clock(halo_clock_end)
            t_halo = t_halo + real(halo_clock_end - halo_clock_start, dp) / real(halo_clock_rate, dp)

            !=========================================================================
            ! CORRECTOR PHASE
            !=========================================================================

            ! 8. Recompute transports with updated thickness
            call profiler_start("Transports")
            call compute_transports(up, vp, h, uh, vh, G, GV)
            call profiler_stop("Transports")

            call system_clock(halo_clock_start, halo_clock_rate)
            call halo_exchange_3d(uh, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call halo_exchange_3d(vh, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call system_clock(halo_clock_end)
            t_halo = t_halo + real(halo_clock_end - halo_clock_start, dp) / real(halo_clock_rate, dp)

            ! 9. Horizontal viscosity with updated state — CUDA C
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

            ! 10. Coriolis with updated state — CUDA C
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

            ! 11. Final velocity update
            call profiler_start("VelUpdate_corr")
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nk
              do j = G%jsc, G%jec
                do i = G%isc, G%iec - 1
                  u(i, j, k) = u(i, j, k) + 0.5_dp * dt * (CAu(i, j, k) + diffu(i, j, k))
                end do
              end do
            end do
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nk
              do j = G%jsc, G%jec - 1
                do i = G%isc, G%iec
                  v(i, j, k) = v(i, j, k) + 0.5_dp * dt * (CAv(i, j, k) + diffv(i, j, k))
                end do
              end do
            end do
            call profiler_stop("VelUpdate_corr")

            ! 12. Vertical viscosity on corrector velocities — CUDA C
            call profiler_start("VertVisc")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(u, v, h, forces%taux, forces%tauy)
            call vert_visc_cra_cuda_c( &
                c_loc(u(G%isd, G%jsd, 1)), c_loc(v(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), 0.5_dp * dt, visc_CS, &
                c_loc(forces%taux(G%isd, G%jsd)), c_loc(forces%tauy(G%isd, G%jsd)))
            !$omp end target data
            call system_clock(clock_end)
            t_vert_visc = t_vert_visc + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("VertVisc")

            ! 13. Barotropic corrector (split API with halo exchanges every substep)
            call profiler_start("Barotropic")
            call system_clock(clock_start, clock_rate)
            !$omp target data use_device_addr(eta_av, ubt_av, vbt_av)
            call btstep_cuda_c_init_state(bt_CS, c_loc(eta_av(G%isd, G%jsd)), &
                                           c_loc(ubt_av(G%isd, G%jsd)), &
                                           c_loc(vbt_av(G%isd, G%jsd)))
            !$omp end target data
            do bt_n = 1, bt_CS%nstep
                call btstep_cuda_c_do_step(bt_CS, bt_n)
                if (bt_n < bt_CS%nstep) then
                    !$omp target data use_device_addr(bt_ubt, bt_vbt, bt_eta)
                    call btstep_cuda_c_export_state(bt_CS, c_loc(bt_ubt(G%isd, G%jsd)), &
                                                    c_loc(bt_vbt(G%isd, G%jsd)), &
                                                    c_loc(bt_eta(G%isd, G%jsd)))
                    !$omp end target data
                    call halo_exchange_2d(bt_ubt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                    call halo_exchange_2d(bt_vbt, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                    call halo_exchange_2d(bt_eta, G%isd, G%ied, G%jsd, G%jed, MD, 1)
                    !$omp target data use_device_addr(bt_ubt, bt_vbt, bt_eta)
                    call btstep_cuda_c_import_state(bt_CS, c_loc(bt_ubt(G%isd, G%jsd)), &
                                                    c_loc(bt_vbt(G%isd, G%jsd)), &
                                                    c_loc(bt_eta(G%isd, G%jsd)))
                    !$omp end target data
                end if
            end do
            !$omp target data use_device_addr(ubt_av, vbt_av, eta)
            call btstep_cuda_c_get_output(bt_CS, c_loc(ubt_av(G%isd, G%jsd)), &
                                           c_loc(vbt_av(G%isd, G%jsd)), &
                                           c_loc(eta(G%isd, G%jsd)))
            !$omp end target data
            call system_clock(clock_end)
            t_barotropic = t_barotropic + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Barotropic")

            ! 14. Final continuity — CUDA C
            call profiler_start("Continuity")
            call system_clock(clock_start, clock_rate)
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nk
              do j = G%jsd, G%jed
                do i = G%isd, G%ied
                  htmp(i, j, k) = h(i, j, k)
                end do
              end do
            end do
            !$omp target data use_device_addr(u, htmp, h, uh, uhbt)
            call continuity_PPM_cuda_c( &
                c_loc(u(G%isd, G%jsd, 1)), c_loc(htmp(G%isd, G%jsd, 1)), &
                c_loc(h(G%isd, G%jsd, 1)), c_loc(uh(G%isd, G%jsd, 1)), &
                0.5_dp * dt, cont_CS, &
                c_loc(uhbt(G%isd, G%jsd)), BT_cont_cc)
            !$omp end target data
            call system_clock(clock_end)
            t_continuity = t_continuity + real(clock_end - clock_start, dp) / real(clock_rate, dp)
            call profiler_stop("Continuity")

            ! Halo exchange after corrector for next iteration
            call system_clock(halo_clock_start, halo_clock_rate)
            call halo_exchange_3d(u, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call halo_exchange_3d(v, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call halo_exchange_3d(h, G%isd, G%ied, G%jsd, G%jed, nk, MD, HALO_WIDTH)
            call halo_exchange_2d(eta, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
            call halo_exchange_2d(ubt, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
            call halo_exchange_2d(vbt, G%isd, G%ied, G%jsd, G%jed, MD, HALO_WIDTH)
            call system_clock(halo_clock_end)
            t_halo = t_halo + real(halo_clock_end - halo_clock_start, dp) / real(halo_clock_rate, dp)

            call system_clock(iter_clock_end)
            if (MD%rank == 0) then
                print '(A,I4,A,F10.6,A)', '  RK2 iteration ', iter, ':  ', &
                    real(iter_clock_end - iter_clock_start, dp) / real(iter_clock_rate, dp), ' s'
            end if
        end do
    end block
    call profiler_stop("RK2_step")
    call system_clock(compute_clock_end)
    t_compute = real(compute_clock_end - compute_clock_start, dp) / real(compute_clock_rate, dp)

    t_total = t_coriolis + t_barotropic + t_continuity + t_vert_visc + t_hor_visc

    if (MD%rank == 0) then
        print '(A)', '=================================================='
        print '(A)', 'Timing Results'
        print '(A)', '=================================================='
        print '(A,F12.6)', 'Compute (wall clock):  ', t_compute
        print '(A,F12.6)', 'Compute (sum of parts):', t_total
        print '(A,F12.6)', 'Halo exchange:         ', t_halo
        print '(A)', '--------------------------------------------------'
        print '(A,F12.6,A,F5.1,A)', '  Coriolis (CUDA C):   ', t_coriolis, &
            '  (', 100.0_dp * t_coriolis / t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Hor viscosity:       ', t_hor_visc, &
            '  (', 100.0_dp * t_hor_visc / t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Vert viscosity:      ', t_vert_visc, &
            '  (', 100.0_dp * t_vert_visc / t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Barotropic:          ', t_barotropic, &
            '  (', 100.0_dp * t_barotropic / t_total, '%)'
        print '(A,F12.6,A,F5.1,A)', '  Continuity:          ', t_continuity, &
            '  (', 100.0_dp * t_continuity / t_total, '%)'
        print '(A)', '--------------------------------------------------'
        print '(A,F12.6)', 'Time per RK2 step:     ', t_compute / real(niter, dp)
        print '(A)', '=================================================='
    end if

    ! Bring final state back to host for verification
    !$omp target update from(u, v, h, eta)

    ! MPI-aware verification (reduce across all PEs)
    call verify_state_mpi(h0, h, u, v, eta, G, GV, MD)

    ! Cleanup
    call continuity_end_cuda_c(cont_CS)
    call dealloc_BT_cont_type_cuda_c(BT_cont_cc)
    call coriolis_end_cuda_c(cor_CS)
    call barotropic_end_cuda_c(bt_CS)
    call end_mech_forcing(forces)
    call end_vertvisc_visc(visc)
    call vert_visc_end_cuda_c(visc_CS)
    call hor_visc_end_cuda_c(hvisc_CS)
    call end_ocean_grid(G)

    call profiler_stop("Total")
    if (MD%rank == 0) then
        call profiler_report("RK2 MPI CUDA C Driver", root_region="Total")
    end if
    call profiler_end()

    ! Release GPU memory for state arrays
    !$omp target exit data map(delete: u, v, h, h0, uh, vh, eta, ubt, vbt, uhbt)
    !$omp target exit data map(delete: CAu, CAv, up, vp, diffu, diffv, htmp)
    !$omp target exit data map(delete: ubt_av, vbt_av, eta_av)
    !$omp target exit data map(delete: bt_ubt, bt_vbt, bt_eta)

    deallocate(u, v, h, h0, uh, vh, CAu, CAv, up, vp)
    deallocate(diffu, diffv, htmp)
    deallocate(eta, ubt, vbt, uhbt, ubt_av, vbt_av, eta_av)
    deallocate(bt_ubt, bt_vbt, bt_eta)

    call halo_cleanup()
    call mpi_domain_end(MD)
    call MPI_Finalize(ierr)

contains

    !> Initialize state using global coordinates for MPI consistency
    subroutine initialize_state_mpi(u, v, h, h0, eta, ubt, vbt, G, GV, MD)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        type(mpi_domain_type), intent(in) :: MD
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: u, v, h, h0
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: eta, ubt, vbt

        integer :: i, j, k, i_global, j_global, halo
        real(dp) :: total_depth

        total_depth = 4000.0_dp
        halo = MD%halo

        !$omp parallel do collapse(3) private(i,j,k,i_global,j_global)
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

        !$omp parallel do collapse(2) private(i,j,i_global,j_global)
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

    !> Initialize forces and visc using global coordinates
    subroutine initialize_forces_visc_mpi(forces, visc, G, GV, MD)
        type(mech_forcing_type), intent(inout) :: forces
        type(vertvisc_type), intent(inout) :: visc
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        type(mpi_domain_type), intent(in) :: MD

        integer :: i, j, k, j_global

        !$omp parallel do collapse(2) private(i,j,j_global)
        do j = G%jsd, G%jed
          do i = G%isd, G%ied
            j_global = j + MD%j_offset
            forces%taux(i, j) = 0.1_dp * sin(real(j_global - 1, dp) / real(MD%nj_global, dp) * PI)
            forces%tauy(i, j) = 0.0_dp
          end do
        end do

        if (visc%has_Rayleigh) then
            !$omp parallel do collapse(3) private(i,j,k)
            do k = 1, GV%ke
              do j = G%jsd, G%jed
                do i = G%isd, G%ied
                  visc%Ray_u(i, j, k) = 0.0_dp
                  visc%Ray_v(i, j, k) = 0.0_dp
                end do
              end do
            end do
            !$omp parallel do collapse(2) private(i,j)
            do j = G%jsd, G%jed
              do i = G%isd, G%ied
                visc%Ray_u(i, j, GV%ke) = 1.0e-4_dp
                visc%Ray_v(i, j, GV%ke) = 1.0e-4_dp
              end do
            end do
        end if
    end subroutine initialize_forces_visc_mpi

    subroutine compute_transports(u, v, h, uh, vh, G, GV)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh, vh

        integer :: i, j, k

        !$omp target teams distribute parallel do collapse(3)
        do k = 1, GV%ke
          do j = G%jsd, G%jed
            do i = G%isd, G%ied
              uh(i, j, k) = u(i, j, k) * 0.5_dp * (h(i, j, k) + h(min(i + 1, G%ied), j, k)) * G%dyCu(i, j)
              vh(i, j, k) = v(i, j, k) * 0.5_dp * (h(i, j, k) + h(i, min(j + 1, G%jed), k)) * G%dxCv(i, j)
            end do
          end do
        end do
    end subroutine compute_transports

    !> MPI-aware state verification using MPI_Allreduce
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

        ! Reduce across all PEs
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

end program rk2_mpi_cuda_c_driver
