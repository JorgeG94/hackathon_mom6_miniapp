!> Unified RK2 driver for MOM6 miniapps
!!
!! This driver orchestrates all solvers (continuity, Coriolis, barotropic, vert_visc)
!! in a simplified split RK2 time-stepping scheme similar to MOM6's step_MOM_dyn_split_RK2.
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
program rk2_driver
   use omp_lib
   use iso_fortran_env, only: dp => real64
   use mom6_types, only: ocean_grid_type, verticalGrid_type, init_ocean_grid, &
                         init_verticalGrid, end_ocean_grid, G_EARTH
   use mom6_continuity
   use mom6_coriolis
   use mom6_barotropic
   use mom6_vert_visc
   use mom6_hor_visc
   use mom6_diag
   use mom6_profiler
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

   ! Diagnostic IDs
   integer :: id_KE, id_diffu_sum, id_diffv_sum, id_mass

   ! 3D state variables
   real(dp), allocatable :: u(:, :, :), v(:, :, :)       ! Velocities
   real(dp), allocatable :: h(:, :, :), h0(:, :, :)      ! Layer thickness (current, initial)
   real(dp), allocatable :: uh(:, :, :), vh(:, :, :)     ! Layer transports
   real(dp), allocatable :: CAu(:, :, :), CAv(:, :, :)   ! Coriolis accelerations
   real(dp), allocatable :: up(:, :, :), vp(:, :, :)     ! Predictor velocities
   real(dp), allocatable :: diffu(:, :, :), diffv(:, :, :) ! Horizontal viscous accelerations

   ! 2D barotropic variables
   real(dp), allocatable :: eta(:, :)                 ! Sea surface height
   real(dp), allocatable :: ubt(:, :), vbt(:, :)       ! Barotropic velocities
   real(dp), allocatable :: ubt_av(:, :), vbt_av(:, :) ! Time-averaged BT velocities
   real(dp), allocatable :: eta_av(:, :)              ! Time-averaged SSH

   ! Timing
   real(dp) :: dt, t_start, t_end, t_total
   real(dp) :: t_coriolis, t_barotropic, t_continuity, t_vert_visc, t_hor_visc
   real(dp) :: t_diag

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
   print '(A)', 'MOM6 Split RK2 Driver'
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
   call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
   call init_verticalGrid(GV, nk)
   call continuity_init(cont_CS, G, GV)
   call coriolis_init(cor_CS, G, SADOURNY75_ENERGY)
   call barotropic_init(bt_CS, G, dt, bt_nsteps)
   call vert_visc_init(visc_CS, G, GV, Kv=1.0e-4_dp, Kv_ml=1.0e-2_dp, Hmix=50.0_dp)
   call hor_visc_init(hvisc_CS, G, GV, Kh=100.0_dp)

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

   !$omp target enter data map(alloc: u, v, h, h0, uh, vh, CAu, CAv, up, vp)
   !$omp target enter data map(alloc: diffu, diffv)
   !$omp target enter data map(alloc: eta, ubt, vbt, ubt_av, vbt_av, eta_av)

   ! Initialize state
   call initialize_state(u, v, h, h0, eta, ubt, vbt, G, GV)

   print '(A)', ''
   print '(A)', 'Running split RK2 time-stepping...'
   print '(A)', ''

   ! Initialize profiler

   t_total = 0.0_dp
   t_coriolis = 0.0_dp
   t_barotropic = 0.0_dp
   t_continuity = 0.0_dp
   t_vert_visc = 0.0_dp
   t_hor_visc = 0.0_dp
   t_diag = 0.0_dp
   block
      real(dp) :: iter_start, iter_end
      call profiler_start("RK2_step", nvtx_only=.true.)
      do iter = 1, niter
         iter_start = omp_get_wtime()

         ! Reset to initial state for timing consistency
         do concurrent(k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
            h(i, j, k) = h0(i, j, k)
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
         t_start = omp_get_wtime()
         call hor_visc(u, v, h, diffu, diffv, G, GV, hvisc_CS)
         t_end = omp_get_wtime()
         t_hor_visc = t_hor_visc + (t_end - t_start)
         call profiler_stop("HorVisc")

         ! 3. Coriolis and momentum advection
         call profiler_start("Coriolis")
         t_start = omp_get_wtime()
         call CorAdCalc(u, v, h, uh, vh, CAu, CAv, G, GV, cor_CS)
         t_end = omp_get_wtime()
         t_coriolis = t_coriolis + (t_end - t_start)
         call profiler_stop("Coriolis")

         ! 4. Predictor velocity update (add Coriolis + horizontal viscosity accelerations)
         call profiler_start("VelUpdate_pred")
         do concurrent(k=1:nk, j=G%jsc:G%jec, i=G%isc:G%iec - 1)
            up(i, j, k) = u(i, j, k) + dt*(CAu(i, j, k) + diffu(i, j, k))
         end do
         do concurrent(k=1:nk, j=G%jsc:G%jec - 1, i=G%isc:G%iec)
            vp(i, j, k) = v(i, j, k) + dt*(CAv(i, j, k) + diffv(i, j, k))
         end do
         call profiler_stop("VelUpdate_pred")

         ! 5. Apply vertical viscosity to predictor velocities
         call profiler_start("VertVisc")
         t_start = omp_get_wtime()
         call vert_visc_coef(up, vp, h, visc_CS, G, GV)
         call vert_visc_apply(up, vp, h, dt, visc_CS, G, GV)
         t_end = omp_get_wtime()
         t_vert_visc = t_vert_visc + (t_end - t_start)
         call profiler_stop("VertVisc")

         ! 6. Barotropic predictor step
         call profiler_start("Barotropic")
         t_start = omp_get_wtime()
         call btstep(eta, ubt, vbt, ubt_av, vbt_av, eta_av, G, bt_CS)
         t_end = omp_get_wtime()
         t_barotropic = t_barotropic + (t_end - t_start)
         call profiler_stop("Barotropic")

         ! 7. Continuity (update thicknesses)
         call profiler_start("Continuity")
         t_start = omp_get_wtime()
         call continuity_PPM(up, vp, h0, h, uh, vh, dt, G, GV, cont_CS, x_first=.true.)
         t_end = omp_get_wtime()
         t_continuity = t_continuity + (t_end - t_start)
         call profiler_stop("Continuity")

         ! Post predictor diagnostics (skip transfers if disabled)
         if (id_diffu_sum > 0 .or. id_diffv_sum > 0) then
            call profiler_start("Diagnostics")
            t_start = omp_get_wtime()
            !$omp target update from(diffu, diffv, h)
            call post_product_sum_u(id_diffu_sum, diffu, h, G, nk, diag_CS)
            call post_product_sum_v(id_diffv_sum, diffv, h, G, nk, diag_CS)
            t_end = omp_get_wtime()
            t_diag = t_diag + (t_end - t_start)
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
         t_start = omp_get_wtime()
         call hor_visc(up, vp, h, diffu, diffv, G, GV, hvisc_CS)
         t_end = omp_get_wtime()
         t_hor_visc = t_hor_visc + (t_end - t_start)
         call profiler_stop("HorVisc")

         ! 10. Coriolis with updated state
         call profiler_start("Coriolis")
         t_start = omp_get_wtime()
         call CorAdCalc(up, vp, h, uh, vh, CAu, CAv, G, GV, cor_CS)
         t_end = omp_get_wtime()
         t_coriolis = t_coriolis + (t_end - t_start)
         call profiler_stop("Coriolis")

         ! 11. Final velocity update (RK2 average with Coriolis + horizontal viscosity)
         call profiler_start("VelUpdate_corr")
         do concurrent(k=1:nk, j=G%jsc:G%jec, i=G%isc:G%iec - 1)
            u(i, j, k) = u(i, j, k) + 0.5_dp*dt*(CAu(i, j, k) + diffu(i, j, k))
         end do
         do concurrent(k=1:nk, j=G%jsc:G%jec - 1, i=G%isc:G%iec)
            v(i, j, k) = v(i, j, k) + 0.5_dp*dt*(CAv(i, j, k) + diffv(i, j, k))
         end do
         call profiler_stop("VelUpdate_corr")

         ! 12. Apply vertical viscosity to corrector velocities
         call profiler_start("VertVisc")
         t_start = omp_get_wtime()
         call vert_visc_coef(u, v, h, visc_CS, G, GV)
         call vert_visc_apply(u, v, h, 0.5_dp*dt, visc_CS, G, GV)
         t_end = omp_get_wtime()
         t_vert_visc = t_vert_visc + (t_end - t_start)
         call profiler_stop("VertVisc")

         ! 13. Barotropic corrector
         call profiler_start("Barotropic")
         t_start = omp_get_wtime()
         call btstep(eta_av, ubt_av, vbt_av, ubt_av, vbt_av, eta, G, bt_CS)
         t_end = omp_get_wtime()
         t_barotropic = t_barotropic + (t_end - t_start)
         call profiler_stop("Barotropic")

         ! 14. Final continuity
         call profiler_start("Continuity")
         t_start = omp_get_wtime()
         call continuity_PPM(u, v, h, h, uh, vh, 0.5_dp*dt, G, GV, cont_CS, x_first=.false.)
         t_end = omp_get_wtime()
         t_continuity = t_continuity + (t_end - t_start)
         call profiler_stop("Continuity")

         ! Post end-of-step diagnostics (only on last iteration, skip if disabled)
         if (iter == niter .and. (id_KE > 0 .or. id_mass > 0)) then
            call profiler_start("Diagnostics")
            t_start = omp_get_wtime()
            !$omp target update from(u, v, h)
            call compute_and_post_KE(id_KE, u, v, h, G, GV, diag_CS)
            call compute_and_post_mass(id_mass, h, G, GV, diag_CS)
            t_end = omp_get_wtime()
            t_diag = t_diag + (t_end - t_start)
            call profiler_stop("Diagnostics")
         end if
         iter_end = omp_get_wtime()
         print '(A,I4,A,F10.6,A)', '  RK2 iteration ', iter, ':  ', iter_end - iter_start, ' s'

      end do
   end block
   call profiler_stop("RK2_step")

   t_total = t_coriolis + t_barotropic + t_continuity + t_vert_visc + t_hor_visc + t_diag

   ! Copy results back from device and delete temporary device arrays
   call profiler_start("D2H_copy_results")
   !$omp target exit data map(from: h, u, v, eta, h0)
   call profiler_stop("D2H_copy_results")

   call profiler_start("GPU_dealloc_temps")
   !$omp target exit data map(delete: uh, vh, CAu, CAv, up, vp)
   !$omp target exit data map(delete: diffu, diffv)
   !$omp target exit data map(delete: ubt, vbt, ubt_av, vbt_av, eta_av)
   call profiler_stop("GPU_dealloc_temps")

   print '(A)', '=================================================='
   print '(A)', 'Timing Results'
   print '(A)', '=================================================='
   print '(A,F12.6)', 'Total time (s):        ', t_total
   print '(A,F12.6)', '  Coriolis time:       ', t_coriolis
   print '(A,F12.6)', '  Hor viscosity time:  ', t_hor_visc
   print '(A,F12.6)', '  Vert viscosity time: ', t_vert_visc
   print '(A,F12.6)', '  Barotropic time:     ', t_barotropic
   print '(A,F12.6)', '  Continuity time:     ', t_continuity
   print '(A,F12.6)', '  Diagnostics time:    ', t_diag
   print '(A)', ''
   print '(A,F12.6)', 'Time per RK2 step:     ', t_total/real(niter, dp)
   print '(A)', '=================================================='
   call diag_report_timing(diag_CS)

   ! Verify results
   call verify_state(h0, h, u, v, eta, G, GV)

   ! Cleanup - each _end() call deallocates device memory
   call profiler_start("Finalize_continuity")
   call continuity_end(cont_CS)
   call profiler_stop("Finalize_continuity")

   call profiler_start("Finalize_coriolis")
   call coriolis_end(cor_CS)
   call profiler_stop("Finalize_coriolis")

   call profiler_start("Finalize_barotropic")
   call barotropic_end(bt_CS)
   call profiler_stop("Finalize_barotropic")

   call profiler_start("Finalize_vert_visc")
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
   call profiler_report("RK2 Driver", root_region="Total")
   call profiler_end()

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

      do concurrent(k=1:GV%ke, j=G%jsd:G%jed, i=G%isd:G%ied)
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

      ! Sea surface height and barotropic velocities
      do concurrent(j=G%jsd:G%jed, i=G%isd:G%ied)
         eta(i, j) = 0.5_dp*sin(real(i - 1, dp)/real(G%ni, dp)*3.14159_dp*2.0_dp)* &
                     cos(real(j - 1, dp)/real(G%nj, dp)*3.14159_dp*2.0_dp)
         ubt(i, j) = 0.05_dp*sin(real(j - 1, dp)/real(G%nj, dp)*3.14159_dp)
         vbt(i, j) = 0.05_dp*cos(real(i - 1, dp)/real(G%ni, dp)*3.14159_dp)
      end do

      ! NOTE: With -gpu=mem:separate, do NOT use "target update to" here!
      ! The do concurrent loops above already computed values on the GPU.
      ! A "target update to" would overwrite device data with uninitialized host data.

   end subroutine initialize_state

   subroutine compute_transports(u, v, h, uh, vh, G, GV)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v, h
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh, vh

      integer :: i, j, k

      do concurrent(k=1:GV%ke, j=G%jsd:G%jed, i=G%isd:G%ied)
         uh(i, j, k) = u(i, j, k)*0.5_dp*(h(i, j, k) + h(min(i + 1, G%ied), j, k))*G%dyCu(i, j)
         vh(i, j, k) = v(i, j, k)*0.5_dp*(h(i, j, k) + h(i, min(j + 1, G%jed), k))*G%dxCv(i, j)
      end do

      ! NOTE: With -gpu=mem:separate, do NOT use "target update to" here!
      ! The do concurrent loop above already computed uh, vh on the GPU.

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
      do concurrent(k=1:GV%ke, j=G%jsc:G%jec, i=G%isc:G%iec)
         KE(i, j, k) = 0.5_dp*( &
                       0.5_dp*(u(i, j, k)**2 + u(i - 1, j, k)**2) + &
                       0.5_dp*(v(i, j, k)**2 + v(i, j - 1, k)**2))
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
         do concurrent(j=G%jsc:G%jec, i=G%isc:G%iec)
            mass(i, j) = mass(i, j) + h(i, j, k)
         end do
      end do

      call post_data_2d(id, mass, G, CS)

      deallocate (mass)

   end subroutine compute_and_post_mass

end program rk2_driver
