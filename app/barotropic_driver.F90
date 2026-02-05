!> Standalone driver for the barotropic solver miniapp
program barotropic_driver
  use omp_lib
  use iso_fortran_env, only: dp => real64
  use mom6_types
  use mom6_barotropic
  implicit none

  type(ocean_grid_type) :: G
  type(barotropic_CS) :: CS

  real(dp), allocatable :: eta_in(:,:), ubt_in(:,:), vbt_in(:,:)
  real(dp), allocatable :: u_av(:,:), v_av(:,:), eta_av(:,:)
  real(dp), allocatable :: eta_init(:,:)

  real(dp) :: t_start, t_end, t_total, dt
  integer :: ni, nj, niter, iter, i, j, nsteps
  character(len=32) :: arg

  ! Default parameters
  ni = 180 ; nj = 180 ; niter = 10 ; nsteps = 30
  dt = 300.0_dp  ! 5 minute baroclinic timestep

  ! Parse command line
  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg) ; read(arg, *) ni
  endif
  if (command_argument_count() >= 2) then
    call get_command_argument(2, arg) ; read(arg, *) nj
  endif
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg) ; read(arg, *) nsteps
  endif
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg) ; read(arg, *) niter
  endif

  print '(A)', '=================================================='
  print '(A)', 'MOM6 Barotropic Mini-App'
  print '(A)', '=================================================='
  print '(A,I5,A,I5)', 'Grid: ', ni, ' x ', nj
  print '(A,I4)', 'Barotropic substeps: ', nsteps
  print '(A,I4)', 'Iterations: ', niter
  print '(A)', '=================================================='

  ! Initialize grid (2D only for barotropic)
  call init_ocean_grid(G, ni, nj, 1, 10.0_dp, 45.0_dp)
  call barotropic_init(CS, G, dt, nsteps)

  ! Allocate state arrays
  allocate(eta_in(G%isd:G%ied, G%jsd:G%jed))
  allocate(ubt_in(G%isd:G%ied, G%jsd:G%jed))
  allocate(vbt_in(G%isd:G%ied, G%jsd:G%jed))
  allocate(u_av(G%isd:G%ied, G%jsd:G%jed))
  allocate(v_av(G%isd:G%ied, G%jsd:G%jed))
  allocate(eta_av(G%isd:G%ied, G%jsd:G%jed))
  allocate(eta_init(G%isd:G%ied, G%jsd:G%jed))

  !$omp target enter data map(alloc: eta_in, ubt_in, vbt_in, u_av, v_av, eta_av)

  ! Initialize state with realistic patterns
  do concurrent (j=G%jsd:G%jed, i=G%isd:G%ied)
    ! Sea surface height anomaly (meters)
    eta_in(i,j) = 0.5_dp * sin(real(i-1, dp)/real(ni, dp) * 3.14159_dp * 2.0_dp) * &
                  cos(real(j-1, dp)/real(nj, dp) * 3.14159_dp * 2.0_dp)
    eta_init(i,j) = eta_in(i,j)
    ! Barotropic velocities (m/s)
    ubt_in(i,j) = 0.05_dp * sin(real(j-1, dp)/real(nj, dp) * 3.14159_dp)
    vbt_in(i,j) = 0.05_dp * cos(real(i-1, dp)/real(ni, dp) * 3.14159_dp)
  end do

  !$omp target update to(eta_in, ubt_in, vbt_in)

  print '(A)', ''
  print '(A)', 'Running barotropic solver...'

  t_total = 0.0_dp

  do iter = 1, niter
    ! Reset initial conditions each iteration
    do concurrent (j=G%jsd:G%jed, i=G%isd:G%ied)
      eta_in(i,j) = eta_init(i,j)
    end do
    !$omp target update to(eta_in)

    t_start = omp_get_wtime()
    call btstep(eta_in, ubt_in, vbt_in, u_av, v_av, eta_av, G, CS)
    t_end = omp_get_wtime()

    t_total = t_total + (t_end - t_start)
  end do

  !$omp target exit data map(from: u_av, v_av, eta_av)
  !$omp target exit data map(delete: eta_in, ubt_in, vbt_in)

  print '(A)', ''
  print '(A)', '=================================================='
  print '(A,F12.6)', 'Total time (s):      ', t_total
  print '(A,F12.6)', 'Time per iteration:  ', t_total / real(niter, dp)
  print '(A,F12.6)', 'Time per substep:    ', t_total / real(niter * nsteps, dp)
  print '(A)', '=================================================='

  ! Verify results
  call verify_barotropic(eta_init, eta_av, u_av, v_av, G)

  ! Cleanup
  call barotropic_end(CS)
  call end_ocean_grid(G)
  deallocate(eta_in, ubt_in, vbt_in, u_av, v_av, eta_av, eta_init)

contains

  subroutine verify_barotropic(eta_init, eta_av, u_av, v_av, G)
    type(ocean_grid_type), intent(in) :: G
    real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: eta_init, eta_av
    real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: u_av, v_av

    real(dp) :: max_eta, max_u, max_v
    real(dp) :: sum_eta_init, sum_eta_final, rel_error
    integer :: i, j

    max_eta = 0.0_dp ; max_u = 0.0_dp ; max_v = 0.0_dp
    sum_eta_init = 0.0_dp ; sum_eta_final = 0.0_dp

    do j = G%jsc, G%jec
      do i = G%isc, G%iec
        max_eta = max(max_eta, abs(eta_av(i,j)))
        max_u = max(max_u, abs(u_av(i,j)))
        max_v = max(max_v, abs(v_av(i,j)))
        sum_eta_init = sum_eta_init + eta_init(i,j) * G%areaT(i,j)
        sum_eta_final = sum_eta_final + eta_av(i,j) * G%areaT(i,j)
      end do
    end do

    ! Volume conservation (eta weighted by area)
    if (abs(sum_eta_init) > 1.0e-20_dp) then
      rel_error = abs(sum_eta_final - sum_eta_init) / abs(sum_eta_init)
    else
      rel_error = abs(sum_eta_final - sum_eta_init)
    endif

    print '(A)', ''
    print '(A)', 'Barotropic Statistics:'
    print '(A,ES15.8)', '  Max |eta_av|: ', max_eta
    print '(A,ES15.8)', '  Max |u_av|:   ', max_u
    print '(A,ES15.8)', '  Max |v_av|:   ', max_v
    print '(A)', ''
    print '(A)', 'Volume Conservation:'
    print '(A,ES15.8)', '  Initial sum:  ', sum_eta_init
    print '(A,ES15.8)', '  Final sum:    ', sum_eta_final
    print '(A,ES15.8)', '  Rel. error:   ', rel_error

    if (max_u > 0.0_dp .and. max_v > 0.0_dp) then
      print '(A)', '  Status: PASS (non-zero velocities computed)'
    else
      print '(A)', '  Status: WARNING (zero velocities)'
    endif
  end subroutine verify_barotropic

end program barotropic_driver
