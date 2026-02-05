!> Standalone driver for the horizontal viscosity miniapp
program hor_visc_driver
  use omp_lib
  use iso_fortran_env, only: dp => real64
  use mom6_types
  use mom6_hor_visc
  implicit none

  type(ocean_grid_type) :: G
  type(verticalGrid_type) :: GV
  type(hor_visc_CS) :: CS

  real(dp), allocatable :: u(:,:,:), v(:,:,:), h(:,:,:)
  real(dp), allocatable :: u_init(:,:,:), v_init(:,:,:)
  real(dp), allocatable :: diffu(:,:,:), diffv(:,:,:)

  real(dp) :: t_start, t_end, t_total
  real(dp) :: Kh
  integer :: ni, nj, nk, niter, iter, i, j, k
  character(len=32) :: arg

  ! Default parameters
  ni = 180 ; nj = 180 ; nk = 75 ; niter = 10
  Kh = 100.0_dp  ! Laplacian viscosity [m2/s]

  ! Parse command line
  if (command_argument_count() >= 1) then
    call get_command_argument(1, arg) ; read(arg, *) ni
  endif
  if (command_argument_count() >= 2) then
    call get_command_argument(2, arg) ; read(arg, *) nj
  endif
  if (command_argument_count() >= 3) then
    call get_command_argument(3, arg) ; read(arg, *) nk
  endif
  if (command_argument_count() >= 4) then
    call get_command_argument(4, arg) ; read(arg, *) niter
  endif
  if (command_argument_count() >= 5) then
    call get_command_argument(5, arg) ; read(arg, *) Kh
  endif

  print '(A)', '=================================================='
  print '(A)', 'MOM6 Horizontal Viscosity Mini-App'
  print '(A)', '=================================================='
  print '(A,I5,A,I5,A,I4)', 'Grid: ', ni, ' x ', nj, ' x ', nk
  print '(A,I4)', 'Iterations: ', niter
  print '(A,F10.1)', 'Laplacian Kh [m2/s]: ', Kh
  print '(A)', '=================================================='

  ! Initialize grid
  call init_ocean_grid(G, ni, nj, nk, 10.0_dp, 45.0_dp)
  call init_verticalGrid(GV, nk)
  call hor_visc_init(CS, G, GV, Kh)

  ! Allocate state arrays
  allocate(u(G%isd:G%ied, G%jsd:G%jed, nk))
  allocate(v(G%isd:G%ied, G%jsd:G%jed, nk))
  allocate(h(G%isd:G%ied, G%jsd:G%jed, nk))
  allocate(u_init(G%isd:G%ied, G%jsd:G%jed, nk))
  allocate(v_init(G%isd:G%ied, G%jsd:G%jed, nk))
  allocate(diffu(G%isd:G%ied, G%jsd:G%jed, nk))
  allocate(diffv(G%isd:G%ied, G%jsd:G%jed, nk))

  !$omp target enter data map(alloc: u, v, h, u_init, v_init, diffu, diffv)

  ! Initialize state with horizontal structure that should be smoothed
  do concurrent (k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
    ! Layer thickness (uniform)
    h(i,j,k) = 4000.0_dp / real(nk, dp)

    ! Velocity with horizontal gradients (jets and eddies)
    u_init(i,j,k) = 0.5_dp * sin(real(j-1, dp)/real(nj, dp) * 6.28318_dp) * &
                    exp(-real(k-1, dp) / 30.0_dp) + &
                    0.2_dp * sin(real(i-1, dp)/real(ni, dp) * 12.56637_dp) * &
                    sin(real(j-1, dp)/real(nj, dp) * 12.56637_dp)
    v_init(i,j,k) = 0.3_dp * cos(real(i-1, dp)/real(ni, dp) * 6.28318_dp) * &
                    exp(-real(k-1, dp) / 30.0_dp) + &
                    0.2_dp * cos(real(i-1, dp)/real(ni, dp) * 12.56637_dp) * &
                    cos(real(j-1, dp)/real(nj, dp) * 12.56637_dp)

    u(i,j,k) = u_init(i,j,k)
    v(i,j,k) = v_init(i,j,k)
  end do

  !$omp target update to(u, v, h)

  print '(A)', ''
  print '(A)', 'Running horizontal viscosity solver...'

  t_total = 0.0_dp

  do iter = 1, niter
    ! Reset velocities each iteration for timing consistency
    do concurrent (k=1:nk, j=G%jsd:G%jed, i=G%isd:G%ied)
      u(i,j,k) = u_init(i,j,k)
      v(i,j,k) = v_init(i,j,k)
    end do
    !$omp target update to(u, v)

    ! Compute horizontal viscous accelerations
    t_start = omp_get_wtime()
    call hor_visc(u, v, h, diffu, diffv, G, GV, CS)
    t_end = omp_get_wtime()
    t_total = t_total + (t_end - t_start)
  end do

  !$omp target exit data map(from: u, v, diffu, diffv)
  !$omp target exit data map(delete: h, u_init, v_init)

  print '(A)', ''
  print '(A)', '=================================================='
  print '(A)', 'Timing Results'
  print '(A)', '=================================================='
  print '(A,F12.6)', 'Total time (s):          ', t_total
  print '(A,F12.6)', 'Time per iteration:      ', t_total / real(niter, dp)
  print '(A)', '=================================================='

  ! Verify viscous accelerations
  call verify_hor_visc(u_init, v_init, diffu, diffv, G, GV)

  ! Cleanup
  call hor_visc_end(CS)
  call end_ocean_grid(G)
  deallocate(u, v, h, u_init, v_init, diffu, diffv)

contains

  subroutine verify_hor_visc(u, v, diffu, diffv, G, GV)
    type(ocean_grid_type), intent(in) :: G
    type(verticalGrid_type), intent(in) :: GV
    real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, v
    real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: diffu, diffv

    real(dp) :: max_u, max_v, max_diffu, max_diffv
    real(dp) :: avg_diffu, avg_diffv, rms_diffu, rms_diffv
    integer :: i, j, k, cnt

    max_u = 0.0_dp ; max_v = 0.0_dp
    max_diffu = 0.0_dp ; max_diffv = 0.0_dp
    avg_diffu = 0.0_dp ; avg_diffv = 0.0_dp
    rms_diffu = 0.0_dp ; rms_diffv = 0.0_dp
    cnt = 0

    ! Compute statistics
    do k = 1, GV%ke
      do j = G%jsc, G%jec
        do i = G%isc, G%iec-1
          max_u = max(max_u, abs(u(i,j,k)))
          max_diffu = max(max_diffu, abs(diffu(i,j,k)))
          avg_diffu = avg_diffu + diffu(i,j,k)
          rms_diffu = rms_diffu + diffu(i,j,k)**2
        end do
      end do
      do j = G%jsc, G%jec-1
        do i = G%isc, G%iec
          max_v = max(max_v, abs(v(i,j,k)))
          max_diffv = max(max_diffv, abs(diffv(i,j,k)))
          avg_diffv = avg_diffv + diffv(i,j,k)
          rms_diffv = rms_diffv + diffv(i,j,k)**2
          cnt = cnt + 1
        end do
      end do
    end do

    if (cnt > 0) then
      avg_diffu = avg_diffu / real(cnt, dp)
      avg_diffv = avg_diffv / real(cnt, dp)
      rms_diffu = sqrt(rms_diffu / real(cnt, dp))
      rms_diffv = sqrt(rms_diffv / real(cnt, dp))
    endif

    print '(A)', ''
    print '(A)', 'Horizontal Viscosity Verification:'
    print '(A)', '--------------------------------------------------'
    print '(A)', 'Input velocities:'
    print '(A,ES15.8)', '  Max |u|:              ', max_u
    print '(A,ES15.8)', '  Max |v|:              ', max_v
    print '(A)', ''
    print '(A)', 'Viscous accelerations:'
    print '(A,ES15.8)', '  Max |diffu|:          ', max_diffu
    print '(A,ES15.8)', '  Max |diffv|:          ', max_diffv
    print '(A,ES15.8)', '  RMS diffu:            ', rms_diffu
    print '(A,ES15.8)', '  RMS diffv:            ', rms_diffv
    print '(A)', ''

    ! Check that accelerations are reasonable (not NaN or huge)
    if (max_diffu < 1.0e10_dp .and. max_diffv < 1.0e10_dp .and. &
        max_diffu == max_diffu .and. max_diffv == max_diffv) then
      print '(A)', '  Status: PASS (accelerations are finite and reasonable)'
    else
      print '(A)', '  Status: FAIL (accelerations are NaN or too large)'
    endif

  end subroutine verify_hor_visc

end program hor_visc_driver
