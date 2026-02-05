!> MOM6 Barotropic Solver Module
!!
!! Fast barotropic (depth-averaged) solver with sub-stepping.
!! Solves the linearized shallow water equations for free surface and
!! depth-averaged velocities.
!!
!! Original code from: src/core/MOM_barotropic.F90
!!
module mom6_barotropic
   use iso_fortran_env, only: dp => real64
   use mom6_types, only: ocean_grid_type, verticalGrid_type, G_EARTH
   implicit none
   private

   public :: btstep, barotropic_init, barotropic_end
   public :: barotropic_CS

   !> Control structure for barotropic solver
   type :: barotropic_CS
      logical :: initialized = .false.
      real(dp) :: dtbt              ! Barotropic timestep [T]
      real(dp) :: bebt              ! Backward Euler parameter [nondim]
      real(dp) :: dgeo_de           ! Geopotential coefficient [nondim]
      integer :: nstep              ! Number of substeps

      ! State arrays (2D)
      real(dp), allocatable :: eta(:, :)      ! Free surface height [H]
      real(dp), allocatable :: eta_pred(:, :) ! Predictor eta [H]
      real(dp), allocatable :: ubt(:, :)      ! Zonal barotropic velocity [L T-1]
      real(dp), allocatable :: vbt(:, :)      ! Meridional barotropic velocity [L T-1]
      real(dp), allocatable :: ubt_prev(:, :) ! Previous ubt [L T-1]
      real(dp), allocatable :: vbt_prev(:, :) ! Previous vbt [L T-1]
      real(dp), allocatable :: uhbt(:, :)     ! Zonal transport [H L2 T-1]
      real(dp), allocatable :: vhbt(:, :)     ! Meridional transport [H L2 T-1]
      real(dp), allocatable :: PFu(:, :)      ! Zonal pressure force [L T-2]
      real(dp), allocatable :: PFv(:, :)      ! Meridional pressure force [L T-2]
      real(dp), allocatable :: Cor_u(:, :)    ! Zonal Coriolis [L T-2]
      real(dp), allocatable :: Cor_v(:, :)    ! Meridional Coriolis [L T-2]

      ! Grid-related arrays
      real(dp), allocatable :: Datu(:, :)     ! Depth*dy at u-points [H L]
      real(dp), allocatable :: Datv(:, :)     ! Depth*dx at v-points [H L]
      real(dp), allocatable :: gtot_E(:, :)   ! Effective gravity (E) [L2 H-1 T-2]
      real(dp), allocatable :: gtot_W(:, :)   ! Effective gravity (W)
      real(dp), allocatable :: gtot_N(:, :)   ! Effective gravity (N)
      real(dp), allocatable :: gtot_S(:, :)   ! Effective gravity (S)
      real(dp), allocatable :: f_4_u(:, :, :)  ! Coriolis coefficients at u [T-1]
      real(dp), allocatable :: f_4_v(:, :, :)  ! Coriolis coefficients at v [T-1]
      real(dp), allocatable :: bt_rem_u(:, :) ! Drag remainder at u [nondim]
      real(dp), allocatable :: bt_rem_v(:, :) ! Drag remainder at v [nondim]

      ! Output arrays
      real(dp), allocatable :: ubt_av(:, :)   ! Time-averaged ubt [L T-1]
      real(dp), allocatable :: vbt_av(:, :)   ! Time-averaged vbt [L T-1]
      real(dp), allocatable :: uhbt_av(:, :)  ! Time-averaged uhbt [H L2 T-1]
      real(dp), allocatable :: vhbt_av(:, :)  ! Time-averaged vhbt [H L2 T-1]
   end type barotropic_CS

contains

   !> Initialize the barotropic solver
   subroutine barotropic_init(CS, G, dt, nstep)
      type(barotropic_CS), intent(inout) :: CS
      type(ocean_grid_type), intent(in) :: G
      real(dp), intent(in) :: dt       ! Baroclinic timestep
      integer, intent(in) :: nstep     ! Number of barotropic substeps

      real(dp) :: f0, depth
      integer :: i, j

      CS%nstep = nstep
      CS%dtbt = dt/real(nstep, dp)
      CS%bebt = 0.2_dp
      CS%dgeo_de = 1.0_dp
      depth = 4000.0_dp

      ! Allocate arrays
      allocate (CS%eta(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%eta_pred(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%ubt(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%vbt(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%ubt_prev(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%vbt_prev(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%uhbt(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%vhbt(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%PFu(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%PFv(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%Cor_u(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%Cor_v(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%Datu(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%Datv(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%gtot_E(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%gtot_W(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%gtot_N(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%gtot_S(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%f_4_u(4, G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%f_4_v(4, G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%bt_rem_u(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%bt_rem_v(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%ubt_av(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%vbt_av(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%uhbt_av(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%vhbt_av(G%isd:G%ied, G%jsd:G%jed))

      !$omp target enter data map(alloc: CS%eta, CS%eta_pred, CS%ubt, CS%vbt)
      !$omp target enter data map(alloc: CS%ubt_prev, CS%vbt_prev, CS%uhbt, CS%vhbt)
      !$omp target enter data map(alloc: CS%PFu, CS%PFv, CS%Cor_u, CS%Cor_v)
      !$omp target enter data map(alloc: CS%Datu, CS%Datv, CS%gtot_E, CS%gtot_W, CS%gtot_N, CS%gtot_S)
      !$omp target enter data map(alloc: CS%f_4_u, CS%f_4_v, CS%bt_rem_u, CS%bt_rem_v)
      !$omp target enter data map(alloc: CS%ubt_av, CS%vbt_av, CS%uhbt_av, CS%vhbt_av)

      ! Initialize grid-related arrays
      do concurrent(j=G%jsd:G%jed, i=G%isd:G%ied)
         CS%Datu(i, j) = depth*G%dyCu(i, j)
         CS%Datv(i, j) = depth*G%dxCv(i, j)
         CS%gtot_E(i, j) = G_EARTH
         CS%gtot_W(i, j) = G_EARTH
         CS%gtot_N(i, j) = G_EARTH
         CS%gtot_S(i, j) = G_EARTH
         CS%bt_rem_u(i, j) = 0.999_dp  ! Small drag
         CS%bt_rem_v(i, j) = 0.999_dp
      end do

      ! Initialize Coriolis coefficients (f/4 at each corner)
      do concurrent(j=G%jsd:G%jed, i=G%isd:G%ied)
         f0 = G%CoriolisBu(i, j)
         CS%f_4_u(1, i, j) = 0.25_dp*f0  ! SW
         CS%f_4_u(2, i, j) = 0.25_dp*f0  ! SE
         CS%f_4_u(3, i, j) = 0.25_dp*f0  ! NW
         CS%f_4_u(4, i, j) = 0.25_dp*f0  ! NE
         CS%f_4_v(1, i, j) = 0.25_dp*f0
         CS%f_4_v(2, i, j) = 0.25_dp*f0
         CS%f_4_v(3, i, j) = 0.25_dp*f0
         CS%f_4_v(4, i, j) = 0.25_dp*f0
      end do

      CS%initialized = .true.

   end subroutine barotropic_init

   !> Finalize the barotropic solver
   subroutine barotropic_end(CS)
      type(barotropic_CS), intent(inout) :: CS

      if (.not. CS%initialized) return

      !$omp target exit data map(delete: CS%eta, CS%eta_pred, CS%ubt, CS%vbt)
      !$omp target exit data map(delete: CS%ubt_prev, CS%vbt_prev, CS%uhbt, CS%vhbt)
      !$omp target exit data map(delete: CS%PFu, CS%PFv, CS%Cor_u, CS%Cor_v)
      !$omp target exit data map(delete: CS%Datu, CS%Datv, CS%gtot_E, CS%gtot_W, CS%gtot_N, CS%gtot_S)
      !$omp target exit data map(delete: CS%f_4_u, CS%f_4_v, CS%bt_rem_u, CS%bt_rem_v)
      !$omp target exit data map(delete: CS%ubt_av, CS%vbt_av, CS%uhbt_av, CS%vhbt_av)

      if (allocated(CS%eta)) deallocate (CS%eta)
      if (allocated(CS%eta_pred)) deallocate (CS%eta_pred)
      if (allocated(CS%ubt)) deallocate (CS%ubt)
      if (allocated(CS%vbt)) deallocate (CS%vbt)
      if (allocated(CS%ubt_prev)) deallocate (CS%ubt_prev)
      if (allocated(CS%vbt_prev)) deallocate (CS%vbt_prev)
      if (allocated(CS%uhbt)) deallocate (CS%uhbt)
      if (allocated(CS%vhbt)) deallocate (CS%vhbt)
      if (allocated(CS%PFu)) deallocate (CS%PFu)
      if (allocated(CS%PFv)) deallocate (CS%PFv)
      if (allocated(CS%Cor_u)) deallocate (CS%Cor_u)
      if (allocated(CS%Cor_v)) deallocate (CS%Cor_v)
      if (allocated(CS%Datu)) deallocate (CS%Datu)
      if (allocated(CS%Datv)) deallocate (CS%Datv)
      if (allocated(CS%gtot_E)) deallocate (CS%gtot_E)
      if (allocated(CS%gtot_W)) deallocate (CS%gtot_W)
      if (allocated(CS%gtot_N)) deallocate (CS%gtot_N)
      if (allocated(CS%gtot_S)) deallocate (CS%gtot_S)
      if (allocated(CS%f_4_u)) deallocate (CS%f_4_u)
      if (allocated(CS%f_4_v)) deallocate (CS%f_4_v)
      if (allocated(CS%bt_rem_u)) deallocate (CS%bt_rem_u)
      if (allocated(CS%bt_rem_v)) deallocate (CS%bt_rem_v)
      if (allocated(CS%ubt_av)) deallocate (CS%ubt_av)
      if (allocated(CS%vbt_av)) deallocate (CS%vbt_av)
      if (allocated(CS%uhbt_av)) deallocate (CS%uhbt_av)
      if (allocated(CS%vhbt_av)) deallocate (CS%vhbt_av)

      CS%initialized = .false.

   end subroutine barotropic_end

   !> Main barotropic time-stepping routine
  !! Performs nstep substeps of the barotropic equations
   subroutine btstep(eta_in, ubt_in, vbt_in, u_av, v_av, eta_av, G, CS)
      type(ocean_grid_type), intent(in) :: G
      type(barotropic_CS), intent(inout) :: CS
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: eta_in    ! Initial eta
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: ubt_in    ! Initial ubt
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: vbt_in    ! Initial vbt
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: u_av     ! Time-averaged u
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: v_av     ! Time-averaged v
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: eta_av   ! Final eta

      real(dp) :: trans_wt1, trans_wt2, inv_nstep
      logical :: v_first
      integer :: i, j, n, is, ie, js, je

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec

      trans_wt1 = 1.0_dp + CS%bebt
      trans_wt2 = -CS%bebt
      inv_nstep = 1.0_dp/real(CS%nstep, dp)

      ! Initialize from input
      do concurrent(j=G%jsd:G%jed, i=G%isd:G%ied)
         CS%eta(i, j) = eta_in(i, j)
         CS%ubt(i, j) = ubt_in(i, j)
         CS%vbt(i, j) = vbt_in(i, j)
         CS%ubt_av(i, j) = 0.0_dp
         CS%vbt_av(i, j) = 0.0_dp
      end do

      ! Barotropic time-stepping loop
      do n = 1, CS%nstep

         ! Store previous velocities
         do concurrent(j=js:je, i=is - 1:ie + 1)
            CS%ubt_prev(i, j) = CS%ubt(i, j)
         end do
         do concurrent(j=js - 1:je + 1, i=is:ie)
            CS%vbt_prev(i, j) = CS%vbt(i, j)
         end do

         ! Eta predictor
         do concurrent(j=js:je, i=is:ie)
            CS%eta_pred(i, j) = CS%eta(i, j) + (CS%dtbt*G%IareaT(i, j))* &
                                (((CS%Datu(i - 1, j)*CS%ubt(i - 1, j)) - (CS%Datu(i, j)*CS%ubt(i, j))) + &
                                 ((CS%Datv(i, j - 1)*CS%vbt(i, j - 1)) - (CS%Datv(i, j)*CS%vbt(i, j))))
         end do

         ! Pressure force
         do concurrent(j=js:je, i=is:ie - 1)
            CS%PFu(i, j) = ((CS%eta_pred(i, j)*CS%gtot_E(i, j)) - &
                            (CS%eta_pred(i + 1, j)*CS%gtot_W(i + 1, j)))* &
                           CS%dgeo_de*G%IdxCu(i, j)
         end do
         do concurrent(j=js:je - 1, i=is:ie)
            CS%PFv(i, j) = ((CS%eta_pred(i, j)*CS%gtot_N(i, j)) - &
                            (CS%eta_pred(i, j + 1)*CS%gtot_S(i, j + 1)))* &
                           CS%dgeo_de*G%IdyCv(i, j)
         end do

         ! Alternating u/v update order
         v_first = (mod(n + G%first_direction, 2) == 1)

         if (v_first) then
            ! Update v first
            call update_v(CS, G, is, ie, js, je - 1)
            call update_u(CS, G, is, ie - 1, js, je)
         else
            ! Update u first
            call update_u(CS, G, is, ie - 1, js, je)
            call update_v(CS, G, is, ie, js, je - 1)
         end if

         ! Compute transports and update eta
         do concurrent(j=js:je, i=is:ie - 1)
            CS%uhbt(i, j) = CS%Datu(i, j)*(trans_wt1*CS%ubt(i, j) + trans_wt2*CS%ubt_prev(i, j))
         end do
         do concurrent(j=js:je - 1, i=is:ie)
            CS%vhbt(i, j) = CS%Datv(i, j)*(trans_wt1*CS%vbt(i, j) + trans_wt2*CS%vbt_prev(i, j))
         end do

         do concurrent(j=js:je, i=is:ie)
            CS%eta(i, j) = CS%eta(i, j) - CS%dtbt*G%IareaT(i, j)* &
                           ((CS%uhbt(i, j) - CS%uhbt(i - 1, j)) + (CS%vhbt(i, j) - CS%vhbt(i, j - 1)))
         end do

         ! Accumulate time averages
         do concurrent(j=js:je, i=is:ie - 1)
            CS%ubt_av(i, j) = CS%ubt_av(i, j) + CS%ubt(i, j)*inv_nstep
         end do
         do concurrent(j=js:je - 1, i=is:ie)
            CS%vbt_av(i, j) = CS%vbt_av(i, j) + CS%vbt(i, j)*inv_nstep
         end do

      end do  ! substep loop

      ! Copy output
      do concurrent(j=G%jsd:G%jed, i=G%isd:G%ied)
         u_av(i, j) = CS%ubt_av(i, j)
         v_av(i, j) = CS%vbt_av(i, j)
         eta_av(i, j) = CS%eta(i, j)
      end do

   end subroutine btstep

   !> Update u velocity
   subroutine update_u(CS, G, is, ie, js, je)
      type(barotropic_CS), intent(inout) :: CS
      type(ocean_grid_type), intent(in) :: G
      integer, intent(in) :: is, ie, js, je

      integer :: i, j

      ! Coriolis for u
      do concurrent(j=js:je, i=is:ie)
         CS%Cor_u(i, j) = (((CS%f_4_u(4, i, j)*CS%vbt(i + 1, j)) + (CS%f_4_u(1, i, j)*CS%vbt(i, j - 1))) + &
                           ((CS%f_4_u(3, i, j)*CS%vbt(i, j)) + (CS%f_4_u(2, i, j)*CS%vbt(i + 1, j - 1))))
      end do

      ! Update u
      do concurrent(j=js:je, i=is:ie)
         CS%ubt(i, j) = CS%bt_rem_u(i, j)*(CS%ubt(i, j) + &
                                           CS%dtbt*(CS%Cor_u(i, j) + CS%PFu(i, j)))
      end do

   end subroutine update_u

   !> Update v velocity
   subroutine update_v(CS, G, is, ie, js, je)
      type(barotropic_CS), intent(inout) :: CS
      type(ocean_grid_type), intent(in) :: G
      integer, intent(in) :: is, ie, js, je

      integer :: i, j

      ! Coriolis for v
      do concurrent(j=js:je, i=is:ie)
         CS%Cor_v(i, j) = -1.0_dp*(((CS%f_4_v(1, i, j)*CS%ubt(i - 1, j)) + (CS%f_4_v(4, i, j)*CS%ubt(i, j + 1))) + &
                                   ((CS%f_4_v(2, i, j)*CS%ubt(i, j)) + (CS%f_4_v(3, i, j)*CS%ubt(i - 1, j + 1))))
      end do

      ! Update v
      do concurrent(j=js:je, i=is:ie)
         CS%vbt(i, j) = CS%bt_rem_v(i, j)*(CS%vbt(i, j) + &
                                           CS%dtbt*(CS%Cor_v(i, j) + CS%PFv(i, j)))
      end do

   end subroutine update_v

end module mom6_barotropic
