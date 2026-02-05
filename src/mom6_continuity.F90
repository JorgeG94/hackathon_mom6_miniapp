!> MOM6 Continuity Solver Module
!!
!! PPM (Piecewise Parabolic Method) continuity solver extracted from MOM6.
!! Solves the layer thickness equation: dh/dt = -div(uh, vh)
!!
!! Original code from: src/core/MOM_continuity_PPM.F90
!!
module mom6_continuity
   use iso_fortran_env, only: dp => real64
   use mom6_types, only: ocean_grid_type, verticalGrid_type
   implicit none
   private

   public :: continuity_PPM, continuity_init, continuity_end
   public :: continuity_CS

   !> Control structure for continuity solver
   type :: continuity_CS
      logical :: initialized = .false.
      real(dp) :: h_min           ! Minimum layer thickness [H]
      ! Work arrays
      real(dp), allocatable :: h_W(:, :, :)   ! West edge thickness
      real(dp), allocatable :: h_E(:, :, :)   ! East edge thickness
      real(dp), allocatable :: h_S(:, :, :)   ! South edge thickness
      real(dp), allocatable :: h_N(:, :, :)   ! North edge thickness
      real(dp), allocatable :: slp(:, :, :)   ! PPM slopes
   end type continuity_CS

contains

   !> Initialize the continuity solver
   subroutine continuity_init(CS, G, GV)
      type(continuity_CS), intent(inout) :: CS
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV

      CS%h_min = GV%Angstrom_H

      allocate (CS%h_W(G%isd:G%ied, G%jsd:G%jed, GV%ke))
      allocate (CS%h_E(G%isd:G%ied, G%jsd:G%jed, GV%ke))
      allocate (CS%h_S(G%isd:G%ied, G%jsd:G%jed, GV%ke))
      allocate (CS%h_N(G%isd:G%ied, G%jsd:G%jed, GV%ke))
      allocate (CS%slp(G%isd:G%ied, G%jsd:G%jed, GV%ke))

      !$omp target enter data map(alloc: CS%h_W, CS%h_E, CS%h_S, CS%h_N, CS%slp)

      CS%initialized = .true.

   end subroutine continuity_init

   !> Finalize the continuity solver
   subroutine continuity_end(CS)
      type(continuity_CS), intent(inout) :: CS

      if (.not. CS%initialized) return

      !$omp target exit data map(delete: CS%h_W, CS%h_E, CS%h_S, CS%h_N, CS%slp)

      if (allocated(CS%h_W)) deallocate (CS%h_W)
      if (allocated(CS%h_E)) deallocate (CS%h_E)
      if (allocated(CS%h_S)) deallocate (CS%h_S)
      if (allocated(CS%h_N)) deallocate (CS%h_N)
      if (allocated(CS%slp)) deallocate (CS%slp)

      CS%initialized = .false.

   end subroutine continuity_end

   !> Main continuity solver using PPM
  !! Updates layer thickness h from hin using velocities u, v
   subroutine continuity_PPM(u, v, hin, h, uh, vh, dt, G, GV, CS, x_first)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u    ! Zonal velocity [L T-1]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: v    ! Meridional velocity [L T-1]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: hin  ! Initial thickness [H]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: h ! Final thickness [H]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh  ! Zonal flux [H L2 T-1]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: vh  ! Meridional flux [H L2 T-1]
      real(dp), intent(in) :: dt        ! Time step [T]
      type(continuity_CS), intent(inout) :: CS
      logical, intent(in), optional :: x_first  ! If true, do x-direction first

      logical :: do_x_first
      integer :: is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

      do_x_first = .true.
      if (present(x_first)) do_x_first = x_first

      if (do_x_first) then
         ! X-direction first
         call PPM_reconstruction_x(hin, CS%h_W, CS%h_E, CS%slp, G, GV)
         call zonal_mass_flux(u, hin, CS%h_W, CS%h_E, uh, dt, G, GV)
         call zonal_convergence(hin, h, uh, dt, CS%h_min, G, GV)

         ! Then Y-direction (using updated h)
         call PPM_reconstruction_y(h, CS%h_S, CS%h_N, CS%slp, G, GV)
         call meridional_mass_flux(v, h, CS%h_S, CS%h_N, vh, dt, G, GV)
         call meridional_convergence(h, h, vh, dt, CS%h_min, G, GV)
      else
         ! Y-direction first
         call PPM_reconstruction_y(hin, CS%h_S, CS%h_N, CS%slp, G, GV)
         call meridional_mass_flux(v, hin, CS%h_S, CS%h_N, vh, dt, G, GV)
         call meridional_convergence(hin, h, vh, dt, CS%h_min, G, GV)

         ! Then X-direction (using updated h)
         call PPM_reconstruction_x(h, CS%h_W, CS%h_E, CS%slp, G, GV)
         call zonal_mass_flux(u, h, CS%h_W, CS%h_E, uh, dt, G, GV)
         call zonal_convergence(h, h, uh, dt, CS%h_min, G, GV)
      end if

   end subroutine continuity_PPM

   !> PPM reconstruction in x-direction with monotonic limiting
   subroutine PPM_reconstruction_x(h, h_W, h_E, slp, G, GV)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: h_W, h_E, slp

      real(dp) :: dMx, dMn
      real(dp), parameter :: oneSixth = 1.0_dp/6.0_dp
      integer :: i, j, k, is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

      ! Phase 1: Compute limited slopes
      do concurrent(k=1:nz, j=js:je, i=is:ie)
         if ((G%mask2dT(i - 1, j)*G%mask2dT(i, j)*G%mask2dT(i + 1, j)) == 0.0_dp) then
            slp(i, j, k) = 0.0_dp
         else
            slp(i, j, k) = 0.5_dp*(h(i + 1, j, k) - h(i - 1, j, k))
            dMx = max(h(i + 1, j, k), h(i - 1, j, k), h(i, j, k)) - h(i, j, k)
            dMn = h(i, j, k) - min(h(i + 1, j, k), h(i - 1, j, k), h(i, j, k))
            slp(i, j, k) = sign(1.0_dp, slp(i, j, k))*min(abs(slp(i, j, k)), 2.0_dp*min(dMx, dMn))
         end if
      end do

      ! Phase 2: Compute edge values
      do concurrent(k=1:nz, j=js:je, i=is:ie)
         h_W(i, j, k) = 0.5_dp*(h(i - 1, j, k) + h(i, j, k)) + oneSixth*(slp(i - 1, j, k) - slp(i, j, k))
         h_E(i, j, k) = 0.5_dp*(h(i + 1, j, k) + h(i, j, k)) + oneSixth*(slp(i, j, k) - slp(i + 1, j, k))
         h_W(i, j, k) = max(h_W(i, j, k), 0.0_dp)
         h_E(i, j, k) = max(h_E(i, j, k), 0.0_dp)
      end do

   end subroutine PPM_reconstruction_x

   !> PPM reconstruction in y-direction
   subroutine PPM_reconstruction_y(h, h_S, h_N, slp, G, GV)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: h_S, h_N, slp

      real(dp) :: dMx, dMn
      real(dp), parameter :: oneSixth = 1.0_dp/6.0_dp
      integer :: i, j, k, is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

      ! Phase 1: Compute limited slopes
      do concurrent(k=1:nz, j=js:je, i=is:ie)
         if ((G%mask2dT(i, j - 1)*G%mask2dT(i, j)*G%mask2dT(i, j + 1)) == 0.0_dp) then
            slp(i, j, k) = 0.0_dp
         else
            slp(i, j, k) = 0.5_dp*(h(i, j + 1, k) - h(i, j - 1, k))
            dMx = max(h(i, j + 1, k), h(i, j - 1, k), h(i, j, k)) - h(i, j, k)
            dMn = h(i, j, k) - min(h(i, j + 1, k), h(i, j - 1, k), h(i, j, k))
            slp(i, j, k) = sign(1.0_dp, slp(i, j, k))*min(abs(slp(i, j, k)), 2.0_dp*min(dMx, dMn))
         end if
      end do

      ! Phase 2: Compute edge values
      do concurrent(k=1:nz, j=js:je, i=is:ie)
         h_S(i, j, k) = 0.5_dp*(h(i, j - 1, k) + h(i, j, k)) + oneSixth*(slp(i, j - 1, k) - slp(i, j, k))
         h_N(i, j, k) = 0.5_dp*(h(i, j + 1, k) + h(i, j, k)) + oneSixth*(slp(i, j, k) - slp(i, j + 1, k))
         h_S(i, j, k) = max(h_S(i, j, k), 0.0_dp)
         h_N(i, j, k) = max(h_N(i, j, k), 0.0_dp)
      end do

   end subroutine PPM_reconstruction_y

   !> Compute zonal mass flux using PPM
   subroutine zonal_mass_flux(u, h, h_W, h_E, uh, dt, G, GV)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, h, h_W, h_E
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh
      real(dp), intent(in) :: dt

      real(dp) :: CFL, curv_3, dh, face_area
      integer :: i, j, k, is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

      do concurrent(k=1:nz, j=js:je, i=is:ie)
         face_area = G%dyCu(i, j)

         if (u(i, j, k) > 0.0_dp) then
            CFL = u(i, j, k)*dt*G%IdxT(i, j)
            curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp*h(i, j, k)
            dh = h_W(i, j, k) - h_E(i, j, k)
            uh(i, j, k) = face_area*u(i, j, k)* &
                          (h_E(i, j, k) + CFL*(0.5_dp*dh + curv_3*(CFL - 1.5_dp)))
         elseif (u(i, j, k) < 0.0_dp) then
            CFL = -u(i, j, k)*dt*G%IdxT(i, j)
            curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp*h(i, j, k)
            dh = h_E(i, j, k) - h_W(i, j, k)
            uh(i, j, k) = face_area*u(i, j, k)* &
                          (h_W(i, j, k) + CFL*(0.5_dp*dh + curv_3*(CFL - 1.5_dp)))
         else
            uh(i, j, k) = 0.0_dp
         end if
      end do

   end subroutine zonal_mass_flux

   !> Compute meridional mass flux using PPM
   subroutine meridional_mass_flux(v, h, h_S, h_N, vh, dt, G, GV)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: v, h, h_S, h_N
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: vh
      real(dp), intent(in) :: dt

      real(dp) :: CFL, curv_3, dh, face_area
      integer :: i, j, k, is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

      do concurrent(k=1:nz, j=js:je, i=is:ie)
         face_area = G%dxCv(i, j)

         if (v(i, j, k) > 0.0_dp) then
            CFL = v(i, j, k)*dt*G%IdyT(i, j)
            curv_3 = (h_S(i, j, k) + h_N(i, j, k)) - 2.0_dp*h(i, j, k)
            dh = h_S(i, j, k) - h_N(i, j, k)
            vh(i, j, k) = face_area*v(i, j, k)* &
                          (h_N(i, j, k) + CFL*(0.5_dp*dh + curv_3*(CFL - 1.5_dp)))
         elseif (v(i, j, k) < 0.0_dp) then
            CFL = -v(i, j, k)*dt*G%IdyT(i, j)
            curv_3 = (h_S(i, j, k) + h_N(i, j, k)) - 2.0_dp*h(i, j, k)
            dh = h_N(i, j, k) - h_S(i, j, k)
            vh(i, j, k) = face_area*v(i, j, k)* &
                          (h_S(i, j, k) + CFL*(0.5_dp*dh + curv_3*(CFL - 1.5_dp)))
         else
            vh(i, j, k) = 0.0_dp
         end if
      end do

   end subroutine meridional_mass_flux

   !> Apply zonal flux convergence
   subroutine zonal_convergence(hin, h, uh, dt, h_min, G, GV)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: hin, uh
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: h
      real(dp), intent(in) :: dt, h_min

      integer :: i, j, k, is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

      do concurrent(k=1:nz, j=js:je, i=is:ie)
         h(i, j, k) = max(hin(i, j, k) - dt*G%IareaT(i, j)*(uh(i + 1, j, k) - uh(i, j, k)), h_min)
      end do

   end subroutine zonal_convergence

   !> Apply meridional flux convergence
   subroutine meridional_convergence(hin, h, vh, dt, h_min, G, GV)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: hin, vh
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: h
      real(dp), intent(in) :: dt, h_min

      integer :: i, j, k, is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke

      do concurrent(k=1:nz, j=js:je, i=is:ie)
         h(i, j, k) = max(hin(i, j, k) - dt*G%IareaT(i, j)*(vh(i, j + 1, k) - vh(i, j, k)), h_min)
      end do

   end subroutine meridional_convergence

end module mom6_continuity
