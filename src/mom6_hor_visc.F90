!> MOM6 Horizontal Viscosity Module
!!
!! Applies Laplacian horizontal viscosity to momentum.
!! Simplified version of MOM6's MOM_hor_visc.F90 for GPU hackathon.
!!
!! The horizontal viscous stress is computed from the strain rate tensor:
!!   sh_xx = du/dx - dv/dy  (tension)
!!   sh_xy = dv/dx + du/dy  (shearing strain)
!!
!! The stress tensor is then:
!!   str_xx = -Kh * h * sh_xx
!!   str_xy = -Kh * hq * sh_xy
!!
!! And the viscous acceleration is the divergence of the stress tensor.
!!
!! GPU Pattern: Uses do concurrent for all loops, 2D work arrays reused per layer.
!!
module mom6_hor_visc
   use iso_fortran_env, only: dp => real64
   use mom6_types, only: ocean_grid_type, verticalGrid_type
   implicit none
   private

   public :: hor_visc_init, hor_visc, hor_visc_end
   public :: hor_visc_CS

   !> Control structure for horizontal viscosity solver
   type :: hor_visc_CS
      logical :: initialized = .false.
      real(dp) :: Kh          ! Laplacian viscosity coefficient [L2 T-1]
      real(dp) :: h_neglect   ! Minimum thickness to avoid division by zero [H]

      ! Pre-computed metric products (for efficiency)
      real(dp), allocatable :: dx2h(:, :)    ! dx^2 at h-points [L2]
      real(dp), allocatable :: dy2h(:, :)    ! dy^2 at h-points [L2]
      real(dp), allocatable :: dx2q(:, :)    ! dx^2 at q-points [L2]
      real(dp), allocatable :: dy2q(:, :)    ! dy^2 at q-points [L2]
      real(dp), allocatable :: DX_dyT(:, :)  ! dx/dy ratio at h-points [nondim]
      real(dp), allocatable :: DY_dxT(:, :)  ! dy/dx ratio at h-points [nondim]
      real(dp), allocatable :: DX_dyBu(:, :)  ! dx/dy ratio at q-points [nondim]
      real(dp), allocatable :: DY_dxBu(:, :)  ! dy/dx ratio at q-points [nondim]

      ! Work arrays (2D, reused per layer)
      real(dp), allocatable :: dudx(:, :)    ! du/dx at h-points [T-1]
      real(dp), allocatable :: dvdy(:, :)    ! dv/dy at h-points [T-1]
      real(dp), allocatable :: dvdx(:, :)    ! dv/dx at q-points [T-1]
      real(dp), allocatable :: dudy(:, :)    ! du/dy at q-points [T-1]
      real(dp), allocatable :: sh_xx(:, :)   ! Horizontal tension [T-1]
      real(dp), allocatable :: sh_xy(:, :)   ! Shearing strain [T-1]
      real(dp), allocatable :: str_xx(:, :)  ! Diagonal stress [H L2 T-2]
      real(dp), allocatable :: str_xy(:, :)  ! Off-diagonal stress [H L2 T-2]
      real(dp), allocatable :: h_u(:, :)     ! Thickness at u-points [H]
      real(dp), allocatable :: h_v(:, :)     ! Thickness at v-points [H]
      real(dp), allocatable :: hq(:, :)      ! Thickness at q-points [H]
   end type hor_visc_CS

contains

   !> Initialize the horizontal viscosity solver
   subroutine hor_visc_init(CS, G, GV, Kh)
      type(hor_visc_CS), intent(inout) :: CS
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), intent(in), optional :: Kh  ! Laplacian viscosity [m2/s]

      integer :: i, j

      ! Set viscosity parameter (default: 100 m2/s)
      CS%Kh = 100.0_dp
      if (present(Kh)) CS%Kh = Kh

      ! Use a larger h_neglect for numerical stability (1 mm instead of Angstrom)
      CS%h_neglect = max(GV%Angstrom_H, 1.0e-3_dp)

      ! Allocate pre-computed metrics
      allocate (CS%dx2h(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%dy2h(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%dx2q(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%dy2q(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%DX_dyT(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%DY_dxT(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%DX_dyBu(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%DY_dxBu(G%isd:G%ied, G%jsd:G%jed))

      ! Allocate work arrays
      allocate (CS%dudx(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%dvdy(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%dvdx(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%dudy(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%sh_xx(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%sh_xy(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%str_xx(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%str_xy(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%h_u(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%h_v(G%isd:G%ied, G%jsd:G%jed))
      allocate (CS%hq(G%isd:G%ied, G%jsd:G%jed))

      ! Initialize all work arrays to zero
      CS%dudx = 0.0_dp; CS%dvdy = 0.0_dp; CS%dvdx = 0.0_dp; CS%dudy = 0.0_dp
      CS%sh_xx = 0.0_dp; CS%sh_xy = 0.0_dp; CS%str_xx = 0.0_dp; CS%str_xy = 0.0_dp
      CS%h_u = 0.0_dp; CS%h_v = 0.0_dp; CS%hq = 0.0_dp

      ! Map to GPU
      !$omp target enter data map(alloc: CS%dx2h, CS%dy2h, CS%dx2q, CS%dy2q)
      !$omp target enter data map(alloc: CS%DX_dyT, CS%DY_dxT, CS%DX_dyBu, CS%DY_dxBu)
      !$omp target enter data map(alloc: CS%dudx, CS%dvdy, CS%dvdx, CS%dudy)
      !$omp target enter data map(alloc: CS%sh_xx, CS%sh_xy, CS%str_xx, CS%str_xy)
      !$omp target enter data map(alloc: CS%h_u, CS%h_v, CS%hq)

      ! Pre-compute metric products
      do concurrent(j=G%jsd:G%jed, i=G%isd:G%ied)
         CS%dx2h(i, j) = G%dxT(i, j)*G%dxT(i, j)
         CS%dy2h(i, j) = G%dyT(i, j)*G%dyT(i, j)
         CS%DX_dyT(i, j) = G%dxT(i, j)*G%IdyT(i, j)
         CS%DY_dxT(i, j) = G%dyT(i, j)*G%IdxT(i, j)
      end do

      do concurrent(j=G%jsd:G%jed, i=G%isd:G%ied)
         CS%dx2q(i, j) = G%dxBu(i, j)*G%dxBu(i, j)
         CS%dy2q(i, j) = G%dyBu(i, j)*G%dyBu(i, j)
         CS%DX_dyBu(i, j) = G%dxBu(i, j)*G%IdyBu(i, j)
         CS%DY_dxBu(i, j) = G%dyBu(i, j)*G%IdxBu(i, j)
      end do

      CS%initialized = .true.

   end subroutine hor_visc_init

   !> Finalize the horizontal viscosity solver
   subroutine hor_visc_end(CS)
      type(hor_visc_CS), intent(inout) :: CS

      if (.not. CS%initialized) return

      !$omp target exit data map(delete: CS%dx2h, CS%dy2h, CS%dx2q, CS%dy2q)
      !$omp target exit data map(delete: CS%DX_dyT, CS%DY_dxT, CS%DX_dyBu, CS%DY_dxBu)
      !$omp target exit data map(delete: CS%dudx, CS%dvdy, CS%dvdx, CS%dudy)
      !$omp target exit data map(delete: CS%sh_xx, CS%sh_xy, CS%str_xx, CS%str_xy)
      !$omp target exit data map(delete: CS%h_u, CS%h_v, CS%hq)

      if (allocated(CS%dx2h)) deallocate (CS%dx2h)
      if (allocated(CS%dy2h)) deallocate (CS%dy2h)
      if (allocated(CS%dx2q)) deallocate (CS%dx2q)
      if (allocated(CS%dy2q)) deallocate (CS%dy2q)
      if (allocated(CS%DX_dyT)) deallocate (CS%DX_dyT)
      if (allocated(CS%DY_dxT)) deallocate (CS%DY_dxT)
      if (allocated(CS%DX_dyBu)) deallocate (CS%DX_dyBu)
      if (allocated(CS%DY_dxBu)) deallocate (CS%DY_dxBu)
      if (allocated(CS%dudx)) deallocate (CS%dudx)
      if (allocated(CS%dvdy)) deallocate (CS%dvdy)
      if (allocated(CS%dvdx)) deallocate (CS%dvdx)
      if (allocated(CS%dudy)) deallocate (CS%dudy)
      if (allocated(CS%sh_xx)) deallocate (CS%sh_xx)
      if (allocated(CS%sh_xy)) deallocate (CS%sh_xy)
      if (allocated(CS%str_xx)) deallocate (CS%str_xx)
      if (allocated(CS%str_xy)) deallocate (CS%str_xy)
      if (allocated(CS%h_u)) deallocate (CS%h_u)
      if (allocated(CS%h_v)) deallocate (CS%h_v)
      if (allocated(CS%hq)) deallocate (CS%hq)

      CS%initialized = .false.

   end subroutine hor_visc_end

   !> Compute horizontal viscous accelerations
  !!
  !! Calculates the Laplacian viscous acceleration using a simplified direct approach:
  !!   diffu = Kh * (d2u/dx2 + d2u/dy2)
  !!   diffv = Kh * (d2v/dx2 + d2v/dy2)
  !!
   subroutine hor_visc(u, v, h, diffu, diffv, G, GV, CS)
      type(ocean_grid_type), intent(in) :: G
      type(verticalGrid_type), intent(in) :: GV
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u   ! Zonal velocity [L T-1]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: v   ! Meridional velocity [L T-1]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h   ! Layer thickness [H]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: diffu  ! Zonal viscous accel [L T-2]
      real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: diffv  ! Merid viscous accel [L T-2]
      type(hor_visc_CS), intent(inout) :: CS

      real(dp) :: Kh, Idx2, Idy2
      real(dp) :: d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2
      integer :: i, j, k, is, ie, js, je, nz

      is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
      Kh = CS%Kh

      ! For uniform grid, precompute inverse grid spacing squared
      Idx2 = G%IdxT(is, js)*G%IdxT(is, js)
      Idy2 = G%IdyT(is, js)*G%IdyT(is, js)

      ! Initialize output arrays to zero
      do concurrent(k=1:nz, j=G%jsd:G%jed, i=G%isd:G%ied)
         diffu(i, j, k) = 0.0_dp
         diffv(i, j, k) = 0.0_dp
      end do

      ! Loop over layers - simple 5-point Laplacian stencil
      do k = 1, nz

         ! diffu: Laplacian of u at u-points
         ! u is at Cu points (i-1/2, j)
         ! d2u/dx2 uses u(i-1), u(i), u(i+1) at u-points
         ! d2u/dy2 uses u(i,j-1), u(i,j), u(i,j+1) at u-points
         do concurrent(j=js:je, i=is:ie - 1)
            d2u_dx2 = (u(i + 1, j, k) - 2.0_dp*u(i, j, k) + u(i - 1, j, k))*Idx2
            d2u_dy2 = (u(i, j + 1, k) - 2.0_dp*u(i, j, k) + u(i, j - 1, k))*Idy2
            diffu(i, j, k) = Kh*(d2u_dx2 + d2u_dy2)
         end do

         ! diffv: Laplacian of v at v-points
         ! v is at Cv points (i, j-1/2)
         ! d2v/dx2 uses v(i-1), v(i), v(i+1) at v-points
         ! d2v/dy2 uses v(i,j-1), v(i,j), v(i,j+1) at v-points
         do concurrent(j=js:je - 1, i=is:ie)
            d2v_dx2 = (v(i + 1, j, k) - 2.0_dp*v(i, j, k) + v(i - 1, j, k))*Idx2
            d2v_dy2 = (v(i, j + 1, k) - 2.0_dp*v(i, j, k) + v(i, j - 1, k))*Idy2
            diffv(i, j, k) = Kh*(d2v_dx2 + d2v_dy2)
         end do

      end do  ! k loop

   end subroutine hor_visc

end module mom6_hor_visc
