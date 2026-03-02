!> MOM6 Horizontal Viscosity Module
!!
!! Applies horizontal viscosity to momentum with realistic branching complexity.
!! Based on real MOM6's MOM_hor_visc.F90.
!!
!! Supported schemes (controlled by flags):
!!   - Laplacian: Simple 2nd-order viscosity (Kh)
!!   - Biharmonic: 4th-order viscosity (Ah) for scale-selective dissipation
!!   - Smagorinsky: Dynamic viscosity based on strain rate
!!   - Leith: Dynamic viscosity based on vorticity gradients (CPU-only!)
!!
!! The strain tensor is computed as:
!!   sh_xx = du/dx - dv/dy  (horizontal tension at h-points)
!!   sh_xy = dv/dx + du/dy  (shearing strain at q-points)
!!
!! Boundary conditions:
!!   no_slip:   sh_xy = (2 - mask) * (dvdx + dudy)  [doubled at boundaries]
!!   free_slip: sh_xy = mask * (dvdx + dudy)        [zero at boundaries]
!!
!! Work arrays are 3D to enable collapse(3) parallelism over the k-direction.
!!
module mom6_hor_visc_omp
    use iso_fortran_env, only: dp => real64, int64
    use mom6_types, only: ocean_grid_type, verticalGrid_type
    implicit none
    private

    public :: hor_visc_init, hor_visc, hor_visc_end
    public :: hor_visc_CS

    !> Control structure for horizontal viscosity solver
   !! Mimics the ~100 fields in real MOM6's hor_visc_CS with ~35 key fields
    type :: hor_visc_CS
        logical :: initialized = .false.

        !----- Scheme selection flags (runtime branching) -----
        logical :: Laplacian = .true.       ! Use Laplacian (harmonic) viscosity
        logical :: biharmonic = .false.     ! Use biharmonic viscosity
        logical :: Smagorinsky_Kh = .false. ! Smagorinsky for Laplacian viscosity
        logical :: Smagorinsky_Ah = .false. ! Smagorinsky for biharmonic viscosity
        logical :: Leith_Kh = .false.       ! Leith vorticity-gradient viscosity
        logical :: Leith_Ah = .false.       ! Leith for biharmonic viscosity
        logical :: compute_FrictWork = .false. ! Whether to compute friction work
        logical :: no_slip = .false.        ! No-slip boundary conditions (vs free-slip)
        logical :: bound_Kh = .false.       ! Apply stability bound to Kh
        logical :: better_bound_Kh = .false. ! Use thickness-aware Kh bounding
        logical :: bound_Ah = .false.       ! Apply stability bound to Ah
        logical :: better_bound_Ah = .false. ! Use thickness-aware Ah bounding
        logical :: use_land_mask = .true.   ! Use land mask for thickness interpolation
        logical :: modified_Leith = .false. ! Include divergence gradient in Leith

        !----- Scalar parameters -----
        real(dp) :: Kh_bg_min = 0.0_dp      ! Minimum background Kh [L2 T-1]
        real(dp) :: Ah_bg_min = 0.0_dp      ! Minimum background Ah [L4 T-1]
        real(dp) :: h_neglect = 1.0e-10_dp  ! Minimum thickness [H]
        real(dp) :: Smag_Lap_const = 0.15_dp ! Smagorinsky constant for Laplacian
        real(dp) :: Smag_bi_const = 0.06_dp  ! Smagorinsky constant for biharmonic
        real(dp) :: Leith_const = 1.0_dp    ! Leith constant

        integer :: nk = 0                   ! Number of vertical layers

        !----- Background viscosity arrays (2D, spatially varying) -----
        real(dp), allocatable :: Kh_bg_xx(:, :)   ! Background Kh at h-points [L2 T-1]
        real(dp), allocatable :: Kh_bg_xy(:, :)   ! Background Kh at q-points [L2 T-1]
        real(dp), allocatable :: Ah_bg_xx(:, :)   ! Background Ah at h-points [L4 T-1]
        real(dp), allocatable :: Ah_bg_xy(:, :)   ! Background Ah at q-points [L4 T-1]

        !----- Maximum viscosity for stability (2D) -----
        real(dp), allocatable :: Kh_Max_xx(:, :)  ! Max Kh at h-points [L2 T-1]
        real(dp), allocatable :: Kh_Max_xy(:, :)  ! Max Kh at q-points [L2 T-1]
        real(dp), allocatable :: Ah_Max_xx(:, :)  ! Max Ah at h-points [L4 T-1]
        real(dp), allocatable :: Ah_Max_xy(:, :)  ! Max Ah at q-points [L4 T-1]

        !----- Smagorinsky/Leith constants (pre-computed, 2D) -----
        real(dp), allocatable :: Laplac2_const_xx(:, :) ! Smag Lap constant at h [L2]
        real(dp), allocatable :: Laplac2_const_xy(:, :) ! Smag Lap constant at q [L2]
        real(dp), allocatable :: Biharm_const_xx(:, :)  ! Smag biharm constant at h [L4]
        real(dp), allocatable :: Biharm_const_xy(:, :)  ! Smag biharm constant at q [L4]
        real(dp), allocatable :: Laplac3_const_xx(:, :) ! Leith constant at h [L3]
        real(dp), allocatable :: Laplac3_const_xy(:, :) ! Leith constant at q [L3]
        real(dp), allocatable :: Biharm6_const_xx(:, :) ! Leith biharmonic constant at h [L6]
        real(dp), allocatable :: Biharm6_const_xy(:, :) ! Leith biharmonic constant at q [L6]

        !----- Pre-computed metric products -----
        real(dp), allocatable :: dx2h(:, :)     ! dx^2 at h-points [L2]
        real(dp), allocatable :: dy2h(:, :)     ! dy^2 at h-points [L2]
        real(dp), allocatable :: dx2q(:, :)     ! dx^2 at q-points [L2]
        real(dp), allocatable :: dy2q(:, :)     ! dy^2 at q-points [L2]
        real(dp), allocatable :: DX_dyT(:, :)   ! dx/dy ratio at h-points [nondim]
        real(dp), allocatable :: DY_dxT(:, :)   ! dy/dx ratio at h-points [nondim]
        real(dp), allocatable :: DX_dyBu(:, :)  ! dx/dy ratio at q-points [nondim]
        real(dp), allocatable :: DY_dxBu(:, :)  ! dy/dx ratio at q-points [nondim]

        !----- Biharmonic metric arrays -----
        real(dp), allocatable :: Idx2dyCu(:, :) ! 1/(dx^2*dy) at u-points [L-3]
        real(dp), allocatable :: Idxdy2u(:, :)  ! 1/(dx*dy^2) at u-points [L-3]
        real(dp), allocatable :: Idx2dyCv(:, :) ! 1/(dx^2*dy) at v-points [L-3]
        real(dp), allocatable :: Idxdy2v(:, :)  ! 1/(dx*dy^2) at v-points [L-3]

        !----- Reduction factors for numerical stability -----
        real(dp), allocatable :: reduction_xx(:, :) ! Reduction factor at h-points [nondim]
        real(dp), allocatable :: reduction_xy(:, :) ! Reduction factor at q-points [nondim]

        !----- Work arrays (3D, one entry per layer for k-parallelism) -----
        ! Velocity gradients
        real(dp), allocatable :: dudx(:, :, :)     ! du/dx at h-points [T-1]
        real(dp), allocatable :: dvdy(:, :, :)     ! dv/dy at h-points [T-1]
        real(dp), allocatable :: dvdx(:, :, :)     ! dv/dx at q-points [T-1]
        real(dp), allocatable :: dudy(:, :, :)     ! du/dy at q-points [T-1]

        ! Strain tensor
        real(dp), allocatable :: sh_xx(:, :, :)    ! Horizontal tension [T-1]
        real(dp), allocatable :: sh_xy(:, :, :)    ! Shearing strain [T-1]

        ! Stress tensor
        real(dp), allocatable :: str_xx(:, :, :)   ! Diagonal stress [H L2 T-2]
        real(dp), allocatable :: str_xy(:, :, :)   ! Off-diagonal stress [H L2 T-2]
        real(dp), allocatable :: bhstr_xx(:, :, :) ! Biharmonic-only stress at h-points [H L2 T-2]
        real(dp), allocatable :: bhstr_xy(:, :, :) ! Biharmonic-only stress at q-points [H L2 T-2]

        ! Thicknesses at staggered points
        real(dp), allocatable :: h_u(:, :, :)      ! Thickness at u-points [H]
        real(dp), allocatable :: h_v(:, :, :)      ! Thickness at v-points [H]
        real(dp), allocatable :: hq(:, :, :)       ! Thickness at q-points [H]

        ! Dynamic viscosity arrays
        real(dp), allocatable :: Kh(:, :, :)       ! Laplacian viscosity [L2 T-1]
        real(dp), allocatable :: Ah(:, :, :)       ! Biharmonic viscosity [L4 T-1]
        real(dp), allocatable :: Shear_mag(:, :, :) ! Strain rate magnitude [T-1]

        ! Biharmonic Laplacian of velocity
        real(dp), allocatable :: Del2u(:, :, :)    ! Laplacian of u [L T-1] (for biharmonic)
        real(dp), allocatable :: Del2v(:, :, :)    ! Laplacian of v [L T-1] (for biharmonic)

        ! Stability bounding arrays
        real(dp), allocatable :: hrat_min(:, :, :)     ! Thickness ratio for bounding [nondim]
        real(dp), allocatable :: visc_bound_rem(:, :, :) ! Remaining viscosity budget [nondim]

        ! Leith vorticity arrays
        real(dp), allocatable :: vort_xy(:, :, :)      ! Vorticity at q-points [T-1]
        real(dp), allocatable :: vort_xy_dx(:, :, :)   ! d(vort)/dx [L-1 T-1]
        real(dp), allocatable :: vort_xy_dy(:, :, :)   ! d(vort)/dy [L-1 T-1]
        real(dp), allocatable :: grad_vort_mag_h(:, :, :) ! |grad(vort)| at h-points [L-1 T-1]
        real(dp), allocatable :: grad_vort_mag_q(:, :, :) ! |grad(vort)| at q-points [L-1 T-1]
        real(dp), allocatable :: vert_vort_mag(:, :, :)   ! Combined vorticity measure [L-1 T-1]
        real(dp), allocatable :: Del2vort_q(:, :, :)      ! Laplacian of vorticity at q-points [L-2 T-1]

        ! Modified Leith divergence arrays
        real(dp), allocatable :: div_xx(:, :, :)       ! Divergence at h-points [T-1]
        real(dp), allocatable :: div_xx_dx(:, :, :)    ! d(div)/dx [L-1 T-1]
        real(dp), allocatable :: div_xx_dy(:, :, :)    ! d(div)/dy [L-1 T-1]
        real(dp), allocatable :: grad_div_mag_h(:, :, :)  ! |grad(div)| at h-points [L-1 T-1]
        real(dp), allocatable :: grad_div_mag_q(:, :, :)  ! |grad(div)| at q-points [L-1 T-1]

        integer(int64) :: nbytes = 0  ! Total bytes allocated for GPU arrays
    end type hor_visc_CS

    real(dp), parameter :: inv_PI3 = 1.0_dp/(3.14159265358979_dp**3)
    real(dp), parameter :: inv_PI6 = 1.0_dp/(3.14159265358979_dp**6)

contains

    !> Initialize the horizontal viscosity solver
    subroutine hor_visc_init(CS, G, GV, Kh, Ah, Laplacian, biharmonic, &
                             Smagorinsky, Leith, no_slip, better_bound)
        type(hor_visc_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), intent(in), optional :: Kh          ! Background Laplacian viscosity [m2/s]
        real(dp), intent(in), optional :: Ah          ! Background biharmonic viscosity [m4/s]
        logical, intent(in), optional :: Laplacian    ! Enable Laplacian viscosity
        logical, intent(in), optional :: biharmonic   ! Enable biharmonic viscosity
        logical, intent(in), optional :: Smagorinsky  ! Enable Smagorinsky dynamic viscosity
        logical, intent(in), optional :: Leith        ! Enable Leith vorticity viscosity
        logical, intent(in), optional :: no_slip      ! Use no-slip boundary conditions
        logical, intent(in), optional :: better_bound ! Use thickness-aware bounding

        real(dp) :: Kh_val, Ah_val, dx_m, grid_sp_h2, grid_sp_h3, grid_sp_h4
        integer :: i, j

        ! Set scheme flags
        if (present(Laplacian)) CS%Laplacian = Laplacian
        if (present(biharmonic)) CS%biharmonic = biharmonic
        if (present(Smagorinsky)) then
            CS%Smagorinsky_Kh = Smagorinsky .and. CS%Laplacian
            CS%Smagorinsky_Ah = Smagorinsky .and. CS%biharmonic
        end if
        if (present(Leith)) then
            CS%Leith_Kh = Leith
            if (Leith .and. CS%biharmonic) CS%Leith_Ah = .true.
        end if
        if (present(no_slip)) CS%no_slip = no_slip
        if (present(better_bound)) then
            CS%better_bound_Kh = better_bound
            CS%better_bound_Ah = better_bound
            CS%bound_Kh = better_bound
            CS%bound_Ah = better_bound
        end if

        ! Set viscosity parameters
        Kh_val = 100.0_dp
        if (present(Kh)) Kh_val = Kh
        Ah_val = 1.0e10_dp  ! Default biharmonic viscosity [m4/s]
        if (present(Ah)) Ah_val = Ah

        CS%Kh_bg_min = 0.0_dp
        CS%h_neglect = max(GV%Angstrom_H, 1.0e-3_dp)
        CS%nk = GV%ke

        ! Allocate all arrays
        call allocate_hor_visc_arrays(CS, G, GV%ke)

        ! Pre-compute metrics
        dx_m = G%dx
        do j=G%jsd,G%jed
            do i=G%isd,G%ied
            ! Grid spacing squared
            CS%dx2h(i, j) = G%dxT(i, j)*G%dxT(i, j)
            CS%dy2h(i, j) = G%dyT(i, j)*G%dyT(i, j)
            CS%dx2q(i, j) = G%dxBu(i, j)*G%dxBu(i, j)
            CS%dy2q(i, j) = G%dyBu(i, j)*G%dyBu(i, j)

            ! Metric ratios
            CS%DX_dyT(i, j) = G%dxT(i, j)*G%IdyT(i, j)
            CS%DY_dxT(i, j) = G%dyT(i, j)*G%IdxT(i, j)
            CS%DX_dyBu(i, j) = G%dxBu(i, j)*G%IdyBu(i, j)
            CS%DY_dxBu(i, j) = G%dyBu(i, j)*G%IdxBu(i, j)

            ! Biharmonic metrics
            CS%Idx2dyCu(i, j) = G%IdxCu(i, j)*G%IdxCu(i, j)*G%IdyCu(i, j)
            CS%Idxdy2u(i, j) = G%IdxCu(i, j)*G%IdyCu(i, j)*G%IdyCu(i, j)
            CS%Idx2dyCv(i, j) = G%IdxCv(i, j)*G%IdxCv(i, j)*G%IdyCv(i, j)
            CS%Idxdy2v(i, j) = G%IdxCv(i, j)*G%IdyCv(i, j)*G%IdyCv(i, j)

            ! Reduction factors (1.0 for now, could be used for partial cells)
            CS%reduction_xx(i, j) = 1.0_dp
            CS%reduction_xy(i, j) = 1.0_dp

            ! Grid spacing for Smagorinsky/Leith constants
            grid_sp_h2 = G%areaT(i, j)  ! dx*dy ~ dx^2
            grid_sp_h3 = sqrt(grid_sp_h2)*grid_sp_h2  ! ~ dx^3
            grid_sp_h4 = grid_sp_h2*grid_sp_h2  ! ~ dx^4

            ! Background viscosities (spatially uniform for now)
            CS%Kh_bg_xx(i, j) = Kh_val
            CS%Kh_bg_xy(i, j) = Kh_val
            CS%Ah_bg_xx(i, j) = Ah_val
            CS%Ah_bg_xy(i, j) = Ah_val

            ! Maximum viscosity for stability (CFL-based)
            ! Kh_max ~ dx^2 / (4*dt), but we use a fixed fraction of grid spacing
            CS%Kh_Max_xx(i, j) = 0.25_dp*grid_sp_h2/1.0_dp  ! Assumes dt=1s for max bound
            CS%Kh_Max_xy(i, j) = 0.25_dp*grid_sp_h2/1.0_dp
            CS%Ah_Max_xx(i, j) = 0.0625_dp*grid_sp_h4/1.0_dp
            CS%Ah_Max_xy(i, j) = 0.0625_dp*grid_sp_h4/1.0_dp

            ! Smagorinsky constants: C_smag^2 * dx^2 for Laplacian
            CS%Laplac2_const_xx(i, j) = CS%Smag_Lap_const*CS%Smag_Lap_const*grid_sp_h2
            CS%Laplac2_const_xy(i, j) = CS%Smag_Lap_const*CS%Smag_Lap_const*grid_sp_h2

            ! Biharmonic Smagorinsky: C_smag^2 * dx^4
            CS%Biharm_const_xx(i, j) = CS%Smag_bi_const*CS%Smag_bi_const*grid_sp_h4
            CS%Biharm_const_xy(i, j) = CS%Smag_bi_const*CS%Smag_bi_const*grid_sp_h4

            ! Leith constants: C_leith * dx^3
            CS%Laplac3_const_xx(i, j) = CS%Leith_const*grid_sp_h3
            CS%Laplac3_const_xy(i, j) = CS%Leith_const*grid_sp_h3

            ! Leith biharmonic constants: C_leith * dx^6
            CS%Biharm6_const_xx(i, j) = CS%Leith_const*grid_sp_h3*grid_sp_h3
            CS%Biharm6_const_xy(i, j) = CS%Leith_const*grid_sp_h3*grid_sp_h3
            end do
        end do

        ! Initialize work arrays to zero
        CS%dudx = 0.0_dp; CS%dvdy = 0.0_dp; CS%dvdx = 0.0_dp; CS%dudy = 0.0_dp
        CS%sh_xx = 0.0_dp; CS%sh_xy = 0.0_dp
        CS%str_xx = 0.0_dp; CS%str_xy = 0.0_dp
        CS%h_u = 0.0_dp; CS%h_v = 0.0_dp; CS%hq = 0.0_dp
        CS%Kh = 0.0_dp; CS%Ah = 0.0_dp; CS%Shear_mag = 0.0_dp
        CS%Del2u = 0.0_dp; CS%Del2v = 0.0_dp
        CS%hrat_min = 1.0_dp; CS%visc_bound_rem = 1.0_dp
        CS%vort_xy = 0.0_dp; CS%vort_xy_dx = 0.0_dp; CS%vort_xy_dy = 0.0_dp
        CS%grad_vort_mag_h = 0.0_dp; CS%grad_vort_mag_q = 0.0_dp; CS%vert_vort_mag = 0.0_dp
        CS%div_xx = 0.0_dp; CS%div_xx_dx = 0.0_dp; CS%div_xx_dy = 0.0_dp
        CS%grad_div_mag_h = 0.0_dp; CS%grad_div_mag_q = 0.0_dp
        CS%bhstr_xx = 0.0_dp; CS%bhstr_xy = 0.0_dp; CS%Del2vort_q = 0.0_dp

        !$omp target enter data map(to: CS)
        !$omp target enter data map(to: CS%Kh_bg_xx, CS%Kh_bg_xy, CS%Ah_bg_xx, CS%Ah_bg_xy)
        !$omp target enter data map(to: CS%Kh_Max_xx, CS%Kh_Max_xy, CS%Ah_Max_xx, CS%Ah_Max_xy)
        !$omp target enter data map(to: CS%Laplac2_const_xx, CS%Laplac2_const_xy)
        !$omp target enter data map(to: CS%Biharm_const_xx, CS%Biharm_const_xy)
        !$omp target enter data map(to: CS%Laplac3_const_xx, CS%Laplac3_const_xy)
        !$omp target enter data map(to: CS%Biharm6_const_xx, CS%Biharm6_const_xy)
        !$omp target enter data map(to: CS%dx2h, CS%dy2h, CS%dx2q, CS%dy2q)
        !$omp target enter data map(to: CS%DX_dyT, CS%DY_dxT, CS%DX_dyBu, CS%DY_dxBu)
        !$omp target enter data map(to: CS%Idx2dyCu, CS%Idxdy2u, CS%Idx2dyCv, CS%Idxdy2v)
        !$omp target enter data map(to: CS%reduction_xx, CS%reduction_xy)
        !$omp target enter data map(alloc: CS%dudx, CS%dvdy, CS%dvdx, CS%dudy)
        !$omp target enter data map(alloc: CS%sh_xx, CS%sh_xy, CS%str_xx, CS%str_xy)
        !$omp target enter data map(alloc: CS%bhstr_xx, CS%bhstr_xy)
        !$omp target enter data map(alloc: CS%h_u, CS%h_v, CS%hq)
        !$omp target enter data map(alloc: CS%Kh, CS%Ah, CS%Shear_mag)
        !$omp target enter data map(alloc: CS%Del2u, CS%Del2v)
        !$omp target enter data map(alloc: CS%hrat_min, CS%visc_bound_rem)
        !$omp target enter data map(alloc: CS%vort_xy, CS%vort_xy_dx, CS%vort_xy_dy)
        !$omp target enter data map(alloc: CS%grad_vort_mag_h, CS%grad_vort_mag_q)
        !$omp target enter data map(alloc: CS%vert_vort_mag, CS%Del2vort_q)
        !$omp target enter data map(alloc: CS%div_xx, CS%div_xx_dx, CS%div_xx_dy)
        !$omp target enter data map(alloc: CS%grad_div_mag_h, CS%grad_div_mag_q)

        CS%initialized = .true.

    end subroutine hor_visc_init

    !> Allocate all arrays in the control structure
    subroutine allocate_hor_visc_arrays(CS, G, nk)
        type(hor_visc_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in) :: nk

        integer :: isd, ied, jsd, jed
        isd = G%isd; ied = G%ied; jsd = G%jsd; jed = G%jed

        ! Background viscosity arrays
        allocate (CS%Kh_bg_xx(isd:ied, jsd:jed))
        allocate (CS%Kh_bg_xy(isd:ied, jsd:jed))
        allocate (CS%Ah_bg_xx(isd:ied, jsd:jed))
        allocate (CS%Ah_bg_xy(isd:ied, jsd:jed))

        ! Maximum viscosity arrays
        allocate (CS%Kh_Max_xx(isd:ied, jsd:jed))
        allocate (CS%Kh_Max_xy(isd:ied, jsd:jed))
        allocate (CS%Ah_Max_xx(isd:ied, jsd:jed))
        allocate (CS%Ah_Max_xy(isd:ied, jsd:jed))

        ! Smagorinsky/Leith constants
        allocate (CS%Laplac2_const_xx(isd:ied, jsd:jed))
        allocate (CS%Laplac2_const_xy(isd:ied, jsd:jed))
        allocate (CS%Biharm_const_xx(isd:ied, jsd:jed))
        allocate (CS%Biharm_const_xy(isd:ied, jsd:jed))
        allocate (CS%Laplac3_const_xx(isd:ied, jsd:jed))
        allocate (CS%Laplac3_const_xy(isd:ied, jsd:jed))
        allocate (CS%Biharm6_const_xx(isd:ied, jsd:jed))
        allocate (CS%Biharm6_const_xy(isd:ied, jsd:jed))

        ! Metric arrays
        allocate (CS%dx2h(isd:ied, jsd:jed))
        allocate (CS%dy2h(isd:ied, jsd:jed))
        allocate (CS%dx2q(isd:ied, jsd:jed))
        allocate (CS%dy2q(isd:ied, jsd:jed))
        allocate (CS%DX_dyT(isd:ied, jsd:jed))
        allocate (CS%DY_dxT(isd:ied, jsd:jed))
        allocate (CS%DX_dyBu(isd:ied, jsd:jed))
        allocate (CS%DY_dxBu(isd:ied, jsd:jed))

        ! Biharmonic metric arrays
        allocate (CS%Idx2dyCu(isd:ied, jsd:jed))
        allocate (CS%Idxdy2u(isd:ied, jsd:jed))
        allocate (CS%Idx2dyCv(isd:ied, jsd:jed))
        allocate (CS%Idxdy2v(isd:ied, jsd:jed))

        ! Reduction factors
        allocate (CS%reduction_xx(isd:ied, jsd:jed))
        allocate (CS%reduction_xy(isd:ied, jsd:jed))

        ! Work arrays (3D for k-parallelism)
        allocate (CS%dudx(isd:ied, jsd:jed, nk))
        allocate (CS%dvdy(isd:ied, jsd:jed, nk))
        allocate (CS%dvdx(isd:ied, jsd:jed, nk))
        allocate (CS%dudy(isd:ied, jsd:jed, nk))
        allocate (CS%sh_xx(isd:ied, jsd:jed, nk))
        allocate (CS%sh_xy(isd:ied, jsd:jed, nk))
        allocate (CS%str_xx(isd:ied, jsd:jed, nk))
        allocate (CS%str_xy(isd:ied, jsd:jed, nk))
        allocate (CS%bhstr_xx(isd:ied, jsd:jed, nk))
        allocate (CS%bhstr_xy(isd:ied, jsd:jed, nk))
        allocate (CS%h_u(isd:ied, jsd:jed, nk))
        allocate (CS%h_v(isd:ied, jsd:jed, nk))
        allocate (CS%hq(isd:ied, jsd:jed, nk))

        ! Dynamic viscosity arrays
        allocate (CS%Kh(isd:ied, jsd:jed, nk))
        allocate (CS%Ah(isd:ied, jsd:jed, nk))
        allocate (CS%Shear_mag(isd:ied, jsd:jed, nk))

        ! Biharmonic arrays
        allocate (CS%Del2u(isd:ied, jsd:jed, nk))
        allocate (CS%Del2v(isd:ied, jsd:jed, nk))

        ! Stability bounding arrays
        allocate (CS%hrat_min(isd:ied, jsd:jed, nk))
        allocate (CS%visc_bound_rem(isd:ied, jsd:jed, nk))

        ! Leith arrays
        allocate (CS%vort_xy(isd:ied, jsd:jed, nk))
        allocate (CS%vort_xy_dx(isd:ied, jsd:jed, nk))
        allocate (CS%vort_xy_dy(isd:ied, jsd:jed, nk))
        allocate (CS%grad_vort_mag_h(isd:ied, jsd:jed, nk))
        allocate (CS%grad_vort_mag_q(isd:ied, jsd:jed, nk))
        allocate (CS%vert_vort_mag(isd:ied, jsd:jed, nk))
        allocate (CS%Del2vort_q(isd:ied, jsd:jed, nk))

        ! Modified Leith arrays
        allocate (CS%div_xx(isd:ied, jsd:jed, nk))
        allocate (CS%div_xx_dx(isd:ied, jsd:jed, nk))
        allocate (CS%div_xx_dy(isd:ied, jsd:jed, nk))
        allocate (CS%grad_div_mag_h(isd:ied, jsd:jed, nk))
        allocate (CS%grad_div_mag_q(isd:ied, jsd:jed, nk))

        ! Compute total bytes: 30 2D arrays + 32 3D arrays of real(dp)
        CS%nbytes = 30_int64 * int(ied - isd + 1, int64) * int(jed - jsd + 1, int64) * 8_int64 + &
                    32_int64 * int(ied - isd + 1, int64) * int(jed - jsd + 1, int64) * int(nk, int64) * 8_int64

    end subroutine allocate_hor_visc_arrays

    !> Finalize the horizontal viscosity solver
    subroutine hor_visc_end(CS)
        type(hor_visc_CS), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! Remove data from device before deallocating
        !$omp target exit data map(delete: CS%Kh_bg_xx, CS%Kh_bg_xy, CS%Ah_bg_xx, CS%Ah_bg_xy)
        !$omp target exit data map(delete: CS%Kh_Max_xx, CS%Kh_Max_xy, CS%Ah_Max_xx, CS%Ah_Max_xy)
        !$omp target exit data map(delete: CS%Laplac2_const_xx, CS%Laplac2_const_xy)
        !$omp target exit data map(delete: CS%Biharm_const_xx, CS%Biharm_const_xy)
        !$omp target exit data map(delete: CS%Laplac3_const_xx, CS%Laplac3_const_xy)
        !$omp target exit data map(delete: CS%Biharm6_const_xx, CS%Biharm6_const_xy)
        !$omp target exit data map(delete: CS%dx2h, CS%dy2h, CS%dx2q, CS%dy2q)
        !$omp target exit data map(delete: CS%DX_dyT, CS%DY_dxT, CS%DX_dyBu, CS%DY_dxBu)
        !$omp target exit data map(delete: CS%Idx2dyCu, CS%Idxdy2u, CS%Idx2dyCv, CS%Idxdy2v)
        !$omp target exit data map(delete: CS%reduction_xx, CS%reduction_xy)
        !$omp target exit data map(delete: CS%dudx, CS%dvdy, CS%dvdx, CS%dudy)
        !$omp target exit data map(delete: CS%sh_xx, CS%sh_xy, CS%str_xx, CS%str_xy)
        !$omp target exit data map(delete: CS%bhstr_xx, CS%bhstr_xy)
        !$omp target exit data map(delete: CS%h_u, CS%h_v, CS%hq)
        !$omp target exit data map(delete: CS%Kh, CS%Ah, CS%Shear_mag)
        !$omp target exit data map(delete: CS%Del2u, CS%Del2v)
        !$omp target exit data map(delete: CS%hrat_min, CS%visc_bound_rem)
        !$omp target exit data map(delete: CS%vort_xy, CS%vort_xy_dx, CS%vort_xy_dy)
        !$omp target exit data map(delete: CS%grad_vort_mag_h, CS%grad_vort_mag_q)
        !$omp target exit data map(delete: CS%vert_vort_mag, CS%Del2vort_q)
        !$omp target exit data map(delete: CS%div_xx, CS%div_xx_dx, CS%div_xx_dy)
        !$omp target exit data map(delete: CS%grad_div_mag_h, CS%grad_div_mag_q)
        !$omp target exit data map(delete: CS)

        ! Deallocate arrays
        if (allocated(CS%Kh_bg_xx)) deallocate (CS%Kh_bg_xx)
        if (allocated(CS%Kh_bg_xy)) deallocate (CS%Kh_bg_xy)
        if (allocated(CS%Ah_bg_xx)) deallocate (CS%Ah_bg_xx)
        if (allocated(CS%Ah_bg_xy)) deallocate (CS%Ah_bg_xy)
        if (allocated(CS%Kh_Max_xx)) deallocate (CS%Kh_Max_xx)
        if (allocated(CS%Kh_Max_xy)) deallocate (CS%Kh_Max_xy)
        if (allocated(CS%Ah_Max_xx)) deallocate (CS%Ah_Max_xx)
        if (allocated(CS%Ah_Max_xy)) deallocate (CS%Ah_Max_xy)
        if (allocated(CS%Laplac2_const_xx)) deallocate (CS%Laplac2_const_xx)
        if (allocated(CS%Laplac2_const_xy)) deallocate (CS%Laplac2_const_xy)
        if (allocated(CS%Biharm_const_xx)) deallocate (CS%Biharm_const_xx)
        if (allocated(CS%Biharm_const_xy)) deallocate (CS%Biharm_const_xy)
        if (allocated(CS%Laplac3_const_xx)) deallocate (CS%Laplac3_const_xx)
        if (allocated(CS%Laplac3_const_xy)) deallocate (CS%Laplac3_const_xy)
        if (allocated(CS%Biharm6_const_xx)) deallocate (CS%Biharm6_const_xx)
        if (allocated(CS%Biharm6_const_xy)) deallocate (CS%Biharm6_const_xy)
        if (allocated(CS%dx2h)) deallocate (CS%dx2h)
        if (allocated(CS%dy2h)) deallocate (CS%dy2h)
        if (allocated(CS%dx2q)) deallocate (CS%dx2q)
        if (allocated(CS%dy2q)) deallocate (CS%dy2q)
        if (allocated(CS%DX_dyT)) deallocate (CS%DX_dyT)
        if (allocated(CS%DY_dxT)) deallocate (CS%DY_dxT)
        if (allocated(CS%DX_dyBu)) deallocate (CS%DX_dyBu)
        if (allocated(CS%DY_dxBu)) deallocate (CS%DY_dxBu)
        if (allocated(CS%Idx2dyCu)) deallocate (CS%Idx2dyCu)
        if (allocated(CS%Idxdy2u)) deallocate (CS%Idxdy2u)
        if (allocated(CS%Idx2dyCv)) deallocate (CS%Idx2dyCv)
        if (allocated(CS%Idxdy2v)) deallocate (CS%Idxdy2v)
        if (allocated(CS%reduction_xx)) deallocate (CS%reduction_xx)
        if (allocated(CS%reduction_xy)) deallocate (CS%reduction_xy)
        if (allocated(CS%dudx)) deallocate (CS%dudx)
        if (allocated(CS%dvdy)) deallocate (CS%dvdy)
        if (allocated(CS%dvdx)) deallocate (CS%dvdx)
        if (allocated(CS%dudy)) deallocate (CS%dudy)
        if (allocated(CS%sh_xx)) deallocate (CS%sh_xx)
        if (allocated(CS%sh_xy)) deallocate (CS%sh_xy)
        if (allocated(CS%str_xx)) deallocate (CS%str_xx)
        if (allocated(CS%str_xy)) deallocate (CS%str_xy)
        if (allocated(CS%bhstr_xx)) deallocate (CS%bhstr_xx)
        if (allocated(CS%bhstr_xy)) deallocate (CS%bhstr_xy)
        if (allocated(CS%h_u)) deallocate (CS%h_u)
        if (allocated(CS%h_v)) deallocate (CS%h_v)
        if (allocated(CS%hq)) deallocate (CS%hq)
        if (allocated(CS%Kh)) deallocate (CS%Kh)
        if (allocated(CS%Ah)) deallocate (CS%Ah)
        if (allocated(CS%Shear_mag)) deallocate (CS%Shear_mag)
        if (allocated(CS%Del2u)) deallocate (CS%Del2u)
        if (allocated(CS%Del2v)) deallocate (CS%Del2v)
        if (allocated(CS%hrat_min)) deallocate (CS%hrat_min)
        if (allocated(CS%visc_bound_rem)) deallocate (CS%visc_bound_rem)
        if (allocated(CS%vort_xy)) deallocate (CS%vort_xy)
        if (allocated(CS%vort_xy_dx)) deallocate (CS%vort_xy_dx)
        if (allocated(CS%vort_xy_dy)) deallocate (CS%vort_xy_dy)
        if (allocated(CS%grad_vort_mag_h)) deallocate (CS%grad_vort_mag_h)
        if (allocated(CS%grad_vort_mag_q)) deallocate (CS%grad_vort_mag_q)
        if (allocated(CS%vert_vort_mag)) deallocate (CS%vert_vort_mag)
        if (allocated(CS%Del2vort_q)) deallocate (CS%Del2vort_q)
        if (allocated(CS%div_xx)) deallocate (CS%div_xx)
        if (allocated(CS%div_xx_dx)) deallocate (CS%div_xx_dx)
        if (allocated(CS%div_xx_dy)) deallocate (CS%div_xx_dy)
        if (allocated(CS%grad_div_mag_h)) deallocate (CS%grad_div_mag_h)
        if (allocated(CS%grad_div_mag_q)) deallocate (CS%grad_div_mag_q)

        CS%initialized = .false.

    end subroutine hor_visc_end

    !> Compute horizontal viscous accelerations
   !!
   !! Computes horizontal viscous accelerations with:
   !! - Multiple viscosity schemes: Laplacian, biharmonic, Smagorinsky, Leith
   !! - Stability bounding with thickness-aware limits
   !! - Full k-parallelism via collapse(3) on all parallel loops
   !!
    subroutine hor_visc(u, v, h, diffu, diffv, G, GV, CS, uh, vh, FrictWork)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u    ! Zonal velocity [L T-1]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: v    ! Meridional velocity [L T-1]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h    ! Layer thickness [H]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: diffu ! Zonal viscous accel [L T-2]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: diffv ! Merid viscous accel [L T-2]
        type(hor_visc_CS), intent(inout) :: CS
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in), optional :: uh ! Volume transport [H L2 T-1]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in), optional :: vh ! Volume transport [H L2 T-1]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out), optional :: FrictWork ! Energy dissipation [H L2 T-3]

        real(dp) :: h_neglect, h_min, Kh_max_here, sh_xx_sq, sh_xy_sq
        real(dp) :: d_del2u, d_del2v, d_str, DY_dxBu_loc, DX_dyBu_loc
        real(dp) :: Del2vort_h
        integer :: i, j, k, is, ie, js, je, nz, Isq, Ieq, Jsq, Jeq

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        Isq = is - 1; Ieq = ie; Jsq = js - 1; Jeq = je
        h_neglect = CS%h_neglect

        !$omp target data map(to: u, v, h) map(from: diffu, diffv)

        ! Initialize output arrays to zero
        !$omp target teams distribute parallel do collapse(3)
        do k=1,nz
            do j=G%jsd,G%jed
                do i=G%isd,G%ied
                    diffu(i, j, k) = 0.0_dp
                    diffv(i, j, k) = 0.0_dp
                end do
            end do
        end do

        !=====================================================================
        ! STEP 1: Calculate velocity gradients
        !=====================================================================
        ! du/dx and dv/dy at h-points (for horizontal tension)
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j=Jsq,Jeq + 1
                do i=Isq,Ieq + 1
                    CS%dudx(i, j, k) = CS%DY_dxT(i, j)*((G%IdyCu(i, j)*u(i, j, k)) - &
                                                         (G%IdyCu(i - 1, j)*u(i - 1, j, k)))
                    CS%dvdy(i, j, k) = CS%DX_dyT(i, j)*((G%IdxCv(i, j)*v(i, j, k)) - &
                                                         (G%IdxCv(i, j - 1)*v(i, j - 1, k)))
                end do
            end do
        end do

        ! dv/dx and du/dy at q-points (for shearing strain)
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do J=Jsq,Jeq
                do I=Isq,Ieq
                    CS%dvdx(I, J, k) = CS%DY_dxBu(I, J)*((v(i + 1, J, k)*G%IdyCv(i + 1, J)) - &
                                                          (v(i, J, k)*G%IdyCv(i, J)))
                    CS%dudy(I, J, k) = CS%DX_dyBu(I, J)*((u(I, j + 1, k)*G%IdxCu(I, j + 1)) - &
                                                          (u(I, j, k)*G%IdxCu(I, j)))
                end do
            end do
        end do

        !=====================================================================
        ! STEP 2: Calculate strain tensor
        !=====================================================================
        ! Horizontal tension (sh_xx) at h-points
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j=Jsq,Jeq + 1
                do i=Isq,Ieq + 1
                    CS%sh_xx(i, j, k) = CS%dudx(i, j, k) - CS%dvdy(i, j, k)
                end do
            end do
        end do

        ! Shearing strain (sh_xy) at q-points with boundary condition branching
        if (CS%no_slip) then
            ! No-slip: double the strain at boundaries (mask=0 means land)
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%sh_xy(I, J, k) = (2.0_dp - G%mask2dBu(I, J))*(CS%dvdx(I, J, k) + CS%dudy(I, J, k))
                    end do
                end do
            end do
        else
            ! Free-slip: zero strain at boundaries
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%sh_xy(I, J, k) = G%mask2dBu(I, J)*(CS%dvdx(I, J, k) + CS%dudy(I, J, k))
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 3: Interpolate thickness to velocity points
        !=====================================================================
        if (CS%use_land_mask) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=js - 1,je + 1
                    do I=Isq,Ieq
                        CS%h_u(I, j, k) = 0.5_dp*(G%mask2dT(i, j)*h(i, j, k) + G%mask2dT(i + 1, j)*h(i + 1, j, k))
                    end do
                end do
            end do
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do i=is - 1,ie + 1
                        CS%h_v(i, J, k) = 0.5_dp*(G%mask2dT(i, j)*h(i, j, k) + G%mask2dT(i, j + 1)*h(i, j + 1, k))
                    end do
                end do
            end do
        else
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=js - 1,je + 1
                    do I=Isq,Ieq
                        CS%h_u(I, j, k) = 0.5_dp*(h(i, j, k) + h(i + 1, j, k))
                    end do
                end do
            end do
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do i=is - 1,ie + 1
                        CS%h_v(i, J, k) = 0.5_dp*(h(i, j, k) + h(i, j + 1, k))
                    end do
                end do
            end do
        end if

        ! Thickness at q-points (average of 4 neighboring h-points)
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do J=Jsq,Jeq
                do I=Isq,Ieq
                    CS%hq(I, J, k) = 0.25_dp*((h(i, j, k) + h(i + 1, j + 1, k)) + &
                                               (h(i + 1, j, k) + h(i, j + 1, k)))
                end do
            end do
        end do

        !=====================================================================
        ! STEP 4: Compute biharmonic Del2u, Del2v if needed
        !=====================================================================
        if (CS%biharmonic) then
            ! Del2u = x.Div(Grad u) at u-points
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=js,je
                    do I=Isq,Ieq
                        CS%Del2u(I, j, k) = CS%Idx2dyCu(I, j)*((CS%dx2q(I, j)*CS%sh_xy(I, j, k)) - &
                                                                (CS%dx2q(I, j - 1)*CS%sh_xy(I, j - 1, k))) + &
                                             CS%Idxdy2u(I, j)*((CS%dy2h(i + 1, j)*CS%sh_xx(i + 1, j, k)) - &
                                                               (CS%dy2h(i, j)*CS%sh_xx(i, j, k)))
                    end do
                end do
            end do

            ! Del2v = y.Div(Grad u) at v-points
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do i=is,ie
                        CS%Del2v(i, J, k) = CS%Idxdy2v(i, J)*((CS%dy2q(i, J)*CS%sh_xy(i, J, k)) - &
                                                               (CS%dy2q(i - 1, J)*CS%sh_xy(i - 1, J, k))) - &
                                             CS%Idx2dyCv(i, J)*((CS%dx2h(i, j + 1)*CS%sh_xx(i, j + 1, k)) - &
                                                                (CS%dx2h(i, j)*CS%sh_xx(i, j, k)))
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 5: Leith vorticity calculation
        !=====================================================================
        if (CS%Leith_Kh) then

            ! Calculate vorticity at q-points
            if (CS%no_slip) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j = Jsq, Jeq
                        do I = Isq, Ieq
                            CS%vort_xy(I, J, k) = (2.0_dp - G%mask2dBu(I, J))*(CS%dvdx(I, J, k) - CS%dudy(I, J, k))
                        end do
                    end do
                end do
            else
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j = Jsq, Jeq
                        do I = Isq, Ieq
                            CS%vort_xy(I, J, k) = G%mask2dBu(I, J)*(CS%dvdx(I, J, k) - CS%dudy(I, J, k))
                        end do
                    end do
                end do
            end if

            ! Vorticity gradient at h-points (requires extended halo)
            !$omp target teams distribute parallel do collapse(3) private(DY_dxBu_loc)
            do k = 1, nz
                do j = js, je
                    do i = is, ie
                        DY_dxBu_loc = G%dyBu(i, j)*G%IdxBu(i, j)
                        CS%vort_xy_dx(i, j, k) = DY_dxBu_loc*((CS%vort_xy(i, j, k)*G%IdyCu(i, j)) - &
                                                               (CS%vort_xy(i - 1, j, k)*G%IdyCu(i - 1, j)))
                    end do
                end do
            end do

            !$omp target teams distribute parallel do collapse(3) private(DX_dyBu_loc)
            do k = 1, nz
                do j = js, je
                    do I = Isq, Ieq
                        DX_dyBu_loc = G%dxBu(I, j)*G%IdyBu(I, j)
                        CS%vort_xy_dy(I, j, k) = DX_dyBu_loc*((CS%vort_xy(I, j, k)*G%IdxCv(i, j)) - &
                                                               (CS%vort_xy(I, j - 1, k)*G%IdxCv(i, j - 1)))
                    end do
                end do
            end do

            ! Laplacian of vorticity at q-points (for Leith biharmonic)
            if (CS%Leith_Ah) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J = Jsq, Jeq
                        do I = Isq, Ieq
                            CS%Del2vort_q(I, J, k) = CS%DY_dxBu(I, J)* &
                                ((CS%vort_xy_dx(i + 1, J, k)*G%IdyCv(i + 1, J)) - &
                                 (CS%vort_xy_dx(i, J, k)*G%IdyCv(i, J))) + &
                                CS%DX_dyBu(I, J)* &
                                ((CS%vort_xy_dy(I, j + 1, k)*G%IdxCu(I, j + 1)) - &
                                 (CS%vort_xy_dy(I, j, k)*G%IdxCu(I, j)))
                        end do
                    end do
                end do
            end if

            ! Modified Leith: include divergence gradient
            if (CS%modified_Leith) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j = js, je
                        do i = is, ie
                            CS%div_xx(i, j, k) = CS%dudx(i, j, k) + CS%dvdy(i, j, k)
                        end do
                    end do
                end do

                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j = js, je
                        do I = Isq, Ieq
                            CS%div_xx_dx(I, j, k) = G%IdxCu(I, j)*(CS%div_xx(i + 1, j, k) - CS%div_xx(i, j, k))
                        end do
                    end do
                end do

                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J = Jsq, Jeq
                        do i = is, ie
                            CS%div_xx_dy(i, J, k) = G%IdyCv(i, J)*(CS%div_xx(i, j + 1, k) - CS%div_xx(i, j, k))
                        end do
                    end do
                end do

                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j = js, je
                        do i = is, ie
                            CS%grad_div_mag_h(i, j, k) = sqrt(((0.5_dp*(CS%div_xx_dx(i, j, k) + CS%div_xx_dx(i - 1, j, k)))**2) + &
                                                               ((0.5_dp*(CS%div_xx_dy(i, j, k) + CS%div_xx_dy(i, j - 1, k)))**2))
                        end do
                    end do
                end do
            else
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j = js, je
                        do i = is, ie
                            CS%grad_div_mag_h(i, j, k) = 0.0_dp
                        end do
                    end do
                end do
            end if

            ! Magnitude of vorticity gradient at h-points
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = js, je
                    do i = is, ie
                        CS%grad_vort_mag_h(i, j, k) = sqrt(((0.5_dp*(CS%vort_xy_dx(i, j, k) + CS%vort_xy_dx(i, j - 1, k)))**2) + &
                                                            ((0.5_dp*(CS%vort_xy_dy(i, j, k) + CS%vort_xy_dy(i - 1, j, k)))**2))
                        CS%vert_vort_mag(i, j, k) = CS%grad_vort_mag_h(i, j, k) + CS%grad_div_mag_h(i, j, k)
                    end do
                end do
            end do

            ! Magnitude of vorticity gradient at q-points
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J = Jsq, Jeq
                    do I = Isq, Ieq
                        CS%grad_vort_mag_q(I, J, k) = sqrt(((0.5_dp*(CS%vort_xy_dx(i, j, k) + CS%vort_xy_dx(i + 1, j, k)))**2) + &
                                                            ((0.5_dp*(CS%vort_xy_dy(i, j, k) + CS%vort_xy_dy(i, j + 1, k)))**2))
                    end do
                end do
            end do

            if (CS%Leith_Ah) then
            end if
        end if

        !=====================================================================
        ! STEP 6: Compute Smagorinsky shear magnitude
        !=====================================================================
        if (CS%Smagorinsky_Kh .or. CS%Smagorinsky_Ah) then
            !$omp target teams distribute parallel do collapse(3) private(sh_xx_sq, sh_xy_sq)
            do k = 1, nz
                do j=js,je
                    do i=is,ie
                        sh_xx_sq = CS%sh_xx(i, j, k)**2
                        sh_xy_sq = 0.25_dp*(((CS%sh_xy(i - 1, j - 1, k)**2) + (CS%sh_xy(i, j, k)**2)) + &
                                            ((CS%sh_xy(i - 1, j, k)**2) + (CS%sh_xy(i, j - 1, k)**2)))
                        CS%Shear_mag(i, j, k) = sqrt(sh_xx_sq + sh_xy_sq)
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 7: Compute thickness ratio for better_bound
        !=====================================================================
        if (CS%better_bound_Kh .or. CS%better_bound_Ah) then
            !$omp target teams distribute parallel do collapse(3) private(h_min)
            do k = 1, nz
                do j=js,je
                    do i=is,ie
                        h_min = min(CS%h_u(i, j, k), CS%h_u(i - 1, j, k), CS%h_v(i, j, k), CS%h_v(i, j - 1, k))
                        CS%hrat_min(i, j, k) = min(1.0_dp, h_min/(h(i, j, k) + h_neglect))
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 8: Determine Laplacian viscosity Kh at h-points
        !=====================================================================
        if (CS%Laplacian) then
            ! Start with background viscosity
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=js,je
                    do i=is,ie
                        CS%Kh(i, j, k) = CS%Kh_bg_xx(i, j)
                    end do
                end do
            end do

            ! Add Smagorinsky contribution
            if (CS%Smagorinsky_Kh) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            CS%Kh(i, j, k) = max(CS%Kh(i, j, k), CS%Laplac2_const_xx(i, j)*CS%Shear_mag(i, j, k))
                        end do
                    end do
                end do
            end if

            ! Add Leith contribution
            if (CS%Leith_Kh) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            CS%Kh(i, j, k) = max(CS%Kh(i, j, k), &
                                                  CS%Laplac3_const_xx(i, j)*CS%vert_vort_mag(i, j, k)*inv_PI3)
                        end do
                    end do
                end do
            end if

            ! Apply minimum floor
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=js,je
                    do i=is,ie
                        CS%Kh(i, j, k) = max(CS%Kh(i, j, k), CS%Kh_bg_min)
                    end do
                end do
            end do

            ! Apply stability bounds
            if (CS%better_bound_Kh .and. CS%better_bound_Ah) then
                ! Track remaining budget for combined Kh + Ah bounding
                !$omp target teams distribute parallel do collapse(3) private(Kh_max_here)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            CS%visc_bound_rem(i, j, k) = 1.0_dp
                            Kh_max_here = CS%hrat_min(i, j, k)*CS%Kh_Max_xx(i, j)
                            if (CS%Kh(i, j, k) >= Kh_max_here) then
                                CS%visc_bound_rem(i, j, k) = 0.0_dp
                                CS%Kh(i, j, k) = Kh_max_here
                            else if (CS%Kh(i, j, k) > 0.0_dp) then
                                CS%visc_bound_rem(i, j, k) = 1.0_dp - CS%Kh(i, j, k)/Kh_max_here
                            end if
                        end do
                    end do
                end do
            else if (CS%better_bound_Kh) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            CS%Kh(i, j, k) = min(CS%Kh(i, j, k), CS%hrat_min(i, j, k)*CS%Kh_Max_xx(i, j))
                        end do
                    end do
                end do
            else if (CS%bound_Kh) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            CS%Kh(i, j, k) = min(CS%Kh(i, j, k), CS%Kh_Max_xx(i, j))
                        end do
                    end do
                end do
            end if

            ! Compute str_xx (Laplacian contribution)
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=Jsq,Jeq + 1
                    do i=Isq,Ieq + 1
                        CS%str_xx(i, j, k) = -CS%Kh(i, j, k)*CS%sh_xx(i, j, k)
                    end do
                end do
            end do
        else
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=Jsq,Jeq + 1
                    do i=Isq,Ieq + 1
                        CS%str_xx(i, j, k) = 0.0_dp
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 9: Determine biharmonic viscosity Ah at h-points
        !=====================================================================
        if (CS%biharmonic) then
            ! Start with background viscosity
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j=js,je
                    do i=is,ie
                        CS%Ah(i, j, k) = CS%Ah_bg_xx(i, j)
                    end do
                end do
            end do

            ! Add Smagorinsky contribution
            if (CS%Smagorinsky_Ah) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            CS%Ah(i, j, k) = max(CS%Ah(i, j, k), CS%Biharm_const_xx(i, j)*CS%Shear_mag(i, j, k))
                        end do
                    end do
                end do
            end if

            ! Add Leith biharmonic contribution
            if (CS%Leith_Ah) then
                !$omp target teams distribute parallel do collapse(3) private(Del2vort_h)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            Del2vort_h = 0.25_dp*((CS%Del2vort_q(i, j, k) + CS%Del2vort_q(i - 1, j - 1, k)) + &
                                                  (CS%Del2vort_q(i - 1, j, k) + CS%Del2vort_q(i, j - 1, k)))
                            CS%Ah(i, j, k) = max(CS%Ah(i, j, k), CS%Biharm6_const_xx(i, j)*abs(Del2vort_h)*inv_PI6)
                        end do
                    end do
                end do
            end if

            ! Apply stability bounds
            if (CS%better_bound_Ah) then
                if (CS%better_bound_Kh) then
                    ! Use remaining viscosity budget
                    !$omp target teams distribute parallel do collapse(3)
                    do k = 1, nz
                        do j=js,je
                            do i=is,ie
                                CS%Ah(i, j, k) = min(CS%Ah(i, j, k), CS%visc_bound_rem(i, j, k)*CS%hrat_min(i, j, k)*CS%Ah_Max_xx(i, j))
                            end do
                        end do
                    end do
                else
                    !$omp target teams distribute parallel do collapse(3)
                    do k = 1, nz
                        do j=js,je
                            do i=is,ie
                                CS%Ah(i, j, k) = min(CS%Ah(i, j, k), CS%hrat_min(i, j, k)*CS%Ah_Max_xx(i, j))
                            end do
                        end do
                    end do
                end if
            else if (CS%bound_Ah) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do j=js,je
                        do i=is,ie
                            CS%Ah(i, j, k) = min(CS%Ah(i, j, k), CS%Ah_Max_xx(i, j))
                        end do
                    end do
                end do
            end if

            ! Add biharmonic contribution to str_xx and store bhstr_xx
            !$omp target teams distribute parallel do collapse(3) private(d_del2u, d_del2v, d_str)
            do k = 1, nz
                do j=Jsq,Jeq + 1
                    do i=Isq,Ieq + 1
                        d_del2u = (G%IdyCu(i, j)*CS%Del2u(i, j, k)) - (G%IdyCu(i - 1, j)*CS%Del2u(i - 1, j, k))
                        d_del2v = (G%IdxCv(i, j)*CS%Del2v(i, j, k)) - (G%IdxCv(i, j - 1)*CS%Del2v(i, j - 1, k))
                        d_str = CS%Ah(i, j, k)*((CS%DY_dxT(i, j)*d_del2u) - (CS%DX_dyT(i, j)*d_del2v))
                        CS%bhstr_xx(i, j, k) = d_str*(h(i, j, k)*CS%reduction_xx(i, j))
                        CS%str_xx(i, j, k) = CS%str_xx(i, j, k) + d_str
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 10: Multiply str_xx by thickness
        !=====================================================================
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j=Jsq,Jeq + 1
                do i=Isq,Ieq + 1
                    CS%str_xx(i, j, k) = CS%str_xx(i, j, k)*(h(i, j, k)*CS%reduction_xx(i, j))
                end do
            end do
        end do

        !=====================================================================
        ! STEP 11: Determine Laplacian viscosity Kh at q-points
        !=====================================================================
        if (CS%Laplacian) then
            ! Start with background viscosity
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%Kh(I, J, k) = CS%Kh_bg_xy(I, J)
                    end do
                end do
            end do

            ! Add Smagorinsky contribution (interpolate shear_mag to q-points)
            if (CS%Smagorinsky_Kh) then
                !$omp target teams distribute parallel do collapse(3) private(sh_xx_sq, sh_xy_sq)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            sh_xx_sq = 0.25_dp*((CS%sh_xx(i, j, k)**2 + CS%sh_xx(i + 1, j + 1, k)**2) + &
                                                (CS%sh_xx(i + 1, j, k)**2 + CS%sh_xx(i, j + 1, k)**2))
                            sh_xy_sq = CS%sh_xy(I, J, k)**2
                            CS%Kh(I, J, k) = max(CS%Kh(I, J, k), CS%Laplac2_const_xy(I, J)*sqrt(sh_xx_sq + sh_xy_sq))
                        end do
                    end do
                end do
            end if

            ! Add Leith contribution
            if (CS%Leith_Kh) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            CS%Kh(I, J, k) = max(CS%Kh(I, J, k), &
                                                  CS%Laplac3_const_xy(I, J)*CS%grad_vort_mag_q(I, J, k)*inv_PI3)
                        end do
                    end do
                end do
            end if

            ! Apply minimum floor
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%Kh(I, J, k) = max(CS%Kh(I, J, k), CS%Kh_bg_min)
                    end do
                end do
            end do

            ! Apply stability bounds at q-points
            if (CS%better_bound_Kh) then
                !$omp target teams distribute parallel do collapse(3) private(h_min)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            h_min = 0.25_dp*((CS%hrat_min(i, j, k) + CS%hrat_min(i + 1, j + 1, k)) + &
                                             (CS%hrat_min(i + 1, j, k) + CS%hrat_min(i, j + 1, k)))
                            CS%Kh(I, J, k) = min(CS%Kh(I, J, k), h_min*CS%Kh_Max_xy(I, J))
                        end do
                    end do
                end do
            else if (CS%bound_Kh) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            CS%Kh(I, J, k) = min(CS%Kh(I, J, k), CS%Kh_Max_xy(I, J))
                        end do
                    end do
                end do
            end if

            ! Compute str_xy (Laplacian contribution)
            if (CS%no_slip) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            CS%str_xy(I, J, k) = -CS%Kh(I, J, k)*CS%sh_xy(I, J, k)
                        end do
                    end do
                end do
            else
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            CS%str_xy(I, J, k) = -CS%Kh(I, J, k)*CS%sh_xy(I, J, k)
                        end do
                    end do
                end do
            end if
        else
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%str_xy(I, J, k) = 0.0_dp
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 12: Determine biharmonic viscosity Ah at q-points
        !=====================================================================
        if (CS%biharmonic) then
            ! Start with background viscosity
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%Ah(I, J, k) = CS%Ah_bg_xy(I, J)
                    end do
                end do
            end do

            ! Add Smagorinsky contribution
            if (CS%Smagorinsky_Ah) then
                !$omp target teams distribute parallel do collapse(3) private(sh_xx_sq, sh_xy_sq)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            sh_xx_sq = 0.25_dp*((CS%sh_xx(i, j, k)**2 + CS%sh_xx(i + 1, j + 1, k)**2) + &
                                                (CS%sh_xx(i + 1, j, k)**2 + CS%sh_xx(i, j + 1, k)**2))
                            sh_xy_sq = CS%sh_xy(I, J, k)**2
                            CS%Ah(I, J, k) = max(CS%Ah(I, J, k), CS%Biharm_const_xy(I, J)*sqrt(sh_xx_sq + sh_xy_sq))
                        end do
                    end do
                end do
            end if

            ! Add Leith biharmonic contribution
            if (CS%Leith_Ah) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            CS%Ah(I, J, k) = max(CS%Ah(I, J, k), CS%Biharm6_const_xy(I, J)*abs(CS%Del2vort_q(I, J, k))*inv_PI6)
                        end do
                    end do
                end do
            end if

            ! Apply stability bounds
            if (CS%better_bound_Ah) then
                !$omp target teams distribute parallel do collapse(3) private(h_min)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            h_min = 0.25_dp*((CS%hrat_min(i, j, k) + CS%hrat_min(i + 1, j + 1, k)) + &
                                             (CS%hrat_min(i + 1, j, k) + CS%hrat_min(i, j + 1, k)))
                            CS%Ah(I, J, k) = min(CS%Ah(I, J, k), h_min*CS%Ah_Max_xy(I, J))
                        end do
                    end do
                end do
            else if (CS%bound_Ah) then
                !$omp target teams distribute parallel do collapse(3)
                do k = 1, nz
                    do J=Jsq,Jeq
                        do I=Isq,Ieq
                            CS%Ah(I, J, k) = min(CS%Ah(I, J, k), CS%Ah_Max_xy(I, J))
                        end do
                    end do
                end do
            end if

            ! Add biharmonic contribution to str_xy and store bhstr_xy
            !$omp target teams distribute parallel do collapse(3) private(d_str)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        d_str = CS%Ah(I, J, k)*((CS%DY_dxBu(I, J)*((CS%Del2v(i + 1, J, k)*G%IdyCv(i + 1, J)) - &
                                                                    (CS%Del2v(i, J, k)*G%IdyCv(i, J)))) + &
                                                 (CS%DX_dyBu(I, J)*((CS%Del2u(I, j + 1, k)*G%IdxCu(I, j + 1)) - &
                                                                    (CS%Del2u(I, j, k)*G%IdxCu(I, j)))))
                        CS%bhstr_xy(I, J, k) = d_str*(CS%hq(I, J, k)*CS%reduction_xy(I, J))
                        CS%str_xy(I, J, k) = CS%str_xy(I, J, k) + d_str
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 13: Multiply str_xy by thickness
        !=====================================================================
        if (CS%no_slip) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%str_xy(I, J, k) = CS%str_xy(I, J, k)*(CS%hq(I, J, k)*CS%reduction_xy(I, J))
                    end do
                end do
            end do
        else
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do J=Jsq,Jeq
                    do I=Isq,Ieq
                        CS%str_xy(I, J, k) = CS%str_xy(I, J, k)*(CS%hq(I, J, k)*G%mask2dBu(I, J)*CS%reduction_xy(I, J))
                    end do
                end do
            end do
        end if

        !=====================================================================
        ! STEP 14: Compute viscous accelerations
        !=====================================================================
        ! diffu = 1/h * x.Div(h * stress tensor)
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do j=js,je
                do I=Isq,Ieq
                    diffu(I, j, k) = ((G%IdxCu(I, j)*((CS%dx2q(I, j - 1)*CS%str_xy(I, j - 1, k)) - &
                                                      (CS%dx2q(I, j)*CS%str_xy(I, j, k))) + &
                                       G%IdyCu(I, j)*((CS%dy2h(i, j)*CS%str_xx(i, j, k)) - &
                                                      (CS%dy2h(i + 1, j)*CS%str_xx(i + 1, j, k))))* &
                                      G%IareaCu(I, j))/(CS%h_u(I, j, k) + h_neglect)
                end do
            end do
        end do

        ! diffv = 1/h * y.Div(h * stress tensor)
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
            do J=Jsq,Jeq
                do i=is,ie
                    diffv(i, J, k) = ((G%IdyCv(i, J)*((CS%dy2q(i - 1, J)*CS%str_xy(i - 1, J, k)) - &
                                                      (CS%dy2q(i, J)*CS%str_xy(i, J, k))) - &
                                       G%IdxCv(i, J)*((CS%dx2h(i, j)*CS%str_xx(i, j, k)) - &
                                                      (CS%dx2h(i, j + 1)*CS%str_xx(i, j + 1, k))))* &
                                      G%IareaCv(i, J))/(CS%h_v(i, J, k) + h_neglect)
                end do
            end do
        end do

        !=====================================================================
        ! STEP 15: Compute friction work (after diffu/diffv)
        !=====================================================================
        if (CS%compute_FrictWork .and. present(FrictWork)) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
                do j = js, je
                    do i = is, ie
                        FrictWork(i, j, k) = ( &
                            ((CS%str_xx(i, j, k)*(u(i, j, k) - u(i - 1, j, k))*G%IdxT(i, j)) &
                            - (CS%str_xx(i, j, k)*(v(i, j, k) - v(i, j - 1, k))*G%IdyT(i, j))) &
                            + 0.25_dp*(( &
                              (CS%str_xy(i, j, k)* &
                                (((u(i, j + 1, k) - u(i, j, k))*G%IdyBu(i, j)) + &
                                 ((v(i + 1, j, k) - v(i, j, k))*G%IdxBu(i, j)))) &
                            + (CS%str_xy(i - 1, j - 1, k)* &
                                (((u(i - 1, j, k) - u(i - 1, j - 1, k))*G%IdyBu(i - 1, j - 1)) + &
                                 ((v(i, j - 1, k) - v(i - 1, j - 1, k))*G%IdxBu(i - 1, j - 1)))) ) &
                            + ( &
                              (CS%str_xy(i - 1, j, k)* &
                                (((u(i - 1, j + 1, k) - u(i - 1, j, k))*G%IdyBu(i - 1, j)) + &
                                 ((v(i, j, k) - v(i - 1, j, k))*G%IdxBu(i - 1, j)))) &
                            + (CS%str_xy(i, j - 1, k)* &
                                (((u(i, j, k) - u(i, j - 1, k))*G%IdyBu(i, j - 1)) + &
                                 ((v(i + 1, j - 1, k) - v(i, j - 1, k))*G%IdxBu(i, j - 1)))) )))
                    end do
                end do
            end do
        end if

        !$omp end target data

    end subroutine hor_visc

end module mom6_hor_visc_omp
