!> MOM6 Coriolis and Momentum Advection Module
!!
!! Computes Coriolis acceleration and kinetic energy gradient.
!! Supports multiple discretization schemes (Sadourny, Arakawa-Hsu, Arakawa-Lamb).
!!
!! Original code from: src/core/MOM_CoriolisAdv.F90
!!
module mom6_coriolis
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type
    implicit none
    private

    public :: CorAdCalc, coriolis_init, coriolis_end
    public :: coriolis_CS
    public :: SADOURNY75_ENERGY, ARAKAWA_HSU90, ARAKAWA_LAMB81

    !> Scheme identifiers
    integer, parameter :: SADOURNY75_ENERGY = 1
    integer, parameter :: ARAKAWA_HSU90 = 2
    integer, parameter :: ARAKAWA_LAMB81 = 3

    !> Control structure for Coriolis solver
    type :: coriolis_CS
        logical :: initialized = .false.
        integer :: Coriolis_Scheme       ! Which scheme to use

        ! Work arrays (3D, enables full k-j-i parallelism)
        real(dp), allocatable :: dvdx(:,:, :)     ! d(v*dy)/dx
        real(dp), allocatable :: dudy(:,:, :)     ! d(u*dx)/dy
        real(dp), allocatable :: rel_vort(:,:, :)  ! Relative vorticity
        real(dp), allocatable :: abs_vort(:,:, :)  ! Absolute vorticity
        real(dp), allocatable :: q(:,:, :)        ! Potential vorticity
        real(dp), allocatable :: Ih_q(:,:, :)     ! Inverse thickness at q-points
        real(dp), allocatable :: hArea_u(:,:, :)  ! h*Area at u-points
        real(dp), allocatable :: hArea_v(:,:, :)  ! h*Area at v-points
        real(dp), allocatable :: Area_q(:,:, :)   ! Area at q-points
        real(dp), allocatable :: KE(:,:, :)       ! Kinetic energy
        ! Arakawa coefficients
        real(dp), allocatable :: a(:,:, :)
        real(dp), allocatable :: b(:,:, :)
        real(dp), allocatable :: c(:,:, :)
        real(dp), allocatable :: d(:,:, :)
    end type coriolis_CS

    real(dp), parameter :: C1_12 = 1.0_dp/12.0_dp
    real(dp), parameter :: C1_24 = 1.0_dp/24.0_dp

contains

    !> Initialize the Coriolis solver
    subroutine coriolis_init(CS, G, scheme)
        type(coriolis_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        integer, intent(in), optional :: scheme

        integer :: i, j, k
        integer :: nk 

        nk = G%nk

        CS%Coriolis_Scheme = SADOURNY75_ENERGY
        if (present(scheme)) CS%Coriolis_Scheme = scheme

        ! Allocate 3D work arrays
        allocate (CS%dvdx(G%isd:G%ied, G%jsd:G%jed, nk))
        allocate (CS%dudy(G%isd:G%ied, G%jsd:G%jed, nk))
        allocate (CS%rel_vort(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%abs_vort(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%q(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%Ih_q(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%hArea_u(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%hArea_v(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%Area_q(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%KE(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%a(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%b(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%c(G%isd:G%ied, G%jsd:G%jed,nk))
        allocate (CS%d(G%isd:G%ied, G%jsd:G%jed,nk))

#ifdef __NVCOMPILER_LLVM__
        !$omp target enter data map(to: CS)
        !$omp target enter data map(alloc: CS%dvdx, CS%dudy, CS%rel_vort, CS%abs_vort)
        !$omp target enter data map(alloc: CS%q, CS%Ih_q, CS%hArea_u, CS%hArea_v, CS%Area_q)
        !$omp target enter data map(alloc: CS%KE, CS%a, CS%b, CS%c, CS%d)
#endif

        ! Precompute Area_q (sum of 4 neighboring h-cell areas)
        do concurrent(k=1:nk, j=G%jsd:G%jed - 1, i=G%isd:G%ied - 1)
            CS%Area_q(i, j, k) = (G%areaT(i, j) + G%areaT(i + 1, j + 1)) + &
                              (G%areaT(i + 1, j) + G%areaT(i, j + 1))
        end do

        CS%initialized = .true.

    end subroutine coriolis_init

    !> Finalize the Coriolis solver
    subroutine coriolis_end(CS)
        type(coriolis_CS), intent(inout) :: CS

        if (.not. CS%initialized) return

#ifdef __NVCOMPILER_LLVM__
        !$omp target exit data map(delete: CS%dvdx, CS%dudy, CS%rel_vort, CS%abs_vort)
        !$omp target exit data map(delete: CS%q, CS%Ih_q, CS%hArea_u, CS%hArea_v, CS%Area_q)
        !$omp target exit data map(delete: CS%KE, CS%a, CS%b, CS%c, CS%d)
#endif

        if (allocated(CS%dvdx)) deallocate (CS%dvdx)
        if (allocated(CS%dudy)) deallocate (CS%dudy)
        if (allocated(CS%rel_vort)) deallocate (CS%rel_vort)
        if (allocated(CS%abs_vort)) deallocate (CS%abs_vort)
        if (allocated(CS%q)) deallocate (CS%q)
        if (allocated(CS%Ih_q)) deallocate (CS%Ih_q)
        if (allocated(CS%hArea_u)) deallocate (CS%hArea_u)
        if (allocated(CS%hArea_v)) deallocate (CS%hArea_v)
        if (allocated(CS%Area_q)) deallocate (CS%Area_q)
        if (allocated(CS%KE)) deallocate (CS%KE)
        if (allocated(CS%a)) deallocate (CS%a)
        if (allocated(CS%b)) deallocate (CS%b)
        if (allocated(CS%c)) deallocate (CS%c)
        if (allocated(CS%d)) deallocate (CS%d)

        CS%initialized = .false.

    end subroutine coriolis_end

    !> Calculate Coriolis and momentum advection accelerations
    subroutine CorAdCalc(u, v, h, uh, vh, CAu, CAv, G, GV, CS)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u   ! Zonal velocity
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: v   ! Meridional velocity
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h   ! Layer thickness
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: uh  ! Zonal transport
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: vh  ! Meridional transport
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: CAu  ! Zonal acceleration
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: CAv  ! Meridional acceleration
        type(coriolis_CS), intent(inout) :: CS

        real(dp) :: hArea_q, vol_neglect, KEx, KEy
        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        vol_neglect = 1.0e-20_dp

        ! Phase 1: Compute circulation terms (independent of everything else)
        do concurrent(k=1:nz, j=js-1:je, i=is-1:ie)
            CS%dvdx(i, j, k) = (v(i + 1, j, k)*G%dyCv(i + 1, j)) - (v(i, j, k)*G%dyCv(i, j))
            CS%dudy(i, j, k) = (u(i, j + 1, k)*G%dxCu(i, j + 1)) - (u(i, j, k)*G%dxCu(i, j))
        end do

        ! Phase 1: Compute thickness-weighted areas at velocity points (independent)
        do concurrent(k=1:nz, j=js-1:je, i=is:ie+1)
            CS%hArea_v(i, j, k) = 0.5_dp*((G%areaT(i, j)*h(i, j, k)) + (G%areaT(i, j + 1)*h(i, j + 1, k)))
        end do

        do concurrent(k=1:nz, j=js:je+1, i=is-1:ie)
            CS%hArea_u(i, j, k) = 0.5_dp*((G%areaT(i, j)*h(i, j, k)) + (G%areaT(i + 1, j)*h(i + 1, j, k)))
        end do

        ! Phase 2: Vorticity + PV (depends on dvdx, dudy, hArea_u, hArea_v)
        ! Merged into one kernel since abs_vort and q only depend on same-point values
        do concurrent(k=1:nz, j=js-1:je, i=is-1:ie)
            CS%rel_vort(i, j, k) = G%mask2dBu(i, j)*(CS%dvdx(i, j, k) - CS%dudy(i, j, k))*G%IareaBu(i, j)
            CS%abs_vort(i, j, k) = G%CoriolisBu(i, j) + CS%rel_vort(i, j, k)
            hArea_q = (CS%hArea_u(i, j, k) + CS%hArea_u(i, j + 1, k)) + &
                      (CS%hArea_v(i, j, k) + CS%hArea_v(i + 1, j, k))
            CS%Ih_q(i, j, k) = CS%Area_q(i, j, k)/(hArea_q + vol_neglect)
            CS%q(i, j, k) = CS%abs_vort(i, j, k)*CS%Ih_q(i, j, k)
        end do

        ! Phase 3: Arakawa scheme coefficients if needed (depends on q)
        if (CS%Coriolis_Scheme == ARAKAWA_HSU90) then
            do concurrent(k=1:nz, j=js:je, i=is-1:ie)
                CS%a(i, j, k) = (CS%q(i, j, k) + (CS%q(i + 1, j, k) + CS%q(i, j - 1, k)))*C1_12
                CS%d(i, j, k) = ((CS%q(i, j, k) + CS%q(i + 1, j - 1, k)) + CS%q(i, j - 1, k))*C1_12
            end do
            do concurrent(k=1:nz, j=js:je, i=is:ie)
                CS%b(i, j, k) = (CS%q(i, j, k) + (CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_12
                CS%c(i, j, k) = ((CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + CS%q(i, j - 1, k))*C1_12
            end do
        elseif (CS%Coriolis_Scheme == ARAKAWA_LAMB81) then
            do concurrent(k=1:nz, j=js:je, i=is:ie)
                CS%a(i - 1, j, k) = (2.0_dp*(CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + (CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
                CS%d(i - 1, j, k) = ((CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + 2.0_dp*(CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
                CS%b(i, j, k) = ((CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + 2.0_dp*(CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
                CS%c(i, j, k) = (2.0_dp*(CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + (CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
            end do
        end if

        ! Phase 3: Kinetic energy (depends only on u, v inputs)
        do concurrent(k=1:nz, j=js:je, i=is:ie)
            CS%KE(i, j, k) = 0.25_dp*( &
                          (G%dyCu(i, j)*u(i, j, k)**2 + G%dyCu(i - 1, j)*u(i - 1, j, k)**2) + &
                          (G%dxCv(i, j)*v(i, j, k)**2 + G%dxCv(i, j - 1)*v(i, j - 1, k)**2) &
                          )/G%areaT(i, j)
        end do

        ! Phase 4: Coriolis accelerations (depends on q, KE, and optionally a/b/c/d)
        if (CS%Coriolis_Scheme == SADOURNY75_ENERGY) then
            ! Energy-conserving Sadourny (1975) scheme
            do concurrent(k=1:nz, j=js:je, i=is:ie-1)
                KEx = (CS%KE(i + 1, j, k) - CS%KE(i, j, k))*G%IdxCu(i, j)
                CAu(i, j, k) = 0.25_dp*( &
                               (CS%q(i, j, k)*(vh(i + 1, j, k) + vh(i, j, k))) + &
                               (CS%q(i, j - 1, k)*(vh(i, j - 1, k) + vh(i + 1, j - 1, k))) &
                               )*G%IdxCu(i, j) - KEx
            end do

            do concurrent(k=1:nz, j=js:je-1, i=is:ie)
                KEy = (CS%KE(i, j + 1, k) - CS%KE(i, j, k))*G%IdyCv(i, j)
                CAv(i, j, k) = -0.25_dp*( &
                               (CS%q(i - 1, j, k)*(uh(i - 1, j, k) + uh(i - 1, j + 1, k))) + &
                               (CS%q(i, j, k)*(uh(i, j, k) + uh(i, j + 1, k))) &
                               )*G%IdyCv(i, j) - KEy
            end do

        else  ! ARAKAWA_HSU90 or ARAKAWA_LAMB81
            do concurrent(k=1:nz, j=js:je, i=is:ie-1)
                KEx = (CS%KE(i + 1, j, k) - CS%KE(i, j, k))*G%IdxCu(i, j)
                CAu(i, j, k) = ( &
                               ((CS%a(i, j, k)*vh(i + 1, j, k)) + (CS%c(i, j, k)*vh(i, j - 1, k))) + &
                               ((CS%b(i, j, k)*vh(i, j, k)) + (CS%d(i, j, k)*vh(i + 1, j - 1, k))) &
                               )*G%IdxCu(i, j) - KEx
            end do

            do concurrent(k=1:nz, j=js:je-1, i=is:ie)
                KEy = (CS%KE(i, j + 1, k) - CS%KE(i, j, k))*G%IdyCv(i, j)
                CAv(i, j, k) = -( &
                               ((CS%a(i - 1, j, k)*uh(i - 1, j, k)) + (CS%c(i, j + 1, k)*uh(i, j + 1, k))) + &
                               ((CS%b(i, j, k)*uh(i, j, k)) + (CS%d(i - 1, j + 1, k)*uh(i - 1, j + 1, k))) &
                               )*G%IdyCv(i, j) - KEy
            end do
        end if

    end subroutine CorAdCalc

end module mom6_coriolis
