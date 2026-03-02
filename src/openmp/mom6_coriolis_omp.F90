!> MOM6 Coriolis and Momentum Advection Module (OpenMP target offloading)
!!
!! Computes Coriolis acceleration and kinetic energy gradient.
!! Supports multiple discretization schemes (Sadourny, Arakawa-Hsu, Arakawa-Lamb).
!!
!! Translated from OpenACC (mom6_coriolis) to OpenMP target offloading.
!!
!! Original code from: src/core/MOM_CoriolisAdv.F90
!!
module mom6_coriolis_omp
    use iso_fortran_env, only: dp => real64, int64
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

        real(dp), allocatable :: Area_q(:, :)        ! 2D, precomputed
        real(dp), allocatable :: q(:, :, :)          ! 3D potential vorticity
        real(dp), allocatable :: KE(:, :, :)         ! 3D kinetic energy
        real(dp), allocatable :: a(:, :, :), b(:, :, :), c(:, :, :), d(:, :, :)  ! 3D, conditional
        integer(int64) :: nbytes = 0  ! Total bytes allocated for GPU arrays
    end type coriolis_CS

    real(dp), parameter :: C1_12 = 1.0_dp/12.0_dp
    real(dp), parameter :: C1_24 = 1.0_dp/24.0_dp

contains

    !> Initialize the Coriolis solver
    subroutine coriolis_init(CS, G, GV, scheme)
        type(coriolis_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        integer, intent(in), optional :: scheme

        integer :: i, j

        CS%Coriolis_Scheme = SADOURNY75_ENERGY
        if (present(scheme)) CS%Coriolis_Scheme = scheme

        ! Allocate 2D precomputed array
        allocate (CS%Area_q(G%isd:G%ied, G%jsd:G%jed))

        ! Allocate 3D work arrays
        allocate (CS%q(G%isd:G%ied, G%jsd:G%jed, GV%ke))
        allocate (CS%KE(G%isd:G%ied, G%jsd:G%jed, GV%ke))

        ! Conditionally allocate Arakawa coefficient arrays
        if (CS%Coriolis_Scheme /= SADOURNY75_ENERGY) then
            allocate (CS%a(G%isd:G%ied, G%jsd:G%jed, GV%ke))
            allocate (CS%b(G%isd:G%ied, G%jsd:G%jed, GV%ke))
            allocate (CS%c(G%isd:G%ied, G%jsd:G%jed, GV%ke))
            allocate (CS%d(G%isd:G%ied, G%jsd:G%jed, GV%ke))
        end if

        ! Compute total bytes: 1 2D + 2 3D always, +4 3D if not Sadourny
        CS%nbytes = (int(G%ied - G%isd + 1, int64) * int(G%jed - G%jsd + 1, int64)) &
            * (1_int64 + 2_int64 * int(GV%ke, int64)) * 8_int64
        if (CS%Coriolis_Scheme /= SADOURNY75_ENERGY) then
            CS%nbytes = CS%nbytes + 4_int64 * int(G%ied - G%isd + 1, int64) &
                * int(G%jed - G%jsd + 1, int64) * int(GV%ke, int64) * 8_int64
        end if

        ! Precompute Area_q (sum of 4 neighboring h-cell areas)
        do j=G%jsd,G%jed - 1
        do i=G%isd,G%ied - 1
            CS%Area_q(i, j) = (G%areaT(i, j) + G%areaT(i + 1, j + 1)) + &
                              (G%areaT(i + 1, j) + G%areaT(i, j + 1))
        end do
        end do

        ! Copy CS to GPU (G is already on GPU from init_ocean_grid)
        !$omp target enter data map(to: CS)
        !$omp target enter data map(alloc: CS%q, CS%KE) map(to: CS%Area_q)
        if (CS%Coriolis_Scheme /= SADOURNY75_ENERGY) then
            !$omp target enter data map(alloc: CS%a, CS%b, CS%c, CS%d)
        end if

        CS%initialized = .true.

    end subroutine coriolis_init

    !> Finalize the Coriolis solver
    subroutine coriolis_end(CS)
        type(coriolis_CS), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! Detach member arrays from GPU, then delete derived type descriptor
        if (allocated(CS%a)) then
            !$omp target exit data map(delete: CS%a, CS%b, CS%c, CS%d)
        end if
        !$omp target exit data map(delete: CS%q, CS%KE, CS%Area_q)
        !$omp target exit data map(delete: CS)

        if (allocated(CS%q)) deallocate (CS%q)
        if (allocated(CS%KE)) deallocate (CS%KE)
        if (allocated(CS%Area_q)) deallocate (CS%Area_q)
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

        real(dp) :: dvdx_l, dudy_l, rel_vort_l, abs_vort_l
        real(dp) :: hArea_q, vol_neglect, Ih_q_l, KEx, KEy
        integer :: i, j, k, is, ie, js, je, nz

        is = G%isc; ie = G%iec; js = G%jsc; je = G%jec; nz = GV%ke
        vol_neglect = 1.0e-20_dp

        ! Kernel 1: Fused PV computation (dvdx, dudy, vorticity, hArea, q)
        !$omp target teams distribute parallel do collapse(3) &
        !$omp&  private(dvdx_l, dudy_l, rel_vort_l, abs_vort_l, hArea_q, Ih_q_l)
        do k = 1, nz
        do j = js - 1, je
        do i = is - 1, ie
            dvdx_l = (v(i + 1, j, k)*G%dyCv(i + 1, j)) - (v(i, j, k)*G%dyCv(i, j))
            dudy_l = (u(i, j + 1, k)*G%dxCu(i, j + 1)) - (u(i, j, k)*G%dxCu(i, j))
            rel_vort_l = G%mask2dBu(i, j)*(dvdx_l - dudy_l)*G%IareaBu(i, j)
            abs_vort_l = G%CoriolisBu(i, j) + rel_vort_l
            hArea_q = (G%areaT(i, j)*h(i, j, k) + G%areaT(i + 1, j)*h(i + 1, j, k)) + &
                      (G%areaT(i, j + 1)*h(i, j + 1, k) + G%areaT(i + 1, j + 1)*h(i + 1, j + 1, k))
            Ih_q_l = CS%Area_q(i, j)/(hArea_q + vol_neglect)
            CS%q(i, j, k) = abs_vort_l*Ih_q_l
        end do
        end do
        end do

        ! Kernel 2: Kinetic energy
        !$omp target teams distribute parallel do collapse(3)
        do k = 1, nz
        do j = js, je
        do i = is, ie
            CS%KE(i, j, k) = 0.25_dp*( &
                              (G%dyCu(i, j)*u(i, j, k)**2 + G%dyCu(i - 1, j)*u(i - 1, j, k)**2) + &
                              (G%dxCv(i, j)*v(i, j, k)**2 + G%dxCv(i, j - 1)*v(i, j - 1, k)**2) &
                              )/G%areaT(i, j)
        end do
        end do
        end do

        ! Compute Arakawa scheme coefficients if needed
        if (CS%Coriolis_Scheme == ARAKAWA_HSU90) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
            do j = js, je
            do i = is - 1, ie
                CS%a(i, j, k) = (CS%q(i, j, k) + (CS%q(i + 1, j, k) + CS%q(i, j - 1, k)))*C1_12
                CS%d(i, j, k) = ((CS%q(i, j, k) + CS%q(i + 1, j - 1, k)) + CS%q(i, j - 1, k))*C1_12
            end do
            end do
            end do
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
            do j = js, je
            do i = is, ie
                CS%b(i, j, k) = (CS%q(i, j, k) + (CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_12
                CS%c(i, j, k) = ((CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + CS%q(i, j - 1, k))*C1_12
            end do
            end do
            end do
        elseif (CS%Coriolis_Scheme == ARAKAWA_LAMB81) then
            !$omp target teams distribute parallel do collapse(3)
            do k = 1, nz
            do j = js, je
            do i = is, ie
                CS%a(i - 1, j, k) = (2.0_dp*(CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + &
                                     (CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
                CS%d(i - 1, j, k) = ((CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + &
                                     2.0_dp*(CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
                CS%b(i, j, k) = ((CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + &
                                 2.0_dp*(CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
                CS%c(i, j, k) = (2.0_dp*(CS%q(i, j, k) + CS%q(i - 1, j - 1, k)) + &
                                 (CS%q(i - 1, j, k) + CS%q(i, j - 1, k)))*C1_24
            end do
            end do
            end do
        end if

        ! Compute Coriolis accelerations based on scheme
        if (CS%Coriolis_Scheme == SADOURNY75_ENERGY) then
            ! Energy-conserving Sadourny (1975) scheme
            !$omp target teams distribute parallel do collapse(3) private(KEx)
            do k = 1, nz
            do j = js, je
            do i = is, ie - 1
                KEx = (CS%KE(i + 1, j, k) - CS%KE(i, j, k))*G%IdxCu(i, j)
                CAu(i, j, k) = 0.25_dp*( &
                               (CS%q(i, j, k)*(vh(i + 1, j, k) + vh(i, j, k))) + &
                               (CS%q(i, j - 1, k)*(vh(i, j - 1, k) + vh(i + 1, j - 1, k))) &
                               )*G%IdxCu(i, j) - KEx
            end do
            end do
            end do

            !$omp target teams distribute parallel do collapse(3) private(KEy)
            do k = 1, nz
            do j = js, je - 1
            do i = is, ie
                KEy = (CS%KE(i, j + 1, k) - CS%KE(i, j, k))*G%IdyCv(i, j)
                CAv(i, j, k) = -0.25_dp*( &
                               (CS%q(i - 1, j, k)*(uh(i - 1, j, k) + uh(i - 1, j + 1, k))) + &
                               (CS%q(i, j, k)*(uh(i, j, k) + uh(i, j + 1, k))) &
                               )*G%IdyCv(i, j) - KEy
            end do
            end do
            end do

        else  ! ARAKAWA_HSU90 or ARAKAWA_LAMB81
            !$omp target teams distribute parallel do collapse(3) private(KEx)
            do k = 1, nz
            do j = js, je
            do i = is, ie - 1
                KEx = (CS%KE(i + 1, j, k) - CS%KE(i, j, k))*G%IdxCu(i, j)
                CAu(i, j, k) = ( &
                               ((CS%a(i, j, k)*vh(i + 1, j, k)) + (CS%c(i, j, k)*vh(i, j - 1, k))) + &
                               ((CS%b(i, j, k)*vh(i, j, k)) + (CS%d(i, j, k)*vh(i + 1, j - 1, k))) &
                               )*G%IdxCu(i, j) - KEx
            end do
            end do
            end do

            !$omp target teams distribute parallel do collapse(3) private(KEy)
            do k = 1, nz
            do j = js, je - 1
            do i = is, ie
                KEy = (CS%KE(i, j + 1, k) - CS%KE(i, j, k))*G%IdyCv(i, j)
                CAv(i, j, k) = -( &
                               ((CS%a(i - 1, j, k)*uh(i - 1, j, k)) + (CS%c(i, j + 1, k)*uh(i, j + 1, k))) + &
                               ((CS%b(i, j, k)*uh(i, j, k)) + (CS%d(i - 1, j + 1, k)*uh(i - 1, j + 1, k))) &
                               )*G%IdyCv(i, j) - KEy
            end do
            end do
            end do
        end if

    end subroutine CorAdCalc

end module mom6_coriolis_omp
