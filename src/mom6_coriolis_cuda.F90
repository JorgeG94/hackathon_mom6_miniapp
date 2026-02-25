!> MOM6 Coriolis and Momentum Advection Module (CUDA Fortran variant)
!!
!! Uses explicit attributes(global) CUDA kernels for GPU execution.
!! Two kernel launches: merged Phase 1+2 (q + KE) then Phase 3
!! (Coriolis acceleration). All intermediates kept in registers.
!!
module mom6_coriolis_cuda
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: CorAdCalc_cuda, coriolis_init_cuda, coriolis_end_cuda
    public :: coriolis_CS_cuda
    public :: SADOURNY75_ENERGY_CUDA, ARAKAWA_HSU90_CUDA, ARAKAWA_LAMB81_CUDA

    integer, parameter :: SADOURNY75_ENERGY_CUDA = 1
    integer, parameter :: ARAKAWA_HSU90_CUDA = 2
    integer, parameter :: ARAKAWA_LAMB81_CUDA = 3

    !> Control structure for CUDA Fortran Coriolis solver
    type :: coriolis_CS_cuda
        logical :: initialized = .false.
        integer :: Coriolis_Scheme
        integer :: is, ie, js, je, nz
        integer :: isd, ied, jsd, jed

        ! 3D device work arrays (outputs of merged Phase 1+2, inputs to Phase 3)
        real(dp), device, allocatable :: q_d(:,:,:)
        real(dp), device, allocatable :: KE_d(:,:,:)

        ! Arakawa scheme coefficients (Phase 3a output, Phase 3b input)
        real(dp), device, allocatable :: a_d(:,:,:)
        real(dp), device, allocatable :: b_d(:,:,:)
        real(dp), device, allocatable :: c_d(:,:,:)
        real(dp), device, allocatable :: d_d(:,:,:)

        ! 2D device grid metrics (copied once at init)
        real(dp), device, allocatable :: dyCv_d(:,:)
        real(dp), device, allocatable :: dxCu_d(:,:)
        real(dp), device, allocatable :: areaT_d(:,:)
        real(dp), device, allocatable :: IareaBu_d(:,:)
        real(dp), device, allocatable :: CoriolisBu_d(:,:)
        real(dp), device, allocatable :: mask2dBu_d(:,:)
        real(dp), device, allocatable :: dyCu_d(:,:)
        real(dp), device, allocatable :: dxCv_d(:,:)
        real(dp), device, allocatable :: IdxCu_d(:,:)
        real(dp), device, allocatable :: IdyCv_d(:,:)
    end type coriolis_CS_cuda

    real(dp), parameter :: C1_12 = 1.0_dp/12.0_dp
    real(dp), parameter :: C1_24 = 1.0_dp/24.0_dp

contains

    !=========================================================================
    ! CUDA Kernels (attributes(global))
    !
    ! All arrays use 1-based local indexing. The caller passes:
    !   n1 = ied - isd + 1   (array dim 1)
    !   n2 = jed - jsd + 1   (array dim 2)
    !   n3 = nz              (array dim 3)
    !   is_l = is - isd + 1  (compute-start offset in dim 1)
    !   ie_l = ie - isd + 1  (compute-end offset in dim 1)
    !   js_l = js - jsd + 1  (compute-start offset in dim 2)
    !   je_l = je - jsd + 1  (compute-end offset in dim 2)
    !=========================================================================

    !> Merged Phase 1+2: Compute q (PV) and KE in a single kernel.
    !! All intermediates (dvdx, dudy, rel_vort, abs_vort, hArea_u, hArea_v)
    !! are kept in registers — never written to global memory.
    !! Eliminates 7 intermediate 3D array writes compared to separate phases.
    attributes(global) subroutine phase12_merged_kernel( &
            q, KE, u, v, h, &
            dyCv, dxCu, dyCu, dxCv, areaT, &
            IareaBu, CoriolisBu, mask2dBu, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: q(n1, n2, n3), KE(n1, n2, n3)
        real(dp), intent(in) :: u(n1, n2, n3), v(n1, n2, n3), h(n1, n2, n3)
        real(dp), intent(in) :: dyCv(n1, n2), dxCu(n1, n2)
        real(dp), intent(in) :: dyCu(n1, n2), dxCv(n1, n2), areaT(n1, n2)
        real(dp), intent(in) :: IareaBu(n1, n2), CoriolisBu(n1, n2), mask2dBu(n1, n2)

        integer :: i, j, k
        real(dp) :: dvdx_val, dudy_val, rel_vort_val, abs_vort_val
        real(dp) :: hau_ij, hau_jp1, hav_ij, hav_ip1, hArea_q_val, aq_val

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        ! --- q: domain [is_l-1 : ie_l, js_l-1 : je_l] ---
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l - 1 .and. j <= je_l) then
            ! Circulation (was dvdx, dudy in separate Phase 1)
            dvdx_val = (v(i + 1, j, k) * dyCv(i + 1, j)) - &
                       (v(i, j, k) * dyCv(i, j))
            dudy_val = (u(i, j + 1, k) * dxCu(i, j + 1)) - &
                       (u(i, j, k) * dxCu(i, j))

            ! Relative and absolute vorticity (was separate Phase 2)
            rel_vort_val = mask2dBu(i, j) * &
                (dvdx_val - dudy_val) * IareaBu(i, j)
            abs_vort_val = CoriolisBu(i, j) + rel_vort_val

            ! Area-weighted thickness at neighbor points (recomputed inline)
            ! Preserves original FP evaluation order for bitwise exactness
            ! hArea_u(i,j)
            hau_ij  = 0.5_dp * ((areaT(i, j) * h(i, j, k)) + &
                                  (areaT(i + 1, j) * h(i + 1, j, k)))
            ! hArea_u(i,j+1)
            hau_jp1 = 0.5_dp * ((areaT(i, j + 1) * h(i, j + 1, k)) + &
                                  (areaT(i + 1, j + 1) * h(i + 1, j + 1, k)))
            ! hArea_v(i,j)
            hav_ij  = 0.5_dp * ((areaT(i, j) * h(i, j, k)) + &
                                  (areaT(i, j + 1) * h(i, j + 1, k)))
            ! hArea_v(i+1,j)
            hav_ip1 = 0.5_dp * ((areaT(i + 1, j) * h(i + 1, j, k)) + &
                                  (areaT(i + 1, j + 1) * h(i + 1, j + 1, k)))

            hArea_q_val = (hau_ij + hau_jp1) + (hav_ij + hav_ip1)

            ! Area_q computed inline (was precomputed 3D array, constant across k)
            aq_val = (areaT(i, j) + areaT(i + 1, j + 1)) + &
                     (areaT(i + 1, j) + areaT(i, j + 1))

            ! PV = abs_vort * Area_q / (hArea_q + epsilon)
            q(i, j, k) = abs_vort_val * aq_val / (hArea_q_val + 1.0e-20_dp)
        end if

        ! --- KE: domain [is_l : ie_l, js_l : je_l] ---
        if (i >= is_l .and. i <= ie_l .and. j >= js_l .and. j <= je_l) then
            KE(i, j, k) = 0.25_dp * ( &
                (dyCu(i, j) * u(i, j, k)**2 + dyCu(i - 1, j) * u(i - 1, j, k)**2) + &
                (dxCv(i, j) * v(i, j, k)**2 + dxCv(i, j - 1) * v(i, j - 1, k)**2) &
                ) / areaT(i, j)
        end if

    end subroutine phase12_merged_kernel

    !> Phase 3a: Compute Arakawa-Hsu90 coefficients a, b, c, d from PV (q).
    attributes(global) subroutine arakawa_hsu90_coef_kernel( &
            a, b, c, d, q, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: a(n1, n2, n3), b(n1, n2, n3)
        real(dp), intent(out) :: c(n1, n2, n3), d(n1, n2, n3)
        real(dp), intent(in) :: q(n1, n2, n3)

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (j < js_l .or. j > je_l .or. k < 1 .or. k > n3) return

        ! a, d: i in [is_l-1 : ie_l]
        if (i >= is_l - 1 .and. i <= ie_l) then
            a(i, j, k) = (q(i, j, k) + (q(i + 1, j, k) + q(i, j - 1, k))) * C1_12
            d(i, j, k) = ((q(i, j, k) + q(i + 1, j - 1, k)) + q(i, j - 1, k)) * C1_12
        end if

        ! b, c: i in [is_l : ie_l]
        if (i >= is_l .and. i <= ie_l) then
            b(i, j, k) = (q(i, j, k) + (q(i - 1, j, k) + q(i, j - 1, k))) * C1_12
            c(i, j, k) = ((q(i, j, k) + q(i - 1, j - 1, k)) + q(i, j - 1, k)) * C1_12
        end if

    end subroutine arakawa_hsu90_coef_kernel

    !> Phase 3a: Compute Arakawa-Lamb81 coefficients a, b, c, d from PV (q).
    attributes(global) subroutine arakawa_lamb81_coef_kernel( &
            a, b, c, d, q, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: a(n1, n2, n3), b(n1, n2, n3)
        real(dp), intent(out) :: c(n1, n2, n3), d(n1, n2, n3)
        real(dp), intent(in) :: q(n1, n2, n3)

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i < is_l .or. i > ie_l .or. j < js_l .or. j > je_l &
            .or. k < 1 .or. k > n3) return

        a(i - 1, j, k) = (2.0_dp * (q(i, j, k) + q(i - 1, j - 1, k)) + &
            (q(i - 1, j, k) + q(i, j - 1, k))) * C1_24
        d(i - 1, j, k) = ((q(i, j, k) + q(i - 1, j - 1, k)) + &
            2.0_dp * (q(i - 1, j, k) + q(i, j - 1, k))) * C1_24
        b(i, j, k) = ((q(i, j, k) + q(i - 1, j - 1, k)) + &
            2.0_dp * (q(i - 1, j, k) + q(i, j - 1, k))) * C1_24
        c(i, j, k) = (2.0_dp * (q(i, j, k) + q(i - 1, j - 1, k)) + &
            (q(i - 1, j, k) + q(i, j - 1, k))) * C1_24

    end subroutine arakawa_lamb81_coef_kernel

    !> Phase 3b: Coriolis acceleration — Sadourny energy-conserving scheme.
    attributes(global) subroutine coriolis_sadourny_kernel( &
            CAu, CAv, q, KE, uh, vh, IdxCu, IdyCv, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: CAu(n1, n2, n3), CAv(n1, n2, n3)
        real(dp), intent(in) :: q(n1, n2, n3), KE(n1, n2, n3)
        real(dp), intent(in) :: uh(n1, n2, n3), vh(n1, n2, n3)
        real(dp), intent(in) :: IdxCu(n1, n2), IdyCv(n1, n2)

        integer :: i, j, k
        real(dp) :: KEx, KEy

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (k < 1 .or. k > n3) return

        ! CAu: i in [is_l : ie_l-1], j in [js_l : je_l]
        if (i >= is_l .and. i <= ie_l - 1 .and. j >= js_l .and. j <= je_l) then
            KEx = (KE(i + 1, j, k) - KE(i, j, k)) * IdxCu(i, j)
            CAu(i, j, k) = 0.25_dp * ( &
                (q(i, j, k) * (vh(i + 1, j, k) + vh(i, j, k))) + &
                (q(i, j - 1, k) * (vh(i, j - 1, k) + vh(i + 1, j - 1, k))) &
                ) * IdxCu(i, j) - KEx
        end if

        ! CAv: i in [is_l : ie_l], j in [js_l : je_l-1]
        if (i >= is_l .and. i <= ie_l .and. j >= js_l .and. j <= je_l - 1) then
            KEy = (KE(i, j + 1, k) - KE(i, j, k)) * IdyCv(i, j)
            CAv(i, j, k) = -0.25_dp * ( &
                (q(i - 1, j, k) * (uh(i - 1, j, k) + uh(i - 1, j + 1, k))) + &
                (q(i, j, k) * (uh(i, j, k) + uh(i, j + 1, k))) &
                ) * IdyCv(i, j) - KEy
        end if

    end subroutine coriolis_sadourny_kernel

    !> Phase 3b: Coriolis acceleration — Arakawa schemes (HSU90 or LAMB81).
    attributes(global) subroutine coriolis_arakawa_kernel( &
            CAu, CAv, a, b, c, d, KE, uh, vh, IdxCu, IdyCv, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: CAu(n1, n2, n3), CAv(n1, n2, n3)
        real(dp), intent(in) :: a(n1, n2, n3), b(n1, n2, n3)
        real(dp), intent(in) :: c(n1, n2, n3), d(n1, n2, n3)
        real(dp), intent(in) :: KE(n1, n2, n3)
        real(dp), intent(in) :: uh(n1, n2, n3), vh(n1, n2, n3)
        real(dp), intent(in) :: IdxCu(n1, n2), IdyCv(n1, n2)

        integer :: i, j, k
        real(dp) :: KEx, KEy

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (k < 1 .or. k > n3) return

        ! CAu: i in [is_l : ie_l-1], j in [js_l : je_l]
        if (i >= is_l .and. i <= ie_l - 1 .and. j >= js_l .and. j <= je_l) then
            KEx = (KE(i + 1, j, k) - KE(i, j, k)) * IdxCu(i, j)
            CAu(i, j, k) = ( &
                ((a(i, j, k) * vh(i + 1, j, k)) + (c(i, j, k) * vh(i, j - 1, k))) + &
                ((b(i, j, k) * vh(i, j, k)) + (d(i, j, k) * vh(i + 1, j - 1, k))) &
                ) * IdxCu(i, j) - KEx
        end if

        ! CAv: i in [is_l : ie_l], j in [js_l : je_l-1]
        if (i >= is_l .and. i <= ie_l .and. j >= js_l .and. j <= je_l - 1) then
            KEy = (KE(i, j + 1, k) - KE(i, j, k)) * IdyCv(i, j)
            CAv(i, j, k) = -( &
                ((a(i - 1, j, k) * uh(i - 1, j, k)) + (c(i, j + 1, k) * uh(i, j + 1, k))) + &
                ((b(i, j, k) * uh(i, j, k)) + (d(i - 1, j + 1, k) * uh(i - 1, j + 1, k))) &
                ) * IdyCv(i, j) - KEy
        end if

    end subroutine coriolis_arakawa_kernel

    !=========================================================================
    ! Host routines
    !=========================================================================

    subroutine coriolis_init_cuda(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nk, &
                                   areaT, IareaBu, CoriolisBu, mask2dBu, &
                                   dyCv, dxCu, dyCu, dxCv, IdxCu, IdyCv, scheme)
        type(coriolis_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec, nk
        real(dp), intent(in) :: areaT(isd:ied, jsd:jed)
        real(dp), intent(in) :: IareaBu(isd:ied, jsd:jed)
        real(dp), intent(in) :: CoriolisBu(isd:ied, jsd:jed)
        real(dp), intent(in) :: mask2dBu(isd:ied, jsd:jed)
        real(dp), intent(in) :: dyCv(isd:ied, jsd:jed)
        real(dp), intent(in) :: dxCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: dyCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: dxCv(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdxCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdyCv(isd:ied, jsd:jed)
        integer, intent(in), optional :: scheme

        CS%Coriolis_Scheme = SADOURNY75_ENERGY_CUDA
        if (present(scheme)) CS%Coriolis_Scheme = scheme

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec; CS%nz = nk

        ! Allocate 3D device work arrays
        allocate(CS%q_d(isd:ied, jsd:jed, nk))
        allocate(CS%KE_d(isd:ied, jsd:jed, nk))
        allocate(CS%a_d(isd:ied, jsd:jed, nk))
        allocate(CS%b_d(isd:ied, jsd:jed, nk))
        allocate(CS%c_d(isd:ied, jsd:jed, nk))
        allocate(CS%d_d(isd:ied, jsd:jed, nk))

        ! Allocate 2D device grid metrics and copy from host
        allocate(CS%dyCv_d(isd:ied, jsd:jed));       CS%dyCv_d = dyCv
        allocate(CS%dxCu_d(isd:ied, jsd:jed));       CS%dxCu_d = dxCu
        allocate(CS%areaT_d(isd:ied, jsd:jed));      CS%areaT_d = areaT
        allocate(CS%IareaBu_d(isd:ied, jsd:jed));    CS%IareaBu_d = IareaBu
        allocate(CS%CoriolisBu_d(isd:ied, jsd:jed)); CS%CoriolisBu_d = CoriolisBu
        allocate(CS%mask2dBu_d(isd:ied, jsd:jed));   CS%mask2dBu_d = mask2dBu
        allocate(CS%dyCu_d(isd:ied, jsd:jed));       CS%dyCu_d = dyCu
        allocate(CS%dxCv_d(isd:ied, jsd:jed));       CS%dxCv_d = dxCv
        allocate(CS%IdxCu_d(isd:ied, jsd:jed));      CS%IdxCu_d = IdxCu
        allocate(CS%IdyCv_d(isd:ied, jsd:jed));      CS%IdyCv_d = IdyCv

        CS%initialized = .true.

    end subroutine coriolis_init_cuda

    subroutine coriolis_end_cuda(CS)
        type(coriolis_CS_cuda), intent(inout) :: CS

        if (.not. CS%initialized) return

        if (allocated(CS%q_d)) deallocate(CS%q_d)
        if (allocated(CS%KE_d)) deallocate(CS%KE_d)
        if (allocated(CS%a_d)) deallocate(CS%a_d)
        if (allocated(CS%b_d)) deallocate(CS%b_d)
        if (allocated(CS%c_d)) deallocate(CS%c_d)
        if (allocated(CS%d_d)) deallocate(CS%d_d)
        if (allocated(CS%dyCv_d)) deallocate(CS%dyCv_d)
        if (allocated(CS%dxCu_d)) deallocate(CS%dxCu_d)
        if (allocated(CS%areaT_d)) deallocate(CS%areaT_d)
        if (allocated(CS%IareaBu_d)) deallocate(CS%IareaBu_d)
        if (allocated(CS%CoriolisBu_d)) deallocate(CS%CoriolisBu_d)
        if (allocated(CS%mask2dBu_d)) deallocate(CS%mask2dBu_d)
        if (allocated(CS%dyCu_d)) deallocate(CS%dyCu_d)
        if (allocated(CS%dxCv_d)) deallocate(CS%dxCv_d)
        if (allocated(CS%IdxCu_d)) deallocate(CS%IdxCu_d)
        if (allocated(CS%IdyCv_d)) deallocate(CS%IdyCv_d)

        CS%initialized = .false.

    end subroutine coriolis_end_cuda

    subroutine CorAdCalc_cuda(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS, bx_in, by_in)
        type(coriolis_CS_cuda), intent(inout) :: CS
        real(dp), device, intent(in)  :: u_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in)  :: v_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in)  :: h_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in)  :: uh_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in)  :: vh_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(out) :: CAu_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(out) :: CAv_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, n3, is_l, ie_l, js_l, je_l, istat, bx, by
        type(dim3) :: grid, tBlock

        ! Configurable block dimensions (default: 32 x 4 x 1 = 128 threads)
        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        ! Array dimensions (1-based for kernel)
        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        n3 = CS%nz

        ! Compute-domain offsets in 1-based local coords
        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        ! Thread block: bx x by x 1 (bx in i for warp coalescing)
        tBlock = dim3(bx, by, 1)
        ! Grid covers full array extent, one k-level per block in z
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), n3)

        ! Merged Phase 1+2: compute q and KE (all intermediates in registers)
        call phase12_merged_kernel<<<grid, tBlock>>>( &
            CS%q_d, CS%KE_d, u_d, v_d, h_d, &
            CS%dyCv_d, CS%dxCu_d, CS%dyCu_d, CS%dxCv_d, CS%areaT_d, &
            CS%IareaBu_d, CS%CoriolisBu_d, CS%mask2dBu_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        ! Phase 3: Coriolis accelerations (scheme-dependent)
        if (CS%Coriolis_Scheme == SADOURNY75_ENERGY_CUDA) then
            call coriolis_sadourny_kernel<<<grid, tBlock>>>( &
                CAu_d, CAv_d, CS%q_d, CS%KE_d, uh_d, vh_d, &
                CS%IdxCu_d, CS%IdyCv_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l)

        else if (CS%Coriolis_Scheme == ARAKAWA_HSU90_CUDA) then
            call arakawa_hsu90_coef_kernel<<<grid, tBlock>>>( &
                CS%a_d, CS%b_d, CS%c_d, CS%d_d, CS%q_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l)
            call coriolis_arakawa_kernel<<<grid, tBlock>>>( &
                CAu_d, CAv_d, CS%a_d, CS%b_d, CS%c_d, CS%d_d, &
                CS%KE_d, uh_d, vh_d, CS%IdxCu_d, CS%IdyCv_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l)

        else if (CS%Coriolis_Scheme == ARAKAWA_LAMB81_CUDA) then
            call arakawa_lamb81_coef_kernel<<<grid, tBlock>>>( &
                CS%a_d, CS%b_d, CS%c_d, CS%d_d, CS%q_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l)
            call coriolis_arakawa_kernel<<<grid, tBlock>>>( &
                CAu_d, CAv_d, CS%a_d, CS%b_d, CS%c_d, CS%d_d, &
                CS%KE_d, uh_d, vh_d, CS%IdxCu_d, CS%IdyCv_d, &
                n1, n2, n3, is_l, ie_l, js_l, je_l)
        end if

        ! Sync to ensure all output is ready before returning
        istat = cudaDeviceSynchronize()

    end subroutine CorAdCalc_cuda

end module mom6_coriolis_cuda
