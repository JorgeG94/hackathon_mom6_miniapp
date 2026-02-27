!> MOM6 Coriolis and Momentum Advection Module (CUDA Fortran variant)
!!
!! Uses explicit attributes(global) CUDA kernels for GPU execution.
!! 3D work arrays for full k-parallelism. Three kernel launches minimum
!! (circulation+KE, vorticity+PV, Coriolis accel) with implicit sync
!! between dependent phases.
!!
module mom6_coriolis_cuda
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: CorAdCalc_cuda, CorAdCalc_cuda_fused, coriolis_init_cuda, coriolis_end_cuda
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

        ! 3D device work arrays
        real(dp), device, allocatable :: dvdx(:,:,:)
        real(dp), device, allocatable :: dudy(:,:,:)
        real(dp), device, allocatable :: rel_vort(:,:,:)
        real(dp), device, allocatable :: abs_vort(:,:,:)
        real(dp), device, allocatable :: q_d(:,:,:)
        real(dp), device, allocatable :: Ih_q(:,:,:)
        real(dp), device, allocatable :: hArea_u(:,:,:)
        real(dp), device, allocatable :: hArea_v(:,:,:)
        real(dp), device, allocatable :: Area_q(:,:,:)
        real(dp), device, allocatable :: KE_d(:,:,:)
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

        ! CUDA streams for concurrent kernel execution
        integer(cuda_stream_kind) :: stream_a = 0   ! Phase 1a → Phase 2
        integer(cuda_stream_kind) :: stream_b = 0   ! Phase 1b (KE)
        type(cudaEvent) :: event_ke_done             ! signaled when KE is ready
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

    !> Phase 1: Compute circulation (dvdx, dudy), area-weighted thickness
    !! (hArea_u, hArea_v), and kinetic energy (KE) from input arrays only.
    !! No cross-dependency — all reads from u, v, h, grid metrics.
    attributes(global) subroutine phase1_kernel( &
            dvdx, dudy, hArea_u, hArea_v, KE, &
            u, v, h, dyCv, dxCu, dyCu, dxCv, areaT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: dvdx(n1, n2, n3), dudy(n1, n2, n3)
        real(dp), intent(out) :: hArea_u(n1, n2, n3), hArea_v(n1, n2, n3), KE(n1, n2, n3)
        real(dp), intent(in) :: u(n1, n2, n3), v(n1, n2, n3), h(n1, n2, n3)
        real(dp), intent(in) :: dyCv(n1, n2), dxCu(n1, n2)
        real(dp), intent(in) :: dyCu(n1, n2), dxCv(n1, n2), areaT(n1, n2)

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        ! dvdx, dudy: i in [is_l-1 : ie_l], j in [js_l-1 : je_l]
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l - 1 .and. j <= je_l) then
            dvdx(i, j, k) = (v(i + 1, j, k) * dyCv(i + 1, j)) - &
                             (v(i, j, k) * dyCv(i, j))
            dudy(i, j, k) = (u(i, j + 1, k) * dxCu(i, j + 1)) - &
                             (u(i, j, k) * dxCu(i, j))
        end if

        ! hArea_v: i in [is_l : ie_l+1], j in [js_l-1 : je_l]
        if (i >= is_l .and. i <= ie_l + 1 .and. j >= js_l - 1 .and. j <= je_l) then
            hArea_v(i, j, k) = 0.5_dp * ((areaT(i, j) * h(i, j, k)) + &
                                           (areaT(i, j + 1) * h(i, j + 1, k)))
        end if

        ! hArea_u: i in [is_l-1 : ie_l], j in [js_l : je_l+1]
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l .and. j <= je_l + 1) then
            hArea_u(i, j, k) = 0.5_dp * ((areaT(i, j) * h(i, j, k)) + &
                                           (areaT(i + 1, j) * h(i + 1, j, k)))
        end if

        ! KE: i in [is_l : ie_l], j in [js_l : je_l]
        if (i >= is_l .and. i <= ie_l .and. j >= js_l .and. j <= je_l) then
            KE(i, j, k) = 0.25_dp * ( &
                (dyCu(i, j) * u(i, j, k)**2 + dyCu(i - 1, j) * u(i - 1, j, k)**2) + &
                (dxCv(i, j) * v(i, j, k)**2 + dxCv(i, j - 1) * v(i, j - 1, k)**2) &
                ) / areaT(i, j)
        end if

    end subroutine phase1_kernel

    !> Phase 1 with shared memory for 3D input data (u, v, h).
    !! Cooperatively loads stencil tile + halo into shared memory so
    !! neighbor accesses come from fast shared mem instead of L1/DRAM.
    !! Requires launch with dim3(32, 4, 1) block size.
    attributes(global) subroutine phase1_kernel_smem( &
            dvdx, dudy, hArea_u, hArea_v, KE, &
            u, v, h, dyCv, dxCu, dyCu, dxCv, areaT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: dvdx(n1, n2, n3), dudy(n1, n2, n3)
        real(dp), intent(out) :: hArea_u(n1, n2, n3), hArea_v(n1, n2, n3), KE(n1, n2, n3)
        real(dp), intent(in) :: u(n1, n2, n3), v(n1, n2, n3), h(n1, n2, n3)
        real(dp), intent(in) :: dyCv(n1, n2), dxCu(n1, n2)
        real(dp), intent(in) :: dyCu(n1, n2), dxCv(n1, n2), areaT(n1, n2)

        ! Tile parameters — must match launch block dimensions
        integer, parameter :: BLK_X = 32, BLK_Y = 4
        ! Shared tile with 1-cell halo: u needs (i-1)/(j+1), v needs (i+1)/(j-1),
        ! h needs (i+1)/(j+1). Union: halo -1..+1 in both dirs.
        integer, parameter :: SX = BLK_X + 2, SY = BLK_Y + 2  ! 34 x 6

        ! Shared memory for 3D stencil data: 3 * 34*6*8 = 4896 bytes
        real(dp), shared :: s_u(SX, SY)
        real(dp), shared :: s_v(SX, SY)
        real(dp), shared :: s_h(SX, SY)

        integer :: i, j, k, tx, ty, tid, si, sj, gi, gj, idx

        tx = threadIdx%x
        ty = threadIdx%y
        i = (blockIdx%x - 1) * BLK_X + tx
        j = (blockIdx%y - 1) * BLK_Y + ty
        k = blockIdx%z

        ! --- Cooperatively load u, v, h tile + halo into shared memory ---
        ! s_*(1,1) = global(i_base-1, j_base-1, k)
        ! s_*(tx+1, ty+1) = global(i, j, k)
        tid = (ty - 1) * BLK_X + tx  ! 1-based [1..128]

        do idx = tid, SX * SY, BLK_X * BLK_Y
            si = mod(idx - 1, SX) + 1
            sj = (idx - 1) / SX + 1
            gi = (blockIdx%x - 1) * BLK_X + si - 1
            gj = (blockIdx%y - 1) * BLK_Y + sj - 1

            if (gi >= 1 .and. gi <= n1 .and. gj >= 1 .and. gj <= n2 &
                .and. k >= 1 .and. k <= n3) then
                s_u(si, sj) = u(gi, gj, k)
                s_v(si, sj) = v(gi, gj, k)
                s_h(si, sj) = h(gi, gj, k)
            end if
        end do

        call syncthreads()

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        ! Shared indexing: s_*(tx+1, ty+1) = (i,j)
        !   (tx, ty+1) = (i-1,j),  (tx+2, ty+1) = (i+1,j)
        !   (tx+1, ty) = (i,j-1),  (tx+1, ty+2) = (i,j+1)

        ! dvdx, dudy
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l - 1 .and. j <= je_l) then
            dvdx(i, j, k) = (s_v(tx + 2, ty + 1) * dyCv(i + 1, j)) - &
                             (s_v(tx + 1, ty + 1) * dyCv(i, j))
            dudy(i, j, k) = (s_u(tx + 1, ty + 2) * dxCu(i, j + 1)) - &
                             (s_u(tx + 1, ty + 1) * dxCu(i, j))
        end if

        ! hArea_v
        if (i >= is_l .and. i <= ie_l + 1 .and. j >= js_l - 1 .and. j <= je_l) then
            hArea_v(i, j, k) = 0.5_dp * ((areaT(i, j) * s_h(tx + 1, ty + 1)) + &
                                           (areaT(i, j + 1) * s_h(tx + 1, ty + 2)))
        end if

        ! hArea_u
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l .and. j <= je_l + 1) then
            hArea_u(i, j, k) = 0.5_dp * ((areaT(i, j) * s_h(tx + 1, ty + 1)) + &
                                           (areaT(i + 1, j) * s_h(tx + 2, ty + 1)))
        end if

        ! KE
        if (i >= is_l .and. i <= ie_l .and. j >= js_l .and. j <= je_l) then
            KE(i, j, k) = 0.25_dp * ( &
                (dyCu(i, j) * s_u(tx + 1, ty + 1)**2 + &
                 dyCu(i - 1, j) * s_u(tx, ty + 1)**2) + &
                (dxCv(i, j) * s_v(tx + 1, ty + 1)**2 + &
                 dxCv(i, j - 1) * s_v(tx + 1, ty)**2) &
                ) / areaT(i, j)
        end if

    end subroutine phase1_kernel_smem

    !> Phase 1a: Compute circulation (dvdx, dudy) and area-weighted thickness
    !! (hArea_u, hArea_v). These are the quantities Phase 2 depends on.
    !! Separated from KE to allow KE to run concurrently with Phase 2.
    attributes(global) subroutine phase1a_circ_kernel( &
            dvdx, dudy, hArea_u, hArea_v, &
            u, v, h, dyCv, dxCu, areaT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: dvdx(n1, n2, n3), dudy(n1, n2, n3)
        real(dp), intent(out) :: hArea_u(n1, n2, n3), hArea_v(n1, n2, n3)
        real(dp), intent(in) :: u(n1, n2, n3), v(n1, n2, n3), h(n1, n2, n3)
        real(dp), intent(in) :: dyCv(n1, n2), dxCu(n1, n2), areaT(n1, n2)

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        ! dvdx, dudy: i in [is_l-1 : ie_l], j in [js_l-1 : je_l]
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l - 1 .and. j <= je_l) then
            dvdx(i, j, k) = (v(i + 1, j, k) * dyCv(i + 1, j)) - &
                             (v(i, j, k) * dyCv(i, j))
            dudy(i, j, k) = (u(i, j + 1, k) * dxCu(i, j + 1)) - &
                             (u(i, j, k) * dxCu(i, j))
        end if

        ! hArea_v: i in [is_l : ie_l+1], j in [js_l-1 : je_l]
        if (i >= is_l .and. i <= ie_l + 1 .and. j >= js_l - 1 .and. j <= je_l) then
            hArea_v(i, j, k) = 0.5_dp * ((areaT(i, j) * h(i, j, k)) + &
                                           (areaT(i, j + 1) * h(i, j + 1, k)))
        end if

        ! hArea_u: i in [is_l-1 : ie_l], j in [js_l : je_l+1]
        if (i >= is_l - 1 .and. i <= ie_l .and. j >= js_l .and. j <= je_l + 1) then
            hArea_u(i, j, k) = 0.5_dp * ((areaT(i, j) * h(i, j, k)) + &
                                           (areaT(i + 1, j) * h(i + 1, j, k)))
        end if

    end subroutine phase1a_circ_kernel

    !> Phase 1b: Compute kinetic energy only. Independent of Phase 2, needed
    !! only by Phase 3. Can run on a separate stream concurrent with Phase 2.
    attributes(global) subroutine phase1b_ke_kernel( &
            KE, u, v, dyCu, dxCv, areaT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: KE(n1, n2, n3)
        real(dp), intent(in) :: u(n1, n2, n3), v(n1, n2, n3)
        real(dp), intent(in) :: dyCu(n1, n2), dxCv(n1, n2), areaT(n1, n2)

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        ! KE: i in [is_l : ie_l], j in [js_l : je_l]
        if (i >= is_l .and. i <= ie_l .and. j >= js_l .and. j <= je_l) then
            KE(i, j, k) = 0.25_dp * ( &
                (dyCu(i, j) * u(i, j, k)**2 + dyCu(i - 1, j) * u(i - 1, j, k)**2) + &
                (dxCv(i, j) * v(i, j, k)**2 + dxCv(i, j - 1) * v(i, j - 1, k)**2) &
                ) / areaT(i, j)
        end if

    end subroutine phase1b_ke_kernel

    !> Phase 2: Compute vorticity and potential vorticity from Phase 1 results.
    !! Reads dvdx, dudy at same point; hArea_u, hArea_v at neighbors.
    attributes(global) subroutine vorticity_pv_kernel( &
            rel_vort, abs_vort, q, Ih_q, &
            dvdx, dudy, hArea_u, hArea_v, Area_q, &
            IareaBu, CoriolisBu, mask2dBu, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: rel_vort(n1, n2, n3), abs_vort(n1, n2, n3)
        real(dp), intent(out) :: q(n1, n2, n3), Ih_q(n1, n2, n3)
        real(dp), intent(in) :: dvdx(n1, n2, n3), dudy(n1, n2, n3)
        real(dp), intent(in) :: hArea_u(n1, n2, n3), hArea_v(n1, n2, n3), Area_q(n1, n2, n3)
        real(dp), intent(in) :: IareaBu(n1, n2), CoriolisBu(n1, n2), mask2dBu(n1, n2)

        integer :: i, j, k
        real(dp) :: hArea_q_val

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        ! Range: i in [is_l-1 : ie_l], j in [js_l-1 : je_l]
        if (i < is_l - 1 .or. i > ie_l .or. j < js_l - 1 .or. j > je_l &
            .or. k < 1 .or. k > n3) return

        rel_vort(i, j, k) = mask2dBu(i, j) * &
            (dvdx(i, j, k) - dudy(i, j, k)) * IareaBu(i, j)
        abs_vort(i, j, k) = CoriolisBu(i, j) + rel_vort(i, j, k)
        hArea_q_val = (hArea_u(i, j, k) + hArea_u(i, j + 1, k)) + &
                      (hArea_v(i, j, k) + hArea_v(i + 1, j, k))
        Ih_q(i, j, k) = Area_q(i, j, k) / (hArea_q_val + 1.0e-20_dp)
        q(i, j, k) = abs_vort(i, j, k) * Ih_q(i, j, k)

    end subroutine vorticity_pv_kernel

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
    ! Device helper functions for fused kernel
    !=========================================================================

    !> Compute potential vorticity q at vorticity point (ii,jj,k) from inputs.
    attributes(device) function compute_pv_inline( &
            u, v, h, dyCv, dxCu, areaT, IareaBu, CoriolisBu, mask2dBu, &
            n1, n2, n3, ii, jj, k) result(q_val)
        integer, value :: n1, n2, n3, ii, jj, k
        real(dp) :: u(n1, n2, n3), v(n1, n2, n3), h(n1, n2, n3)
        real(dp) :: dyCv(n1, n2), dxCu(n1, n2), areaT(n1, n2)
        real(dp) :: IareaBu(n1, n2), CoriolisBu(n1, n2), mask2dBu(n1, n2)
        real(dp) :: q_val

        real(dp) :: dvdx, dudy, rel_vort, abs_vort
        real(dp) :: hau, hau_jp1, hav, hav_ip1, haq, aq

        dvdx = v(ii + 1, jj, k) * dyCv(ii + 1, jj) - v(ii, jj, k) * dyCv(ii, jj)
        dudy = u(ii, jj + 1, k) * dxCu(ii, jj + 1) - u(ii, jj, k) * dxCu(ii, jj)
        rel_vort = mask2dBu(ii, jj) * (dvdx - dudy) * IareaBu(ii, jj)
        abs_vort = CoriolisBu(ii, jj) + rel_vort

        hau     = 0.5_dp * (areaT(ii, jj)     * h(ii, jj, k)     + areaT(ii + 1, jj)     * h(ii + 1, jj, k))
        hau_jp1 = 0.5_dp * (areaT(ii, jj + 1) * h(ii, jj + 1, k) + areaT(ii + 1, jj + 1) * h(ii + 1, jj + 1, k))
        hav     = 0.5_dp * (areaT(ii, jj)     * h(ii, jj, k)     + areaT(ii, jj + 1)     * h(ii, jj + 1, k))
        hav_ip1 = 0.5_dp * (areaT(ii + 1, jj) * h(ii + 1, jj, k) + areaT(ii + 1, jj + 1) * h(ii + 1, jj + 1, k))
        haq = hau + hau_jp1 + hav + hav_ip1
        aq  = areaT(ii, jj) + areaT(ii + 1, jj + 1) + areaT(ii + 1, jj) + areaT(ii, jj + 1)

        q_val = abs_vort * aq / (haq + 1.0e-20_dp)
    end function compute_pv_inline

    !> Compute kinetic energy at tracer point (ii,jj,k) from inputs.
    attributes(device) function compute_ke_inline( &
            u, v, dyCu, dxCv, areaT, &
            n1, n2, n3, ii, jj, k) result(ke_val)
        integer, value :: n1, n2, n3, ii, jj, k
        real(dp) :: u(n1, n2, n3), v(n1, n2, n3)
        real(dp) :: dyCu(n1, n2), dxCv(n1, n2), areaT(n1, n2)
        real(dp) :: ke_val

        ke_val = 0.25_dp * ( &
            (dyCu(ii, jj) * u(ii, jj, k)**2 + dyCu(ii - 1, jj) * u(ii - 1, jj, k)**2) + &
            (dxCv(ii, jj) * v(ii, jj, k)**2 + dxCv(ii, jj - 1) * v(ii, jj - 1, k)**2) &
            ) / areaT(ii, jj)
    end function compute_ke_inline

    !=========================================================================
    ! Fused single-kernel Sadourny (no intermediate arrays)
    !=========================================================================

    !> Fused Coriolis kernel: computes CAu and CAv in a single launch by
    !! recomputing all intermediates (PV, KE) inline from input arrays.
    !! Trades extra FLOPs for eliminating 9 intermediate 3D array writes/reads.
    attributes(global) subroutine coriolis_fused_sadourny_kernel( &
            CAu, CAv, u, v, h, uh, vh, &
            dyCv, dxCu, dyCu, dxCv, areaT, &
            IareaBu, CoriolisBu, mask2dBu, IdxCu, IdyCv, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: CAu(n1, n2, n3), CAv(n1, n2, n3)
        real(dp), intent(in) :: u(n1, n2, n3), v(n1, n2, n3), h(n1, n2, n3)
        real(dp), intent(in) :: uh(n1, n2, n3), vh(n1, n2, n3)
        real(dp), intent(in) :: dyCv(n1, n2), dxCu(n1, n2), dyCu(n1, n2), dxCv(n1, n2)
        real(dp), intent(in) :: areaT(n1, n2)
        real(dp), intent(in) :: IareaBu(n1, n2), CoriolisBu(n1, n2), mask2dBu(n1, n2)
        real(dp), intent(in) :: IdxCu(n1, n2), IdyCv(n1, n2)

        integer :: i, j, k
        real(dp) :: q_ij, q_jm1, q_im1, KE_ij, KE_ip1, KE_jp1, KEx, KEy
        logical :: do_cau, do_cav

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (k < 1 .or. k > n3) return

        do_cau = (i >= is_l .and. i <= ie_l - 1 .and. j >= js_l .and. j <= je_l)
        do_cav = (i >= is_l .and. i <= ie_l .and. j >= js_l .and. j <= je_l - 1)

        if (.not. do_cau .and. .not. do_cav) return

        ! q(i,j,k) — needed by both CAu and CAv
        q_ij = compute_pv_inline(u, v, h, dyCv, dxCu, areaT, &
            IareaBu, CoriolisBu, mask2dBu, n1, n2, n3, i, j, k)

        ! KE(i,j,k) — needed by both CAu and CAv
        KE_ij = compute_ke_inline(u, v, dyCu, dxCv, areaT, n1, n2, n3, i, j, k)

        ! --- CAu(i,j,k) ---
        if (do_cau) then
            q_jm1 = compute_pv_inline(u, v, h, dyCv, dxCu, areaT, &
                IareaBu, CoriolisBu, mask2dBu, n1, n2, n3, i, j - 1, k)
            KE_ip1 = compute_ke_inline(u, v, dyCu, dxCv, areaT, n1, n2, n3, i + 1, j, k)
            KEx = (KE_ip1 - KE_ij) * IdxCu(i, j)
            CAu(i, j, k) = 0.25_dp * ( &
                (q_ij  * (vh(i + 1, j, k) + vh(i, j, k))) + &
                (q_jm1 * (vh(i, j - 1, k) + vh(i + 1, j - 1, k))) &
                ) * IdxCu(i, j) - KEx
        end if

        ! --- CAv(i,j,k) ---
        if (do_cav) then
            q_im1 = compute_pv_inline(u, v, h, dyCv, dxCu, areaT, &
                IareaBu, CoriolisBu, mask2dBu, n1, n2, n3, i - 1, j, k)
            KE_jp1 = compute_ke_inline(u, v, dyCu, dxCv, areaT, n1, n2, n3, i, j + 1, k)
            KEy = (KE_jp1 - KE_ij) * IdyCv(i, j)
            CAv(i, j, k) = -0.25_dp * ( &
                (q_im1 * (uh(i - 1, j, k) + uh(i - 1, j + 1, k))) + &
                (q_ij  * (uh(i, j, k) + uh(i, j + 1, k))) &
                ) * IdyCv(i, j) - KEy
        end if

    end subroutine coriolis_fused_sadourny_kernel

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

        integer :: i, j, k, istat
        real(dp), allocatable :: Area_q_h(:,:,:)

        CS%Coriolis_Scheme = SADOURNY75_ENERGY_CUDA
        if (present(scheme)) CS%Coriolis_Scheme = scheme

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec; CS%nz = nk

        ! Allocate 3D device work arrays
        allocate(CS%dvdx(isd:ied, jsd:jed, nk))
        allocate(CS%dudy(isd:ied, jsd:jed, nk))
        allocate(CS%rel_vort(isd:ied, jsd:jed, nk))
        allocate(CS%abs_vort(isd:ied, jsd:jed, nk))
        allocate(CS%q_d(isd:ied, jsd:jed, nk))
        allocate(CS%Ih_q(isd:ied, jsd:jed, nk))
        allocate(CS%hArea_u(isd:ied, jsd:jed, nk))
        allocate(CS%hArea_v(isd:ied, jsd:jed, nk))
        allocate(CS%Area_q(isd:ied, jsd:jed, nk))
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

        ! Precompute Area_q on host, then copy to device
        allocate(Area_q_h(isd:ied, jsd:jed, nk))
        do k = 1, nk
            do j = jsd, jed - 1
                do i = isd, ied - 1
                    Area_q_h(i, j, k) = (areaT(i, j) + areaT(i + 1, j + 1)) + &
                                        (areaT(i + 1, j) + areaT(i, j + 1))
                end do
            end do
        end do
        CS%Area_q = Area_q_h
        deallocate(Area_q_h)

        ! Create CUDA streams and event for concurrent execution
        istat = cudaStreamCreate(CS%stream_a)
        istat = cudaStreamCreate(CS%stream_b)
        istat = cudaEventCreate(CS%event_ke_done)

        CS%initialized = .true.

    end subroutine coriolis_init_cuda

    subroutine coriolis_end_cuda(CS)
        type(coriolis_CS_cuda), intent(inout) :: CS
        integer :: istat

        if (.not. CS%initialized) return

        ! Destroy streams and events
        istat = cudaStreamDestroy(CS%stream_a)
        istat = cudaStreamDestroy(CS%stream_b)
        istat = cudaEventDestroy(CS%event_ke_done)

        ! Deallocate device arrays
        if (allocated(CS%dvdx)) deallocate(CS%dvdx)
        if (allocated(CS%dudy)) deallocate(CS%dudy)
        if (allocated(CS%rel_vort)) deallocate(CS%rel_vort)
        if (allocated(CS%abs_vort)) deallocate(CS%abs_vort)
        if (allocated(CS%q_d)) deallocate(CS%q_d)
        if (allocated(CS%Ih_q)) deallocate(CS%Ih_q)
        if (allocated(CS%hArea_u)) deallocate(CS%hArea_u)
        if (allocated(CS%hArea_v)) deallocate(CS%hArea_v)
        if (allocated(CS%Area_q)) deallocate(CS%Area_q)
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

        ! Configurable block dimensions (default: 32 x 8 x 1 = 256 threads)
        bx = 32; by = 8
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

        ! Phase 1: circulation, area-weighted thickness, KE (all from inputs)
        call phase1_kernel<<<grid, tBlock>>>( &
            CS%dvdx, CS%dudy, CS%hArea_u, CS%hArea_v, CS%KE_d, &
            u_d, v_d, h_d, CS%dyCv_d, CS%dxCu_d, CS%dyCu_d, CS%dxCv_d, CS%areaT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        ! Phase 2: vorticity and PV (needs Phase 1 results at neighbors)
        call vorticity_pv_kernel<<<grid, tBlock>>>( &
            CS%rel_vort, CS%abs_vort, CS%q_d, CS%Ih_q, &
            CS%dvdx, CS%dudy, CS%hArea_u, CS%hArea_v, CS%Area_q, &
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

    !> Fused single-kernel Coriolis solver (Sadourny scheme only).
    !! Computes CAu and CAv in one kernel launch by recomputing all intermediates
    !! (PV, KE) inline — eliminates intermediate 3D array traffic.
    subroutine CorAdCalc_cuda_fused(u_d, v_d, h_d, uh_d, vh_d, CAu_d, CAv_d, CS, bx_in, by_in)
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

        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        n3 = CS%nz

        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), n3)

        call coriolis_fused_sadourny_kernel<<<grid, tBlock>>>( &
            CAu_d, CAv_d, u_d, v_d, h_d, uh_d, vh_d, &
            CS%dyCv_d, CS%dxCu_d, CS%dyCu_d, CS%dxCv_d, CS%areaT_d, &
            CS%IareaBu_d, CS%CoriolisBu_d, CS%mask2dBu_d, &
            CS%IdxCu_d, CS%IdyCv_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        istat = cudaDeviceSynchronize()

    end subroutine CorAdCalc_cuda_fused

end module mom6_coriolis_cuda
