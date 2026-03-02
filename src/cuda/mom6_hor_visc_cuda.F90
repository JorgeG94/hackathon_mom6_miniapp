!> MOM6 Horizontal Viscosity Module (CUDA Fortran variant)
!!
!! Implements the Laplacian-only horizontal viscosity path using explicit
!! attributes(global) CUDA kernels. Uses 3D kernels that parallelize over
!! the vertical (k) direction via blockIdx%z, eliminating the host k-loop.
!!
!! Five 3D kernels:
!!   1. vel_grad_kernel  -- velocity gradients (dudx, dvdy, dvdx, dudy)
!!   2. strain_kernel    -- strain tensor (sh_xx, sh_xy) with free-slip BC
!!   3. stress_xx_kernel -- diagonal stress str_xx at h-points
!!   4. stress_xy_kernel -- off-diagonal stress str_xy at q-points
!!   5. divergence_kernel-- viscous acceleration (diffu, diffv)
!!
!! 1-based local indexing, block size 32x4, k in blockIdx%z.
!!
module mom6_hor_visc_cuda
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: hor_visc_init_cuda, hor_visc_cuda, hor_visc_end_cuda
    public :: hor_visc_CS_cuda

    !> Control structure for CUDA Fortran horizontal viscosity solver
    type :: hor_visc_CS_cuda
        logical :: initialized = .false.
        integer :: is, ie, js, je, isd, ied, jsd, jed, nz

        ! Scalar parameters
        real(dp) :: Kh_bg                    ! Background Laplacian viscosity
        real(dp) :: h_neglect

        ! Device 2D metric arrays (copied once at init)
        real(dp), device, allocatable :: DY_dxT_d(:,:)   ! dy/dx at h-points
        real(dp), device, allocatable :: DX_dyT_d(:,:)   ! dx/dy at h-points
        real(dp), device, allocatable :: DY_dxBu_d(:,:)  ! dy/dx at q-points
        real(dp), device, allocatable :: DX_dyBu_d(:,:)  ! dx/dy at q-points
        real(dp), device, allocatable :: IdyCu_d(:,:)
        real(dp), device, allocatable :: IdxCu_d(:,:)
        real(dp), device, allocatable :: IdyCv_d(:,:)
        real(dp), device, allocatable :: IdxCv_d(:,:)
        real(dp), device, allocatable :: IareaCu_d(:,:)
        real(dp), device, allocatable :: IareaCv_d(:,:)
        real(dp), device, allocatable :: mask2dT_d(:,:)
        real(dp), device, allocatable :: mask2dBu_d(:,:)
        real(dp), device, allocatable :: reduction_xx_d(:,:)
        real(dp), device, allocatable :: reduction_xy_d(:,:)
        real(dp), device, allocatable :: dy2h_d(:,:)
        real(dp), device, allocatable :: dx2h_d(:,:)
        real(dp), device, allocatable :: dy2q_d(:,:)
        real(dp), device, allocatable :: dx2q_d(:,:)

        ! Device 3D work arrays (one per layer)
        real(dp), device, allocatable :: dudx_d(:,:,:)
        real(dp), device, allocatable :: dvdy_d(:,:,:)
        real(dp), device, allocatable :: dvdx_d(:,:,:)
        real(dp), device, allocatable :: dudy_d(:,:,:)
        real(dp), device, allocatable :: sh_xx_d(:,:,:)
        real(dp), device, allocatable :: sh_xy_d(:,:,:)
        real(dp), device, allocatable :: str_xx_d(:,:,:)
        real(dp), device, allocatable :: str_xy_d(:,:,:)
    end type hor_visc_CS_cuda

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
    !
    ! Index conventions (matching MOM6):
    !   Isq_l = is_l - 1,  Ieq_l = ie_l   (staggered u-point range)
    !   Jsq_l = js_l - 1,  Jeq_l = je_l   (staggered v-point range)
    !
    ! k = blockIdx%z indexes the vertical layer.
    !=========================================================================

    !> Kernel 1: Compute velocity gradients (3D)
    !! dudx, dvdy at h-points (i in [Isq:Ieq+1], j in [Jsq:Jeq+1])
    !! dvdx, dudy at q-points (I in [Isq:Ieq], J in [Jsq:Jeq])
    attributes(global) subroutine vel_grad_kernel( &
            dudx, dvdy, dvdx, dudy, &
            u, v, &
            DY_dxT, DX_dyT, DY_dxBu, DX_dyBu, &
            IdyCu, IdxCu, IdyCv, IdxCv, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: dudx(n1, n2, n3)
        real(dp), intent(out) :: dvdy(n1, n2, n3)
        real(dp), intent(out) :: dvdx(n1, n2, n3)
        real(dp), intent(out) :: dudy(n1, n2, n3)
        real(dp), intent(in) :: u(n1, n2, n3)
        real(dp), intent(in) :: v(n1, n2, n3)
        real(dp), intent(in) :: DY_dxT(n1, n2)
        real(dp), intent(in) :: DX_dyT(n1, n2)
        real(dp), intent(in) :: DY_dxBu(n1, n2)
        real(dp), intent(in) :: DX_dyBu(n1, n2)
        real(dp), intent(in) :: IdyCu(n1, n2)
        real(dp), intent(in) :: IdxCu(n1, n2)
        real(dp), intent(in) :: IdyCv(n1, n2)
        real(dp), intent(in) :: IdxCv(n1, n2)

        integer :: i, j, k
        integer :: Isq_l, Ieq_l, Jsq_l, Jeq_l

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        Isq_l = is_l - 1
        Ieq_l = ie_l
        Jsq_l = js_l - 1
        Jeq_l = je_l

        ! dudx, dvdy at h-points: i in [Isq:Ieq+1], j in [Jsq:Jeq+1]
        if (i >= Isq_l .and. i <= Ieq_l + 1 .and. &
            j >= Jsq_l .and. j <= Jeq_l + 1) then
            dudx(i, j, k) = DY_dxT(i, j) * (IdyCu(i, j) * u(i, j, k) - &
                                               IdyCu(i - 1, j) * u(i - 1, j, k))
            dvdy(i, j, k) = DX_dyT(i, j) * (IdxCv(i, j) * v(i, j, k) - &
                                               IdxCv(i, j - 1) * v(i, j - 1, k))
        end if

        ! dvdx, dudy at q-points: I in [Isq:Ieq], J in [Jsq:Jeq]
        if (i >= Isq_l .and. i <= Ieq_l .and. &
            j >= Jsq_l .and. j <= Jeq_l) then
            dvdx(i, j, k) = DY_dxBu(i, j) * (v(i + 1, j, k) * IdyCv(i + 1, j) - &
                                                v(i, j, k) * IdyCv(i, j))
            dudy(i, j, k) = DX_dyBu(i, j) * (u(i, j + 1, k) * IdxCu(i, j + 1) - &
                                                u(i, j, k) * IdxCu(i, j))
        end if

    end subroutine vel_grad_kernel

    !> Kernel 2: Compute strain tensor (3D)
    !! sh_xx at h-points (i in [Isq:Ieq+1], j in [Jsq:Jeq+1])
    !! sh_xy at q-points (I in [Isq:Ieq], J in [Jsq:Jeq]) with free-slip BC
    attributes(global) subroutine strain_kernel( &
            sh_xx, sh_xy, &
            dudx, dvdy, dvdx, dudy, &
            mask2dBu, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: sh_xx(n1, n2, n3)
        real(dp), intent(out) :: sh_xy(n1, n2, n3)
        real(dp), intent(in) :: dudx(n1, n2, n3)
        real(dp), intent(in) :: dvdy(n1, n2, n3)
        real(dp), intent(in) :: dvdx(n1, n2, n3)
        real(dp), intent(in) :: dudy(n1, n2, n3)
        real(dp), intent(in) :: mask2dBu(n1, n2)

        integer :: i, j, k
        integer :: Isq_l, Ieq_l, Jsq_l, Jeq_l

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        Isq_l = is_l - 1
        Ieq_l = ie_l
        Jsq_l = js_l - 1
        Jeq_l = je_l

        ! sh_xx at h-points: i in [Isq:Ieq+1], j in [Jsq:Jeq+1]
        if (i >= Isq_l .and. i <= Ieq_l + 1 .and. &
            j >= Jsq_l .and. j <= Jeq_l + 1) then
            sh_xx(i, j, k) = dudx(i, j, k) - dvdy(i, j, k)
        end if

        ! sh_xy at q-points: I in [Isq:Ieq], J in [Jsq:Jeq]
        ! Free-slip BC: zero strain at boundaries (mask=0 means land)
        if (i >= Isq_l .and. i <= Ieq_l .and. &
            j >= Jsq_l .and. j <= Jeq_l) then
            sh_xy(i, j, k) = mask2dBu(i, j) * (dvdx(i, j, k) + dudy(i, j, k))
        end if

    end subroutine strain_kernel

    !> Kernel 3: Compute diagonal stress str_xx at h-points (3D)
    !! str_xx = -Kh * sh_xx * h * reduction_xx
    !! Range: i in [Isq:Ieq+1], j in [Jsq:Jeq+1]
    attributes(global) subroutine stress_xx_kernel( &
            str_xx, sh_xx, h, reduction_xx, &
            Kh_bg, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: Kh_bg
        real(dp), intent(out) :: str_xx(n1, n2, n3)
        real(dp), intent(in) :: sh_xx(n1, n2, n3)
        real(dp), intent(in) :: h(n1, n2, n3)
        real(dp), intent(in) :: reduction_xx(n1, n2)

        integer :: i, j, k
        integer :: Isq_l, Ieq_l, Jsq_l, Jeq_l

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        Isq_l = is_l - 1
        Ieq_l = ie_l
        Jsq_l = js_l - 1
        Jeq_l = je_l

        if (i >= Isq_l .and. i <= Ieq_l + 1 .and. &
            j >= Jsq_l .and. j <= Jeq_l + 1) then
            str_xx(i, j, k) = -Kh_bg * sh_xx(i, j, k) * h(i, j, k) * reduction_xx(i, j)
        end if

    end subroutine stress_xx_kernel

    !> Kernel 4: Compute off-diagonal stress str_xy at q-points (3D)
    !! str_xy = -Kh * sh_xy * hq * mask2dBu * reduction_xy
    !! hq = 0.25 * (h(i,j,k) + h(i+1,j+1,k) + h(i+1,j,k) + h(i,j+1,k))
    !! Range: I in [Isq:Ieq], J in [Jsq:Jeq]
    attributes(global) subroutine stress_xy_kernel( &
            str_xy, sh_xy, h, mask2dBu, reduction_xy, &
            Kh_bg, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: Kh_bg
        real(dp), intent(out) :: str_xy(n1, n2, n3)
        real(dp), intent(in) :: sh_xy(n1, n2, n3)
        real(dp), intent(in) :: h(n1, n2, n3)
        real(dp), intent(in) :: mask2dBu(n1, n2)
        real(dp), intent(in) :: reduction_xy(n1, n2)

        integer :: i, j, k
        integer :: Isq_l, Ieq_l, Jsq_l, Jeq_l
        real(dp) :: hq_val

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        Isq_l = is_l - 1
        Ieq_l = ie_l
        Jsq_l = js_l - 1
        Jeq_l = je_l

        if (i >= Isq_l .and. i <= Ieq_l .and. &
            j >= Jsq_l .and. j <= Jeq_l) then
            hq_val = 0.25_dp * ((h(i, j, k) + h(i + 1, j + 1, k)) + &
                                 (h(i + 1, j, k) + h(i, j + 1, k)))
            str_xy(i, j, k) = -Kh_bg * sh_xy(i, j, k) * hq_val * &
                                 mask2dBu(i, j) * reduction_xy(i, j)
        end if

    end subroutine stress_xy_kernel

    !> Kernel 5: Compute viscous acceleration from stress divergence (3D)
    !! diffu at u-points (I in [Isq:Ieq], j in [js:je])
    !! diffv at v-points (i in [is:ie], J in [Jsq:Jeq])
    attributes(global) subroutine divergence_kernel( &
            diffu, diffv, &
            str_xx, str_xy, h, &
            mask2dT, &
            IdyCu, IdxCu, IdyCv, IdxCv, &
            IareaCu, IareaCv, &
            dy2h, dx2h, dy2q, dx2q, &
            h_neglect, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), value, intent(in) :: h_neglect
        real(dp), intent(out) :: diffu(n1, n2, n3)
        real(dp), intent(out) :: diffv(n1, n2, n3)
        real(dp), intent(in) :: str_xx(n1, n2, n3)
        real(dp), intent(in) :: str_xy(n1, n2, n3)
        real(dp), intent(in) :: h(n1, n2, n3)
        real(dp), intent(in) :: mask2dT(n1, n2)
        real(dp), intent(in) :: IdyCu(n1, n2)
        real(dp), intent(in) :: IdxCu(n1, n2)
        real(dp), intent(in) :: IdyCv(n1, n2)
        real(dp), intent(in) :: IdxCv(n1, n2)
        real(dp), intent(in) :: IareaCu(n1, n2)
        real(dp), intent(in) :: IareaCv(n1, n2)
        real(dp), intent(in) :: dy2h(n1, n2)
        real(dp), intent(in) :: dx2h(n1, n2)
        real(dp), intent(in) :: dy2q(n1, n2)
        real(dp), intent(in) :: dx2q(n1, n2)

        integer :: i, j, k
        integer :: Isq_l, Ieq_l, Jsq_l, Jeq_l
        real(dp) :: h_u, h_v

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        if (i > n1 .or. j > n2 .or. k < 1 .or. k > n3) return

        Isq_l = is_l - 1
        Ieq_l = ie_l
        Jsq_l = js_l - 1
        Jeq_l = je_l

        ! diffu at u-points: I in [Isq:Ieq], j in [js:je]
        if (i >= Isq_l .and. i <= Ieq_l .and. &
            j >= js_l .and. j <= je_l) then
            h_u = 0.5_dp * (mask2dT(i, j) * h(i, j, k) + &
                             mask2dT(i + 1, j) * h(i + 1, j, k))
            diffu(i, j, k) = ((IdxCu(i, j) * (dx2q(i, j - 1) * str_xy(i, j - 1, k) - &
                                                 dx2q(i, j) * str_xy(i, j, k)) + &
                                 IdyCu(i, j) * (dy2h(i, j) * str_xx(i, j, k) - &
                                                 dy2h(i + 1, j) * str_xx(i + 1, j, k))) * &
                                IareaCu(i, j)) / (h_u + h_neglect)
        end if

        ! diffv at v-points: i in [is:ie], J in [Jsq:Jeq]
        if (i >= is_l .and. i <= ie_l .and. &
            j >= Jsq_l .and. j <= Jeq_l) then
            h_v = 0.5_dp * (mask2dT(i, j) * h(i, j, k) + &
                             mask2dT(i, j + 1) * h(i, j + 1, k))
            diffv(i, j, k) = ((IdyCv(i, j) * (dy2q(i - 1, j) * str_xy(i - 1, j, k) - &
                                                 dy2q(i, j) * str_xy(i, j, k)) - &
                                 IdxCv(i, j) * (dx2h(i, j) * str_xx(i, j, k) - &
                                                 dx2h(i, j + 1) * str_xx(i, j + 1, k))) * &
                                IareaCv(i, j)) / (h_v + h_neglect)
        end if

    end subroutine divergence_kernel

    !=========================================================================
    ! Host routines
    !=========================================================================

    !> Initialize the CUDA horizontal viscosity solver.
    !! Copies metric arrays to device once. Allocates 3D device work arrays.
    subroutine hor_visc_init_cuda(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nz, &
                                   Kh, h_neglect, &
                                   DY_dxT, DX_dyT, DY_dxBu, DX_dyBu, &
                                   IdyCu, IdxCu, IdyCv, IdxCv, &
                                   IareaCu, IareaCv, &
                                   mask2dT, mask2dBu, &
                                   reduction_xx, reduction_xy, &
                                   dy2h, dx2h, dy2q, dx2q)
        type(hor_visc_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed
        integer, intent(in) :: isc, iec, jsc, jec, nz
        real(dp), intent(in) :: Kh
        real(dp), intent(in) :: h_neglect
        real(dp), intent(in) :: DY_dxT(isd:ied, jsd:jed)
        real(dp), intent(in) :: DX_dyT(isd:ied, jsd:jed)
        real(dp), intent(in) :: DY_dxBu(isd:ied, jsd:jed)
        real(dp), intent(in) :: DX_dyBu(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdyCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdxCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdyCv(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdxCv(isd:ied, jsd:jed)
        real(dp), intent(in) :: IareaCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: IareaCv(isd:ied, jsd:jed)
        real(dp), intent(in) :: mask2dT(isd:ied, jsd:jed)
        real(dp), intent(in) :: mask2dBu(isd:ied, jsd:jed)
        real(dp), intent(in) :: reduction_xx(isd:ied, jsd:jed)
        real(dp), intent(in) :: reduction_xy(isd:ied, jsd:jed)
        real(dp), intent(in) :: dy2h(isd:ied, jsd:jed)
        real(dp), intent(in) :: dx2h(isd:ied, jsd:jed)
        real(dp), intent(in) :: dy2q(isd:ied, jsd:jed)
        real(dp), intent(in) :: dx2q(isd:ied, jsd:jed)

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec
        CS%nz = nz

        ! Store scalar parameters
        CS%Kh_bg = Kh
        CS%h_neglect = h_neglect

        ! Allocate 2D device metric arrays and copy from host
        allocate(CS%DY_dxT_d(isd:ied, jsd:jed));     CS%DY_dxT_d = DY_dxT
        allocate(CS%DX_dyT_d(isd:ied, jsd:jed));     CS%DX_dyT_d = DX_dyT
        allocate(CS%DY_dxBu_d(isd:ied, jsd:jed));    CS%DY_dxBu_d = DY_dxBu
        allocate(CS%DX_dyBu_d(isd:ied, jsd:jed));    CS%DX_dyBu_d = DX_dyBu
        allocate(CS%IdyCu_d(isd:ied, jsd:jed));      CS%IdyCu_d = IdyCu
        allocate(CS%IdxCu_d(isd:ied, jsd:jed));      CS%IdxCu_d = IdxCu
        allocate(CS%IdyCv_d(isd:ied, jsd:jed));      CS%IdyCv_d = IdyCv
        allocate(CS%IdxCv_d(isd:ied, jsd:jed));      CS%IdxCv_d = IdxCv
        allocate(CS%IareaCu_d(isd:ied, jsd:jed));    CS%IareaCu_d = IareaCu
        allocate(CS%IareaCv_d(isd:ied, jsd:jed));    CS%IareaCv_d = IareaCv
        allocate(CS%mask2dT_d(isd:ied, jsd:jed));    CS%mask2dT_d = mask2dT
        allocate(CS%mask2dBu_d(isd:ied, jsd:jed));   CS%mask2dBu_d = mask2dBu
        allocate(CS%reduction_xx_d(isd:ied, jsd:jed)); CS%reduction_xx_d = reduction_xx
        allocate(CS%reduction_xy_d(isd:ied, jsd:jed)); CS%reduction_xy_d = reduction_xy
        allocate(CS%dy2h_d(isd:ied, jsd:jed));       CS%dy2h_d = dy2h
        allocate(CS%dx2h_d(isd:ied, jsd:jed));       CS%dx2h_d = dx2h
        allocate(CS%dy2q_d(isd:ied, jsd:jed));       CS%dy2q_d = dy2q
        allocate(CS%dx2q_d(isd:ied, jsd:jed));       CS%dx2q_d = dx2q

        ! Allocate 3D device work arrays (one entry per layer)
        allocate(CS%dudx_d(isd:ied, jsd:jed, nz))
        allocate(CS%dvdy_d(isd:ied, jsd:jed, nz))
        allocate(CS%dvdx_d(isd:ied, jsd:jed, nz))
        allocate(CS%dudy_d(isd:ied, jsd:jed, nz))
        allocate(CS%sh_xx_d(isd:ied, jsd:jed, nz))
        allocate(CS%sh_xy_d(isd:ied, jsd:jed, nz))
        allocate(CS%str_xx_d(isd:ied, jsd:jed, nz))
        allocate(CS%str_xy_d(isd:ied, jsd:jed, nz))

        CS%initialized = .true.

    end subroutine hor_visc_init_cuda

    !> Compute horizontal viscous accelerations on the GPU.
    !! Launches 5 3D kernels covering all layers simultaneously.
    subroutine hor_visc_cuda(u_d, v_d, h_d, diffu_d, diffv_d, CS, nz, bx_in, by_in)
        type(hor_visc_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: nz
        real(dp), device, intent(in)  :: u_d(CS%isd:CS%ied, CS%jsd:CS%jed, nz)
        real(dp), device, intent(in)  :: v_d(CS%isd:CS%ied, CS%jsd:CS%jed, nz)
        real(dp), device, intent(in)  :: h_d(CS%isd:CS%ied, CS%jsd:CS%jed, nz)
        real(dp), device, intent(out) :: diffu_d(CS%isd:CS%ied, CS%jsd:CS%jed, nz)
        real(dp), device, intent(out) :: diffv_d(CS%isd:CS%ied, CS%jsd:CS%jed, nz)
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, n3, is_l, ie_l, js_l, je_l, istat, bx, by
        type(dim3) :: grid, tBlock

        ! Block dimensions (default 32x4 for good occupancy)
        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        ! Array dimensions (1-based for kernel)
        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        n3 = nz

        ! Compute-domain offsets in 1-based local coords
        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        ! Thread block: bx x by x 1, with k in blockIdx%z
        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), n3)

        ! Kernel 1: Velocity gradients (all layers)
        call vel_grad_kernel<<<grid, tBlock>>>( &
            CS%dudx_d, CS%dvdy_d, CS%dvdx_d, CS%dudy_d, &
            u_d, v_d, &
            CS%DY_dxT_d, CS%DX_dyT_d, CS%DY_dxBu_d, CS%DX_dyBu_d, &
            CS%IdyCu_d, CS%IdxCu_d, CS%IdyCv_d, CS%IdxCv_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        ! Kernel 2: Strain tensor (all layers)
        call strain_kernel<<<grid, tBlock>>>( &
            CS%sh_xx_d, CS%sh_xy_d, &
            CS%dudx_d, CS%dvdy_d, CS%dvdx_d, CS%dudy_d, &
            CS%mask2dBu_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        ! Kernel 3: Diagonal stress at h-points (all layers)
        call stress_xx_kernel<<<grid, tBlock>>>( &
            CS%str_xx_d, CS%sh_xx_d, h_d, CS%reduction_xx_d, &
            CS%Kh_bg, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        ! Kernel 4: Off-diagonal stress at q-points (all layers)
        call stress_xy_kernel<<<grid, tBlock>>>( &
            CS%str_xy_d, CS%sh_xy_d, h_d, &
            CS%mask2dBu_d, CS%reduction_xy_d, &
            CS%Kh_bg, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        ! Kernel 5: Stress divergence -> viscous acceleration (all layers)
        call divergence_kernel<<<grid, tBlock>>>( &
            diffu_d, diffv_d, &
            CS%str_xx_d, CS%str_xy_d, h_d, &
            CS%mask2dT_d, &
            CS%IdyCu_d, CS%IdxCu_d, CS%IdyCv_d, CS%IdxCv_d, &
            CS%IareaCu_d, CS%IareaCv_d, &
            CS%dy2h_d, CS%dx2h_d, CS%dy2q_d, CS%dx2q_d, &
            CS%h_neglect, &
            n1, n2, n3, is_l, ie_l, js_l, je_l)

        ! Sync to ensure all output is ready before returning
        istat = cudaDeviceSynchronize()

    end subroutine hor_visc_cuda

    !> Finalize and deallocate all device arrays.
    subroutine hor_visc_end_cuda(CS)
        type(hor_visc_CS_cuda), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! Deallocate device metric arrays
        if (allocated(CS%DY_dxT_d)) deallocate(CS%DY_dxT_d)
        if (allocated(CS%DX_dyT_d)) deallocate(CS%DX_dyT_d)
        if (allocated(CS%DY_dxBu_d)) deallocate(CS%DY_dxBu_d)
        if (allocated(CS%DX_dyBu_d)) deallocate(CS%DX_dyBu_d)
        if (allocated(CS%IdyCu_d)) deallocate(CS%IdyCu_d)
        if (allocated(CS%IdxCu_d)) deallocate(CS%IdxCu_d)
        if (allocated(CS%IdyCv_d)) deallocate(CS%IdyCv_d)
        if (allocated(CS%IdxCv_d)) deallocate(CS%IdxCv_d)
        if (allocated(CS%IareaCu_d)) deallocate(CS%IareaCu_d)
        if (allocated(CS%IareaCv_d)) deallocate(CS%IareaCv_d)
        if (allocated(CS%mask2dT_d)) deallocate(CS%mask2dT_d)
        if (allocated(CS%mask2dBu_d)) deallocate(CS%mask2dBu_d)
        if (allocated(CS%reduction_xx_d)) deallocate(CS%reduction_xx_d)
        if (allocated(CS%reduction_xy_d)) deallocate(CS%reduction_xy_d)
        if (allocated(CS%dy2h_d)) deallocate(CS%dy2h_d)
        if (allocated(CS%dx2h_d)) deallocate(CS%dx2h_d)
        if (allocated(CS%dy2q_d)) deallocate(CS%dy2q_d)
        if (allocated(CS%dx2q_d)) deallocate(CS%dx2q_d)

        ! Deallocate device work arrays
        if (allocated(CS%dudx_d)) deallocate(CS%dudx_d)
        if (allocated(CS%dvdy_d)) deallocate(CS%dvdy_d)
        if (allocated(CS%dvdx_d)) deallocate(CS%dvdx_d)
        if (allocated(CS%dudy_d)) deallocate(CS%dudy_d)
        if (allocated(CS%sh_xx_d)) deallocate(CS%sh_xx_d)
        if (allocated(CS%sh_xy_d)) deallocate(CS%sh_xy_d)
        if (allocated(CS%str_xx_d)) deallocate(CS%str_xx_d)
        if (allocated(CS%str_xy_d)) deallocate(CS%str_xy_d)

        CS%initialized = .false.

    end subroutine hor_visc_end_cuda

end module mom6_hor_visc_cuda
