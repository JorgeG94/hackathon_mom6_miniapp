!> MOM6 Continuity PPM Solver Module (CUDA Fortran variant)
!!
!! Uses explicit attributes(global) CUDA kernels for GPU execution.
!! Three kernel launches: PPM reconstruction, zonal flux computation,
!! and zonal convergence (thickness update).
!!
!! Simplified version: zonal direction only (matching OpenACC code which
!! ignores meridional). Uses 3D collapse(3) kernels for PPM and flux,
!! plus a 3D kernel for convergence.
!!
!! PPM reconstruction follows Colella & Woodward (1984) with monotonic
!! limiting. Edge values computed via limited slopes, then Colella-Woodward
!! monotonicity constraint applied.
!!
module mom6_continuity_cuda
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: continuity_init_cuda, continuity_PPM_cuda, continuity_end_cuda
    public :: continuity_CS_cuda

    real(dp), parameter :: oneSixth = 1.0_dp / 6.0_dp

    !> Control structure for CUDA Fortran continuity solver
    type :: continuity_CS_cuda
        logical :: initialized = .false.
        integer :: is, ie, js, je, nz
        integer :: isd, ied, jsd, jed

        !> PPM parameters
        logical :: monotonic

        !> Device 2D grid metrics (copied at init)
        real(dp), device, allocatable :: IareaT_d(:,:)
        real(dp), device, allocatable :: IdxT_d(:,:)
        real(dp), device, allocatable :: dy_Cu_d(:,:)
        real(dp), device, allocatable :: mask2dT_d(:,:)

        !> Device 3D work arrays for PPM edge values
        real(dp), device, allocatable :: h_W(:,:,:)   ! West edge thickness
        real(dp), device, allocatable :: h_E(:,:,:)   ! East edge thickness
    end type continuity_CS_cuda

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

    !> PPM reconstruction kernel: computes west (h_W) and east (h_E) edge
    !! values for each cell using limited slopes and optional Colella-Woodward
    !! monotonic limiting.
    !!
    !! Compute range: i in [is_l-1 : ie_l+1], j in [js_l : je_l], k in [1 : n3]
    !! Reads h_in at stencil i-2..i+2 (requires 2-cell halo).
    !! The monotonic flag selects between CW84 monotonic limiter and a
    !! simple positive-definite limiter.
    attributes(global) subroutine ppm_reconstruction_3d_kernel( &
            h_W, h_E, h_in, mask2dT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, h_min, monotonic)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: h_W(n1, n2, n3), h_E(n1, n2, n3)
        real(dp), intent(in)  :: h_in(n1, n2, n3)
        real(dp), intent(in)  :: mask2dT(n1, n2)
        real(dp), value, intent(in) :: h_min
        integer, value, intent(in) :: monotonic

        integer :: i, j, k
        real(dp) :: h_im1, h_ip1, h_i
        real(dp) :: dMx, dMn, slp_im1, slp_i, slp_ip1
        real(dp) :: RLdiff, RLdiff2, RLmean, FunFac
        real(dp) :: curv, dh, scale_val

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        ! Compute range: i in [is_l-1 : ie_l+1], j in [js_l : je_l]
        if (i < is_l - 1 .or. i > ie_l + 1 .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        ! ---------------------------------------------------------
        ! Step 1: Compute limited slopes at i-1, i, i+1
        ! ---------------------------------------------------------

        ! Slope at i-1
        if (i - 2 >= 1 .and. i >= 1 .and. &
            (mask2dT(i - 2, j) * mask2dT(i - 1, j) * mask2dT(i, j)) /= 0.0_dp) then
            slp_im1 = 0.5_dp * (h_in(i, j, k) - h_in(i - 2, j, k))
            dMx = max(h_in(i, j, k), h_in(i - 2, j, k), h_in(i - 1, j, k)) - h_in(i - 1, j, k)
            dMn = h_in(i - 1, j, k) - min(h_in(i, j, k), h_in(i - 2, j, k), h_in(i - 1, j, k))
            slp_im1 = sign(1.0_dp, slp_im1) * min(abs(slp_im1), 2.0_dp * min(dMx, dMn))
        else
            slp_im1 = 0.0_dp
        end if

        ! Slope at i
        if (i - 1 >= 1 .and. i + 1 <= n1 .and. &
            (mask2dT(i - 1, j) * mask2dT(i, j) * mask2dT(i + 1, j)) /= 0.0_dp) then
            slp_i = 0.5_dp * (h_in(i + 1, j, k) - h_in(i - 1, j, k))
            dMx = max(h_in(i + 1, j, k), h_in(i - 1, j, k), h_in(i, j, k)) - h_in(i, j, k)
            dMn = h_in(i, j, k) - min(h_in(i + 1, j, k), h_in(i - 1, j, k), h_in(i, j, k))
            slp_i = sign(1.0_dp, slp_i) * min(abs(slp_i), 2.0_dp * min(dMx, dMn))
        else
            slp_i = 0.0_dp
        end if

        ! Slope at i+1
        if (i + 2 <= n1 .and. i >= 1 .and. &
            (mask2dT(i, j) * mask2dT(i + 1, j) * mask2dT(i + 2, j)) /= 0.0_dp) then
            slp_ip1 = 0.5_dp * (h_in(i + 2, j, k) - h_in(i, j, k))
            dMx = max(h_in(i + 2, j, k), h_in(i, j, k), h_in(i + 1, j, k)) - h_in(i + 1, j, k)
            dMn = h_in(i + 1, j, k) - min(h_in(i + 2, j, k), h_in(i, j, k), h_in(i + 1, j, k))
            slp_ip1 = sign(1.0_dp, slp_ip1) * min(abs(slp_ip1), 2.0_dp * min(dMx, dMn))
        else
            slp_ip1 = 0.0_dp
        end if

        ! ---------------------------------------------------------
        ! Step 2: Compute edge values using slopes
        ! ---------------------------------------------------------
        h_im1 = mask2dT(max(i - 1, 1), j) * h_in(max(i - 1, 1), j, k) + &
                 (1.0_dp - mask2dT(max(i - 1, 1), j)) * h_in(i, j, k)
        h_ip1 = mask2dT(min(i + 1, n1), j) * h_in(min(i + 1, n1), j, k) + &
                 (1.0_dp - mask2dT(min(i + 1, n1), j)) * h_in(i, j, k)

        h_W(i, j, k) = 0.5_dp * (h_im1 + h_in(i, j, k)) + oneSixth * (slp_im1 - slp_i)
        h_E(i, j, k) = 0.5_dp * (h_ip1 + h_in(i, j, k)) + oneSixth * (slp_i - slp_ip1)

        ! ---------------------------------------------------------
        ! Step 3: Apply limiter
        ! ---------------------------------------------------------
        h_i = h_in(i, j, k)

        if (monotonic == 1) then
            ! Colella-Woodward monotonic limiter
            if ((h_E(i, j, k) - h_i) * (h_i - h_W(i, j, k)) <= 0.0_dp) then
                h_W(i, j, k) = h_i
                h_E(i, j, k) = h_i
            else
                RLdiff = h_E(i, j, k) - h_W(i, j, k)
                RLmean = 0.5_dp * (h_E(i, j, k) + h_W(i, j, k))
                FunFac = 6.0_dp * RLdiff * (h_i - RLmean)
                RLdiff2 = RLdiff * RLdiff
                if (FunFac > RLdiff2) h_W(i, j, k) = 3.0_dp * h_i - 2.0_dp * h_E(i, j, k)
                if (FunFac < -RLdiff2) h_E(i, j, k) = 3.0_dp * h_i - 2.0_dp * h_W(i, j, k)
            end if
        else
            ! Simple positive-definite limiter
            curv = 3.0_dp * ((h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_i)
            if (curv > 0.0_dp) then
                dh = h_E(i, j, k) - h_W(i, j, k)
                if (abs(dh) < curv) then
                    if (h_i <= h_min) then
                        h_W(i, j, k) = h_i
                        h_E(i, j, k) = h_i
                    elseif (12.0_dp * curv * (h_i - h_min) < (curv**2 + 3.0_dp * dh**2)) then
                        scale_val = 12.0_dp * curv * (h_i - h_min) / (curv**2 + 3.0_dp * dh**2)
                        h_W(i, j, k) = h_i + scale_val * (h_W(i, j, k) - h_i)
                        h_E(i, j, k) = h_i + scale_val * (h_E(i, j, k) - h_i)
                    end if
                end if
            end if
        end if

    end subroutine ppm_reconstruction_3d_kernel

    !> Zonal flux kernel: computes the volume/mass flux uh at each u-face
    !! using the PPM reconstruction (h_W, h_E).
    !!
    !! Compute range: I in [is_l-1 : ie_l], j in [js_l : je_l], k in [1 : n3]
    !! For u > 0, flux uses upstream (west) cell PPM profile.
    !! For u < 0, flux uses downstream (east) cell PPM profile.
    attributes(global) subroutine zonal_flux_layer_3d_kernel( &
            uh, u, h_in, h_W, h_E, dy_Cu, IdxT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: uh(n1, n2, n3)
        real(dp), intent(in)  :: u(n1, n2, n3)
        real(dp), intent(in)  :: h_in(n1, n2, n3)
        real(dp), intent(in)  :: h_W(n1, n2, n3), h_E(n1, n2, n3)
        real(dp), intent(in)  :: dy_Cu(n1, n2)
        real(dp), intent(in)  :: IdxT(n1, n2)
        real(dp), value, intent(in) :: dt

        integer :: i, j, k
        real(dp) :: CFL, curv_3

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        ! Flux compute range: I in [is_l-1 : ie_l], j in [js_l : je_l]
        if (i < is_l - 1 .or. i > ie_l .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        if (u(i, j, k) > 0.0_dp) then
            ! Upwind cell is i (west side of face)
            CFL = u(i, j, k) * dt * IdxT(i, j)
            curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
            uh(i, j, k) = dy_Cu(i, j) * u(i, j, k) * &
                (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + &
                 curv_3 * (CFL - 1.5_dp)))
        elseif (u(i, j, k) < 0.0_dp) then
            ! Upwind cell is i+1 (east side of face)
            CFL = -u(i, j, k) * dt * IdxT(i + 1, j)
            curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
            uh(i, j, k) = dy_Cu(i, j) * u(i, j, k) * &
                (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                 curv_3 * (CFL - 1.5_dp)))
        else
            uh(i, j, k) = 0.0_dp
        end if

    end subroutine zonal_flux_layer_3d_kernel

    !> Zonal convergence kernel: updates layer thickness from initial
    !! thickness and zonal flux divergence.
    !!
    !! h(i,j,k) = max(hin(i,j,k) - dt * IareaT(i,j) * (uh(i,j,k) - uh(i-1,j,k)), 0)
    !!
    !! Compute range: i in [is_l : ie_l], j in [js_l : je_l], k in [1 : n3]
    attributes(global) subroutine zonal_convergence_kernel( &
            h, hin, uh, IareaT, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt)
        integer, value, intent(in) :: n1, n2, n3, is_l, ie_l, js_l, je_l
        real(dp), intent(out) :: h(n1, n2, n3)
        real(dp), intent(in)  :: hin(n1, n2, n3)
        real(dp), intent(in)  :: uh(n1, n2, n3)
        real(dp), intent(in)  :: IareaT(n1, n2)
        real(dp), value, intent(in) :: dt

        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        k = blockIdx%z

        ! Convergence range: i in [is_l : ie_l], j in [js_l : je_l]
        if (i < is_l .or. i > ie_l .or. &
            j < js_l .or. j > je_l .or. &
            k < 1 .or. k > n3) return

        h(i, j, k) = max(hin(i, j, k) - dt * IareaT(i, j) * &
                     (uh(i, j, k) - uh(i - 1, j, k)), 0.0_dp)

    end subroutine zonal_convergence_kernel

    !=========================================================================
    ! Host routines
    !=========================================================================

    !> Initialize the CUDA continuity solver.
    !! Stores dimensions, allocates device arrays, and copies grid metrics
    !! to the device.
    subroutine continuity_init_cuda(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nk, &
                                     IareaT, IdxT, dy_Cu, mask2dT, monotonic)
        type(continuity_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec, nk
        real(dp), intent(in) :: IareaT(isd:ied, jsd:jed)
        real(dp), intent(in) :: IdxT(isd:ied, jsd:jed)
        real(dp), intent(in) :: dy_Cu(isd:ied, jsd:jed)
        real(dp), intent(in) :: mask2dT(isd:ied, jsd:jed)
        logical, intent(in)  :: monotonic

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec; CS%nz = nk

        ! Store PPM parameters
        CS%monotonic = monotonic

        ! Allocate 3D device work arrays for PPM edge values
        allocate(CS%h_W(isd:ied, jsd:jed, nk))
        allocate(CS%h_E(isd:ied, jsd:jed, nk))

        ! Allocate 2D device grid metrics and copy from host
        allocate(CS%IareaT_d(isd:ied, jsd:jed));  CS%IareaT_d = IareaT
        allocate(CS%IdxT_d(isd:ied, jsd:jed));    CS%IdxT_d = IdxT
        allocate(CS%dy_Cu_d(isd:ied, jsd:jed));   CS%dy_Cu_d = dy_Cu
        allocate(CS%mask2dT_d(isd:ied, jsd:jed)); CS%mask2dT_d = mask2dT

        CS%initialized = .true.

    end subroutine continuity_init_cuda

    !> Run the PPM continuity solver on the GPU.
    !! Launches three kernels in sequence:
    !!   1. PPM reconstruction (compute edge values h_W, h_E)
    !!   2. Zonal flux computation (uh from u, h_in, h_W, h_E)
    !!   3. Zonal convergence (update h from hin and flux divergence)
    !!
    !! All input/output arrays must be device-resident.
    subroutine continuity_PPM_cuda(u_d, hin_d, h_d, uh_d, dt, CS, bx_in, by_in)
        type(continuity_CS_cuda), intent(inout) :: CS
        real(dp), device, intent(in)  :: u_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in)  :: hin_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(out) :: h_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(out) :: uh_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), intent(in)          :: dt
        integer, intent(in), optional :: bx_in, by_in

        integer :: n1, n2, n3, is_l, ie_l, js_l, je_l, istat, bx, by
        integer :: mono_flag
        real(dp) :: h_min
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

        ! PPM parameters
        h_min = 2.0e-10_dp   ! 2 * Angstrom_H
        mono_flag = 0
        if (CS%monotonic) mono_flag = 1

        ! Thread block: bx x by x 1, with k in blockIdx%z
        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), n3)

        ! Kernel 1: PPM reconstruction of edge values
        call ppm_reconstruction_3d_kernel<<<grid, tBlock>>>( &
            CS%h_W, CS%h_E, hin_d, CS%mask2dT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, h_min, mono_flag)

        ! Kernel 2: Zonal flux computation
        call zonal_flux_layer_3d_kernel<<<grid, tBlock>>>( &
            uh_d, u_d, hin_d, CS%h_W, CS%h_E, CS%dy_Cu_d, CS%IdxT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt)

        ! Kernel 3: Zonal convergence (thickness update)
        call zonal_convergence_kernel<<<grid, tBlock>>>( &
            h_d, hin_d, uh_d, CS%IareaT_d, &
            n1, n2, n3, is_l, ie_l, js_l, je_l, dt)

        ! Sync to ensure all output is ready before returning
        istat = cudaDeviceSynchronize()

    end subroutine continuity_PPM_cuda

    !> Finalize the CUDA continuity solver.
    !! Deallocates all device arrays.
    subroutine continuity_end_cuda(CS)
        type(continuity_CS_cuda), intent(inout) :: CS

        if (.not. CS%initialized) return

        ! Deallocate 3D work arrays
        if (allocated(CS%h_W)) deallocate(CS%h_W)
        if (allocated(CS%h_E)) deallocate(CS%h_E)

        ! Deallocate 2D grid metrics
        if (allocated(CS%IareaT_d)) deallocate(CS%IareaT_d)
        if (allocated(CS%IdxT_d)) deallocate(CS%IdxT_d)
        if (allocated(CS%dy_Cu_d)) deallocate(CS%dy_Cu_d)
        if (allocated(CS%mask2dT_d)) deallocate(CS%mask2dT_d)

        CS%initialized = .false.

    end subroutine continuity_end_cuda

end module mom6_continuity_cuda
