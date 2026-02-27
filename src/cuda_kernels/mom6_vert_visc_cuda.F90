!> MOM6 Vertical Viscosity Module (CUDA Fortran variant)
!!
!! Uses explicit attributes(global) CUDA kernels for GPU execution.
!! Column-parallel pattern: collapse(2) over j,i with serial k-loop per thread.
!! Each thread owns a full water column. Two private arrays per thread:
!! hvel(nz) and z_i(nz+1) declared as automatic arrays inside the kernel.
!! Coupling coefficients a_col are computed on-the-fly as scalars; the
!! tridiagonal c1 scratch is stored into z_i (dead after coef phase).
!!
!! Fused coef+remnant+apply eliminates all intermediate global-memory traffic
!! for coupling coefficients, effective thicknesses, and remnant fractions.
!! Only outputs written to global memory are: u/v (velocity), visc_rem_u/v,
!! taux_bot/tauy_bot.
!!
!! Simplified version: no Rayleigh drag.
!!
module mom6_vert_visc_cuda
    use cudafor
    use iso_fortran_env, only: dp => real64
    implicit none
    private

    public :: vert_visc_init_cuda, vert_visc_cra_cuda, vert_visc_end_cuda
    public :: vert_visc_CS_cuda

    real(dp), parameter :: RHO_0 = 1035.0_dp

    !> Control structure for CUDA Fortran vertical viscosity solver
    type :: vert_visc_CS_cuda
        logical :: initialized = .false.
        integer :: is, ie, js, je, nz
        integer :: isd, ied, jsd, jed

        ! Viscosity parameters (scalars, host-side)
        real(dp) :: Kv, Kv_extra_bbl, Hmix, Hbbl, Kv_ml

        ! Output arrays on device
        real(dp), device, allocatable :: visc_rem_u(:,:,:)  ! (isd:ied, jsd:jed, nz)
        real(dp), device, allocatable :: visc_rem_v(:,:,:)
        real(dp), device, allocatable :: taux_bot(:,:)      ! (isd:ied, jsd:jed)
        real(dp), device, allocatable :: tauy_bot(:,:)

        ! Grid metrics on device (2D, copied once at init)
        real(dp), device, allocatable :: mask2dCu_d(:,:)
        real(dp), device, allocatable :: mask2dCv_d(:,:)
    end type vert_visc_CS_cuda

contains

    !=========================================================================
    ! CUDA Kernels (attributes(global))
    !
    ! All arrays use 1-based local indexing. The caller passes:
    !   n1 = ied - isd + 1   (array dim 1)
    !   n2 = jed - jsd + 1   (array dim 2)
    !   nz                    (array dim 3 / number of layers)
    !   is_l = is - isd + 1  (compute-start offset in dim 1)
    !   ie_l = ie - isd + 1  (compute-end offset in dim 1)
    !   js_l = js - jsd + 1  (compute-start offset in dim 2)
    !   je_l = je - jsd + 1  (compute-end offset in dim 2)
    !
    ! Thread mapping: i = threadIdx%x, j = threadIdx%y.
    ! Grid is 2D only (no blockIdx%z). Each thread handles one full column.
    !=========================================================================

    !> Fused coef+remnant+apply kernel for u-points.
    !! Each thread handles one (i,j) column: bottom-up coefficient computation,
    !! then combined remnant + velocity tridiagonal solve, then bottom stress.
    attributes(global) subroutine vert_visc_cra_u_kernel( &
            u, h, mask2dCu, visc_rem_u, taux_bot, taux, &
            Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl, &
            dt, dt_Rho0, h_neglect, I_Hbbl, &
            n1, n2, nz, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, nz, is_l, ie_l, js_l, je_l
        real(dp), intent(inout) :: u(n1, n2, nz)
        real(dp), intent(in)    :: h(n1, n2, nz)
        real(dp), intent(in)    :: mask2dCu(n1, n2)
        real(dp), intent(out)   :: visc_rem_u(n1, n2, nz)
        real(dp), intent(out)   :: taux_bot(n1, n2)
        real(dp), intent(in)    :: taux(n1, n2)
        real(dp), value, intent(in) :: Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl
        real(dp), value, intent(in) :: dt, dt_Rho0, h_neglect, I_Hbbl

        ! Thread-local private arrays (automatic, on-stack / local memory)
        real(dp) :: hvel(nz), z_i(nz + 1)

        ! Local scalars
        real(dp) :: h_harm, h_arith, h_delta, z2, botfn
        real(dp) :: z_top, Kv_tot, topfn, h_shear
        real(dp) :: sfc_stress, b1, d1, b_denom_1
        real(dp) :: a_k, a_k1, h_eff
        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        ! U-points range: i in [is_l-1 : ie_l], j in [js_l : je_l]
        if (i < is_l - 1 .or. i > ie_l .or. j < js_l .or. j > je_l) return

        if (mask2dCu(i, j) > 0.0_dp) then

            ! ============================================================
            ! Coefficient phase: bottom-up hvel and z_i computation
            ! ============================================================
            z_i(nz + 1) = 0.0_dp

            do k = nz, 1, -1
                h_harm = 2.0_dp * h(i, j, k) * h(i + 1, j, k) / &
                         (h(i, j, k) + h(i + 1, j, k) + h_neglect)
                h_arith = 0.5_dp * (h(i + 1, j, k) + h(i, j, k))
                h_delta = h(i + 1, j, k) - h(i, j, k)

                hvel(k) = h_harm

                if (u(i, j, k) * h_delta < 0.0_dp) then
                    z2 = z_i(k + 1)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    hvel(k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                end if

                z_i(k) = z_i(k + 1) + h_harm * I_Hbbl
            end do

            ! ============================================================
            ! Combined remnant + apply: single tridiagonal solve
            ! ============================================================
            sfc_stress = dt_Rho0 * taux(i, j) * mask2dCu(i, j)

            a_k = 0.0_dp  ! a_col(1) = 0 (surface)

            ! Compute a_col(2) on the fly
            if (nz > 1) then
                z_top = hvel(1)
                Kv_tot = Kv
                if (z_top < Hmix) then
                    topfn = 1.0_dp - z_top / Hmix
                    Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn
                end if
                z2 = z_i(2)
                botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                Kv_tot = Kv_tot + Kv_extra_bbl * botfn
                h_shear = 0.5_dp * (hvel(2) + hvel(1) + h_neglect)
                a_k1 = Kv_tot / h_shear
            else
                a_k1 = (Kv + Kv_extra_bbl) / (0.5_dp * hvel(nz) + h_neglect)
            end if

            ! Layer 1
            h_eff = hvel(1) + h_neglect
            b_denom_1 = h_eff + dt * a_k
            b1 = 1.0_dp / (b_denom_1 + dt * a_k1)
            d1 = b_denom_1 * b1
            visc_rem_u(i, j, 1) = b1 * h_eff
            u(i, j, 1) = b1 * (h_eff * u(i, j, 1) + sfc_stress)

            ! Interior layers -- a_col computed on the fly, c1 stored in z_i
            do k = 2, nz
                ! Store c1(k) in z_i(k) (z_i(k) dead after coef phase for k-1)
                z_i(k) = dt * a_k1 * b1

                a_k = a_k1  ! shift: a_col(k) = previous a_col(k+1)

                ! Compute a_col(k+1) on the fly
                if (k < nz) then
                    z_top = z_top + hvel(k)
                    Kv_tot = Kv
                    if (z_top < Hmix) then
                        topfn = 1.0_dp - z_top / Hmix
                        Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn
                    end if
                    z2 = z_i(k + 1)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    Kv_tot = Kv_tot + Kv_extra_bbl * botfn
                    h_shear = 0.5_dp * (hvel(k + 1) + hvel(k) + h_neglect)
                    a_k1 = Kv_tot / h_shear
                else
                    a_k1 = (Kv + Kv_extra_bbl) / (0.5_dp * hvel(nz) + h_neglect)
                end if

                h_eff = hvel(k) + h_neglect
                b_denom_1 = h_eff + dt * (a_k * d1)
                b1 = 1.0_dp / (b_denom_1 + dt * a_k1)
                d1 = b_denom_1 * b1
                visc_rem_u(i, j, k) = (h_eff + &
                    dt * a_k * visc_rem_u(i, j, k - 1)) * b1
                u(i, j, k) = (h_eff * u(i, j, k) + &
                    dt * a_k * u(i, j, k - 1)) * b1
            end do

            ! Combined back substitution (c1 stored in z_i)
            do k = nz - 1, 1, -1
                visc_rem_u(i, j, k) = visc_rem_u(i, j, k) + &
                    z_i(k + 1) * visc_rem_u(i, j, k + 1)
                u(i, j, k) = u(i, j, k) + z_i(k + 1) * u(i, j, k + 1)
            end do

            ! Bottom stress (a_k1 is a_col(nz+1) after the loop)
            taux_bot(i, j) = RHO_0 * u(i, j, nz) * a_k1

        else
            ! Masked points: remnant = 1, stress = 0
            do k = 1, nz
                visc_rem_u(i, j, k) = 1.0_dp
            end do
            taux_bot(i, j) = 0.0_dp
        end if

    end subroutine vert_visc_cra_u_kernel

    !> Fused coef+remnant+apply kernel for v-points.
    !! Same pattern as u-kernel but uses h(i,j,k)/h(i,j+1,k) for thickness
    !! and mask2dCv for masking.
    attributes(global) subroutine vert_visc_cra_v_kernel( &
            v, h, mask2dCv, visc_rem_v, tauy_bot, tauy, &
            Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl, &
            dt, dt_Rho0, h_neglect, I_Hbbl, &
            n1, n2, nz, is_l, ie_l, js_l, je_l)
        integer, value, intent(in) :: n1, n2, nz, is_l, ie_l, js_l, je_l
        real(dp), intent(inout) :: v(n1, n2, nz)
        real(dp), intent(in)    :: h(n1, n2, nz)
        real(dp), intent(in)    :: mask2dCv(n1, n2)
        real(dp), intent(out)   :: visc_rem_v(n1, n2, nz)
        real(dp), intent(out)   :: tauy_bot(n1, n2)
        real(dp), intent(in)    :: tauy(n1, n2)
        real(dp), value, intent(in) :: Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl
        real(dp), value, intent(in) :: dt, dt_Rho0, h_neglect, I_Hbbl

        ! Thread-local private arrays (automatic, on-stack / local memory)
        real(dp) :: hvel(nz), z_i(nz + 1)

        ! Local scalars
        real(dp) :: h_harm, h_arith, h_delta, z2, botfn
        real(dp) :: z_top, Kv_tot, topfn, h_shear
        real(dp) :: sfc_stress, b1, d1, b_denom_1
        real(dp) :: a_k, a_k1, h_eff
        integer :: i, j, k

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y

        ! V-points range: i in [is_l : ie_l], j in [js_l-1 : je_l]
        if (i < is_l .or. i > ie_l .or. j < js_l - 1 .or. j > je_l) return

        if (mask2dCv(i, j) > 0.0_dp) then

            ! ============================================================
            ! Coefficient phase: bottom-up hvel and z_i computation
            ! ============================================================
            z_i(nz + 1) = 0.0_dp

            do k = nz, 1, -1
                h_harm = 2.0_dp * h(i, j, k) * h(i, j + 1, k) / &
                         (h(i, j, k) + h(i, j + 1, k) + h_neglect)
                h_arith = 0.5_dp * (h(i, j + 1, k) + h(i, j, k))
                h_delta = h(i, j + 1, k) - h(i, j, k)

                hvel(k) = h_harm

                if (v(i, j, k) * h_delta < 0.0_dp) then
                    z2 = z_i(k + 1)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    hvel(k) = (1.0_dp - botfn) * h_harm + botfn * h_arith
                end if

                z_i(k) = z_i(k + 1) + h_harm * I_Hbbl
            end do

            ! ============================================================
            ! Combined remnant + apply: single tridiagonal solve
            ! ============================================================
            sfc_stress = dt_Rho0 * tauy(i, j) * mask2dCv(i, j)

            a_k = 0.0_dp  ! a_col(1) = 0 (surface)

            ! Compute a_col(2) on the fly
            if (nz > 1) then
                z_top = hvel(1)
                Kv_tot = Kv
                if (z_top < Hmix) then
                    topfn = 1.0_dp - z_top / Hmix
                    Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn
                end if
                z2 = z_i(2)
                botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                Kv_tot = Kv_tot + Kv_extra_bbl * botfn
                h_shear = 0.5_dp * (hvel(2) + hvel(1) + h_neglect)
                a_k1 = Kv_tot / h_shear
            else
                a_k1 = (Kv + Kv_extra_bbl) / (0.5_dp * hvel(nz) + h_neglect)
            end if

            ! Layer 1
            h_eff = hvel(1) + h_neglect
            b_denom_1 = h_eff + dt * a_k
            b1 = 1.0_dp / (b_denom_1 + dt * a_k1)
            d1 = b_denom_1 * b1
            visc_rem_v(i, j, 1) = b1 * h_eff
            v(i, j, 1) = b1 * (h_eff * v(i, j, 1) + sfc_stress)

            ! Interior layers
            do k = 2, nz
                z_i(k) = dt * a_k1 * b1  ! c1(k) stored in z_i(k)

                a_k = a_k1

                if (k < nz) then
                    z_top = z_top + hvel(k)
                    Kv_tot = Kv
                    if (z_top < Hmix) then
                        topfn = 1.0_dp - z_top / Hmix
                        Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn
                    end if
                    z2 = z_i(k + 1)
                    botfn = 1.0_dp / (1.0_dp + 0.09_dp * z2*z2*z2*z2*z2*z2)
                    Kv_tot = Kv_tot + Kv_extra_bbl * botfn
                    h_shear = 0.5_dp * (hvel(k + 1) + hvel(k) + h_neglect)
                    a_k1 = Kv_tot / h_shear
                else
                    a_k1 = (Kv + Kv_extra_bbl) / (0.5_dp * hvel(nz) + h_neglect)
                end if

                h_eff = hvel(k) + h_neglect
                b_denom_1 = h_eff + dt * (a_k * d1)
                b1 = 1.0_dp / (b_denom_1 + dt * a_k1)
                d1 = b_denom_1 * b1
                visc_rem_v(i, j, k) = (h_eff + &
                    dt * a_k * visc_rem_v(i, j, k - 1)) * b1
                v(i, j, k) = (h_eff * v(i, j, k) + &
                    dt * a_k * v(i, j, k - 1)) * b1
            end do

            ! Combined back substitution (c1 stored in z_i)
            do k = nz - 1, 1, -1
                visc_rem_v(i, j, k) = visc_rem_v(i, j, k) + &
                    z_i(k + 1) * visc_rem_v(i, j, k + 1)
                v(i, j, k) = v(i, j, k) + z_i(k + 1) * v(i, j, k + 1)
            end do

            ! Bottom stress (a_k1 is a_col(nz+1) after the loop)
            tauy_bot(i, j) = RHO_0 * v(i, j, nz) * a_k1

        else
            ! Masked points: remnant = 1, stress = 0
            do k = 1, nz
                visc_rem_v(i, j, k) = 1.0_dp
            end do
            tauy_bot(i, j) = 0.0_dp
        end if

    end subroutine vert_visc_cra_v_kernel

    !=========================================================================
    ! Host routines
    !=========================================================================

    !> Initialize the CUDA vertical viscosity solver.
    !! Stores dimensions and scalar parameters, allocates device arrays,
    !! and copies mask data to the device.
    subroutine vert_visc_init_cuda(CS, isd, ied, jsd, jed, isc, iec, jsc, jec, nk, &
                                    mask2dCu, mask2dCv, Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl)
        type(vert_visc_CS_cuda), intent(inout) :: CS
        integer, intent(in) :: isd, ied, jsd, jed, isc, iec, jsc, jec, nk
        real(dp), intent(in) :: mask2dCu(isd:ied, jsd:jed)
        real(dp), intent(in) :: mask2dCv(isd:ied, jsd:jed)
        real(dp), intent(in) :: Kv, Kv_ml, Kv_extra_bbl, Hmix, Hbbl

        ! Store dimensions
        CS%isd = isd; CS%ied = ied; CS%jsd = jsd; CS%jed = jed
        CS%is = isc; CS%ie = iec; CS%js = jsc; CS%je = jec; CS%nz = nk

        ! Store viscosity parameters
        CS%Kv = Kv
        CS%Kv_ml = Kv_ml
        CS%Kv_extra_bbl = Kv_extra_bbl
        CS%Hmix = Hmix
        CS%Hbbl = Hbbl

        ! Allocate 3D device output arrays
        allocate(CS%visc_rem_u(isd:ied, jsd:jed, nk))
        allocate(CS%visc_rem_v(isd:ied, jsd:jed, nk))

        ! Allocate 2D device output arrays
        allocate(CS%taux_bot(isd:ied, jsd:jed))
        allocate(CS%tauy_bot(isd:ied, jsd:jed))

        ! Allocate 2D device grid metrics and copy from host
        allocate(CS%mask2dCu_d(isd:ied, jsd:jed)); CS%mask2dCu_d = mask2dCu
        allocate(CS%mask2dCv_d(isd:ied, jsd:jed)); CS%mask2dCv_d = mask2dCv

        CS%initialized = .true.

    end subroutine vert_visc_init_cuda

    !> Launch fused coef+remnant+apply kernels for both u and v points.
    !! u_d, v_d are device arrays (inout), h_d is device (in),
    !! taux_d, tauy_d are device surface stress arrays (in).
    subroutine vert_visc_cra_cuda(u_d, v_d, h_d, dt, CS, taux_d, tauy_d, bx_in, by_in)
        type(vert_visc_CS_cuda), intent(inout) :: CS
        real(dp), device, intent(inout) :: u_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(inout) :: v_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), device, intent(in)    :: h_d(CS%isd:CS%ied, CS%jsd:CS%jed, CS%nz)
        real(dp), intent(in)            :: dt
        real(dp), device, intent(in)    :: taux_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        real(dp), device, intent(in)    :: tauy_d(CS%isd:CS%ied, CS%jsd:CS%jed)
        integer, intent(in), optional   :: bx_in, by_in

        integer :: n1, n2, nz, is_l, ie_l, js_l, je_l, istat, bx, by
        real(dp) :: dt_Rho0, h_neglect, I_Hbbl
        type(dim3) :: grid, tBlock

        ! Configurable block dimensions (default: 32 x 4 = 128 threads)
        bx = 32; by = 4
        if (present(bx_in)) bx = bx_in
        if (present(by_in)) by = by_in

        ! Array dimensions (1-based for kernel)
        n1 = CS%ied - CS%isd + 1
        n2 = CS%jed - CS%jsd + 1
        nz = CS%nz

        ! Compute-domain offsets in 1-based local coords
        is_l = CS%is - CS%isd + 1
        ie_l = CS%ie - CS%isd + 1
        js_l = CS%js - CS%jsd + 1
        je_l = CS%je - CS%jsd + 1

        ! Derived constants
        h_neglect = 1.0e-30_dp
        I_Hbbl = 1.0_dp / (CS%Hbbl + h_neglect)
        dt_Rho0 = dt / RHO_0

        ! Thread block: bx x by (2D grid, no z-dimension)
        tBlock = dim3(bx, by, 1)
        grid = dim3(ceiling(real(n1) / real(bx)), ceiling(real(n2) / real(by)), 1)

        ! Launch u-points kernel
        call vert_visc_cra_u_kernel<<<grid, tBlock>>>( &
            u_d, h_d, CS%mask2dCu_d, CS%visc_rem_u, CS%taux_bot, taux_d, &
            CS%Kv, CS%Kv_ml, CS%Kv_extra_bbl, CS%Hmix, CS%Hbbl, &
            dt, dt_Rho0, h_neglect, I_Hbbl, &
            n1, n2, nz, is_l, ie_l, js_l, je_l)

        ! Launch v-points kernel
        call vert_visc_cra_v_kernel<<<grid, tBlock>>>( &
            v_d, h_d, CS%mask2dCv_d, CS%visc_rem_v, CS%tauy_bot, tauy_d, &
            CS%Kv, CS%Kv_ml, CS%Kv_extra_bbl, CS%Hmix, CS%Hbbl, &
            dt, dt_Rho0, h_neglect, I_Hbbl, &
            n1, n2, nz, is_l, ie_l, js_l, je_l)

        ! Sync to ensure all output is ready before returning
        istat = cudaDeviceSynchronize()

    end subroutine vert_visc_cra_cuda

    !> Finalize the CUDA vertical viscosity solver.
    !! Deallocates all device arrays.
    subroutine vert_visc_end_cuda(CS)
        type(vert_visc_CS_cuda), intent(inout) :: CS

        if (.not. CS%initialized) return

        if (allocated(CS%visc_rem_u)) deallocate(CS%visc_rem_u)
        if (allocated(CS%visc_rem_v)) deallocate(CS%visc_rem_v)
        if (allocated(CS%taux_bot)) deallocate(CS%taux_bot)
        if (allocated(CS%tauy_bot)) deallocate(CS%tauy_bot)
        if (allocated(CS%mask2dCu_d)) deallocate(CS%mask2dCu_d)
        if (allocated(CS%mask2dCv_d)) deallocate(CS%mask2dCv_d)

        CS%initialized = .false.

    end subroutine vert_visc_end_cuda

end module mom6_vert_visc_cuda
