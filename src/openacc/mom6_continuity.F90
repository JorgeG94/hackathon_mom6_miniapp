!> MOM6 Continuity Solver Module
!!
!! PPM (Piecewise Parabolic Method) continuity solver extracted from MOM6.
!! Solves the layer thickness equation: dh/dt = -div(uh, vh)
!!
!! Original code from: src/core/MOM_continuity_PPM.F90
!!
module mom6_continuity
    use iso_fortran_env, only: dp => real64, int64
    use mom6_types, only: ocean_grid_type, verticalGrid_type, BT_cont_type
    implicit none
    private

    public :: continuity_PPM, continuity_init, continuity_end, continuity_CS

    !> Control structure for continuity solver
    type :: continuity_CS
        logical :: initialized = .false. !< True if this control structure has been initialized.
        logical :: upwind_1st      !< If true, use a first-order upwind scheme.
        logical :: monotonic       !< If true, use the Colella & Woodward monotonic
                                 !! limiter; otherwise use a simple positive
                                 !! definite limiter.
        logical :: simple_2nd      !< If true, use a simple second order (arithmetic
                                 !! mean) interpolation of the edge values instead
                                 !! of the higher order interpolation.
        real(dp) :: tol_eta            !< The tolerance for free-surface height
                                 !! discrepancies between the barotropic solution and
                                 !! the sum of the layer thicknesses [H ~> m or kg m-2].
        real(dp) :: tol_vel            !< The tolerance for barotropic velocity
                                 !! discrepancies between the barotropic solution and
                                 !! the sum of the layer thicknesses [L T-1 ~> m s-1].
        real(dp) :: CFL_limit_adjust   !< The maximum CFL of the adjusted velocities [nondim]
        logical :: aggress_adjust  !< If true, allow the adjusted velocities to have a
                                 !! relative CFL change up to 0.5.  False by default.
        logical :: vol_CFL         !< If true, use the ratio of the open face lengths
                                 !! to the tracer cell areas when estimating CFL
                                 !! numbers.  Without aggress_adjust, the default is
                                 !! false; it is always true with.
        logical :: better_iter     !< If true, stop corrective iterations using a
                                 !! velocity-based criterion and only stop if the
                                 !! iteration is better than all predecessors.
        logical :: use_visc_rem_max !< If true, use more appropriate limiting bounds
                                 !! for corrections in strongly viscous columns.
        logical :: marginal_faces  !< If true, use the marginal face areas from the
                                 !! continuity solver for use as the weights in the
                                 !! barotropic solver.  Otherwise use the transport
                                 !! averaged areas.

        ! GPU work arrays for fully-ported zonal_mass_flux
        real(dp), allocatable :: duhdu(:,:,:)      !< Partial derivative of uh with u [H L] (isd:ied, jsd:jed, ke)
        real(dp), allocatable :: du(:,:)           !< Barotropic velocity correction [L T-1] (isd:ied, jsd:jed)
        real(dp), allocatable :: du_max_CFL(:,:)   !< Upper CFL limit on du [L T-1] (isd:ied, jsd:jed)
        real(dp), allocatable :: du_min_CFL(:,:)   !< Lower CFL limit on du [L T-1] (isd:ied, jsd:jed)
        real(dp), allocatable :: duhdu_tot_0(:,:)  !< Sum of duhdu over k [H L] (isd:ied, jsd:jed)
        real(dp), allocatable :: uh_tot_0(:,:)     !< Sum of uh over k [H L2 T-1] (isd:ied, jsd:jed)
        real(dp), allocatable :: visc_rem_max(:,:) !< Column max of visc_rem [nondim] (isd:ied, jsd:jed)
        integer(int64) :: nbytes = 0  ! Total bytes allocated for GPU arrays
    end type continuity_CS

    interface
        !> Newton iteration to adjust zonal fluxes to match barotropic transport (GPU version).
        !! Implemented in submodule mom6_continuity_adjust.
        module subroutine zonal_flux_adjust_gpu(u, h_in, h_W, h_E, uhbt, uh, CS, &
                                         visc_rem_u, dt, G, GV, por_face_areaU, &
                                         use_visc_rem)
            type(ocean_grid_type), intent(in) :: G
            type(verticalGrid_type), intent(in) :: GV
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, h_in, h_W, h_E
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in) :: uhbt
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: uh
            type(continuity_CS), intent(inout) :: CS
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: visc_rem_u
            real(dp), intent(in) :: dt
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: por_face_areaU
            logical, intent(in) :: use_visc_rem
        end subroutine zonal_flux_adjust_gpu

        !> GPU kernel: Compute BT_cont face areas (Newton + 3 test velocities).
        !! Implemented in submodule mom6_continuity_adjust.
        module subroutine set_zonal_BT_cont_gpu(u, h_in, h_W, h_E, BT_cont, CS, &
                                         visc_rem_u, dt, G, GV, por_face_areaU, use_visc_rem)
            type(ocean_grid_type), intent(in) :: G
            type(verticalGrid_type), intent(in) :: GV
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, h_in, h_W, h_E
            type(BT_cont_type), intent(inout) :: BT_cont
            type(continuity_CS), intent(inout) :: CS
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: visc_rem_u
            real(dp), intent(in) :: dt
            real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: por_face_areaU
            logical, intent(in) :: use_visc_rem
        end subroutine set_zonal_BT_cont_gpu
    end interface

contains

    !> Initialize the continuity solver
    subroutine continuity_init(CS, G, GV, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
        type(continuity_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(:, :), intent(inout), allocatable:: uhbt, du_cor
        real(dp), dimension(:, :, :), intent(inout), allocatable:: u_cor, por_face_areaU, visc_rem_u

        allocate (uhbt(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate (u_cor(G%isd:G%ied, G%jsd:G%jed, GV%ke), source=0.0_dp)
        allocate (du_cor(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate (por_face_areaU(G%isd:G%ied, G%jsd:G%jed, GV%ke), source=1._dp)
        allocate (visc_rem_u(G%isd:G%ied, G%jsd:G%jed, GV%ke), source=1._dp)

        CS%initialized = .true.
        CS%upwind_1st = .false.
        CS%monotonic = .false.
        CS%simple_2nd = .false.
        CS%tol_eta = 1.d-12
        CS%tol_vel = 3.d8
        CS%CFL_limit_adjust = 0.5d0
        CS%aggress_adjust = .false.
        CS%vol_CFL = .false.
        CS%better_iter = .true.
        CS%use_visc_rem_max = .true.
        CS%marginal_faces = .true.

        ! Allocate GPU work arrays
        allocate(CS%duhdu(G%isd:G%ied, G%jsd:G%jed, GV%ke), source=0.0_dp)
        allocate(CS%du(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate(CS%du_max_CFL(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate(CS%du_min_CFL(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate(CS%duhdu_tot_0(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate(CS%uh_tot_0(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)
        allocate(CS%visc_rem_max(G%isd:G%ied, G%jsd:G%jed), source=0.0_dp)

        ! Compute total bytes: CS arrays (1 3D + 6 2D) + external arrays (2 2D + 3 3D)
        CS%nbytes = int(G%ied - G%isd + 1, int64) * int(G%jed - G%jsd + 1, int64) &
            * (8_int64 + 4_int64 * int(GV%ke, int64)) * 8_int64

        !$acc enter data copyin(CS)
        !$acc enter data copyin(uhbt, por_face_areaU, visc_rem_u)
        !$acc enter data create(u_cor, du_cor)
        !$acc enter data create(CS%duhdu, CS%du, CS%du_max_CFL, CS%du_min_CFL)
        !$acc enter data create(CS%duhdu_tot_0, CS%uh_tot_0, CS%visc_rem_max)

    end subroutine continuity_init

    !> Finalize the continuity solver
    subroutine continuity_end(CS, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
        type(continuity_CS), intent(inout) :: CS
        real(dp), dimension(:, :), intent(inout), allocatable:: uhbt, du_cor
        real(dp), dimension(:, :, :), intent(inout), allocatable:: u_cor, por_face_areaU, visc_rem_u

        if (.not. CS%initialized) return

        !$acc exit data delete(CS%duhdu, CS%du, CS%du_max_CFL, CS%du_min_CFL)
        !$acc exit data delete(CS%duhdu_tot_0, CS%uh_tot_0, CS%visc_rem_max)
        !$acc exit data delete(uhbt, por_face_areaU, visc_rem_u)
        !$acc exit data delete(u_cor, du_cor)
        !$acc exit data delete(CS)

        if (allocated(CS%duhdu)) deallocate(CS%duhdu)
        if (allocated(CS%du)) deallocate(CS%du)
        if (allocated(CS%du_max_CFL)) deallocate(CS%du_max_CFL)
        if (allocated(CS%du_min_CFL)) deallocate(CS%du_min_CFL)
        if (allocated(CS%duhdu_tot_0)) deallocate(CS%duhdu_tot_0)
        if (allocated(CS%uh_tot_0)) deallocate(CS%uh_tot_0)
        if (allocated(CS%visc_rem_max)) deallocate(CS%visc_rem_max)
        if (allocated(uhbt)) deallocate (uhbt)
        if (allocated(u_cor)) deallocate (u_cor)
        if (allocated(du_cor)) deallocate (du_cor)
        if (allocated(por_face_areaU)) deallocate (por_face_areaU)
        if (allocated(visc_rem_u)) deallocate (visc_rem_u)

        CS%initialized = .false.

    end subroutine continuity_end

    !> Main continuity solver using PPM
  !! Updates layer thickness h from hin using velocities u, v
    subroutine continuity_PPM(u, hin, h, uh, dt, G, GV, CS, por_face_areaU, uhbt, visc_rem_u, u_cor, BT_cont, du_cor)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u    ! Zonal velocity [L T-1]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: hin  ! Initial thickness [H]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: h  ! Final thickness [H]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh  ! Zonal flux [H L2 T-1]
        real(dp), intent(in) :: dt        ! Time step [T]
        type(continuity_CS), intent(inout) :: CS
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)  :: por_face_areaU !< pointers to porous barrier fractional cell metrics
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), optional, intent(in)    :: uhbt !< The summed volume flux through zonal faces
                                                      !! [H L2 T-1 ~> m3 s-1 or kg s-1].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), optional, intent(in)    :: visc_rem_u
        !< The fraction of zonal momentum originally
                                 !! in a layer that remains after a time-step of viscosity, and the
                                 !! fraction of a time-step's worth of a barotropic acceleration that
                                 !! a layer experiences after viscosity is applied [nondim].
                                 !! Visc_rem_u is between 0 (at the bottom) and 1 (far above the bottom).
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), optional, intent(out)   :: u_cor
        !< The zonal velocities that give uhbt as the depth-integrated transport [L T-1 ~> m s-1].
        type(BT_cont_type), optional, pointer :: BT_cont !< A structure with elements that describe
                                 !!  the effective open face areas as a function of barotropic flow.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), optional, intent(out)   :: du_cor !< The zonal velocity increments from u that give uhbt
                                                      !! as the depth-integrated transports [L T-1 ~> m s-1].
        ! Local variables
        real(dp) :: h_W(G%isd:G%ied, G%jsd:G%jed, GV%ke) ! West edge thicknesses in the zonal PPM reconstruction [H ~> m or kg m-2]
        real(dp) :: h_E(G%isd:G%ied, G%jsd:G%jed, GV%ke) ! East edge thicknesses in the zonal PPM reconstruction [H ~> m or kg m-2]
        real(dp) :: h_min  ! The minimum layer thickness [H ~> m or kg m-2].  h_min could be 0.

        integer :: is, ie, js, je, nz

        is = G%isc
        ie = G%iec
        js = G%jsc
        je = G%jec
        nz = GV%ke

        ! advect zonally — all GPU, zero memcpys
        !$acc data create(h_W, h_E)
        call zonal_edge_thickness(hin, h_W, h_E, G, GV, CS)
        call zonal_mass_flux(u, hin, h_W, h_E, uh, dt, G, GV, CS, por_face_areaU, &
                             uhbt, visc_rem_u, u_cor, BT_cont, du_cor)
        call continuity_zonal_convergence(h, uh, dt, G, GV, hin)
        !$acc end data

        ! ignoring meridional direction for this example

    end subroutine continuity_PPM

!> Updates the thicknesses due to zonal thickness fluxes.
    subroutine continuity_zonal_convergence(h, uh, dt, G, GV, hin, hmin)
        type(ocean_grid_type), intent(in)    :: G    !< Ocean's grid structure
        type(verticalGrid_type), intent(in)    :: GV   !< Ocean's vertical grid structure
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(inout) :: h    !< Final layer thickness [H ~> m or kg m-2]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: uh   !< Zonal thickness flux, u*h*dy [H L2 T-1 ~> m3 s-1 or kg s-1]
        real(dp), intent(in)    :: dt   !< Time increment [T ~> s]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            optional, intent(in)    :: hin  !< Initial layer thickness [H ~> m or kg m-2].
                                                     !! If hin is absent, h is also the initial thickness.
        real(dp), optional, intent(in)    :: hmin !< The minimum layer thickness [H ~> m or kg m-2]

        real(dp) :: h_min  ! The minimum layer thickness [H ~> m or kg m-2].  h_min could be 0.
        integer :: i, j, k

        h_min = 0.0_dp; if (present(hmin)) h_min = hmin

        if (present(hin)) then
            !$acc parallel loop collapse(3) default(present)
            do k = 1, GV%ke; do j = G%jsc, G%jec; do i = G%isc, G%iec
                    h(i, j, k) = max(hin(i, j, k) - dt*G%IareaT(i, j)*(uh(I, j, k) - uh(I - 1, j, k)), h_min)
            end do; end do; end do
        else
            !$acc parallel loop collapse(3) default(present)
            do k = 1, GV%ke; do j = G%jsc, G%jec; do i = G%isc, G%iec
                    h(i, j, k) = max(h(i, j, k) - dt*G%IareaT(i, j)*(uh(I, j, k) - uh(I - 1, j, k)), h_min)
                end do; end do; end do
        end if

    end subroutine continuity_zonal_convergence

!> Set the reconstructed thicknesses at the eastern and western edges of tracer cells.
    subroutine zonal_edge_thickness(h_in, h_W, h_E, G, GV, CS)
        type(ocean_grid_type), intent(in)    :: G    !< Ocean's grid structure.
        type(verticalGrid_type), intent(in)    :: GV   !< Ocean's vertical grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: h_in !< Tracer cell layer thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(out)   :: h_W  !< Western edge layer thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(out)   :: h_E  !< Eastern edge layer thickness [H ~> m or kg m-2].
        type(continuity_CS), intent(in)    :: CS   !< This module's control structure.

        ! Local variables
        integer :: i, j, k, ish, ieh, jsh, jeh, nz

        ish = G%isc
        ieh = G%iec
        jsh = G%jsc
        jeh = G%jec
        nz = GV%ke

        if (CS%upwind_1st) then
            !$acc parallel loop collapse(3) default(present)
            do k = 1, nz
                do j = jsh, jeh
                    do i = ish - 1, ieh + 1
                        h_W(i, j, k) = h_in(i, j, k)
                        h_E(i, j, k) = h_in(i, j, k)
                    end do
                end do
            end do
        else
            call PPM_reconstruction_x_3d(h_in, h_W, h_E, G, GV, &
                                         2.0_dp*GV%Angstrom_H, CS%monotonic, CS%simple_2nd)
        end if

    end subroutine zonal_edge_thickness

!> Calculates the mass or volume fluxes through the zonal faces, and other related quantities.
!! Fully GPU-ported version — all computation runs on device with default(present).
    subroutine zonal_mass_flux(u, h_in, h_W, h_E, uh, dt, G, GV, CS, por_face_areaU, &
                               uhbt, visc_rem_u, u_cor, BT_cont, du_cor)
        type(ocean_grid_type), intent(in)    :: G
        type(verticalGrid_type), intent(in)    :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_in
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_W
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_E
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh
        real(dp), intent(in) :: dt
        type(continuity_CS), intent(inout) :: CS
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: por_face_areaU
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), optional, intent(in) :: uhbt
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), optional, intent(in) :: visc_rem_u
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), optional, intent(out) :: u_cor
        type(BT_cont_type), optional, pointer :: BT_cont
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), optional, intent(out) :: du_cor

        ! Local variables
        real(dp) :: CFL_dt, I_dt
        integer :: i, j, k, ish, ieh, jsh, jeh, nz
        logical :: use_visc_rem, set_BT_cont
        ! Newton iteration locals (for inlined zonal_flux_adjust)
        real(dp) :: du_val, du_prev, ddu, uh_err_val, uh_err_best_val
        real(dp) :: duhdu_tot_val, du_max_val, du_min_val
        real(dp) :: tol_eta_n, tol_vel_n
        real(dp) :: CFL_n, curv_3_n, h_marg_n, u_adj_n, uh_k_n, duhdu_k_n, visc_rem_val_n
        logical :: do_more
        integer :: itt
        integer, parameter :: max_itts = 20

        use_visc_rem = present(visc_rem_u)
        set_BT_cont = .false.
        if (present(BT_cont)) set_BT_cont = (associated(BT_cont))

        ish = G%isc ; ieh = G%iec ; jsh = G%jsc ; jeh = G%jec ; nz = GV%ke

        CFL_dt = CS%CFL_limit_adjust / dt
        I_dt = 1.0_dp / dt
        if (CS%aggress_adjust) CFL_dt = I_dt

        ! --- Step 2: Compute initial uh and duhdu on GPU (collapse(3)) ---
        call zonal_flux_layer_3d(u, h_in, h_W, h_E, uh, CS, visc_rem_u, &
                                 dt, G, GV, CS%vol_CFL, por_face_areaU, use_visc_rem)

        if (present(uhbt) .or. set_BT_cont) then
            ! --- Step 3a: visc_rem_max ---
            if (use_visc_rem .and. CS%use_visc_rem_max) then
                !$acc parallel loop collapse(2) default(present)
                do j = jsh, jeh
                    do I = ish - 1, ieh
                        CS%visc_rem_max(I, j) = 0.0_dp
                        do k = 1, nz
                            CS%visc_rem_max(I, j) = max(CS%visc_rem_max(I, j), visc_rem_u(I, j, k))
                        end do
                    end do
                end do
            else
                !$acc parallel loop collapse(2) default(present)
                do j = jsh, jeh
                    do I = ish - 1, ieh
                        CS%visc_rem_max(I, j) = 1.0_dp
                    end do
                end do
            end if

            ! --- Step 3b: uh_tot_0 + duhdu_tot_0 ---
            !$acc parallel loop collapse(2) default(present)
            do j = jsh, jeh
                do I = ish - 1, ieh
                    CS%uh_tot_0(I, j) = 0.0_dp
                    CS%duhdu_tot_0(I, j) = 0.0_dp
                    do k = 1, nz
                        CS%uh_tot_0(I, j) = CS%uh_tot_0(I, j) + uh(I, j, k)
                        CS%duhdu_tot_0(I, j) = CS%duhdu_tot_0(I, j) + CS%duhdu(I, j, k)
                    end do
                end do
            end do

            ! --- Step 3c: CFL limits ---
            call zonal_CFL_limits_gpu(u, CS, visc_rem_u, dt, G, GV, CFL_dt, I_dt, use_visc_rem)

            ! --- Step 4: Newton iteration (call subroutine with non-optional args to avoid nvfortran bug) ---
            if (present(uhbt)) then
                call zonal_flux_adjust_gpu(u, h_in, h_W, h_E, uhbt, uh, CS, &
                                           visc_rem_u, dt, G, GV, por_face_areaU, use_visc_rem)
                ! Write u_cor and du_cor from converged CS%du (explicit present to avoid optional arg bug)
                if (present(u_cor)) then
                    !$acc parallel loop collapse(3) default(present)
                    do k = 1, nz
                        do j = jsh, jeh
                            do I = ish - 1, ieh
                                if (use_visc_rem) then
                                    u_cor(I, j, k) = u(I, j, k) + CS%du(I, j) * visc_rem_u(I, j, k)
                                else
                                    u_cor(I, j, k) = u(I, j, k) + CS%du(I, j)
                                end if
                            end do
                        end do
                    end do
                end if
                if (present(du_cor)) then
                    !$acc parallel loop collapse(2) default(present)
                    do j = jsh, jeh
                        do I = ish - 1, ieh
                            du_cor(I, j) = CS%du(I, j)
                        end do
                    end do
                end if
            end if

            ! --- Step 5: set_zonal_BT_cont on GPU ---
            if (set_BT_cont) then
                call set_zonal_BT_cont_gpu(u, h_in, h_W, h_E, BT_cont, CS, &
                                           visc_rem_u, dt, G, GV, por_face_areaU, use_visc_rem)
            end if
        end if

        ! zonal_flux_thickness is already GPU-ported
        if (set_BT_cont) then
            if (allocated(BT_cont%h_u)) then
                if (present(u_cor)) then
                    call zonal_flux_thickness(u_cor, h_in, h_W, h_E, BT_cont%h_u, dt, G, GV, &
                                              CS%vol_CFL, CS%marginal_faces, por_face_areaU, visc_rem_u)
                else
                    call zonal_flux_thickness(u, h_in, h_W, h_E, BT_cont%h_u, dt, G, GV, &
                                              CS%vol_CFL, CS%marginal_faces, por_face_areaU, visc_rem_u)
                end if
            end if
        end if

    end subroutine zonal_mass_flux

!> GPU kernel: Compute uh and duhdu for all layers — replaces per-j zonal_flux_layer calls
    subroutine zonal_flux_layer_3d(u, h_in, h_W, h_E, uh, CS, visc_rem_u, &
                                   dt, G, GV, vol_CFL, por_face_areaU, use_visc_rem)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u, h_in, h_W, h_E
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: uh
        type(continuity_CS), intent(inout) :: CS
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: visc_rem_u
        real(dp), intent(in) :: dt
        logical, intent(in) :: vol_CFL, use_visc_rem
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: por_face_areaU

        real(dp) :: CFL, curv_3, h_marg, visc_rem_val
        integer :: i, j, k, ish, ieh, jsh, jeh, nz

        ish = G%isc ; ieh = G%iec ; jsh = G%jsc ; jeh = G%jec ; nz = GV%ke

        !$acc parallel loop collapse(3) default(present) private(CFL, curv_3, h_marg, visc_rem_val)
        do k = 1, nz
            do j = jsh, jeh
                do I = ish - 1, ieh
                    if (use_visc_rem) then
                        visc_rem_val = visc_rem_u(I, j, k)
                    else
                        visc_rem_val = 1.0_dp
                    end if

                    if (u(I, j, k) > 0.0_dp) then
                        if (vol_CFL) then
                            CFL = (u(I, j, k) * dt) * (G%dy_Cu(I, j) * G%IareaT(i, j))
                        else
                            CFL = u(I, j, k) * dt * G%IdxT(i, j)
                        end if
                        curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k)
                        uh(I, j, k) = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u(I, j, k) * &
                            (h_E(i, j, k) + CFL * (0.5_dp * (h_W(i, j, k) - h_E(i, j, k)) + curv_3 * (CFL - 1.5_dp)))
                        h_marg = h_E(i, j, k) + CFL * ((h_W(i, j, k) - h_E(i, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    elseif (u(I, j, k) < 0.0_dp) then
                        if (vol_CFL) then
                            CFL = (-u(I, j, k) * dt) * (G%dy_Cu(I, j) * G%IareaT(i + 1, j))
                        else
                            CFL = -u(I, j, k) * dt * G%IdxT(i + 1, j)
                        end if
                        curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0_dp * h_in(i + 1, j, k)
                        uh(I, j, k) = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * u(I, j, k) * &
                            (h_W(i + 1, j, k) + CFL * (0.5_dp * (h_E(i + 1, j, k) - h_W(i + 1, j, k)) + curv_3 * (CFL - 1.5_dp)))
                        h_marg = h_W(i + 1, j, k) + CFL * ((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + 3.0_dp * curv_3 * (CFL - 1.0_dp))
                    else
                        uh(I, j, k) = 0.0_dp
                        h_marg = 0.5_dp * (h_W(i + 1, j, k) + h_E(i, j, k))
                    end if
                    CS%duhdu(I, j, k) = (G%dy_Cu(I, j) * por_face_areaU(I, j, k)) * h_marg * visc_rem_val
                end do
            end do
        end do
    end subroutine zonal_flux_layer_3d

!> GPU kernel: Compute CFL limits for du correction
    subroutine zonal_CFL_limits_gpu(u, CS, visc_rem_u, dt, G, GV, CFL_dt, I_dt, use_visc_rem)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: u
        type(continuity_CS), intent(inout) :: CS
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: visc_rem_u
        real(dp), intent(in) :: dt, CFL_dt, I_dt
        logical, intent(in) :: use_visc_rem

        real(dp) :: I_vrm, dx_W, dx_E, du_lim, vrm_k
        integer :: i, j, k, ish, ieh, jsh, jeh, nz

        ish = G%isc ; ieh = G%iec ; jsh = G%jsc ; jeh = G%jec ; nz = GV%ke

        !$acc parallel loop collapse(2) default(present) &
        !$acc   private(I_vrm, dx_W, dx_E, du_lim, vrm_k)
        do j = jsh, jeh
            do I = ish - 1, ieh
                ! Initial CFL limits from visc_rem_max
                I_vrm = 0.0_dp
                if (CS%visc_rem_max(I, j) > 0.0_dp) I_vrm = 1.0_dp / CS%visc_rem_max(I, j)
                if (CS%vol_CFL) then
                    dx_W = G%areaT(i, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                    if (abs(dx_W) > 1000.0_dp * G%dxT(i, j)) dx_W = 1000.0_dp * G%dxT(i, j)
                    dx_E = G%areaT(i + 1, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                    if (abs(dx_E) > 1000.0_dp * G%dxT(i + 1, j)) dx_E = 1000.0_dp * G%dxT(i + 1, j)
                else
                    dx_W = G%dxT(i, j)
                    dx_E = G%dxT(i + 1, j)
                end if
                CS%du_max_CFL(I, j) = 2.0_dp * (CFL_dt * dx_W) * I_vrm
                CS%du_min_CFL(I, j) = -2.0_dp * (CFL_dt * dx_E) * I_vrm

                ! Tighten CFL limits over k
                if (use_visc_rem) then
                    if (CS%aggress_adjust) then
                        do k = 1, nz
                            vrm_k = visc_rem_u(I, j, k)
                            if (CS%vol_CFL) then
                                dx_W = G%areaT(i, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_W) > 1000.0_dp * G%dxT(i, j)) dx_W = 1000.0_dp * G%dxT(i, j)
                                dx_E = G%areaT(i + 1, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_E) > 1000.0_dp * G%dxT(i + 1, j)) dx_E = 1000.0_dp * G%dxT(i + 1, j)
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if
                            du_lim = 0.499_dp * ((dx_W * I_dt - u(I, j, k)) + min(0.0_dp, u(I - 1, j, k)))
                            if (CS%du_max_CFL(I, j) * vrm_k > du_lim) &
                                CS%du_max_CFL(I, j) = du_lim / vrm_k
                            du_lim = 0.499_dp * ((-dx_E * I_dt - u(I, j, k)) + max(0.0_dp, u(I + 1, j, k)))
                            if (CS%du_min_CFL(I, j) * vrm_k < du_lim) &
                                CS%du_min_CFL(I, j) = du_lim / vrm_k
                        end do
                    else
                        do k = 1, nz
                            vrm_k = visc_rem_u(I, j, k)
                            if (CS%vol_CFL) then
                                dx_W = G%areaT(i, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_W) > 1000.0_dp * G%dxT(i, j)) dx_W = 1000.0_dp * G%dxT(i, j)
                                dx_E = G%areaT(i + 1, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_E) > 1000.0_dp * G%dxT(i + 1, j)) dx_E = 1000.0_dp * G%dxT(i + 1, j)
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if
                            if (CS%du_max_CFL(I, j) * vrm_k > dx_W * CFL_dt - u(I, j, k) * G%mask2dCu(I, j)) &
                                CS%du_max_CFL(I, j) = (dx_W * CFL_dt - u(I, j, k)) / vrm_k
                            if (CS%du_min_CFL(I, j) * vrm_k < -dx_E * CFL_dt - u(I, j, k) * G%mask2dCu(I, j)) &
                                CS%du_min_CFL(I, j) = -(dx_E * CFL_dt + u(I, j, k)) / vrm_k
                        end do
                    end if
                else
                    if (CS%aggress_adjust) then
                        do k = 1, nz
                            if (CS%vol_CFL) then
                                dx_W = G%areaT(i, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_W) > 1000.0_dp * G%dxT(i, j)) dx_W = 1000.0_dp * G%dxT(i, j)
                                dx_E = G%areaT(i + 1, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_E) > 1000.0_dp * G%dxT(i + 1, j)) dx_E = 1000.0_dp * G%dxT(i + 1, j)
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if
                            CS%du_max_CFL(I, j) = min(CS%du_max_CFL(I, j), 0.499_dp * &
                                ((dx_W * I_dt - u(I, j, k)) + min(0.0_dp, u(I - 1, j, k))))
                            CS%du_min_CFL(I, j) = max(CS%du_min_CFL(I, j), 0.499_dp * &
                                ((-dx_E * I_dt - u(I, j, k)) + max(0.0_dp, u(I + 1, j, k))))
                        end do
                    else
                        do k = 1, nz
                            if (CS%vol_CFL) then
                                dx_W = G%areaT(i, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_W) > 1000.0_dp * G%dxT(i, j)) dx_W = 1000.0_dp * G%dxT(i, j)
                                dx_E = G%areaT(i + 1, j) / (G%dy_Cu(I, j) + 1.0e-30_dp)
                                if (abs(dx_E) > 1000.0_dp * G%dxT(i + 1, j)) dx_E = 1000.0_dp * G%dxT(i + 1, j)
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if
                            CS%du_max_CFL(I, j) = min(CS%du_max_CFL(I, j), dx_W * CFL_dt - u(I, j, k))
                            CS%du_min_CFL(I, j) = max(CS%du_min_CFL(I, j), -(dx_E * CFL_dt + u(I, j, k)))
                        end do
                    end if
                end if

                ! Ensure bounds include 0
                CS%du_max_CFL(I, j) = max(CS%du_max_CFL(I, j), 0.0_dp)
                CS%du_min_CFL(I, j) = min(CS%du_min_CFL(I, j), 0.0_dp)
            end do
        end do
    end subroutine zonal_CFL_limits_gpu

!> Sets the effective interface thickness associated with the fluxes at each zonal velocity point,
!! optionally scaling back these thicknesses to account for viscosity and fractional open areas.
    subroutine zonal_flux_thickness(u, h, h_W, h_E, h_u, dt, G, GV, vol_CFL, &
                                    marginal, por_face_areaU, visc_rem_u)
        type(ocean_grid_type), intent(in)    :: G    !< Ocean's grid structure.
        type(verticalGrid_type), intent(in)    :: GV   !< Ocean's vertical grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)   :: u    !< Zonal velocity [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h    !< Layer thickness used to
                                                                   !! calculate fluxes [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_W  !< West edge thickness in the
                                                                   !! reconstruction [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_E  !< East edge thickness in the
                                                                   !! reconstruction [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(inout) :: h_u !< Effective thickness at zonal faces,
                                                                   !! scaled down to account for the effects of
                                                                   !! viscosity and the fractional open area
                                                                   !! [H ~> m or kg m-2].
        real(dp), intent(in)    :: dt   !< Time increment [T ~> s].
        logical, intent(in)    :: vol_CFL !< If true, rescale the ratio
                          !! of face areas to the cell areas when estimating the CFL number.
        logical, intent(in)    :: marginal !< If true, report the
                          !! marginal face thicknesses; otherwise report transport-averaged thicknesses.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: por_face_areaU !< fractional open area of U-faces [nondim]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            optional, intent(in)    :: visc_rem_u
        !< Both the fraction of the momentum originally in a layer that remains after
                          !! a time-step of viscosity, and the fraction of a time-step's worth of a
                          !! barotropic acceleration that a layer experiences after viscosity is applied [nondim].
                          !! Visc_rem_u is between 0 (at the bottom) and 1 (far above the bottom).

        ! Local variables
        real(dp) :: CFL  ! The CFL number based on the local velocity and grid spacing [nondim]
        real(dp) :: curv_3 ! A measure of the thickness curvature over a grid length [H ~> m or kg m-2]
        real(dp) :: h_avg  ! The average thickness of a flux [H ~> m or kg m-2].
        real(dp) :: h_marg ! The marginal thickness of a flux [H ~> m or kg m-2].
        logical :: local_open_BC
        integer :: i, j, k, ish, ieh, jsh, jeh, nz, n
        ish = G%isc
        ieh = G%iec
        jsh = G%jsc
        jeh = G%jec
        nz = GV%ke

        !$acc parallel loop collapse(3) default(present) private(CFL,curv_3,h_marg,h_avg)
        do k = 1, nz
            do j = jsh, jeh
                do I = ish - 1, ieh
                    if (u(I, j, k) > 0.0_dp) then
                        if (vol_CFL) then
                            CFL = (u(I, j, k)*dt)*(G%dy_Cu(I, j)*G%IareaT(i, j))
                        else
                            CFL = u(I, j, k)*dt*G%IdxT(i, j)
                        end if
                        curv_3 = (h_W(i, j, k) + h_E(i, j, k)) - 2.0*h(i, j, k)
                        h_avg = h_E(i, j, k) + CFL*(0.5*(h_W(i, j, k) - h_E(i, j, k)) + curv_3*(CFL - 1.5))
                        h_marg = h_E(i, j, k) + CFL*((h_W(i, j, k) - h_E(i, j, k)) + 3.0*curv_3*(CFL - 1.0))
                    elseif (u(I, j, k) < 0.0_dp) then
                        if (vol_CFL) then
                            CFL = (-u(I, j, k)*dt)*(G%dy_Cu(I, j)*G%IareaT(i + 1, j))
                        else
                            CFL = -u(I, j, k)*dt*G%IdxT(i + 1, j)
                        end if
                        curv_3 = (h_W(i + 1, j, k) + h_E(i + 1, j, k)) - 2.0*h(i + 1, j, k)
                        h_avg = h_W(i + 1, j, k) + CFL*(0.5*(h_E(i + 1, j, k) - h_W(i + 1, j, k)) + curv_3*(CFL - 1.5))
                        h_marg = h_W(i + 1, j, k) + CFL*((h_E(i + 1, j, k) - h_W(i + 1, j, k)) + &
                                                         3.0*curv_3*(CFL - 1.0))
                    else
                        h_avg = 0.5*(h_W(i + 1, j, k) + h_E(i, j, k))
                        !   The choice to use the arithmetic mean here is somewhat arbitrarily, but
                        ! it should be noted that h_W(i+1,j,k) and h_E(i,j,k) are usually the same.
                        h_marg = 0.5*(h_W(i + 1, j, k) + h_E(i, j, k))
                        !    h_marg = (2.0 * h_W(i+1,j,k) * h_E(i,j,k)) / &
                        !             (h_W(i+1,j,k) + h_E(i,j,k) + GV%H_subroundoff)
                    end if

                    if (marginal) then
                        h_u(I, j, k) = h_marg
                    else
                        h_u(I, j, k) = h_avg
                    end if
                end do
            end do
        end do
        if (present(visc_rem_u)) then
            ! Scale back the thickness to account for the effects of viscosity and the fractional open
            ! thickness to give an appropriate non-normalized weight for each layer in determining the
            ! barotropic acceleration.
            !$acc parallel loop collapse(3) default(present)
            do k = 1, nz
                do j = jsh, jeh
                    do I = ish - 1, ieh
                        h_u(I, j, k) = h_u(I, j, k)*(visc_rem_u(I, j, k)*por_face_areaU(I, j, k))
                    end do
                end do
            end do
        else
            !$acc parallel loop collapse(3) default(present)
            do k = 1, nz
                do j = jsh, jeh
                    do I = ish - 1, ieh
                        h_u(I, j, k) = h_u(I, j, k)*por_face_areaU(I, j, k)
                    end do
                end do
            end do
        end if

!   local_open_BC = .false.
!   if (associated(OBC)) local_open_BC = OBC%open_u_BCs_exist_globally
!   if (local_open_BC) then
!     do n = 1, OBC%number_of_segments
!       if (OBC%segment(n)%open .and. OBC%segment(n)%is_E_or_W) then
!         I = OBC%segment(n)%HI%IsdB
!         if (OBC%segment(n)%direction == OBC_DIRECTION_E) then
!           if (present(visc_rem_u)) then ; do k=1,nz
!             do j = OBC%segment(n)%HI%jsd, OBC%segment(n)%HI%jed
!               h_u(I,j,k) = h(i,j,k) * (visc_rem_u(I,j,k) * por_face_areaU(I,j,k))
!             enddo
!           enddo ; else ; do k=1,nz
!             do j = OBC%segment(n)%HI%jsd, OBC%segment(n)%HI%jed
!               h_u(I,j,k) = h(i,j,k) * por_face_areaU(I,j,k)
!             enddo
!           enddo ; endif
!         else
!           if (present(visc_rem_u)) then ; do k=1,nz
!             do j = OBC%segment(n)%HI%jsd, OBC%segment(n)%HI%jed
!               h_u(I,j,k) = h(i+1,j,k) * (visc_rem_u(I,j,k) * por_face_areaU(I,j,k))
!             enddo
!           enddo ; else ; do k=1,nz
!             do j = OBC%segment(n)%HI%jsd, OBC%segment(n)%HI%jed
!               h_u(I,j,k) = h(i+1,j,k) * por_face_areaU(I,j,k)
!             enddo
!           enddo ; endif
!         endif
!       endif
!     enddo
!   endif

    end subroutine zonal_flux_thickness


!> Evaluates the zonal mass or volume fluxes in a layer (CPU version for legacy callers).
    subroutine zonal_flux_layer(u, h, h_W, h_E, uh, duhdu, visc_rem, dt, G, j, &
                                ish, ieh, do_I, vol_CFL, por_face_areaU)
        type(ocean_grid_type), intent(in) :: G
        real(dp), dimension(G%isd:G%ied), intent(in) :: u, visc_rem, h, h_W, h_E
        real(dp), dimension(G%isd:G%ied), intent(inout) :: uh, duhdu
        real(dp), intent(in) :: dt
        integer, intent(in) :: j, ish, ieh
        logical, dimension(G%isd:G%ied), intent(in) :: do_I
        logical, intent(in) :: vol_CFL
        real(dp), dimension(G%isd:G%ied), intent(in) :: por_face_areaU

        real(dp) :: CFL, curv_3, h_marg
        integer :: i

        do I = ish - 1, ieh
            if (do_I(I)) then
                if (u(I) > 0.0_dp) then
                    if (vol_CFL) then
                        CFL = (u(I)*dt)*(G%dy_Cu(I, j)*G%IareaT(i, j))
                    else
                        CFL = u(I)*dt*G%IdxT(i, j)
                    end if
                    curv_3 = (h_W(i) + h_E(i)) - 2.0_dp*h(i)
                    uh(I) = (G%dy_Cu(I, j)*por_face_areaU(I))*u(I)* &
                            (h_E(i) + CFL*(0.5_dp*(h_W(i) - h_E(i)) + curv_3*(CFL - 1.5_dp)))
                    h_marg = h_E(i) + CFL*((h_W(i) - h_E(i)) + 3.0_dp*curv_3*(CFL - 1.0_dp))
                elseif (u(I) < 0.0_dp) then
                    if (vol_CFL) then
                        CFL = (-u(I)*dt)*(G%dy_Cu(I, j)*G%IareaT(i + 1, j))
                    else
                        CFL = -u(I)*dt*G%IdxT(i + 1, j)
                    end if
                    curv_3 = (h_W(i + 1) + h_E(i + 1)) - 2.0_dp*h(i + 1)
                    uh(I) = (G%dy_Cu(I, j)*por_face_areaU(I))*u(I)* &
                            (h_W(i + 1) + CFL*(0.5_dp*(h_E(i + 1) - h_W(i + 1)) + curv_3*(CFL - 1.5_dp)))
                    h_marg = h_W(i + 1) + CFL*((h_E(i + 1) - h_W(i + 1)) + 3.0_dp*curv_3*(CFL - 1.0_dp))
                else
                    uh(I) = 0.0_dp
                    h_marg = 0.5_dp*(h_W(i + 1) + h_E(i))
                end if
                duhdu(I) = (G%dy_Cu(I, j)*por_face_areaU(I))*h_marg*visc_rem(I)
            end if
        end do
    end subroutine zonal_flux_layer

!> Returns the barotropic velocity adjustment (LEGACY CPU version - kept for continuity_driver)
    subroutine zonal_flux_adjust(u, h_in, h_W, h_E, uhbt, uh_tot_0, duhdu_tot_0, &
                                 du, du_max_CFL, du_min_CFL, dt, G, GV, CS, visc_rem, &
                                 j, ish, ieh, do_I_in, por_face_areaU, uh_3d)

        type(ocean_grid_type), intent(in)    :: G    !< Ocean's grid structure.
        type(verticalGrid_type), intent(in)    :: GV   !< Ocean's vertical grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)   :: u    !< Zonal velocity [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_in !< Layer thickness used to
                                                                   !! calculate fluxes [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_W  !< West edge thickness in the
                                                                   !! reconstruction [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_E  !< East edge thickness in the
                                                                   !! reconstruction [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, GV%ke), intent(in)    :: visc_rem !< Both the fraction of the
                       !! momentum originally in a layer that remains after a time-step of viscosity, and
                       !! the fraction of a time-step's worth of a barotropic acceleration that a layer
                       !! experiences after viscosity is applied [nondim].
                       !! Visc_rem is between 0 (at the bottom) and 1 (far above the bottom).
        real(dp), dimension(G%isd:G%ied), intent(in)    :: uhbt !< The summed volume flux
                       !! through zonal faces [H L2 T-1 ~> m3 s-1 or kg s-1].

        real(dp), dimension(G%isd:G%ied), intent(in)    :: du_max_CFL  !< Maximum acceptable
                       !! value of du [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: du_min_CFL  !< Minimum acceptable
                       !! value of du [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: uh_tot_0    !< The summed transport
                       !! with 0 adjustment [H L2 T-1 ~> m3 s-1 or kg s-1].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: duhdu_tot_0 !< The partial derivative
                       !! of du_err with du at 0 adjustment [H L ~> m2 or kg m-1].
        real(dp), dimension(G%isd:G%ied), intent(out)   :: du !<
                       !! The barotropic velocity adjustment [L T-1 ~> m s-1].
        real(dp), intent(in)    :: dt   !< Time increment [T ~> s].
        type(continuity_CS), intent(in)    :: CS   !< This module's control structure.
        integer, intent(in)    :: j    !< Spatial index.
        integer, intent(in)    :: ish  !< Start of index range.
        integer, intent(in)    :: ieh  !< End of index range.
        logical, dimension(G%isd:G%ied), intent(in)    :: do_I_in     !<
                       !! A logical flag indicating which I values to work on.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in) :: por_face_areaU !< fractional open area of U-faces [nondim]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), optional, intent(inout) :: uh_3d !<
                       !! Volume flux through zonal faces = u*h*dy [H L2 T-1 ~> m3 s-1 or kg s-1].
        ! Local variables
        real(dp), dimension(G%isd:G%ied, GV%ke) :: &
            uh_aux, &  ! An auxiliary zonal volume flux [H L2 T-1 ~> m3 s-1 or kg s-1].
            duhdu      ! Partial derivative of uh with u [H L ~> m2 or kg m-1].
        real(dp), dimension(G%isd:G%ied) :: &
            uh_err, &  ! Difference between uhbt and the summed uh [H L2 T-1 ~> m3 s-1 or kg s-1].
            uh_err_best, & ! The smallest value of uh_err found so far [H L2 T-1 ~> m3 s-1 or kg s-1].
            u_new, &   ! The velocity with the correction added [L T-1 ~> m s-1].
            duhdu_tot, &! Summed partial derivative of uh with u [H L ~> m2 or kg m-1].
            du_min, &  ! Lower limit on du correction based on CFL limits and previous iterations [L T-1 ~> m s-1]
            du_max     ! Upper limit on du correction based on CFL limits and previous iterations [L T-1 ~> m s-1]
        real(dp) :: du_prev ! The previous value of du [L T-1 ~> m s-1].
        real(dp) :: ddu     ! The change in du from the previous iteration [L T-1 ~> m s-1].
        real(dp) :: tol_eta ! The tolerance for the current iteration [H ~> m or kg m-2].
        real(dp) :: tol_vel ! The tolerance for velocity in the current iteration [L T-1 ~> m s-1].
        integer :: i, k, nz, itt
        integer, parameter :: max_itts = 20
        logical :: domore, do_I(G%isd:G%ied)

        nz = GV%ke

        uh_aux(:, :) = 0.0_dp
        duhdu(:, :) = 0.0_dp

        if (present(uh_3d)) then
            do k = 1, nz
            do I = ish - 1, ieh
                uh_aux(i, k) = uh_3d(I, j, k)
            end do
            end do
        end if

        do I = ish - 1, ieh
            du(I) = 0.0_dp
            do_I(I) = do_I_in(I)
            du_max(I) = du_max_CFL(I)
            du_min(I) = du_min_CFL(I)
            uh_err(I) = uh_tot_0(I) - uhbt(I)
            duhdu_tot(I) = duhdu_tot_0(I)
            uh_err_best(I) = abs(uh_err(I))
        end do

        do itt = 1, max_itts
            select case (itt)
            case (:1)
                tol_eta = 1e-6*CS%tol_eta
            case (2)
                tol_eta = 1e-4*CS%tol_eta
            case (3)
                tol_eta = 1e-2*CS%tol_eta
            case default
                tol_eta = CS%tol_eta
            end select
            tol_vel = CS%tol_vel

            do I = ish - 1, ieh
                if (uh_err(I) > 0.0_dp) then
                    du_max(I) = du(I)
                elseif (uh_err(I) < 0.0_dp) then
                    du_min(I) = du(I)
                else
                    do_I(I) = .false.
                end if
            end do
            domore = .false.
            do I = ish - 1, ieh
                if (do_I(I)) then
                    if ((dt*min(G%IareaT(i, j), G%IareaT(i + 1, j))*abs(uh_err(I)) > tol_eta) .or. &
                        (CS%better_iter .and. ((abs(uh_err(I)) > tol_vel*duhdu_tot(I)) .or. &
                                               (abs(uh_err(I)) > uh_err_best(I))))) then
                        !   Use Newton's method, provided it stays bounded.  Otherwise bisect
                        ! the value with the appropriate bound.
                        ddu = -uh_err(I)/duhdu_tot(I)
                        du_prev = du(I)
                        du(I) = du(I) + ddu
                        if (abs(ddu) < 1.0e-15*abs(du(I))) then
                            do_I(I) = .false. ! ddu is small enough to quit.
                        elseif (ddu > 0.0_dp) then
                            if (du(I) >= du_max(I)) then
                                du(I) = 0.5*(du_prev + du_max(I))
                                if (du_max(I) - du_prev < 1.0e-15*abs(du(I))) do_I(I) = .false.
                            end if
                        else ! ddu < 0.0_dp
                            if (du(I) <= du_min(I)) then
                                du(I) = 0.5*(du_prev + du_min(I))
                                if (du_prev - du_min(I) < 1.0e-15*abs(du(I))) do_I(I) = .false.
                            end if
                        end if
                        if (do_I(I)) domore = .true.
                    else
                        do_I(I) = .false.
                    end if
                end if
            end do
            if (.not. domore) exit

            if ((itt < max_itts) .or. present(uh_3d)) then
                do k = 1, nz
                    do I = ish - 1, ieh
                        u_new(I) = u(I, j, k) + du(I)*visc_rem(I, k)
                    end do
                    call zonal_flux_layer(u_new, h_in(:, j, k), h_W(:, j, k), h_E(:, j, k), &
                                          uh_aux(:, k), duhdu(:, k), visc_rem(:, k), &
                                          dt, G, j, ish, ieh, do_I, CS%vol_CFL, por_face_areaU(:, j, k))
                end do
            end if

            if (itt < max_itts) then
                do I = ish - 1, ieh
                    uh_err(I) = -uhbt(I)
                    duhdu_tot(I) = 0.0_dp
                end do
                do k = 1, nz
                    do I = ish - 1, ieh
                        uh_err(I) = uh_err(I) + uh_aux(I, k)
                        duhdu_tot(I) = duhdu_tot(I) + duhdu(I, k)
                    end do
                end do
                do I = ish - 1, ieh
                    uh_err_best(I) = min(uh_err_best(I), abs(uh_err(I)))
                end do
            end if
        end do ! itt-loop
        ! If there are any faces which have not converged to within the tolerance,
        ! so-be-it, or else use a final upwind correction?
        ! This never seems to happen with 20 iterations as max_itt.

        if (present(uh_3d)) then
            do k = 1, nz
            do I = ish - 1, ieh
                uh_3d(I, j, k) = uh_aux(I, k)
            end do
            end do
        end if

    end subroutine zonal_flux_adjust

!> Sets a structure that describes the zonal barotropic volume or mass fluxes as a
!! function of barotropic flow to agree closely with the sum of the layer's transports.
    subroutine set_zonal_BT_cont(u, h_in, h_W, h_E, BT_cont, uh_tot_0, duhdu_tot_0, &
                                 du_max_CFL, du_min_CFL, dt, G, GV, CS, visc_rem, &
                                 visc_rem_max, j, ish, ieh, do_I, por_face_areaU)
        type(ocean_grid_type), intent(in)    :: G    !< Ocean's grid structure.
        type(verticalGrid_type), intent(in)    :: GV   !< Ocean's vertical grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: u    !< Zonal velocity [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_in !< Layer thickness used to
                                                                   !! calculate fluxes [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_W  !< West edge thickness in the
                                                                   !! reconstruction [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in)    :: h_E  !< East edge thickness in the
                                                                   !! reconstruction [H ~> m or kg m-2].
        type(BT_cont_type), intent(inout) :: BT_cont !< A structure with elements
                       !! that describe the effective open face areas as a function of barotropic flow.
        real(dp), dimension(G%isd:G%ied), intent(in)    :: uh_tot_0    !< The summed transport
                       !! with 0 adjustment [H L2 T-1 ~> m3 s-1 or kg s-1].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: duhdu_tot_0 !< The partial derivative
                       !! of du_err with du at 0 adjustment [H L ~> m2 or kg m-1].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: du_max_CFL  !< Maximum acceptable
                       !! value of du [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: du_min_CFL  !< Minimum acceptable
                       !! value of du [L T-1 ~> m s-1].
        real(dp), intent(in)    :: dt   !< Time increment [T ~> s].
        type(continuity_CS), intent(in)    :: CS   !< This module's control structure.
        real(dp), dimension(G%isd:G%ied, GV%ke), intent(in)    :: visc_rem !< Both the fraction of the
                       !! momentum originally in a layer that remains after a time-step of viscosity, and
                       !! the fraction of a time-step's worth of a barotropic acceleration that a layer
                       !! experiences after viscosity is applied [nondim].
                       !! Visc_rem is between 0 (at the bottom) and 1 (far above the bottom).
        real(dp), dimension(G%isd:G%ied), intent(in)    :: visc_rem_max !< Maximum allowable visc_rem [nondim].
        integer, intent(in)    :: j        !< Spatial index.
        integer, intent(in)    :: ish      !< Start of index range.
        integer, intent(in)    :: ieh      !< End of index range.
        logical, dimension(G%isd:G%ied), intent(in)    :: do_I     !< A logical flag indicating
                       !! which I values to work on.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in) :: por_face_areaU !< fractional open area of U-faces [nondim]
        ! Local variables
        real(dp), dimension(G%isd:G%ied) :: &
            du0, &        ! The barotropic velocity increment that gives 0 transport [L T-1 ~> m s-1].
            duL, duR, &   ! The barotropic velocity increments that give the westerly
            ! (duL) and easterly (duR) test velocities [L T-1 ~> m s-1].
            zeros, &      ! An array of full of 0 transports [H L2 T-1 ~> m3 s-1 or kg s-1]
            du_CFL, &     ! The velocity increment that corresponds to CFL_min [L T-1 ~> m s-1].
            u_L, u_R, &   ! The westerly (u_L), easterly (u_R), and zero-barotropic
            u_0, &        ! transport (u_0) layer test velocities [L T-1 ~> m s-1].
            duhdu_L, &    ! The effective layer marginal face areas with the westerly
            duhdu_R, &    ! (_L), easterly (_R), and zero-barotropic (_0) test
            duhdu_0, &    ! velocities [H L ~> m2 or kg m-1].
            uh_L, uh_R, & ! The layer transports with the westerly (_L), easterly (_R),
            uh_0, &       ! and zero-barotropic (_0) test velocities [H L2 T-1 ~> m3 s-1 or kg s-1].
            FAmt_L, FAmt_R, & ! The summed effective marginal face areas for the 3
            FAmt_0, &     ! test velocities [H L ~> m2 or kg m-1].
            uhtot_L, &    ! The summed transport with the westerly (uhtot_L) and
            uhtot_R       ! and easterly (uhtot_R) test velocities [H L2 T-1 ~> m3 s-1 or kg s-1].
        real(dp) :: FA_0    ! The effective face area with 0 barotropic transport [L H ~> m2 or kg m-1].
        real(dp) :: FA_avg  ! The average effective face area [L H ~> m2 or kg m-1], nominally given by
        ! the realized transport divided by the barotropic velocity.
        real(dp) :: visc_rem_lim ! The larger of visc_rem and min_visc_rem [nondim]. This
        ! limiting is necessary to keep the inverse of visc_rem
        ! from leading to large CFL numbers.
        real(dp) :: min_visc_rem ! The smallest permitted value for visc_rem that is used
        ! in finding the barotropic velocity that changes the
        ! flow direction [nondim].  This is necessary to keep the inverse
        ! of visc_rem from leading to large CFL numbers.
        real(dp) :: CFL_min ! A minimal increment in the CFL to try to ensure that the
        ! flow is truly upwind [nondim]
        real(dp) :: Idt     ! The inverse of the time step [T-1 ~> s-1].
        logical :: domore
        integer :: i, k, nz

        nz = GV%ke
        Idt = 1.0/dt
        min_visc_rem = 0.1
        CFL_min = 1e-6

        ! Diagnose the zero-transport correction, du0.
        do I = ish - 1, ieh
            zeros(I) = 0.0_dp
        end do
        call zonal_flux_adjust(u, h_in, h_W, h_E, zeros, uh_tot_0, duhdu_tot_0, du0, &
                               du_max_CFL, du_min_CFL, dt, G, GV, CS, visc_rem, &
                               j, ish, ieh, do_I, por_face_areaU)

        ! Determine the westerly- and easterly- fluxes.  Choose a sufficiently
        ! negative velocity correction for the easterly-flux, and a sufficiently
        ! positive correction for the westerly-flux.
        domore = .false.
        do I = ish - 1, ieh
            if (do_I(I)) domore = .true.
            du_CFL(I) = (CFL_min*Idt)*G%dxCu(I, j)
            duR(I) = min(0.0_dp, du0(I) - du_CFL(I))
            duL(I) = max(0.0_dp, du0(I) + du_CFL(I))
            FAmt_L(I) = 0.0_dp
            FAmt_R(I) = 0.0_dp
            FAmt_0(I) = 0.0_dp
            uhtot_L(I) = 0.0_dp
            uhtot_R(I) = 0.0_dp
        end do

        if (.not. domore) then
            do k = 1, nz
                do I = ish - 1, ieh
                    BT_cont%FA_u_W0(I, j) = 0.0_dp
                    BT_cont%FA_u_WW(I, j) = 0.0_dp
                    BT_cont%FA_u_E0(I, j) = 0.0_dp
                    BT_cont%FA_u_EE(I, j) = 0.0_dp
                    BT_cont%uBT_WW(I, j) = 0.0_dp
                    BT_cont%uBT_EE(I, j) = 0.0_dp
                end do
            end do
            return
        end if

        do k = 1, nz
            do I = ish - 1, ieh
            if (do_I(I)) then
                visc_rem_lim = max(visc_rem(I, k), min_visc_rem*visc_rem_max(I))
                if (visc_rem_lim > 0.0_dp) then ! This is almost always true for ocean points.
                    if (u(I, j, k) + duR(I)*visc_rem_lim > -du_CFL(I)*visc_rem(I, k)) &
                        duR(I) = -(u(I, j, k) + du_CFL(I)*visc_rem(I, k))/visc_rem_lim
                    if (u(I, j, k) + duL(I)*visc_rem_lim < du_CFL(I)*visc_rem(I, k)) &
                        duL(I) = -(u(I, j, k) - du_CFL(I)*visc_rem(I, k))/visc_rem_lim
                end if
            end if
            end do
        end do

        do k = 1, nz
            do I = ish - 1, ieh
                if (do_I(I)) then
                    u_L(I) = u(I, j, k) + duL(I)*visc_rem(I, k)
                    u_R(I) = u(I, j, k) + duR(I)*visc_rem(I, k)
                    u_0(I) = u(I, j, k) + du0(I)*visc_rem(I, k)
                end if
            end do
            call zonal_flux_layer(u_0, h_in(:, j, k), h_W(:, j, k), h_E(:, j, k), uh_0, duhdu_0, &
                                  visc_rem(:, k), dt, G, j, ish, ieh, do_I, CS%vol_CFL, por_face_areaU(:, j, k))
            call zonal_flux_layer(u_L, h_in(:, j, k), h_W(:, j, k), h_E(:, j, k), uh_L, duhdu_L, &
                                  visc_rem(:, k), dt, G, j, ish, ieh, do_I, CS%vol_CFL, por_face_areaU(:, j, k))
            call zonal_flux_layer(u_R, h_in(:, j, k), h_W(:, j, k), h_E(:, j, k), uh_R, duhdu_R, &
                                  visc_rem(:, k), dt, G, j, ish, ieh, do_I, CS%vol_CFL, por_face_areaU(:, j, k))
            do I = ish - 1, ieh
                if (do_I(I)) then
                    FAmt_0(I) = FAmt_0(I) + duhdu_0(I)
                    FAmt_L(I) = FAmt_L(I) + duhdu_L(I)
                    FAmt_R(I) = FAmt_R(I) + duhdu_R(I)
                    uhtot_L(I) = uhtot_L(I) + uh_L(I)
                    uhtot_R(I) = uhtot_R(I) + uh_R(I)
                end if
            end do
        end do
        do I = ish - 1, ieh
            if (do_I(I)) then
                FA_0 = FAmt_0(I)
                FA_avg = FAmt_0(I)
                if ((duL(I) - du0(I)) /= 0.0_dp) &
                    FA_avg = uhtot_L(I)/(duL(I) - du0(I))
                if (FA_avg > max(FA_0, FAmt_L(I))) then
                    FA_avg = max(FA_0, FAmt_L(I))
                elseif (FA_avg < min(FA_0, FAmt_L(I))) then
                    FA_0 = FA_avg
                end if

                BT_cont%FA_u_W0(I, j) = FA_0
                BT_cont%FA_u_WW(I, j) = FAmt_L(I)
                if (abs(FA_0 - FAmt_L(I)) <= 1e-12*FA_0) then
                    BT_cont%uBT_WW(I, j) = 0.0_dp
                else
                    BT_cont%uBT_WW(I, j) = (1.5*(duL(I) - du0(I)))* &
                                           ((FAmt_L(I) - FA_avg)/(FAmt_L(I) - FA_0))
                end if

                FA_0 = FAmt_0(I)
                FA_avg = FAmt_0(I)
                if ((duR(I) - du0(I)) /= 0.0_dp) &
                    FA_avg = uhtot_R(I)/(duR(I) - du0(I))
                if (FA_avg > max(FA_0, FAmt_R(I))) then
                    FA_avg = max(FA_0, FAmt_R(I))
                elseif (FA_avg < min(FA_0, FAmt_R(I))) then
                    FA_0 = FA_avg
                end if

                BT_cont%FA_u_E0(I, j) = FA_0
                BT_cont%FA_u_EE(I, j) = FAmt_R(I)
                if (abs(FAmt_R(I) - FA_0) <= 1e-12*FA_0) then
                    BT_cont%uBT_EE(I, j) = 0.0_dp
                else
                    BT_cont%uBT_EE(I, j) = (1.5*(duR(I) - du0(I)))* &
                                           ((FAmt_R(I) - FA_avg)/(FAmt_R(I) - FA_0))
                end if
            else
                BT_cont%FA_u_W0(I, j) = 0.0_dp
                BT_cont%FA_u_WW(I, j) = 0.0_dp
                BT_cont%FA_u_E0(I, j) = 0.0_dp
                BT_cont%FA_u_EE(I, j) = 0.0_dp
                BT_cont%uBT_WW(I, j) = 0.0_dp
                BT_cont%uBT_EE(I, j) = 0.0_dp
            end if
        end do

    end subroutine set_zonal_BT_cont

!> 3D PPM reconstruction — single collapse(3) kernel instead of nz separate kernel launches.
    subroutine PPM_reconstruction_x_3d(h_in, h_W, h_E, G, GV, h_min, monotonic, simple_2nd)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(in) :: h_in
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), intent(out) :: h_W, h_E
        real(dp), intent(in) :: h_min
        logical, intent(in) :: monotonic, simple_2nd

        real(dp), parameter :: oneSixth = 1._dp / 6._dp
        real(dp) :: h_ip1, h_im1, dMx, dMn, slp_im1, slp_i, slp_ip1
        real(dp) :: h_i, RLdiff, RLdiff2, RLmean, FunFac
        real(dp) :: curv, dh, scale_val
        integer :: i, j, k, isl, iel, jsl, jel, nz

        isl = G%isc - 1 ; iel = G%iec + 1
        jsl = G%jsc ; jel = G%jec
        nz = GV%ke

        if (simple_2nd) then
            !$acc parallel loop collapse(3) default(present) private(h_im1, h_ip1)
            do k = 1, nz
                do j = jsl, jel
                    do i = isl, iel
                        h_im1 = G%mask2dT(i - 1, j) * h_in(i - 1, j, k) + (1.0_dp - G%mask2dT(i - 1, j)) * h_in(i, j, k)
                        h_ip1 = G%mask2dT(i + 1, j) * h_in(i + 1, j, k) + (1.0_dp - G%mask2dT(i + 1, j)) * h_in(i, j, k)
                        h_W(i, j, k) = 0.5_dp * (h_im1 + h_in(i, j, k))
                        h_E(i, j, k) = 0.5_dp * (h_ip1 + h_in(i, j, k))
                    end do
                end do
            end do
        else
            ! Full PPM with slopes computed inline (avoids separate slp array)
            !$acc parallel loop collapse(3) default(present) &
            !$acc   private(h_im1, h_ip1, dMx, dMn, slp_im1, slp_i, slp_ip1)
            do k = 1, nz
                do j = jsl, jel
                    do i = isl, iel
                        ! Compute slope at i-1
                        if ((G%mask2dT(i - 2, j) * G%mask2dT(i - 1, j) * G%mask2dT(i, j)) == 0.0_dp) then
                            slp_im1 = 0.0_dp
                        else
                            slp_im1 = 0.5_dp * (h_in(i, j, k) - h_in(i - 2, j, k))
                            dMx = max(h_in(i, j, k), h_in(i - 2, j, k), h_in(i - 1, j, k)) - h_in(i - 1, j, k)
                            dMn = h_in(i - 1, j, k) - min(h_in(i, j, k), h_in(i - 2, j, k), h_in(i - 1, j, k))
                            slp_im1 = sign(1._dp, slp_im1) * min(abs(slp_im1), 2._dp * min(dMx, dMn))
                        end if
                        ! Compute slope at i
                        if ((G%mask2dT(i - 1, j) * G%mask2dT(i, j) * G%mask2dT(i + 1, j)) == 0.0_dp) then
                            slp_i = 0.0_dp
                        else
                            slp_i = 0.5_dp * (h_in(i + 1, j, k) - h_in(i - 1, j, k))
                            dMx = max(h_in(i + 1, j, k), h_in(i - 1, j, k), h_in(i, j, k)) - h_in(i, j, k)
                            dMn = h_in(i, j, k) - min(h_in(i + 1, j, k), h_in(i - 1, j, k), h_in(i, j, k))
                            slp_i = sign(1._dp, slp_i) * min(abs(slp_i), 2._dp * min(dMx, dMn))
                        end if
                        ! Compute slope at i+1
                        if ((G%mask2dT(i, j) * G%mask2dT(i + 1, j) * G%mask2dT(i + 2, j)) == 0.0_dp) then
                            slp_ip1 = 0.0_dp
                        else
                            slp_ip1 = 0.5_dp * (h_in(i + 2, j, k) - h_in(i, j, k))
                            dMx = max(h_in(i + 2, j, k), h_in(i, j, k), h_in(i + 1, j, k)) - h_in(i + 1, j, k)
                            dMn = h_in(i + 1, j, k) - min(h_in(i + 2, j, k), h_in(i, j, k), h_in(i + 1, j, k))
                            slp_ip1 = sign(1._dp, slp_ip1) * min(abs(slp_ip1), 2._dp * min(dMx, dMn))
                        end if

                        h_im1 = G%mask2dT(i - 1, j) * h_in(i - 1, j, k) + (1.0_dp - G%mask2dT(i - 1, j)) * h_in(i, j, k)
                        h_ip1 = G%mask2dT(i + 1, j) * h_in(i + 1, j, k) + (1.0_dp - G%mask2dT(i + 1, j)) * h_in(i, j, k)
                        h_W(i, j, k) = 0.5_dp * (h_im1 + h_in(i, j, k)) + oneSixth * (slp_im1 - slp_i)
                        h_E(i, j, k) = 0.5_dp * (h_ip1 + h_in(i, j, k)) + oneSixth * (slp_i - slp_ip1)
                    end do
                end do
            end do
        end if

        ! Apply limiter
        if (monotonic) then
            !$acc parallel loop collapse(3) default(present) &
            !$acc   private(h_i, RLdiff, RLdiff2, RLmean, FunFac)
            do k = 1, nz
                do j = jsl, jel
                    do i = isl, iel
                        h_i = h_in(i, j, k)
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
                    end do
                end do
            end do
        else
            !$acc parallel loop collapse(3) default(present) private(curv, dh, scale_val)
            do k = 1, nz
                do j = jsl, jel
                    do i = isl, iel
                        curv = 3.0_dp * ((h_W(i, j, k) + h_E(i, j, k)) - 2.0_dp * h_in(i, j, k))
                        if (curv > 0.0_dp) then
                            dh = h_E(i, j, k) - h_W(i, j, k)
                            if (abs(dh) < curv) then
                                if (h_in(i, j, k) <= h_min) then
                                    h_W(i, j, k) = h_in(i, j, k)
                                    h_E(i, j, k) = h_in(i, j, k)
                                elseif (12.0_dp * curv * (h_in(i, j, k) - h_min) < (curv**2 + 3.0_dp * dh**2)) then
                                    scale_val = 12.0_dp * curv * (h_in(i, j, k) - h_min) / (curv**2 + 3.0_dp * dh**2)
                                    h_W(i, j, k) = h_in(i, j, k) + scale_val * (h_W(i, j, k) - h_in(i, j, k))
                                    h_E(i, j, k) = h_in(i, j, k) + scale_val * (h_E(i, j, k) - h_in(i, j, k))
                                end if
                            end if
                        end if
                    end do
                end do
            end do
        end if

    end subroutine PPM_reconstruction_x_3d

!> Calculates left/right edge values for PPM reconstruction.
    subroutine PPM_reconstruction_x(h_in, h_W, h_E, G, h_min, monotonic, simple_2nd)
        type(ocean_grid_type), intent(in)  :: G    !< Ocean's grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in)  :: h_in !< Layer thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: h_W  !< West edge thickness in the reconstruction,
                                                         !! [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(out) :: h_E  !< East edge thickness in the reconstruction,
                                                         !! [H ~> m or kg m-2].
        real(dp), intent(in)  :: h_min !< The minimum thickness
                    !! that can be obtained by a concave parabolic fit [H ~> m or kg m-2]
        logical, intent(in)  :: monotonic !< If true, use the
                    !! Colella & Woodward monotonic limiter.
                    !! Otherwise use a simple positive-definite limiter.
        logical, intent(in)  :: simple_2nd !< If true, use the
                    !! arithmetic mean thicknesses as the default edge values
                    !! for a simple 2nd order scheme.

        ! Local variables with useful mnemonic names.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed)  :: slp ! The slopes per grid point [H ~> m or kg m-2]
        real(dp), parameter :: oneSixth = 1._dp/6._dp  ! [nondim]
        real(dp) :: h_ip1, h_im1 ! Neighboring thicknesses or sensibly extrapolated values [H ~> m or kg m-2]
        real(dp) :: dMx, dMn     ! The difference between the local thickness and the maximum (dMx) or
        ! minimum (dMn) of the surrounding values [H ~> m or kg m-2]
        integer :: i, j, isl, iel, jsl, jel, n
!   logical :: local_open_BC
!   type(OBC_segment_type), pointer :: segment => NULL()

!   local_open_BC = .false.
!   if (associated(OBC)) then
!     local_open_BC = OBC%open_u_BCs_exist_globally
!   endif

        isl = G%isc - 1
        iel = G%iec + 1
        jsl = G%jsc
        jel = G%jec

        if (simple_2nd) then
            !$acc parallel loop collapse(2) default(present) private(h_im1, h_ip1)
            do j = jsl, jel
                do i = isl, iel
                    h_im1 = G%mask2dT(i - 1, j)*h_in(i - 1, j) + (1.0 - G%mask2dT(i - 1, j))*h_in(i, j)
                    h_ip1 = G%mask2dT(i + 1, j)*h_in(i + 1, j) + (1.0 - G%mask2dT(i + 1, j))*h_in(i, j)
                    h_W(i, j) = 0.5*(h_im1 + h_in(i, j))
                    h_E(i, j) = 0.5*(h_ip1 + h_in(i, j))
                end do
            end do
        else
            !$acc data create(slp)
            !$acc parallel loop collapse(2) default(present) private(dMx, dMn)
            do j = jsl, jel
                do i = isl - 1, iel + 1
                    if ((G%mask2dT(i - 1, j)*G%mask2dT(i, j)*G%mask2dT(i + 1, j)) == 0.0_dp) then
                        slp(i, j) = 0.0_dp
                    else
                        ! This uses a simple 2nd order slope.
                        slp(i, j) = 0.5*(h_in(i + 1, j) - h_in(i - 1, j))
                        ! Monotonic constraint, see Eq. B2 in Lin 1994, MWR (132)
                        dMx = max(h_in(i + 1, j), h_in(i - 1, j), h_in(i, j)) - h_in(i, j)
                        dMn = h_in(i, j) - min(h_in(i + 1, j), h_in(i - 1, j), h_in(i, j))
                        slp(i, j) = sign(1._dp, slp(i, j))*min(abs(slp(i, j)), 2._dp*min(dMx, dMn))
                        ! * (G%mask2dT(i-1,j) * G%mask2dT(i,j) * G%mask2dT(i+1,j))
                    end if
                end do
            end do

            !  if (local_open_BC) then
            !    do n=1, OBC%number_of_segments
            !      segment => OBC%segment(n)
            !      if (.not. segment%on_pe) cycle
            !      if (segment%is_E_or_W) then
            !        I=segment%HI%IsdB
            !        do j=segment%HI%jsd,segment%HI%jed
            !          slp(i+1,j) = 0.0_dp
            !          slp(i,j) = 0.0_dp
            !        enddo
            !      endif
            !    enddo
            !  endif

            !$acc parallel loop collapse(2) default(present) private(h_im1, h_ip1)
            do j = jsl, jel
                do i = isl, iel
                    ! Neighboring values should take into account any boundaries.  The 3
                    ! following sets of expressions are equivalent.
                    ! h_im1 = h_in(i-1,j,k) ; if (G%mask2dT(i-1,j) < 0.5) h_im1 = h_in(i,j)
                    ! h_ip1 = h_in(i+1,j,k) ; if (G%mask2dT(i+1,j) < 0.5) h_ip1 = h_in(i,j)
                    h_im1 = G%mask2dT(i - 1, j)*h_in(i - 1, j) + (1.0 - G%mask2dT(i - 1, j))*h_in(i, j)
                    h_ip1 = G%mask2dT(i + 1, j)*h_in(i + 1, j) + (1.0 - G%mask2dT(i + 1, j))*h_in(i, j)
                    ! Left/right values following Eq. B2 in Lin 1994, MWR (132)
                    h_W(i, j) = 0.5*(h_im1 + h_in(i, j)) + oneSixth*(slp(i - 1, j) - slp(i, j))
                    h_E(i, j) = 0.5*(h_ip1 + h_in(i, j)) + oneSixth*(slp(i, j) - slp(i + 1, j))
                end do
            end do
            !$acc end data
        end if

! not functional in this example, but we must be cognisant of.
!   if (local_open_BC) then
!     do n=1, OBC%number_of_segments
!       segment => OBC%segment(n)
!       if (.not. segment%on_pe) cycle
!       if (segment%direction == OBC_DIRECTION_E) then
!         I=segment%HI%IsdB
!         do j=segment%HI%jsd,segment%HI%jed
!           h_W(i+1,j) = h_in(i,j)
!           h_E(i+1,j) = h_in(i,j)
!           h_W(i,j) = h_in(i,j)
!           h_E(i,j) = h_in(i,j)
!         enddo
!       elseif (segment%direction == OBC_DIRECTION_W) then
!         I=segment%HI%IsdB
!         do j=segment%HI%jsd,segment%HI%jed
!           h_W(i,j) = h_in(i+1,j)
!           h_E(i,j) = h_in(i+1,j)
!           h_W(i+1,j) = h_in(i+1,j)
!           h_E(i+1,j) = h_in(i+1,j)
!         enddo
!       endif
!     enddo
!   endif

        if (monotonic) then
            call PPM_limit_CW84(h_in, h_W, h_E, G, isl, iel, jsl, jel)
        else
            call PPM_limit_pos(h_in, h_W, h_E, h_min, G, isl, iel, jsl, jel)
        end if

        return
    end subroutine PPM_reconstruction_x

!> This subroutine limits the left/right edge values of the PPM reconstruction
!! to give a reconstruction that is positive-definite.  Here this is
!! reinterpreted as giving a constant thickness if the mean thickness is less
!! than h_min, with a minimum of h_min otherwise.
    subroutine PPM_limit_pos(h_in, h_L, h_R, h_min, G, iis, iie, jis, jie)
        type(ocean_grid_type), intent(in)  :: G    !< Ocean's grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in)  :: h_in !< Layer thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(inout) :: h_L !< Left thickness in the reconstruction [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(inout) :: h_R !< Right thickness in the reconstruction [H ~> m or kg m-2].
        real(dp), intent(in)  :: h_min !< The minimum thickness
                    !! that can be obtained by a concave parabolic fit [H ~> m or kg m-2]
        integer, intent(in)  :: iis      !< Start of i index range.
        integer, intent(in)  :: iie      !< End of i index range.
        integer, intent(in)  :: jis      !< Start of j index range.
        integer, intent(in)  :: jie      !< End of j index range.

! Local variables
        real    :: curv  ! The grid-normalized curvature of the three thicknesses  [H ~> m or kg m-2]
        real    :: dh    ! The difference between the edge thicknesses             [H ~> m or kg m-2]
        real    :: scale ! A scaling factor to reduce the curvature of the fit               [nondim]
        integer :: i, j

        !$acc parallel loop collapse(2) default(present) private(curv, dh, scale)
        do j = jis, jie
            do i = iis, iie
                ! This limiter prevents undershooting minima within the domain with
                ! values less than h_min.
                curv = 3.0*((h_L(i, j) + h_R(i, j)) - 2.0*h_in(i, j))
                if (curv > 0.0_dp) then ! Only minima are limited.
                    dh = h_R(i, j) - h_L(i, j)
                    if (abs(dh) < curv) then ! The parabola's minimum is within the cell.
                        if (h_in(i, j) <= h_min) then
                            h_L(i, j) = h_in(i, j)
                            h_R(i, j) = h_in(i, j)
                        elseif (12.0*curv*(h_in(i, j) - h_min) < (curv**2 + 3.0*dh**2)) then
                            ! The minimum value is h_in - (curv^2 + 3*dh^2)/(12*curv), and must
                            ! be limited in this case.  0 < scale < 1.
                            scale = 12.0*curv*(h_in(i, j) - h_min)/(curv**2 + 3.0*dh**2)
                            h_L(i, j) = h_in(i, j) + scale*(h_L(i, j) - h_in(i, j))
                            h_R(i, j) = h_in(i, j) + scale*(h_R(i, j) - h_in(i, j))
                        end if
                    end if
                end if
            end do
        end do

    end subroutine PPM_limit_pos

    !> This subroutine limits the left/right edge values of the PPM reconstruction
   !! according to the monotonic prescription of Colella and Woodward, 1984.
    subroutine PPM_limit_CW84(h_in, h_L, h_R, G, iis, iie, jis, jie)
        type(ocean_grid_type), intent(in)  :: G     !< Ocean's grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(in)  :: h_in  !< Layer thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(inout) :: h_L !< Left thickness in the reconstruction,
                                                            !! [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), intent(inout) :: h_R !< Right thickness in the reconstruction,
                                                            !! [H ~> m or kg m-2].
        integer, intent(in)  :: iis   !< Start of i index range.
        integer, intent(in)  :: iie   !< End of i index range.
        integer, intent(in)  :: jis   !< Start of j index range.
        integer, intent(in)  :: jie   !< End of j index range.

        ! Local variables
        real    :: h_i      ! A copy of the cell-average layer thickness                [H ~> m or kg m-2]
        real    :: RLdiff   ! The difference between the input edge values              [H ~> m or kg m-2]
        real    :: RLdiff2  ! The squared difference between the input edge values   [H2 ~> m2 or kg2 m-4]
        real    :: RLmean   ! The average of the input edge thicknesses                 [H ~> m or kg m-2]
        real    :: FunFac   ! A curious product of the thickness slope and curvature [H2 ~> m2 or kg2 m-4]
        integer :: i, j

        !$acc parallel loop collapse(2) default(present) private(h_i, RLdiff, RLdiff2, RLmean, FunFac)
        do j = jis, jie
            do i = iis, iie
                ! This limiter monotonizes the parabola following
                ! Colella and Woodward, 1984, Eq. 1.10
                h_i = h_in(i, j)
                if ((h_R(i, j) - h_i)*(h_i - h_L(i, j)) <= 0.) then
                    h_L(i, j) = h_i
                    h_R(i, j) = h_i
                else
                    RLdiff = h_R(i, j) - h_L(i, j)            ! Difference of edge values
                    RLmean = 0.5*(h_R(i, j) + h_L(i, j))  ! Mean of edge values
                    FunFac = 6.*RLdiff*(h_i - RLmean) ! Some funny factor
                    RLdiff2 = RLdiff*RLdiff               ! Square of difference
                    if (FunFac > RLdiff2) h_L(i, j) = 3.*h_i - 2.*h_R(i, j)
                    if (FunFac < -RLdiff2) h_R(i, j) = 3.*h_i - 2.*h_L(i, j)
                end if
            end do
        end do

        return
    end subroutine PPM_limit_CW84

    !> Return the maximum ratio of a/b or maxrat.
    function ratio_max(a, b, maxrat) result(ratio)
        real(dp), intent(in) :: a       !< Numerator, in arbitrary units [A]
        real(dp), intent(in) :: b       !< Denominator, in arbitrary units [B]
        real(dp), intent(in) :: maxrat  !< Maximum value of ratio [A B-1]
        real(dp) :: ratio               !< Return value [A B-1]

        if (abs(a) > abs(maxrat*b)) then
            ratio = maxrat
        else
            ratio = a/b
        end if
    end function ratio_max

end module mom6_continuity
