!> MOM6 Continuity Solver Module
!!
!! PPM (Piecewise Parabolic Method) continuity solver extracted from MOM6.
!! Solves the layer thickness equation: dh/dt = -div(uh, vh)
!!
!! Original code from: src/core/MOM_continuity_PPM.F90
!!
module mom6_continuity
    use iso_fortran_env, only: dp => real64
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
    end type continuity_CS

contains

    !> Initialize the continuity solver
    subroutine continuity_init(CS, G, GV, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
        type(continuity_CS), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        real(dp), dimension(:, :), intent(inout), allocatable:: uhbt, du_cor
        real(dp), dimension(:, :, :), intent(inout), allocatable:: u_cor, por_face_areaU, visc_rem_u

        allocate (uhbt(G%isd:G%ied, G%jsd:G%jed))
        allocate (u_cor(G%isd:G%ied, G%jsd:G%jed, GV%ke))
        allocate (du_cor(G%isd:G%ied, G%jsd:G%jed))
        allocate (por_face_areaU(G%isd:G%ied, G%jsd:G%jed, GV%ke), source=1._dp)
        allocate (visc_rem_u(G%isd:G%ied, G%jsd:G%jed, GV%ke), source=1._dp)

        CS%initialized = .true.
        CS%upwind_1st = .false.
        CS%simple_2nd = .false.
        CS%tol_eta = 1.d-12
        CS%tol_vel = 3.d8
        CS%CFL_limit_adjust = 0.5d0
        CS%aggress_adjust = .false.
        CS%vol_CFL = .false.
        CS%better_iter = .true.
        CS%use_visc_rem_max = .true.
        CS%marginal_faces = .true.

    end subroutine continuity_init

    !> Finalize the continuity solver
    subroutine continuity_end(CS, uhbt, u_cor, du_cor, por_face_areaU, visc_rem_u)
        type(continuity_CS), intent(inout) :: CS
        real(dp), dimension(:, :), intent(inout), allocatable:: uhbt, du_cor
        real(dp), dimension(:, :, :), intent(inout), allocatable:: u_cor, por_face_areaU, visc_rem_u

        if (.not. CS%initialized) return

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
        type(continuity_CS), intent(in) :: CS
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

        ! advect zonally
        call zonal_edge_thickness(hin, h_W, h_E, G, GV, CS)
        call zonal_mass_flux(u, hin, h_W, h_E, uh, dt, G, GV, CS, por_face_areaU, &
                             uhbt, visc_rem_u, u_cor, BT_cont, du_cor)
        call continuity_zonal_convergence(h, uh, dt, G, GV, hin)

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
            !$OMP parallel do default(shared)
            do k = 1, GV%ke; do j = G%jsc, G%jec; do i = G%isc, G%iec
                    h(i, j, k) = max(hin(i, j, k) - dt*G%IareaT(i, j)*(uh(I, j, k) - uh(I - 1, j, k)), h_min)
                end do; end do; end do
        else
            !$OMP parallel do default(shared)
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
            !$OMP parallel do default(shared)
            do k = 1, nz
                do j = jsh, jeh
                    do i = ish - 1, ieh + 1
                        h_W(i, j, k) = h_in(i, j, k)
                        h_E(i, j, k) = h_in(i, j, k)
                    end do
                end do
            end do
        else
            !$OMP parallel do default(shared)
            do k = 1, nz
                call PPM_reconstruction_x(h_in(:, :, k), h_W(:, :, k), h_E(:, :, k), G, &
                                          2.0*GV%Angstrom_H, CS%monotonic, CS%simple_2nd)
            end do
        end if

    end subroutine zonal_edge_thickness

!> Calculates the mass or volume fluxes through the zonal faces, and other related quantities.
    subroutine zonal_mass_flux(u, h_in, h_W, h_E, uh, dt, G, GV, CS, por_face_areaU, &
                               uhbt, visc_rem_u, u_cor, BT_cont, du_cor)
        type(ocean_grid_type), intent(in)    :: G    !< Ocean's grid structure.
        type(verticalGrid_type), intent(in)    :: GV   !< Ocean's vertical grid structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: u    !< Zonal velocity [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: h_in !< Layer thickness used to calculate fluxes [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: h_W !< Western edge thicknesses [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: h_E !< Eastern edge thicknesses [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(out)   :: uh   !< Volume flux through zonal faces = u*h*dy
                                                 !! [H L2 T-1 ~> m3 s-1 or kg s-1].
        real(dp), intent(in)    :: dt   !< Time increment [T ~> s].
        type(continuity_CS), intent(in)    :: CS   !< This module's control structure.
!   type(ocean_OBC_type),    pointer       :: OBC  !< Open boundaries control structure.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            intent(in)    :: por_face_areaU !< fractional open area of U-faces [nondim]
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), &
            optional, intent(in)    :: uhbt !< The summed volume flux through zonal faces
                                                 !! [H L2 T-1 ~> m3 s-1 or kg s-1].
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            optional, intent(in)    :: visc_rem_u
        !< The fraction of zonal momentum originally in a layer that remains after a
                     !! time-step of viscosity, and the fraction of a time-step's worth of a barotropic
                     !! acceleration that a layer experiences after viscosity is applied [nondim].
                     !! Visc_rem_u is between 0 (at the bottom) and 1 (far above the bottom).
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed, GV%ke), &
            optional, intent(out)   :: u_cor
        !< The zonal velocities (u with a barotropic correction)
                     !! that give uhbt as the depth-integrated transport [L T-1 ~> m s-1]
        type(BT_cont_type), optional, pointer  :: BT_cont !< A structure with elements that describe the
                     !! effective open face areas as a function of barotropic flow.
        real(dp), dimension(G%isd:G%ied, G%jsd:G%jed), &
            optional, intent(out)   :: du_cor !< The zonal velocity increments from u that give uhbt
                                                 !! as the depth-integrated transports [L T-1 ~> m s-1].

        ! Local variables
        real(dp), dimension(G%isd:G%ied, GV%ke) :: duhdu ! Partial derivative of uh with u [H L ~> m2 or kg m-1].
        real(dp), dimension(G%isd:G%ied) :: &
            du, &         ! Corrective barotropic change in the velocity to give uhbt [L T-1 ~> m s-1].
            du_min_CFL, & ! Lower limit on du correction to avoid CFL violations [L T-1 ~> m s-1]
            du_max_CFL, & ! Upper limit on du correction to avoid CFL violations [L T-1 ~> m s-1]
            duhdu_tot_0, & ! Summed partial derivative of uh with u [H L ~> m2 or kg m-1].
            uh_tot_0, &   ! Summed transport with no barotropic correction [H L2 T-1 ~> m3 s-1 or kg s-1].
            visc_rem_max  ! The column maximum of visc_rem [nondim].
        logical, dimension(G%isd:G%ied) :: do_I
        real(dp), dimension(G%isd:G%ied, GV%ke) :: &
            visc_rem      ! A 2-D copy of visc_rem_u or an array of 1's [nondim].
        real(dp), dimension(G%isd:G%ied) :: FAuI  ! A list of sums of zonal face areas [H L ~> m2 or kg m-1].
        real(dp) :: FA_u    ! A sum of zonal face areas [H L ~> m2 or kg m-1].
        real(dp) :: I_vrm   ! 1.0 / visc_rem_max [nondim]
        real(dp) :: CFL_dt  ! The maximum CFL ratio of the adjusted velocities divided by
        ! the time step [T-1 ~> s-1].
        real(dp) :: I_dt    ! 1.0 / dt [T-1 ~> s-1].
        real(dp) :: du_lim  ! The velocity change that give a relative CFL of 1 [L T-1 ~> m s-1].
        real(dp) :: dx_E, dx_W ! Effective x-grid spacings to the east and west [L ~> m].
        integer :: i, j, k, ish, ieh, jsh, jeh, n, nz
!   integer :: l_seg ! The OBC segment number
        logical :: use_visc_rem, set_BT_cont
!   logical :: simple_OBC_pt(G%isd:G%ied)  ! Indicates points in a row with specified transport OBCs

        use_visc_rem = present(visc_rem_u)

        set_BT_cont = .false.
        if (present(BT_cont)) set_BT_cont = (associated(BT_cont))

        if (present(du_cor)) du_cor(:, :) = 0.0_dp

        ish = G%isc
        ieh = G%iec
        jsh = G%jsc
        jeh = G%jec
        nz = GV%ke

        CFL_dt = CS%CFL_limit_adjust/dt
        I_dt = 1.0/dt
        if (CS%aggress_adjust) CFL_dt = I_dt

        if (.not. use_visc_rem) visc_rem(:, :) = 1.0
        do j = jsh, jeh
            do I = ish - 1, ieh
                do_I(I) = .true.
            end do
            ! Set uh and duhdu.
            do k = 1, nz
                if (use_visc_rem) then
                    do I = ish - 1, ieh
                        visc_rem(I, k) = visc_rem_u(I, j, k)
                    end do
                end if
                call zonal_flux_layer(u(:, j, k), h_in(:, j, k), h_W(:, j, k), h_E(:, j, k), &
                                      uh(:, j, k), duhdu(:, k), visc_rem(:, k), &
                                      dt, G, j, ish, ieh, do_I, CS%vol_CFL, por_face_areaU(:, j, k))
                ! if (local_specified_BC) then
                !   do I=ish-1,ieh ; if (OBC%segnum_u(I,j) /= 0) then
                !     l_seg = abs(OBC%segnum_u(I,j))
                !     if (OBC%segment(l_seg)%specified) uh(I,j,k) = OBC%segment(l_seg)%normal_trans(I,j,k)
                !   endif ; enddo
                ! endif
            end do

            if (present(uhbt) .or. set_BT_cont) then
                if (use_visc_rem .and. CS%use_visc_rem_max) then
                    visc_rem_max(:) = 0.0_dp
                    do k = 1, nz
                        do I = ish - 1, ieh
                            visc_rem_max(I) = max(visc_rem_max(I), visc_rem(I, k))
                        end do
                    end do
                else
                    visc_rem_max(:) = 1.0
                end if
                !   Set limits on du that will keep the CFL number between -1 and 1.
                ! This should be adequate to keep the root bracketed in all cases.
                do I = ish - 1, ieh
                    I_vrm = 0.0_dp
                    if (visc_rem_max(I) > 0.0_dp) I_vrm = 1.0/visc_rem_max(I)
                    if (CS%vol_CFL) then
                        dx_W = ratio_max(G%areaT(i, j), G%dy_Cu(I, j), 1000.0*G%dxT(i, j))
                        dx_E = ratio_max(G%areaT(i + 1, j), G%dy_Cu(I, j), 1000.0*G%dxT(i + 1, j))
                    else
                        dx_W = G%dxT(i, j)
                        dx_E = G%dxT(i + 1, j)
                    end if
                    du_max_CFL(I) = 2.0*(CFL_dt*dx_W)*I_vrm
                    du_min_CFL(I) = -2.0*(CFL_dt*dx_E)*I_vrm
                    uh_tot_0(I) = 0.0_dp
                    duhdu_tot_0(I) = 0.0_dp
                end do
                do k = 1, nz
                    do I = ish - 1, ieh
                        duhdu_tot_0(I) = duhdu_tot_0(I) + duhdu(I, k)
                        uh_tot_0(I) = uh_tot_0(I) + uh(I, j, k)
                    end do
                end do
                if (use_visc_rem) then
                    if (CS%aggress_adjust) then
                        do k = 1, nz
                            do I = ish - 1, ieh
                            if (CS%vol_CFL) then
                                dx_W = ratio_max(G%areaT(i, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i, j))
                                dx_E = ratio_max(G%areaT(i + 1, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i + 1, j))
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if

                            du_lim = 0.499_dp*((dx_W*I_dt - u(I, j, k)) + MIN(0.0_dp, u(I - 1, j, k)))
                            if (du_max_CFL(I)*visc_rem(I, k) > du_lim) &
                                du_max_CFL(I) = du_lim/visc_rem(I, k)

                            du_lim = 0.499_dp*((-dx_E*I_dt - u(I, j, k)) + MAX(0.0_dp, u(I + 1, j, k)))
                            if (du_min_CFL(I)*visc_rem(I, k) < du_lim) &
                                du_min_CFL(I) = du_lim/visc_rem(I, k)
                            end do
                        end do
                    else
                        do k = 1, nz
                            do I = ish - 1, ieh
                            if (CS%vol_CFL) then
                                dx_W = ratio_max(G%areaT(i, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i, j))
                                dx_E = ratio_max(G%areaT(i + 1, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i + 1, j))
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if

                            if (du_max_CFL(I)*visc_rem(I, k) > dx_W*CFL_dt - u(I, j, k)*G%mask2dCu(I, j)) &
                                du_max_CFL(I) = (dx_W*CFL_dt - u(I, j, k))/visc_rem(I, k)
                            if (du_min_CFL(I)*visc_rem(I, k) < -dx_E*CFL_dt - u(I, j, k)*G%mask2dCu(I, j)) &
                                du_min_CFL(I) = -(dx_E*CFL_dt + u(I, j, k))/visc_rem(I, k)
                            end do
                        end do
                    end if
                else
                    if (CS%aggress_adjust) then
                        do k = 1, nz
                            do I = ish - 1, ieh
                            if (CS%vol_CFL) then
                                dx_W = ratio_max(G%areaT(i, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i, j))
                                dx_E = ratio_max(G%areaT(i + 1, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i + 1, j))
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if

                            du_max_CFL(I) = MIN(du_max_CFL(I), 0.499_dp* &
                                                ((dx_W*I_dt - u(I, j, k)) + MIN(0.0_dp, u(I - 1, j, k))))
                            du_min_CFL(I) = MAX(du_min_CFL(I), 0.499_dp* &
                                                ((-dx_E*I_dt - u(I, j, k)) + MAX(0.0_dp, u(I + 1, j, k))))
                            end do
                        end do
                    else
                        do k = 1, nz
                            do I = ish - 1, ieh
                            if (CS%vol_CFL) then
                                dx_W = ratio_max(G%areaT(i, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i, j))
                                dx_E = ratio_max(G%areaT(i + 1, j), G%dy_Cu(I, j), 1000.0_dp*G%dxT(i + 1, j))
                            else
                                dx_W = G%dxT(i, j)
                                dx_E = G%dxT(i + 1, j)
                            end if

                            du_max_CFL(I) = MIN(du_max_CFL(I), dx_W*CFL_dt - u(I, j, k))
                            du_min_CFL(I) = MAX(du_min_CFL(I), -(dx_E*CFL_dt + u(I, j, k)))
                            end do
                        end do
                    end if
                end if
                do I = ish - 1, ieh
                    du_max_CFL(I) = max(du_max_CFL(I), 0.0_dp)
                    du_min_CFL(I) = min(du_min_CFL(I), 0.0_dp)
                end do

                ! any_simple_OBC = .false.
                if (present(uhbt) .or. set_BT_cont) then
                    !   if (local_specified_BC .or. local_Flather_OBC) then ; do I=ish-1,ieh
                    !     l_seg = abs(OBC%segnum_u(I,j))

                    !     ! Avoid reconciling barotropic/baroclinic transports if transport is specified
                    !     simple_OBC_pt(I) = .false.
                    !     if (l_seg /= OBC_NONE) simple_OBC_pt(I) = OBC%segment(l_seg)%specified
                    !     do_I(I) = .not.simple_OBC_pt(I)
                    !     any_simple_OBC = any_simple_OBC .or. simple_OBC_pt(I)
                    !   enddo ; else ; do I=ish-1,ieh
                    do_I(I) = .true.
                    !   enddo ; endif
                end if

                if (present(uhbt)) then
                    ! Find du and uh.
                    call zonal_flux_adjust(u, h_in, h_W, h_E, uhbt(:, j), uh_tot_0, duhdu_tot_0, du, &
                                           du_max_CFL, du_min_CFL, dt, G, GV, CS, visc_rem, &
                                           j, ish, ieh, do_I, por_face_areaU, uh)

                    if (present(u_cor)) then
                        do k = 1, nz
                            do I = ish - 1, ieh
                                u_cor(I, j, k) = u(I, j, k) + du(I)*visc_rem(I, k)
                            end do
                            !  if (any_simple_OBC) then ; do I=ish-1,ieh ; if (simple_OBC_pt(I)) then
                            !    u_cor(I,j,k) = OBC%segment(abs(OBC%segnum_u(I,j)))%normal_vel(I,j,k)
                            !  endif ; enddo ; endif
                        end do
                    end if ! u-corrected

                    if (present(du_cor)) then
                        do I = ish - 1, ieh
                            du_cor(I, j) = du(I)
                        end do
                    end if

                end if

                if (set_BT_cont) then
                    call set_zonal_BT_cont(u, h_in, h_W, h_E, BT_cont, uh_tot_0, duhdu_tot_0, &
                                           du_max_CFL, du_min_CFL, dt, G, GV, CS, visc_rem, &
                                           visc_rem_max, j, ish, ieh, do_I, por_face_areaU)
                    !   if (any_simple_OBC) then
                    !     do I=ish-1,ieh
                    !       if (simple_OBC_pt(I)) FAuI(I) = GV%H_subroundoff*G%dy_Cu(I,j)
                    !     enddo
                    !     ! NOTE: simple_OBC_pt(I) should prevent access to segment OBC_NONE
                    !     do k=1,nz ; do I=ish-1,ieh ; if (simple_OBC_pt(I)) then
                    !       l_seg = abs(OBC%segnum_u(I,j))
                    !       if ((abs(OBC%segment(l_seg)%normal_vel(I,j,k)) > 0.0_dp) .and. (OBC%segment(l_seg)%specified)) &
                    !         FAuI(I) = FAuI(I) + OBC%segment(l_seg)%normal_trans(I,j,k) / OBC%segment(l_seg)%normal_vel(I,j,k)
                    !     endif ; enddo ; enddo
                    !     do I=ish-1,ieh ; if (simple_OBC_pt(I)) then
                    !       BT_cont%FA_u_W0(I,j) = FAuI(I) ; BT_cont%FA_u_E0(I,j) = FAuI(I)
                    !       BT_cont%FA_u_WW(I,j) = FAuI(I) ; BT_cont%FA_u_EE(I,j) = FAuI(I)
                    !       BT_cont%uBT_WW(I,j) = 0.0_dp ; BT_cont%uBT_EE(I,j) = 0.0_dp
                    !     endif ; enddo
                    !   endif
                end if ! set_BT_cont

            end if ! present(uhbt) or set_BT_cont

        end do ! j-loop

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

!> Evaluates the zonal mass or volume fluxes in a layer.
    subroutine zonal_flux_layer(u, h, h_W, h_E, uh, duhdu, visc_rem, dt, G, j, &
                                ish, ieh, do_I, vol_CFL, por_face_areaU)
        type(ocean_grid_type), intent(in)    :: G        !< Ocean's grid structure.
        real(dp), dimension(G%isd:G%ied), intent(in)    :: u        !< Zonal velocity [L T-1 ~> m s-1].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: visc_rem !< Both the fraction of the
                        !! momentum originally in a layer that remains after a time-step
                        !! of viscosity, and the fraction of a time-step's worth of a barotropic
                        !! acceleration that a layer experiences after viscosity is applied [nondim].
                        !! Visc_rem is between 0 (at the bottom) and 1 (far above the bottom).
        real(dp), dimension(G%isd:G%ied), intent(in)    :: h        !< Layer thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: h_W      !< West edge thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied), intent(in)    :: h_E      !< East edge thickness [H ~> m or kg m-2].
        real(dp), dimension(G%isd:G%ied), intent(inout) :: uh       !< Zonal mass or volume
                                                          !! transport [H L2 T-1 ~> m3 s-1 or kg s-1].
        real(dp), dimension(G%isd:G%ied), intent(inout) :: duhdu    !< Partial derivative of uh
                                                          !! with u [H L ~> m2 or kg m-1].
        real(dp), intent(in)    :: dt       !< Time increment [T ~> s]
        integer, intent(in)    :: j        !< Spatial index.
        integer, intent(in)    :: ish      !< Start of index range.
        integer, intent(in)    :: ieh      !< End of index range.
        logical, dimension(G%isd:G%ied), intent(in)    :: do_I     !< Which i values to work on.
        logical, intent(in)    :: vol_CFL  !< If true, rescale the
        real(dp), dimension(G%isd:G%ied), intent(in)    :: por_face_areaU !< fractional open area of U-faces [nondim]
          !! ratio of face areas to the cell areas when estimating the CFL number.
!   type(ocean_OBC_type), optional, pointer     :: OBC !< Open boundaries control structure.
        ! Local variables
        real(dp) :: CFL  ! The CFL number based on the local velocity and grid spacing [nondim]
        real(dp) :: curv_3 ! A measure of the thickness curvature over a grid length [H ~> m or kg m-2]
        real(dp) :: h_marg ! The marginal thickness of a flux [H ~> m or kg m-2].
        integer :: i
!   integer :: l_seg
!   logical :: local_open_BC

!   local_open_BC = .false.
!   if (present(OBC)) then ; if (associated(OBC)) then
!     local_open_BC = OBC%open_u_BCs_exist_globally
!   endif ; endif

        do I = ish - 1, ieh
            if (do_I(I)) then
                ! Set new values of uh and duhdu.
                if (u(I) > 0.0_dp) then
                    if (vol_CFL) then
                        CFL = (u(I)*dt)*(G%dy_Cu(I, j)*G%IareaT(i, j))
                    else
                        CFL = u(I)*dt*G%IdxT(i, j)
                    end if
                    curv_3 = (h_W(i) + h_E(i)) - 2.0*h(i)
                    uh(I) = (G%dy_Cu(I, j)*por_face_areaU(I))*u(I)* &
                            (h_E(i) + CFL*(0.5*(h_W(i) - h_E(i)) + curv_3*(CFL - 1.5)))
                    h_marg = h_E(i) + CFL*((h_W(i) - h_E(i)) + 3.0*curv_3*(CFL - 1.0))
                elseif (u(I) < 0.0_dp) then
                    if (vol_CFL) then
                        CFL = (-u(I)*dt)*(G%dy_Cu(I, j)*G%IareaT(i + 1, j))
                    else
                        CFL = -u(I)*dt*G%IdxT(i + 1, j)
                    end if
                    curv_3 = (h_W(i + 1) + h_E(i + 1)) - 2.0*h(i + 1)
                    uh(I) = (G%dy_Cu(I, j)*por_face_areaU(I))*u(I)* &
                            (h_W(i + 1) + CFL*(0.5*(h_E(i + 1) - h_W(i + 1)) + curv_3*(CFL - 1.5)))
                    h_marg = h_W(i + 1) + CFL*((h_E(i + 1) - h_W(i + 1)) + 3.0*curv_3*(CFL - 1.0))
                else
                    uh(I) = 0.0_dp
                    h_marg = 0.5*(h_W(i + 1) + h_E(i))
                end if
                duhdu(I) = (G%dy_Cu(I, j)*por_face_areaU(I))*h_marg*visc_rem(I)
            end if
        end do

!   if (local_open_BC) then
!     do I=ish-1,ieh ; if (do_I(I)) then ; if (OBC%segnum_u(I,j) /= 0) then
!       if (OBC%segment(abs(OBC%segnum_u(I,j)))%open) then
!         if (OBC%segnum_u(I,j) > 0) then !  OBC_DIRECTION_E
!           uh(I) = (G%dy_Cu(I,j) * por_face_areaU(I)) * u(I) * h(i)
!           duhdu(I) = (G%dy_Cu(I,j) * por_face_areaU(I)) * h(i) * visc_rem(I)
!         else !  OBC_DIRECTION_W
!           uh(I) = (G%dy_Cu(I,j) * por_face_areaU(I)) * u(I) * h(i+1)
!           duhdu(I) = (G%dy_Cu(I,j)* por_face_areaU(I)) * h(i+1) * visc_rem(I)
!         endif
!       endif
!     endif ; endif ; enddo
!   endif
    end subroutine zonal_flux_layer

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

        !$OMP parallel do default(shared) private(CFL,curv_3,h_marg,h_avg)
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
            !$OMP parallel do default(shared)
            do k = 1, nz
                do j = jsh, jeh
                    do I = ish - 1, ieh
                        h_u(I, j, k) = h_u(I, j, k)*(visc_rem_u(I, j, k)*por_face_areaU(I, j, k))
                    end do
                end do
            end do
        else
            !$OMP parallel do default(shared)
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

!> Returns the barotropic velocity adjustment that gives the
!! desired barotropic (layer-summed) transport.
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
            do j = jsl, jel
                do i = isl, iel
                    h_im1 = G%mask2dT(i - 1, j)*h_in(i - 1, j) + (1.0 - G%mask2dT(i - 1, j))*h_in(i, j)
                    h_ip1 = G%mask2dT(i + 1, j)*h_in(i + 1, j) + (1.0 - G%mask2dT(i + 1, j))*h_in(i, j)
                    h_W(i, j) = 0.5*(h_im1 + h_in(i, j))
                    h_E(i, j) = 0.5*(h_ip1 + h_in(i, j))
                end do
            end do
        else
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
