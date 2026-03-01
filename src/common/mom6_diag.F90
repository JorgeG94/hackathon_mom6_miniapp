!> MOM6 Simplified Diagnostic Module
!!
!! Provides a simplified diagnostic system for the MOM6 miniapp that mimics
!! MOM6's diagnostic patterns but focuses on benchmarking-relevant outputs.
!!
!! Key Features:
!! - Register diagnostics at initialization, returns integer ID
!! - ID > 0 means diagnostic is active; ID <= 0 means skip
!! - post_data generic interface handles 2D/3D fields
!! - post_product_sum_u/v compute vertical sums of 3D products (momentum budget)
!!
!!
!! Output Modes:
!! - DIAG_NONE:  Disabled (no computation)
!! - DIAG_STATS: Print statistics only (min/max/mean/rms)
!! - DIAG_FILE:  Write to binary file
!! - DIAG_BOTH:  Stats + file output
!!
module mom6_diag
    use iso_fortran_env, only: dp => real64
    use mom6_types, only: ocean_grid_type, verticalGrid_type
    implicit none
    private

    !> Diagnostic output modes
    integer, parameter, public :: DIAG_NONE = 0      ! Disabled
    integer, parameter, public :: DIAG_STATS = 1     ! Print statistics only
    integer, parameter, public :: DIAG_FILE = 2      ! Write to binary file
    integer, parameter, public :: DIAG_BOTH = 3      ! Stats + file

    !> Maximum number of registered diagnostics
    integer, parameter :: MAX_DIAGS = 100

    !> Single diagnostic descriptor
    type :: diag_field_type
        character(len=32) :: name = ''
        character(len=64) :: long_name = ''
        character(len=16) :: units = ''
        integer :: id = -1
        integer :: output_mode = DIAG_NONE
        integer :: ndims = 0           ! 2 or 3
        character(len=4) :: position   ! 'h', 'u', 'v', 'q'
    end type diag_field_type

    !> Diagnostic control structure
    type, public :: diag_ctrl
        logical :: initialized = .false.
        integer :: num_diags = 0
        type(diag_field_type) :: fields(MAX_DIAGS)

        ! Timing
        real(dp) :: total_time = 0.0_dp
        integer :: total_calls = 0

        ! Output settings
        character(len=256) :: output_dir = './'
        integer :: output_freq = 1       ! Output every N calls
        integer :: call_count = 0

        ! Pre-allocated work arrays for reductions
        real(dp), allocatable :: work_2d(:, :)
        real(dp), allocatable :: work_sum(:)   ! For vertical sums

        ! Grid bounds (cached for convenience)
        integer :: isd, ied, jsd, jed
        integer :: isc, iec, jsc, jec
    end type diag_ctrl

    public :: diag_init, diag_end
    public :: register_diag_field
    public :: post_data_2d, post_data_3d
    public :: post_product_sum_u, post_product_sum_v
    public :: diag_report_timing

contains

    !> Initialize diagnostic system, allocate work arrays
    subroutine diag_init(CS, G, GV, output_dir, output_freq)
        type(diag_ctrl), intent(inout) :: CS
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        character(len=*), intent(in), optional :: output_dir
        integer, intent(in), optional :: output_freq

        if (CS%initialized) return

        ! Store output settings
        if (present(output_dir)) CS%output_dir = trim(output_dir)
        if (present(output_freq)) CS%output_freq = output_freq

        ! Cache grid bounds
        CS%isd = G%isd; CS%ied = G%ied
        CS%jsd = G%jsd; CS%jed = G%jed
        CS%isc = G%isc; CS%iec = G%iec
        CS%jsc = G%jsc; CS%jec = G%jec

        ! Allocate work arrays
        allocate (CS%work_2d(G%isd:G%ied, G%jsd:G%jed))
        allocate (CS%work_sum(GV%ke))

        ! Initialize to zero
        CS%work_2d = 0.0_dp
        CS%work_sum = 0.0_dp

        ! Initialize timing
        CS%total_time = 0.0_dp
        CS%total_calls = 0
        CS%call_count = 0
        CS%num_diags = 0

        CS%initialized = .true.

    end subroutine diag_init

    !> Finalize diagnostic system
    subroutine diag_end(CS)
        type(diag_ctrl), intent(inout) :: CS

        if (.not. CS%initialized) return

        if (allocated(CS%work_2d)) deallocate (CS%work_2d)
        if (allocated(CS%work_sum)) deallocate (CS%work_sum)

        CS%initialized = .false.

    end subroutine diag_end

    !> Register a diagnostic field
  !!
  !! Returns positive ID if active, -1 if disabled.
    function register_diag_field(CS, name, long_name, units, ndims, position, output_mode) result(id)
        type(diag_ctrl), intent(inout) :: CS
        character(len=*), intent(in) :: name
        character(len=*), intent(in) :: long_name
        character(len=*), intent(in) :: units
        integer, intent(in) :: ndims
        character(len=*), intent(in) :: position
        integer, intent(in) :: output_mode
        integer :: id

        if (output_mode == DIAG_NONE) then
            id = -1
            return
        end if

        if (CS%num_diags >= MAX_DIAGS) then
            print '(A,I4)', 'WARNING: Maximum diagnostics reached: ', MAX_DIAGS
            id = -1
            return
        end if

        CS%num_diags = CS%num_diags + 1
        id = CS%num_diags

        CS%fields(id)%name = trim(name)
        CS%fields(id)%long_name = trim(long_name)
        CS%fields(id)%units = trim(units)
        CS%fields(id)%id = id
        CS%fields(id)%ndims = ndims
        CS%fields(id)%position = trim(position)
        CS%fields(id)%output_mode = output_mode

    end function register_diag_field

    !> Post a 2D diagnostic field
    subroutine post_data_2d(id, field, G, CS)
        integer, intent(in) :: id
        real(dp), intent(in) :: field(:, :)
        type(ocean_grid_type), intent(in) :: G
        type(diag_ctrl), intent(inout) :: CS

        real(dp) :: fmin, fmax, fmean, frms
        integer :: i, j, cnt

        if (id <= 0) return
        if (.not. CS%initialized) return

        CS%total_calls = CS%total_calls + 1
        CS%call_count = CS%call_count + 1

        ! Compute statistics
        fmin = huge(1.0_dp)
        fmax = -huge(1.0_dp)
        fmean = 0.0_dp
        frms = 0.0_dp
        cnt = 0

        ! Use explicit loops with reduction
        do j = G%jsc, G%jec
            do i = G%isc, G%iec
                fmin = min(fmin, field(i, j))
                fmax = max(fmax, field(i, j))
                fmean = fmean + field(i, j)
                frms = frms + field(i, j)**2
                cnt = cnt + 1
            end do
        end do

        if (cnt > 0) then
            fmean = fmean/real(cnt, dp)
            frms = sqrt(frms/real(cnt, dp))
        end if

        ! Output based on mode
        if (iand(CS%fields(id)%output_mode, DIAG_STATS) /= 0) then
            print '(A,A,A,4ES14.6)', '  Diag: ', trim(CS%fields(id)%name), &
                ' min/max/mean/rms: ', fmin, fmax, fmean, frms
        end if

        if (iand(CS%fields(id)%output_mode, DIAG_FILE) /= 0) then
            if (mod(CS%call_count, CS%output_freq) == 0) then
                call write_binary_2d(CS, id, field, G)
            end if
        end if

    end subroutine post_data_2d

    !> Post a 3D diagnostic field
    subroutine post_data_3d(id, field, G, GV, CS)
        integer, intent(in) :: id
        real(dp), intent(in) :: field(:, :, :)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV
        type(diag_ctrl), intent(inout) :: CS

        real(dp) :: fmin, fmax, fmean, frms
        integer :: i, j, k, cnt, nz

        if (id <= 0) return
        if (.not. CS%initialized) return

        CS%total_calls = CS%total_calls + 1
        CS%call_count = CS%call_count + 1

        nz = GV%ke

        ! Compute statistics over all layers
        fmin = huge(1.0_dp)
        fmax = -huge(1.0_dp)
        fmean = 0.0_dp
        frms = 0.0_dp
        cnt = 0

        ! Use explicit loops with reduction
        do k = 1, nz
            do j = G%jsc, G%jec
                do i = G%isc, G%iec
                    fmin = min(fmin, field(i, j, k))
                    fmax = max(fmax, field(i, j, k))
                    fmean = fmean + field(i, j, k)
                    frms = frms + field(i, j, k)**2
                    cnt = cnt + 1
                end do
            end do
        end do

        if (cnt > 0) then
            fmean = fmean/real(cnt, dp)
            frms = sqrt(frms/real(cnt, dp))
        end if

        ! Output based on mode
        if (iand(CS%fields(id)%output_mode, DIAG_STATS) /= 0) then
            print '(A,A,A,4ES14.6)', '  Diag: ', trim(CS%fields(id)%name), &
                ' min/max/mean/rms: ', fmin, fmax, fmean, frms
        end if

        if (iand(CS%fields(id)%output_mode, DIAG_FILE) /= 0) then
            if (mod(CS%call_count, CS%output_freq) == 0) then
                call write_binary_3d(CS, id, field, G, GV)
            end if
        end if

    end subroutine post_data_3d

    !> Compute and post vertically-integrated product of two 3D u-point fields
  !!
  !! This mimics MOM6's momentum budget diagnostics (e.g., post_product_sum_u
  !! for PFu_visc_rem = visc_rem_u * PFu vertically integrated).
    subroutine post_product_sum_u(id, u_a, u_b, G, nz, CS)
        integer, intent(in) :: id
        integer, intent(in) :: nz
        real(dp), intent(in) :: u_a(:, :, :)
        real(dp), intent(in) :: u_b(:, :, :)
        type(ocean_grid_type), intent(in) :: G
        type(diag_ctrl), intent(inout) :: CS

        integer :: i, j, k

        if (id <= 0) return
        if (.not. CS%initialized) return

        ! Compute vertical sum of product
        ! Initialize work array to zero
        do j = G%jsc, G%jec
            do i = G%isc, G%iec - 1
                CS%work_2d(i, j) = 0.0_dp
            end do
        end do

        ! Accumulate product over layers
        do k = 1, nz
            do j = G%jsc, G%jec
                do i = G%isc, G%iec - 1
                    CS%work_2d(i, j) = CS%work_2d(i, j) + u_a(i, j, k)*u_b(i, j, k)
                end do
            end do
        end do

        call post_data_2d(id, CS%work_2d, G, CS)

    end subroutine post_product_sum_u

    !> Compute and post vertically-integrated product of two 3D v-point fields
    subroutine post_product_sum_v(id, v_a, v_b, G, nz, CS)
        integer, intent(in) :: id
        integer, intent(in) :: nz
        real(dp), intent(in) :: v_a(:, :, :)
        real(dp), intent(in) :: v_b(:, :, :)
        type(ocean_grid_type), intent(in) :: G
        type(diag_ctrl), intent(inout) :: CS

        integer :: i, j, k

        if (id <= 0) return
        if (.not. CS%initialized) return

        ! Compute vertical sum of product
        ! Initialize work array to zero
        do j = G%jsc, G%jec - 1
            do i = G%isc, G%iec
                CS%work_2d(i, j) = 0.0_dp
            end do
        end do

        ! Accumulate product over layers
        do k = 1, nz
            do j = G%jsc, G%jec - 1
                do i = G%isc, G%iec
                    CS%work_2d(i, j) = CS%work_2d(i, j) + v_a(i, j, k)*v_b(i, j, k)
                end do
            end do
        end do

        call post_data_2d(id, CS%work_2d, G, CS)

    end subroutine post_product_sum_v

    !> Report timing statistics for diagnostics
    subroutine diag_report_timing(CS)
        type(diag_ctrl), intent(in) :: CS

        print '(A)', ''
        print '(A)', 'Diagnostic Summary:'
        print '(A)', '--------------------------------------------------'
        print '(A,I8)', '  Registered diagnostics:  ', CS%num_diags
        print '(A,I8)', '  Total post_data calls:   ', CS%total_calls

    end subroutine diag_report_timing

    !> Write 2D field to binary file
    subroutine write_binary_2d(CS, id, field, G)
        type(diag_ctrl), intent(in) :: CS
        integer, intent(in) :: id
        real(dp), intent(in) :: field(:, :)
        type(ocean_grid_type), intent(in) :: G

        character(len=512) :: filename
        integer :: iunit, ierr
        integer :: ni, nj

        ni = G%iec - G%isc + 1
        nj = G%jec - G%jsc + 1

        ! Construct filename
        write (filename, '(A,A,A,I6.6,A)') trim(CS%output_dir), '/', &
            trim(CS%fields(id)%name), CS%call_count, '.bin'

        ! Open file for binary write
        open (newunit=iunit, file=trim(filename), form='unformatted', &
              access='stream', status='replace', iostat=ierr)

        if (ierr /= 0) then
            print '(A,A)', 'WARNING: Could not open file: ', trim(filename)
            return
        end if

        ! Write dimensions and data (computational domain only)
        write (iunit) ni, nj
        write (iunit) field(G%isc:G%iec, G%jsc:G%jec)

        close (iunit)

    end subroutine write_binary_2d

    !> Write 3D field to binary file
    subroutine write_binary_3d(CS, id, field, G, GV)
        type(diag_ctrl), intent(in) :: CS
        integer, intent(in) :: id
        real(dp), intent(in) :: field(:, :, :)
        type(ocean_grid_type), intent(in) :: G
        type(verticalGrid_type), intent(in) :: GV

        character(len=512) :: filename
        integer :: iunit, ierr
        integer :: ni, nj, nk

        ni = G%iec - G%isc + 1
        nj = G%jec - G%jsc + 1
        nk = GV%ke

        ! Construct filename
        write (filename, '(A,A,A,I6.6,A)') trim(CS%output_dir), '/', &
            trim(CS%fields(id)%name), CS%call_count, '.bin'

        ! Open file for binary write
        open (newunit=iunit, file=trim(filename), form='unformatted', &
              access='stream', status='replace', iostat=ierr)

        if (ierr /= 0) then
            print '(A,A)', 'WARNING: Could not open file: ', trim(filename)
            return
        end if

        ! Write dimensions and data (computational domain only)
        write (iunit) ni, nj, nk
        write (iunit) field(G%isc:G%iec, G%jsc:G%jec, 1:nk)

        close (iunit)

    end subroutine write_binary_3d

end module mom6_diag
