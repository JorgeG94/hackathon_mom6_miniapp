!> MOM6 Portable Profiler Module
!!
!! Provides a unified interface for profiling that wraps:
!! - NVTX ranges (for NVIDIA Nsight profiling)
!! - Wall-clock timers (for performance measurement)
!!
!! Portability:
!! - Compile with -DUSE_NVTX to enable NVTX (requires nvtx module from NVIDIA HPC SDK)
!! - Without USE_NVTX, NVTX calls become no-ops but timers still work
!! - Compile with -DDISABLE_PROFILER to disable everything (zero overhead)
!!
!! Usage:
!!   use mom6_profiler
!!
!!   call profiler_init()
!!
!!   call profiler_start("MyRegion")
!!   ... code to profile ...
!!   call profiler_stop("MyRegion")
!!
!!   call profiler_report()
!!   call profiler_end()
!!
!! Or use the region type directly:
!!   type(profile_region) :: rgn
!!   call rgn%start("MyRegion")
!!   ... code ...
!!   call rgn%stop()
!!   print *, 'Elapsed: ', rgn%elapsed
!!
module mom6_profiler
   use iso_fortran_env, only: dp => real64, int64
#ifdef USE_NVTX
   use nvtx
#endif
   implicit none
   private

   !> Maximum number of named regions to track
   integer, parameter :: MAX_REGIONS = 256
   integer, parameter :: MAX_NAME_LEN = 64

   !> Profile region type for direct use
   type, public :: profile_region
      character(len=MAX_NAME_LEN) :: name = ''
      real(dp) :: start_time = 0.0_dp
      real(dp) :: elapsed = 0.0_dp
      logical :: active = .false.
   contains
      procedure :: start => region_start
      procedure :: stop => region_stop
      procedure :: reset => region_reset
   end type profile_region

   !> Internal region tracking for named API
   type :: tracked_region
      character(len=MAX_NAME_LEN) :: name = ''
      real(dp) :: start_time = 0.0_dp
      real(dp) :: total_time = 0.0_dp
      integer :: call_count = 0
      logical :: active = .false.
      logical :: nvtx_only = .false.  ! If true, only shows in NVTX, not in text report
   end type tracked_region

   !> Global profiler state
   type :: profiler_state
      logical :: initialized = .false.
      logical :: enabled = .true.
      integer :: num_regions = 0
      type(tracked_region) :: regions(MAX_REGIONS)
   end type profiler_state

   type(profiler_state), save :: state

   public :: profiler_init, profiler_end
   public :: profiler_start, profiler_stop
   public :: profiler_enable, profiler_disable
   public :: profiler_report, profiler_reset
   public :: profiler_get_time

contains

   !---------------------------------------------------------------------------
   ! Wall clock timer (portable)
   !---------------------------------------------------------------------------

   function get_wall_time() result(t)
      real(dp) :: t
      integer(int64) :: count, count_rate
      call system_clock(count, count_rate)
      t = real(count, dp)/real(count_rate, dp)
   end function get_wall_time

   !---------------------------------------------------------------------------
   ! NVTX wrappers (no-ops when USE_NVTX not defined)
   !---------------------------------------------------------------------------

   subroutine nvtx_range_push(name)
      character(len=*), intent(in) :: name
#ifdef USE_NVTX
      call nvtxStartRange(name)
#endif
   end subroutine nvtx_range_push

   subroutine nvtx_range_pop()
#ifdef USE_NVTX
      call nvtxEndRange()
#endif
   end subroutine nvtx_range_pop

   !---------------------------------------------------------------------------
   ! Profile region type methods
   !---------------------------------------------------------------------------

   subroutine region_start(this, name)
      class(profile_region), intent(inout) :: this
      character(len=*), intent(in) :: name

#ifdef DISABLE_PROFILER
      return
#endif

      this%name = trim(name)
      this%active = .true.
      call nvtx_range_push(name)
      this%start_time = get_wall_time()

   end subroutine region_start

   subroutine region_stop(this)
      class(profile_region), intent(inout) :: this

#ifdef DISABLE_PROFILER
      return
#endif

      if (.not. this%active) return

      this%elapsed = get_wall_time() - this%start_time
      call nvtx_range_pop()
      this%active = .false.

   end subroutine region_stop

   subroutine region_reset(this)
      class(profile_region), intent(inout) :: this

      this%name = ''
      this%start_time = 0.0_dp
      this%elapsed = 0.0_dp
      this%active = .false.

   end subroutine region_reset

   !---------------------------------------------------------------------------
   ! Global profiler API
   !---------------------------------------------------------------------------

   !> Initialize the profiler
   subroutine profiler_init(enabled)
      logical, intent(in), optional :: enabled

      state%initialized = .true.
      state%enabled = .true.
      if (present(enabled)) state%enabled = enabled
      state%num_regions = 0

   end subroutine profiler_init

   !> Finalize the profiler
   subroutine profiler_end()
      state%initialized = .false.
      state%num_regions = 0
   end subroutine profiler_end

   !> Enable profiling
   subroutine profiler_enable()
      state%enabled = .true.
   end subroutine profiler_enable

   !> Disable profiling (timers and NVTX become no-ops)
   subroutine profiler_disable()
      state%enabled = .false.
   end subroutine profiler_disable

   !> Find or create a region by name
   function find_or_create_region(name) result(idx)
      character(len=*), intent(in) :: name
      integer :: idx
      integer :: i

      ! Search for existing region
      do i = 1, state%num_regions
         if (trim(state%regions(i)%name) == trim(name)) then
            idx = i
            return
         end if
      end do

      ! Create new region
      if (state%num_regions < MAX_REGIONS) then
         state%num_regions = state%num_regions + 1
         idx = state%num_regions
         state%regions(idx)%name = trim(name)
         state%regions(idx)%total_time = 0.0_dp
         state%regions(idx)%call_count = 0
         state%regions(idx)%active = .false.
      else
         print '(A)', 'WARNING: profiler_start - max regions exceeded'
         idx = -1
      end if

   end function find_or_create_region

   !> Start a named profiling region
  !! If nvtx_only is true, the region only appears in NVTX timeline, not in text report
   subroutine profiler_start(name, nvtx_only)
      character(len=*), intent(in) :: name
      logical, intent(in), optional :: nvtx_only
      integer :: idx

#ifdef DISABLE_PROFILER
      return
#endif

      if (.not. state%enabled) return

      idx = find_or_create_region(name)
      if (idx < 0) return

      if (state%regions(idx)%active) then
         print '(A,A)', 'WARNING: profiler_start called on active region: ', trim(name)
         return
      end if

      ! Set nvtx_only flag if specified (only on first call)
      if (present(nvtx_only) .and. state%regions(idx)%call_count == 0) then
         state%regions(idx)%nvtx_only = nvtx_only
      end if

      state%regions(idx)%active = .true.
      call nvtx_range_push(name)
      state%regions(idx)%start_time = get_wall_time()

   end subroutine profiler_start

   !> Stop a named profiling region
   subroutine profiler_stop(name)
      character(len=*), intent(in) :: name
      integer :: idx
      real(dp) :: elapsed

#ifdef DISABLE_PROFILER
      return
#endif

      if (.not. state%enabled) return

      ! Find the region
      do idx = 1, state%num_regions
         if (trim(state%regions(idx)%name) == trim(name)) exit
      end do

      if (idx > state%num_regions) then
         print '(A,A)', 'WARNING: profiler_stop called on unknown region: ', trim(name)
         return
      end if

      if (.not. state%regions(idx)%active) then
         print '(A,A)', 'WARNING: profiler_stop called on inactive region: ', trim(name)
         return
      end if

      elapsed = get_wall_time() - state%regions(idx)%start_time
      call nvtx_range_pop()

      state%regions(idx)%total_time = state%regions(idx)%total_time + elapsed
      state%regions(idx)%call_count = state%regions(idx)%call_count + 1
      state%regions(idx)%active = .false.

   end subroutine profiler_stop

   !> Get accumulated time for a named region
   function profiler_get_time(name) result(t)
      character(len=*), intent(in) :: name
      real(dp) :: t
      integer :: idx

      t = 0.0_dp

      do idx = 1, state%num_regions
         if (trim(state%regions(idx)%name) == trim(name)) then
            t = state%regions(idx)%total_time
            return
         end if
      end do

   end function profiler_get_time

   !> Reset all timing data
   subroutine profiler_reset()
      integer :: i

      do i = 1, state%num_regions
         state%regions(i)%total_time = 0.0_dp
         state%regions(i)%call_count = 0
         state%regions(i)%active = .false.
      end do

   end subroutine profiler_reset

   !> Print profiling report
  !! If root_region is specified, percentages are calculated relative to that region's time
  !! and the root region is shown separately as the total.
  !! Regions are sorted by time descending.
   subroutine profiler_report(title, root_region)
      character(len=*), intent(in), optional :: title
      character(len=*), intent(in), optional :: root_region
      integer :: i, j, root_idx, n_print, tmp_idx
      integer :: sorted_idx(MAX_REGIONS)
      real(dp) :: total_time, pct

      if (state%num_regions == 0) then
         print '(A)', 'Profiler: No regions recorded'
         return
      end if

      ! Find root region if specified
      root_idx = -1
      if (present(root_region)) then
         do i = 1, state%num_regions
            if (trim(state%regions(i)%name) == trim(root_region)) then
               root_idx = i
               exit
            end if
         end do
      end if

      ! Use root region time as total, or sum all regions if no root specified
      if (root_idx > 0) then
         total_time = state%regions(root_idx)%total_time
      else
         total_time = 0.0_dp
         do i = 1, state%num_regions
            total_time = total_time + state%regions(i)%total_time
         end do
      end if

      ! Build list of indices to print (excluding root and nvtx_only regions)
      n_print = 0
      do i = 1, state%num_regions
         if (i /= root_idx .and. .not. state%regions(i)%nvtx_only) then
            n_print = n_print + 1
            sorted_idx(n_print) = i
         end if
      end do

      ! Sort by time descending (simple insertion sort)
      do i = 2, n_print
         tmp_idx = sorted_idx(i)
         j = i - 1
         do while (j >= 1 .and. state%regions(sorted_idx(j))%total_time < state%regions(tmp_idx)%total_time)
            sorted_idx(j + 1) = sorted_idx(j)
            j = j - 1
         end do
         sorted_idx(j + 1) = tmp_idx
      end do

      print '(A)', ''
      print '(A)', '============================================================'
      if (present(title)) then
         print '(A,A)', 'Profiler Report: ', trim(title)
      else
         print '(A)', 'Profiler Report'
      end if
      print '(A)', '============================================================'
      print '(A)', '  Region                          Time (s)    Calls    %    '
      print '(A)', '------------------------------------------------------------'

      do i = 1, n_print
         j = sorted_idx(i)
         if (total_time > 0.0_dp) then
            pct = 100.0_dp*state%regions(j)%total_time/total_time
         else
            pct = 0.0_dp
         end if
         print '(A,A32,F12.6,I8,F8.1)', '  ', &
            state%regions(j)%name, &
            state%regions(j)%total_time, &
            state%regions(j)%call_count, &
            pct
      end do

      print '(A)', '------------------------------------------------------------'
      print '(A,F12.6)', '  Total:                        ', total_time
      print '(A)', '============================================================'

#ifdef USE_NVTX
      print '(A)', '  (NVTX enabled - use Nsight Systems for GPU timeline)'
#else
      print '(A)', '  (NVTX disabled - compile with -DUSE_NVTX to enable)'
#endif

   end subroutine profiler_report

end module mom6_profiler
