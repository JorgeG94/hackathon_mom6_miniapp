cmake_minimum_required(VERSION 3.18)

project(
  mom6_miniapps
  VERSION 1.0.0
  LANGUAGES Fortran
  DESCRIPTION "MOM6 Mini-Apps")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

# Compiler-specific flags
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
  message(STATUS "Detected NVIDIA HPC compiler (nvfortran)")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -acc=multicore,gpu -Minfo=accel -gpu=mem:separate")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  message(STATUS "Detected GNU Fortran compiler (gfortran)")
  set(CMAKE_Fortran_FLAGS
      "${CMAKE_Fortran_FLAGS} -O3 -fallow-argument-mismatch")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  message(STATUS "Detected Intel Fortran compiler")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -heap-arrays")

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
  message(STATUS "Detected Flang compiler")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3")

else()
  message(WARNING "Unknown compiler: ${CMAKE_Fortran_COMPILER_ID}")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3")
endif()

message(STATUS "Fortran flags: ${CMAKE_Fortran_FLAGS}")

# =============================================================================
# CUDA Fortran kernels (opt-in, nvfortran only)
# =============================================================================

option(ENABLE_CUDA_KERNELS "Build CUDA Fortran kernel variants (requires NVHPC)" OFF)

if(ENABLE_CUDA_KERNELS)
  if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
    message(FATAL_ERROR "ENABLE_CUDA_KERNELS requires the NVHPC (nvfortran) compiler")
  endif()
  message(STATUS "CUDA Fortran kernels: ENABLED")
  # nvfortran handles CUDA linking internally via -cuda flag, no need for find_package(CUDAToolkit)
endif()

# =============================================================================
# Module library (shared solvers)
# =============================================================================

add_library(
  mom6_modules STATIC
  src/mom6_types.F90
  src/mom6_profiler.F90
  src/mom6_continuity.F90
  src/mom6_continuity_adjust.F90
  src/mom6_coriolis.F90
  src/mom6_barotropic.F90
  src/mom6_vert_visc.F90
  src/mom6_hor_visc.F90
  src/mom6_diag.F90)

# Work around nvfortran >=O2 codegen bug in complex OpenACC GPU kernels
# (CUDA_EXCEPTION_14 Warp Illegal Address). Only the two routines in the
# submodule file are affected; everything else stays at -O3.
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
  set_source_files_properties(src/mom6_continuity_adjust.F90
    PROPERTIES COMPILE_OPTIONS "-O1")
endif()

# Set module output directory
set_target_properties(mom6_modules PROPERTIES Fortran_MODULE_DIRECTORY
                                              ${CMAKE_BINARY_DIR}/mod)

# Include directory for modules
target_include_directories(mom6_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)

# =============================================================================
# Driver executables
# =============================================================================

# Continuity driver
add_executable(continuity_driver app/continuity_driver.F90)
target_link_libraries(continuity_driver PRIVATE mom6_modules)
target_include_directories(continuity_driver PRIVATE ${CMAKE_BINARY_DIR}/mod)

# Coriolis driver
add_executable(coriolis_driver app/coriolis_driver.F90)
target_link_libraries(coriolis_driver PRIVATE mom6_modules)
target_include_directories(coriolis_driver PRIVATE ${CMAKE_BINARY_DIR}/mod)

# Barotropic driver
add_executable(barotropic_driver app/barotropic_driver.F90)
target_link_libraries(barotropic_driver PRIVATE mom6_modules)
target_include_directories(barotropic_driver PRIVATE ${CMAKE_BINARY_DIR}/mod)

# Vertical viscosity driver
add_executable(vert_visc_driver app/vert_visc_driver.F90)
target_link_libraries(vert_visc_driver PRIVATE mom6_modules)
target_include_directories(vert_visc_driver PRIVATE ${CMAKE_BINARY_DIR}/mod)

# Horizontal viscosity driver
add_executable(hor_visc_driver app/hor_visc_driver.F90)
target_link_libraries(hor_visc_driver PRIVATE mom6_modules)
target_include_directories(hor_visc_driver PRIVATE ${CMAKE_BINARY_DIR}/mod)

# RK2 unified driver
add_executable(rk2_driver app/rk2_driver.F90)
target_link_libraries(rk2_driver PRIVATE mom6_modules)
target_include_directories(rk2_driver PRIVATE ${CMAKE_BINARY_DIR}/mod)

# =============================================================================
# CUDA Fortran kernel library + compare drivers (opt-in)
# =============================================================================

if(ENABLE_CUDA_KERNELS)
  # CUDA kernels are compiled with the same base flags + -cuda appended.
  # This matches what works in the Makefile: $(FFLAGS) -cuda ... $(LDFLAGS) -cuda
  # Do NOT use separate isolated CUDA flags — mixing OpenACC .o with CUDA .o
  # requires the linker to see both -acc and -cuda together.

  add_library(
    mom6_cuda_kernels STATIC
    src/cuda_kernels/mom6_coriolis_cuda.F90
    src/cuda_kernels/mom6_vert_visc_cuda.F90
    src/cuda_kernels/mom6_barotropic_cuda.F90
    src/cuda_kernels/mom6_hor_visc_cuda.F90
    src/cuda_kernels/mom6_continuity_cuda.F90)

  set_target_properties(mom6_cuda_kernels PROPERTIES Fortran_MODULE_DIRECTORY
                                                     ${CMAKE_BINARY_DIR}/mod)
  target_include_directories(mom6_cuda_kernels PUBLIC ${CMAKE_BINARY_DIR}/mod)
  # Append -cuda to the base flags (inherits -O3 -acc -Minfo=accel from global)
  # -cuda only at compile time; -acc already links the CUDA runtime
  target_compile_options(mom6_cuda_kernels PRIVATE "-cuda")

  # Helper function to create a compare driver target
  function(add_compare_driver name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mom6_modules mom6_cuda_kernels)
    target_include_directories(${name} PRIVATE ${CMAKE_BINARY_DIR}/mod)
    # Compile and link with -cuda appended to the base flags
    target_compile_options(${name} PRIVATE "-cuda")
    target_link_options(${name} PRIVATE "-cuda" "-cudalib=nvtx")
  endfunction()

  add_compare_driver(coriolis_compare_driver app/coriolis_compare_driver.F90)
  add_compare_driver(vert_visc_compare_driver app/vert_visc_compare_driver.F90)
  add_compare_driver(barotropic_compare_driver app/barotropic_compare_driver.F90)
  add_compare_driver(hor_visc_compare_driver app/hor_visc_compare_driver.F90)
  add_compare_driver(continuity_compare_driver app/continuity_compare_driver.F90)
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS continuity_driver coriolis_driver barotropic_driver
                vert_visc_driver hor_visc_driver rk2_driver
        RUNTIME DESTINATION bin)
install(TARGETS mom6_modules ARCHIVE DESTINATION lib)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/mod/
  DESTINATION include/mom6_miniapps
  FILES_MATCHING
  PATTERN "*.mod")

# =============================================================================
# Testing / CTest
# =============================================================================

enable_testing()

add_test(NAME continuity_driver_small COMMAND continuity_driver 60 60 30 2)
add_test(NAME coriolis_driver_small COMMAND coriolis_driver 60 60 30 2)
add_test(NAME barotropic_driver_small COMMAND barotropic_driver 60 60 15 2)
add_test(NAME vert_visc_driver_small COMMAND vert_visc_driver 60 60 30 2)
add_test(NAME hor_visc_driver_small COMMAND hor_visc_driver 60 60 30 2)
add_test(NAME rk2_driver_small COMMAND rk2_driver 60 60 30 2 15)

if(ENABLE_CUDA_KERNELS)
  add_test(NAME coriolis_compare_small COMMAND coriolis_compare_driver 60 60 30 2)
  add_test(NAME vert_visc_compare_small COMMAND vert_visc_compare_driver 60 60 30 2)
  add_test(NAME barotropic_compare_small COMMAND barotropic_compare_driver 60 60 15 2)
  add_test(NAME hor_visc_compare_small COMMAND hor_visc_compare_driver 60 60 30 2)
  add_test(NAME continuity_compare_small COMMAND continuity_compare_driver 60 60 30 2)
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "")
message(STATUS "=== MOM6 Mini-Apps Configuration ===")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(
  STATUS
    "Fortran compiler: ${CMAKE_Fortran_COMPILER_ID} (${CMAKE_Fortran_COMPILER})"
)
message(STATUS "CUDA kernels:     ${ENABLE_CUDA_KERNELS}")
message(STATUS "Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
