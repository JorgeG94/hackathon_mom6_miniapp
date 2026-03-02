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
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -mp=multicore,gpu -acc=multicore,gpu  -gpu=mem:separate -cudalib=nvtx")

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

option(MOM6_ENABLE_CUDA_KERNELS "Build CUDA Fortran kernel variants (requires NVHPC)" OFF)

if(MOM6_ENABLE_CUDA_KERNELS)
  if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
    message(FATAL_ERROR "MOM6_ENABLE_CUDA_KERNELS requires the NVHPC (nvfortran) compiler")
  endif()
  message(STATUS "CUDA Fortran kernels: ENABLED")
  # nvfortran handles CUDA linking internally via -cuda flag, no need for find_package(CUDAToolkit)
endif()

# =============================================================================
# Module library (shared solvers)
# =============================================================================

add_library(mom6_modules STATIC)
add_subdirectory(src/common)
add_subdirectory(src/openacc)

# Set module output directory
set_target_properties(mom6_modules PROPERTIES Fortran_MODULE_DIRECTORY
                                              ${CMAKE_BINARY_DIR}/mod)

target_compile_definitions(mom6_modules PRIVATE USE_NVTX)
# Include directory for modules
target_include_directories(mom6_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)

# =============================================================================
# Driver executables
# =============================================================================

option(MOM6_ENABLE_MODULE_DRIVERS "Build individual module drivers (continuity, coriolis, etc.)" OFF)

add_executable(rk2_driver app/rk2_driver.F90)
target_link_libraries(rk2_driver PRIVATE mom6_modules)

set(OPENACC_DRIVERS rk2_driver)

if(MOM6_ENABLE_MODULE_DRIVERS)
  set(MODULE_DRIVERS
    continuity_driver coriolis_driver barotropic_driver
    vert_visc_driver hor_visc_driver)

  foreach(drv ${MODULE_DRIVERS})
    add_executable(${drv} app/${drv}.F90)
    target_link_libraries(${drv} PRIVATE mom6_modules)
  endforeach()

  list(APPEND OPENACC_DRIVERS ${MODULE_DRIVERS})
endif()

# =============================================================================
# CUDA Fortran kernel library + drivers (opt-in)
# =============================================================================

if(MOM6_ENABLE_CUDA_KERNELS)
  add_subdirectory(src/cuda)

  # Helper function to create a CUDA driver target
  function(add_cuda_driver name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE mom6_modules mom6_cuda_kernels)
    target_compile_options(${name} PRIVATE "-cuda")
    target_link_options(${name} PRIVATE "-cuda" "-cudalib=nvtx")
  endfunction()

  add_cuda_driver(rk2_cuda_driver app/rk2_cuda_driver.F90)

  if(MOM6_ENABLE_MODULE_DRIVERS)
    set(MODULE_CUDA_DRIVERS
      continuity_cuda_driver coriolis_cuda_driver barotropic_cuda_driver
      vert_visc_cuda_driver hor_visc_cuda_driver)

    foreach(drv ${MODULE_CUDA_DRIVERS})
      add_cuda_driver(${drv} app/${drv}.F90)
    endforeach()
  endif()
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS ${OPENACC_DRIVERS} RUNTIME DESTINATION bin)
install(TARGETS mom6_modules ARCHIVE DESTINATION lib)
install(
  DIRECTORY ${CMAKE_BINARY_DIR}/mod/
  DESTINATION include/mom6_miniapps
  FILES_MATCHING
  PATTERN "*.mod")

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
message(STATUS "CUDA kernels:     ${MOM6_ENABLE_CUDA_KERNELS}")
message(STATUS "Module drivers:   ${MOM6_ENABLE_MODULE_DRIVERS}")
message(STATUS "Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
