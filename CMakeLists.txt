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

elseif(CMAKE_Fortran_COMPILER_ID MATCHES "LLVMFlang")
  message(STATUS "Detected Flang compiler")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -fopenmp --offload-arch=gfx90a -fopenmp-offload-mandatory -fPIC")

else()
  message(WARNING "Unknown compiler: ${CMAKE_Fortran_COMPILER_ID}")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3")
endif()

message(STATUS "Fortran flags: ${CMAKE_Fortran_FLAGS}")

# =============================================================================
# Build options
# =============================================================================

option(MOM6_ENABLE_ALL "Build all backends for super duper benchmarking" OFF)
option(MOM6_ENABLE_CUDA_KERNELS "Build CUDA Fortran kernel variants (requires NVHPC)" OFF)
option(MOM6_ENABLE_CUDA_C_KERNELS "Build CUDA C kernel variants (nvcc + any Fortran compiler)" OFF)
option(MOM6_ENABLE_OPENACC_KERNELS "Build OpenACC kernel variants (requires NVHPC)" OFF)
option(MOM6_ENABLE_SERIAL_KERNELS "Build pure CPU serial kernel variants (no GPU)" OFF)
option(MOM6_ENABLE_OMP_KERNELS "Build OpenMP target offloading kernel variants (default backend)" ON)
option(MOM6_ENABLE_MPI "Build MPI-parallel drivers (requires MPI Fortran)" OFF)
option(MOM6_ENABLE_SINGLE_GPU_DRIVERS "Build single-GPU drivers (rk2_driver, rk2_cuda_driver, rk2_omp_driver)" ON)
option(MOM6_ENABLE_MODULE_DRIVERS "Build individual module drivers (continuity, coriolis, etc.)" OFF)

if(MOM6_ENABLE_ALL) 
  set(MOM6_ENABLE_CUDA_KERNELS ON)
  set(MOM6_ENABLE_CUDA_C_KERNELS ON)
  set(MOM6_ENABLE_OPENACC_KERNELS ON)
  set(MOM6_ENABLE_SERIAL_KERNELS ON)
  set(MOM6_ENABLE_OMP_KERNELS ON)
  set(MOM6_ENABLE_MPI ON)
  set(MOM6_ENABLE_SINGLE_GPU_DRIVERS ON)
  set(MOM6_ENABLE_MODULE_DRIVERS ON)
endif()

if(MOM6_ENABLE_CUDA_KERNELS)
  if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
    message(FATAL_ERROR "MOM6_ENABLE_CUDA_KERNELS requires the NVHPC (nvfortran) compiler")
  endif()
  message(STATUS "CUDA Fortran kernels: ENABLED")
endif()

if(MOM6_ENABLE_CUDA_C_KERNELS)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Default to sm_70 (V100); override with -DCMAKE_CUDA_ARCHITECTURES=80 etc.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
  endif()
  message(STATUS "CUDA C kernels: ENABLED (nvcc + iso_c_binding, arch=${CMAKE_CUDA_ARCHITECTURES})")
endif()

if(MOM6_ENABLE_OPENACC_KERNELS)
  if(NOT CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
    message(FATAL_ERROR "MOM6_ENABLE_OPENACC_KERNELS requires the NVHPC (nvfortran) compiler")
  endif()
  message(STATUS "OpenACC kernels: ENABLED")
endif()

# =============================================================================
# Module library (shared solvers — OpenMP default backend)
# =============================================================================
if(CMAKE_Fortran_COMPILER_ID MATCHES "LLVMFlang")
  message(STATUS "To build on AMD the common modules need to be a shared library!")
  add_library(mom6_modules SHARED)
else()
  add_library(mom6_modules STATIC)
endif()
add_subdirectory(src/common)

# OpenMP is the default portable backend (works on AMD, Intel, NVIDIA)
if(MOM6_ENABLE_OMP_KERNELS)
  message(STATUS "OpenMP target kernels: ENABLED (default backend)")
  add_subdirectory(src/openmp)
endif()

set_target_properties(mom6_modules PROPERTIES Fortran_MODULE_DIRECTORY
                                              ${CMAKE_BINARY_DIR}/mod)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
  target_compile_definitions(mom6_modules PRIVATE USE_NVTX)
endif()
# OMP target data management is needed for both OMP kernels and CUDA C
# (the CUDA C driver uses !$omp target enter/exit data for state arrays)
if(MOM6_ENABLE_OMP_KERNELS OR MOM6_ENABLE_CUDA_C_KERNELS)
  target_compile_definitions(mom6_modules PRIVATE USE_OMP_OFFLOAD)
endif()
target_include_directories(mom6_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)

# =============================================================================
# Serial CPU kernel library (no GPU directives)
# =============================================================================

if(MOM6_ENABLE_SERIAL_KERNELS)
  message(STATUS "Serial CPU kernels: ENABLED")

  add_library(mom6_serial_modules STATIC
    src/serial/mom6_continuity_serial.F90
    src/serial/mom6_coriolis_serial.F90
    src/serial/mom6_barotropic_serial.F90
    src/serial/mom6_vert_visc_serial.F90
    src/serial/mom6_hor_visc_serial.F90)

  set_target_properties(mom6_serial_modules PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
  target_include_directories(mom6_serial_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)
  target_link_libraries(mom6_serial_modules PUBLIC mom6_modules)
endif()

# =============================================================================
# OpenACC kernel library (opt-in, requires NVHPC)
# =============================================================================

if(MOM6_ENABLE_OPENACC_KERNELS)
  add_subdirectory(src/openacc)
endif()

# =============================================================================
# CUDA Fortran kernel library (opt-in)
# =============================================================================

if(MOM6_ENABLE_CUDA_KERNELS)
  add_subdirectory(src/cudafor)
endif()

# =============================================================================
# CUDA C kernel library (opt-in — nvcc for .cu, any Fortran for wrappers)
# =============================================================================

if(MOM6_ENABLE_CUDA_C_KERNELS)
  add_subdirectory(src/cuda_c)
endif()

# =============================================================================
# Single-GPU drivers (opt-in, OFF by default)
# =============================================================================

if(MOM6_ENABLE_SINGLE_GPU_DRIVERS)
  message(STATUS "Single-GPU drivers: ENABLED")

  # OpenACC single-GPU driver (requires OPENACC kernels)
  if(MOM6_ENABLE_OPENACC_KERNELS)
    add_executable(rk2_driver app/rk2_driver.F90)
    target_link_libraries(rk2_driver PRIVATE mom6_modules)

    set(OPENACC_DRIVERS rk2_driver)

    if(MOM6_ENABLE_MODULE_DRIVERS)
      set(MODULE_DRIVERS
        continuity_driver coriolis_driver barotropic_driver
        vert_visc_driver hor_visc_driver)

      foreach(drv ${MODULE_DRIVERS})
        add_executable(${drv} app/module_drivers/${drv}.F90)
        target_link_libraries(${drv} PRIVATE mom6_modules)
      endforeach()

      list(APPEND OPENACC_DRIVERS ${MODULE_DRIVERS})
    endif()
  endif()

  # CUDA single-GPU driver
  if(MOM6_ENABLE_CUDA_KERNELS)
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
        add_cuda_driver(${drv} app/module_drivers/${drv}.F90)
      endforeach()
    endif()
  endif()

  # CUDA C single-GPU driver (nvcc kernels + OMP target data management)
  if(MOM6_ENABLE_CUDA_C_KERNELS)
    add_executable(rk2_cuda_c_driver app/rk2_cuda_c_driver.F90)
    target_link_libraries(rk2_cuda_c_driver PRIVATE
      mom6_modules mom6_cuda_c_wrappers)
    # Link cudart and stdc++ explicitly — CUDA::cudart causes warnings with nvfortran
    target_link_options(rk2_cuda_c_driver PRIVATE
      "-L${CUDAToolkit_LIBRARY_DIR}" "-lcudart" "-lstdc++")
  endif()

  # OpenMP single-GPU driver
  if(MOM6_ENABLE_OMP_KERNELS)
    add_executable(rk2_omp_driver app/rk2_omp_driver.F90)
    target_link_libraries(rk2_omp_driver PRIVATE mom6_modules mom6_omp_modules)
  endif()

  # Serial CPU single-node driver
  if(MOM6_ENABLE_SERIAL_KERNELS)
    add_executable(rk2_serial_driver app/rk2_serial_driver.F90)
    target_link_libraries(rk2_serial_driver PRIVATE mom6_modules mom6_serial_modules)
  endif()
endif()

# =============================================================================
# MPI modules and drivers (ON by default — realistic multi-GPU)
# =============================================================================

if(MOM6_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS Fortran)
  message(STATUS "MPI Fortran: ENABLED (${MPI_Fortran_COMPILER})")

  # MPI infrastructure library (domain decomposition + halo exchange)
  add_library(mom6_mpi_modules STATIC
    src/common/mom6_mpi_domain.F90
    src/common/mom6_mpi_halo.F90)

  set_target_properties(mom6_mpi_modules PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
  target_include_directories(mom6_mpi_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)
  target_link_libraries(mom6_mpi_modules PUBLIC mom6_modules MPI::MPI_Fortran)

  # OpenACC MPI driver (requires OpenACC kernels)
  if(MOM6_ENABLE_OPENACC_KERNELS)
    add_executable(rk2_mpi_driver app/rk2_mpi_driver.F90)
    target_link_libraries(rk2_mpi_driver PRIVATE mom6_modules mom6_mpi_modules MPI::MPI_Fortran)
  endif()

  # CUDA MPI (requires both MPI and CUDA enabled)
  if(MOM6_ENABLE_CUDA_KERNELS)
    # CUDA MPI halo exchange library
    add_library(mom6_mpi_cuda_modules STATIC
      src/cudafor/mom6_mpi_halo_cuda.F90)

    set_target_properties(mom6_mpi_cuda_modules PROPERTIES
      Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
    target_include_directories(mom6_mpi_cuda_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)
    target_compile_options(mom6_mpi_cuda_modules PRIVATE "-cuda")
    target_link_libraries(mom6_mpi_cuda_modules PUBLIC mom6_mpi_modules MPI::MPI_Fortran)

    # CUDA MPI driver
    add_executable(rk2_mpi_cuda_driver app/rk2_mpi_cuda_driver.F90)
    target_link_libraries(rk2_mpi_cuda_driver PRIVATE
      mom6_modules mom6_cuda_kernels mom6_mpi_modules mom6_mpi_cuda_modules MPI::MPI_Fortran)
    target_compile_options(rk2_mpi_cuda_driver PRIVATE "-cuda")
    target_link_options(rk2_mpi_cuda_driver PRIVATE "-cuda" "-cudalib=nvtx")
  endif()

  # OpenMP target MPI (requires both MPI and OMP kernels enabled)
  if(MOM6_ENABLE_OMP_KERNELS)
    # OpenMP MPI halo exchange library
    add_library(mom6_mpi_omp_modules STATIC
      src/openmp/mom6_mpi_halo_omp.F90)

    set_target_properties(mom6_mpi_omp_modules PROPERTIES
      Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
    target_include_directories(mom6_mpi_omp_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)
    target_link_libraries(mom6_mpi_omp_modules PUBLIC mom6_mpi_modules MPI::MPI_Fortran)

    # OpenMP MPI driver
    add_executable(rk2_mpi_omp_driver app/rk2_mpi_omp_driver.F90)
    target_link_libraries(rk2_mpi_omp_driver PRIVATE
      mom6_modules mom6_omp_modules mom6_mpi_modules mom6_mpi_omp_modules MPI::MPI_Fortran)
  endif()

  # Serial CPU MPI (requires MPI and serial kernels)
  if(MOM6_ENABLE_SERIAL_KERNELS)
    add_executable(rk2_mpi_serial_driver app/rk2_mpi_serial_driver.F90)
    target_link_libraries(rk2_mpi_serial_driver PRIVATE
      mom6_modules mom6_serial_modules mom6_mpi_modules MPI::MPI_Fortran)
  endif()

  # CUDA C MPI (requires both MPI and CUDA C kernels + OMP halo exchange)
  if(MOM6_ENABLE_CUDA_C_KERNELS)
    # Needs OMP halo exchange library — build it if not already built
    if(NOT TARGET mom6_mpi_omp_modules)
      add_library(mom6_mpi_omp_modules STATIC
        src/openmp/mom6_mpi_halo_omp.F90)

      set_target_properties(mom6_mpi_omp_modules PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
      target_include_directories(mom6_mpi_omp_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)
      target_link_libraries(mom6_mpi_omp_modules PUBLIC mom6_mpi_modules MPI::MPI_Fortran)
    endif()

    # CUDA C MPI driver
    add_executable(rk2_mpi_cuda_c_driver app/rk2_mpi_cuda_c_driver.F90)
    target_link_libraries(rk2_mpi_cuda_c_driver PRIVATE
      mom6_modules mom6_cuda_c_wrappers mom6_mpi_modules mom6_mpi_omp_modules MPI::MPI_Fortran)
    target_link_options(rk2_mpi_cuda_c_driver PRIVATE
      "-L${CUDAToolkit_LIBRARY_DIR}" "-lcudart" "-lstdc++")
  endif()
endif()

# =============================================================================
# Installation
# =============================================================================

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
message(STATUS "OpenMP target:    ${MOM6_ENABLE_OMP_KERNELS} (default backend)")
message(STATUS "OpenACC kernels:  ${MOM6_ENABLE_OPENACC_KERNELS}")
message(STATUS "CUDA kernels:     ${MOM6_ENABLE_CUDA_KERNELS}")
message(STATUS "Serial CPU:       ${MOM6_ENABLE_SERIAL_KERNELS}")
message(STATUS "CUDA C kernels:   ${MOM6_ENABLE_CUDA_C_KERNELS}")
message(STATUS "MPI parallel:     ${MOM6_ENABLE_MPI}")
message(STATUS "Single-GPU:       ${MOM6_ENABLE_SINGLE_GPU_DRIVERS}")
message(STATUS "Module drivers:   ${MOM6_ENABLE_MODULE_DRIVERS}")
message(STATUS "Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
