cmake_minimum_required(VERSION 3.18)

project(
  mom6_miniapps
  VERSION 1.0.0
  LANGUAGES Fortran
  DESCRIPTION "MOM6 Mini-Apps")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

# Compiler-specific flags
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
  message(STATUS "Detected NVIDIA HPC compiler (nvfortran)")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -acc -Minfo=accel")

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
# Module library (shared solvers)
# =============================================================================

add_library(
  mom6_modules STATIC
  src/mom6_types.F90
  src/mom6_profiler.F90
  src/mom6_continuity.F90
  src/mom6_coriolis.F90
  src/mom6_barotropic.F90
  src/mom6_vert_visc.F90
  src/mom6_hor_visc.F90
  src/mom6_diag.F90)

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
message(STATUS "Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
