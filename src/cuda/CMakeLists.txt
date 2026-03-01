add_library(
  mom6_cuda_kernels STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_coriolis_cuda.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_vert_visc_cuda.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_barotropic_cuda.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_hor_visc_cuda.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_continuity_cuda.F90)

set_target_properties(mom6_cuda_kernels PROPERTIES Fortran_MODULE_DIRECTORY
                                                   ${CMAKE_BINARY_DIR}/mod)
target_include_directories(mom6_cuda_kernels PUBLIC ${CMAKE_BINARY_DIR}/mod)
# Append -cuda to the base flags (inherits -O3 -acc -Minfo=accel from global)
# -cuda only at compile time; -acc already links the CUDA runtime
target_compile_options(mom6_cuda_kernels PRIVATE "-cuda")
