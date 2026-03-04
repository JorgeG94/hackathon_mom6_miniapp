# CUDA C kernels library: nvcc-compiled .cu + Fortran iso_c_binding wrappers
# No cudafor dependency — any OpenMP-offload-capable Fortran compiler can link.

# --- CUDA C kernel objects (compiled with nvcc) ---
add_library(mom6_cuda_c_kernels STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_helpers.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_coriolis_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_hor_visc_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_vert_visc_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_barotropic_kernels.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_continuity_kernels.cu)

target_include_directories(mom6_cuda_c_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(mom6_cuda_c_kernels PROPERTIES
  CUDA_STANDARD 11
  POSITION_INDEPENDENT_CODE ON)

# --- Fortran wrapper library (iso_c_binding interfaces) ---
add_library(mom6_cuda_c_wrappers STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_cuda_c_common.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_coriolis_cuda_c.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_hor_visc_cuda_c.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_vert_visc_cuda_c.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_barotropic_cuda_c.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_continuity_cuda_c.F90)

set_target_properties(mom6_cuda_c_wrappers PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
target_include_directories(mom6_cuda_c_wrappers PUBLIC ${CMAKE_BINARY_DIR}/mod)
target_link_libraries(mom6_cuda_c_wrappers PUBLIC mom6_cuda_c_kernels)
