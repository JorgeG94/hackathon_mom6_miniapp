add_library(mom6_omp_modules STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_continuity_omp.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_continuity_adjust_omp.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_coriolis_omp.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_barotropic_omp.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_vert_visc_omp.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_hor_visc_omp.F90)

set_target_properties(mom6_omp_modules PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
target_include_directories(mom6_omp_modules PUBLIC ${CMAKE_BINARY_DIR}/mod)
target_link_libraries(mom6_omp_modules PUBLIC mom6_modules)
