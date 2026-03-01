target_sources(mom6_modules PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_continuity.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_continuity_adjust.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_coriolis.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_barotropic.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_vert_visc.F90
  ${CMAKE_CURRENT_SOURCE_DIR}/mom6_hor_visc.F90)
