# MOM6 Mini-Apps Makefile
#
# Targets:
#   make              - Build build/rk2_driver (OpenACC, GPU+multicore)
#   make cuda         - Build build/rk2_cuda_driver
#   make modules      - Build all 5 OpenACC module drivers into build/
#   make modules-cuda - Build all 6 CUDA module drivers into build/
#   make small_scaling - Run scripts/benchmark_scaling.sh
#   make large_scaling - Run scripts/strong_scaling.sh
#   make plots        - Run plotting scripts
#   make clean        - Remove all build artifacts
#   make info         - Print compiler/flags

# Default compiler
FC ?= gfortran

# Directories
SRCDIR_COMMON  = src/common
SRCDIR_OPENACC = src/openacc
SRCDIR_CUDA    = src/cuda
APPDIR   = app
BUILDDIR = build

# Compiler-specific flags (use findstring to match full paths)
ifneq (,$(findstring nvfortran,$(FC)))
  FFLAGS = -O3  -mp=multicore,gpu -acc=multicore,gpu -gpu=mem:separate
  LDFLAGS = -cudalib=nvtx
  MODFLAG = -module
else ifneq (,$(findstring gfortran,$(FC)))
  # GNU Fortran
  FFLAGS = -O3 -fallow-argument-mismatch -ffree-line-length-none
  LDFLAGS =
  MODFLAG = -J
else ifneq (,$(findstring ifx,$(FC)))
  # Intel ifx
  FFLAGS = -O3 -heap-arrays
  LDFLAGS =
  MODFLAG = -module
else ifneq (,$(findstring flang-new,$(FC)))
  # LLVM Flang
  FFLAGS = -O3
  LDFLAGS =
  MODFLAG = -J
else ifneq (,$(findstring lfortran,$(FC)))
  FFLAGS = -O3 --cpp
  LDFLAGS =
  MODFLAG = -J
else
  # Default flags
  FFLAGS = -O3
  LDFLAGS =
  MODFLAG = -J
endif

# Module include path
MODFLAGS = -I$(BUILDDIR)

# CUDA kernel objects
CUDA_MODULES = $(BUILDDIR)/mom6_coriolis_cuda.o \
               $(BUILDDIR)/mom6_vert_visc_cuda.o \
               $(BUILDDIR)/mom6_barotropic_cuda.o \
               $(BUILDDIR)/mom6_hor_visc_cuda.o \
               $(BUILDDIR)/mom6_continuity_cuda.o

# Module objects
MODULES = $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o \
          $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o \
          $(BUILDDIR)/mom6_coriolis.o $(BUILDDIR)/mom6_barotropic.o \
          $(BUILDDIR)/mom6_vert_visc.o $(BUILDDIR)/mom6_hor_visc.o \
          $(BUILDDIR)/mom6_diag.o

# OpenACC module drivers (individual kernels)
MODULE_DRIVERS = $(BUILDDIR)/continuity_driver $(BUILDDIR)/coriolis_driver \
                 $(BUILDDIR)/barotropic_driver $(BUILDDIR)/vert_visc_driver \
                 $(BUILDDIR)/hor_visc_driver

# CUDA module drivers (individual kernels)
CUDA_DRIVERS = $(BUILDDIR)/continuity_cuda_driver $(BUILDDIR)/coriolis_cuda_driver \
               $(BUILDDIR)/barotropic_cuda_driver $(BUILDDIR)/vert_visc_cuda_driver \
               $(BUILDDIR)/hor_visc_cuda_driver $(BUILDDIR)/rk2_cuda_driver

.PHONY: all cuda modules modules-cuda clean info small_scaling large_scaling plots

all: $(BUILDDIR)/rk2_driver

cuda: $(BUILDDIR)/rk2_cuda_driver

modules: $(MODULE_DRIVERS)

modules-cuda: $(CUDA_DRIVERS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

#==============================================================================
# Module compilation (order matters due to dependencies)
#==============================================================================

$(BUILDDIR)/mom6_types.o: $(SRCDIR_COMMON)/mom6_types.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_profiler.o: $(SRCDIR_COMMON)/mom6_profiler.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_continuity.o: $(SRCDIR_OPENACC)/mom6_continuity.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

# Submodule with zonal_flux_adjust_gpu and set_zonal_BT_cont_gpu
$(BUILDDIR)/mom6_continuity_adjust.o: $(SRCDIR_OPENACC)/mom6_continuity_adjust.F90 $(BUILDDIR)/mom6_continuity.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_coriolis.o: $(SRCDIR_OPENACC)/mom6_coriolis.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_barotropic.o: $(SRCDIR_OPENACC)/mom6_barotropic.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_vert_visc.o: $(SRCDIR_OPENACC)/mom6_vert_visc.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_hor_visc.o: $(SRCDIR_OPENACC)/mom6_hor_visc.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_diag.o: $(SRCDIR_COMMON)/mom6_diag.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

#==============================================================================
# OpenACC driver compilation (all output to build/)
#==============================================================================

$(BUILDDIR)/continuity_driver: $(APPDIR)/continuity_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o $(LDFLAGS)

$(BUILDDIR)/coriolis_driver: $(APPDIR)/coriolis_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o $(LDFLAGS)

$(BUILDDIR)/barotropic_driver: $(APPDIR)/barotropic_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(LDFLAGS)

$(BUILDDIR)/vert_visc_driver: $(APPDIR)/vert_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o $(LDFLAGS)

$(BUILDDIR)/hor_visc_driver: $(APPDIR)/hor_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(LDFLAGS)

$(BUILDDIR)/rk2_driver: $(APPDIR)/rk2_driver.F90 $(MODULES)
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(MODULES) $(LDFLAGS)

#==============================================================================
# CUDA Fortran kernel compilation (nvfortran only, -cuda flag)
#==============================================================================

$(BUILDDIR)/mom6_coriolis_cuda.o: $(SRCDIR_CUDA)/mom6_coriolis_cuda.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_vert_visc_cuda.o: $(SRCDIR_CUDA)/mom6_vert_visc_cuda.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_barotropic_cuda.o: $(SRCDIR_CUDA)/mom6_barotropic_cuda.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_hor_visc_cuda.o: $(SRCDIR_CUDA)/mom6_hor_visc_cuda.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_continuity_cuda.o: $(SRCDIR_CUDA)/mom6_continuity_cuda.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

#==============================================================================
# CUDA driver compilation (all output to build/)
#==============================================================================

$(BUILDDIR)/continuity_cuda_driver: $(APPDIR)/continuity_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity_cuda.o $(LDFLAGS) -cuda

$(BUILDDIR)/coriolis_cuda_driver: $(APPDIR)/coriolis_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis_cuda.o $(LDFLAGS) -cuda

$(BUILDDIR)/barotropic_cuda_driver: $(APPDIR)/barotropic_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_barotropic_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_barotropic_cuda.o $(LDFLAGS) -cuda

$(BUILDDIR)/vert_visc_cuda_driver: $(APPDIR)/vert_visc_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc_cuda.o $(LDFLAGS) -cuda

$(BUILDDIR)/hor_visc_cuda_driver: $(APPDIR)/hor_visc_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(BUILDDIR)/mom6_hor_visc_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(BUILDDIR)/mom6_hor_visc_cuda.o $(LDFLAGS) -cuda

$(BUILDDIR)/rk2_cuda_driver: $(APPDIR)/rk2_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_hor_visc.o $(CUDA_MODULES)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_hor_visc.o $(CUDA_MODULES) $(LDFLAGS) -cuda

#==============================================================================
# Benchmarking and plotting
#==============================================================================

small_scaling:
	bash scripts/benchmark_scaling.sh

large_scaling:
	bash scripts/strong_scaling.sh

plots:
	python scripts/plot_benchmark.py
	python scripts/plot_larger_scaling.py

#==============================================================================
# Utility targets
#==============================================================================

clean:
	rm -rf $(BUILDDIR) benchmark_plots/ benchmark_logs/ *_scaling_logs/ *.csv

info:
	@echo "========================================"
	@echo "MOM6 Mini-Apps Build Configuration"
	@echo "========================================"
	@echo "Compiler: $(FC)"
	@echo "FFLAGS:   $(FFLAGS)"
	@echo "LDFLAGS:  $(LDFLAGS)"
	@echo "========================================"
