# MOM6 Mini-Apps Makefile
#
# Build mini-apps for benchmarking
#
# Usage:
#   make              - Build all with gfortran
#   make FC=gfortran  - Build with gfortran
#   make FC=ifx       - Build with Intel ifx
#   make clean        - Remove binaries
#
# Examples:
#   make FC=gfortran        # GNU build
#   make FC=ifx             # Intel build
#   make FC=nvfortran       # NVIDIA build

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

# Driver executables
DRIVERS = continuity_driver coriolis_driver barotropic_driver vert_visc_driver hor_visc_driver rk2_driver

# CUDA-only driver executables (explicit CUDA Fortran kernels)
CUDA_DRIVERS = continuity_cuda_driver coriolis_cuda_driver barotropic_cuda_driver \
               vert_visc_cuda_driver hor_visc_cuda_driver rk2_cuda_driver

# CUDA compare driver executables (OpenACC vs CUDA side-by-side)
COMPARE_DRIVERS = coriolis_compare_driver vert_visc_compare_driver \
                  barotropic_compare_driver hor_visc_compare_driver \
                  continuity_compare_driver

.PHONY: all cuda compare clean info run-continuity run-coriolis run-barotropic run-vert-visc run-rk2 run-all

all: $(BUILDDIR) $(DRIVERS)

cuda: $(BUILDDIR) $(DRIVERS) $(CUDA_DRIVERS)

compare: $(BUILDDIR) $(DRIVERS) $(COMPARE_DRIVERS)

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
# Driver compilation
#==============================================================================

continuity_driver: $(APPDIR)/continuity_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o $(LDFLAGS)

coriolis_driver: $(APPDIR)/coriolis_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o $(LDFLAGS)

barotropic_driver: $(APPDIR)/barotropic_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(LDFLAGS)

vert_visc_driver: $(APPDIR)/vert_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o $(LDFLAGS)

hor_visc_driver: $(APPDIR)/hor_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(LDFLAGS)

rk2_driver: $(APPDIR)/rk2_driver.F90 $(MODULES)
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
# CUDA compare driver compilation (link both OpenACC modules + CUDA kernels)
#==============================================================================

coriolis_compare_driver: $(APPDIR)/coriolis_compare_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o $(BUILDDIR)/mom6_coriolis_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o $(BUILDDIR)/mom6_coriolis_cuda.o $(LDFLAGS) -cuda

vert_visc_compare_driver: $(APPDIR)/vert_visc_compare_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o $(BUILDDIR)/mom6_vert_visc_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o $(BUILDDIR)/mom6_vert_visc_cuda.o $(LDFLAGS) -cuda

barotropic_compare_driver: $(APPDIR)/barotropic_compare_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_barotropic_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_barotropic_cuda.o $(LDFLAGS) -cuda

hor_visc_compare_driver: $(APPDIR)/hor_visc_compare_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(BUILDDIR)/mom6_hor_visc_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(BUILDDIR)/mom6_hor_visc_cuda.o $(LDFLAGS) -cuda

continuity_compare_driver: $(APPDIR)/continuity_compare_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o $(BUILDDIR)/mom6_continuity_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o $(BUILDDIR)/mom6_continuity_cuda.o $(LDFLAGS) -cuda

#==============================================================================
# CUDA-only driver compilation (explicit CUDA Fortran kernels, no OpenACC compute)
#==============================================================================

continuity_cuda_driver: $(APPDIR)/continuity_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity_cuda.o $(LDFLAGS) -cuda

coriolis_cuda_driver: $(APPDIR)/coriolis_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis_cuda.o $(LDFLAGS) -cuda

barotropic_cuda_driver: $(APPDIR)/barotropic_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_barotropic_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_barotropic_cuda.o $(LDFLAGS) -cuda

vert_visc_cuda_driver: $(APPDIR)/vert_visc_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc_cuda.o $(LDFLAGS) -cuda

hor_visc_cuda_driver: $(APPDIR)/hor_visc_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(BUILDDIR)/mom6_hor_visc_cuda.o
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(BUILDDIR)/mom6_hor_visc_cuda.o $(LDFLAGS) -cuda

rk2_cuda_driver: $(APPDIR)/rk2_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_hor_visc.o $(CUDA_MODULES)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_hor_visc.o $(CUDA_MODULES) $(LDFLAGS) -cuda

#==============================================================================
# Utility targets
#==============================================================================

clean:
	rm -f $(DRIVERS) $(CUDA_DRIVERS) $(COMPARE_DRIVERS) *.o *.mod
	rm -rf $(BUILDDIR)

info:
	@echo "========================================"
	@echo "MOM6 Mini-Apps Build Configuration"
	@echo "========================================"
	@echo "Compiler: $(FC)"
	@echo "FFLAGS:   $(FFLAGS)"
	@echo "LDFLAGS:  $(LDFLAGS)"
	@echo "========================================"
	@echo ""
	@echo "Drivers: $(DRIVERS)"
	@echo "========================================"

#==============================================================================
# Run targets
#==============================================================================

run-continuity: continuity_driver
	@echo "Running continuity driver (180x180x75, 10 iters)..."
	./continuity_driver 180 180 75 10

run-coriolis: coriolis_driver
	@echo "Running Coriolis driver (180x180x75, 10 iters, Sadourny)..."
	./coriolis_driver 180 180 75 10 sadourny

run-barotropic: barotropic_driver
	@echo "Running Barotropic driver (180x180, 30 substeps, 10 iters)..."
	./barotropic_driver 180 180 30 10

run-vert-visc: vert_visc_driver
	@echo "Running Vertical Viscosity driver (180x180x75, 10 iters)..."
	./vert_visc_driver 180 180 75 10

run-hor-visc: hor_visc_driver
	@echo "Running Horizontal Viscosity driver (180x180x75, 10 iters)..."
	./hor_visc_driver 180 180 75 10

run-rk2: rk2_driver
	@echo "Running RK2 driver (180x180x75, 10 iters, 30 BT substeps)..."
	./rk2_driver 180 180 75 10 30

run-all: $(DRIVERS)
	@echo ""
	@echo "========================================"
	@echo "Running Continuity Driver"
	@echo "========================================"
	./continuity_driver 180 180 75 10
	@echo ""
	@echo "========================================"
	@echo "Running Coriolis Driver"
	@echo "========================================"
	./coriolis_driver 180 180 75 10
	@echo ""
	@echo "========================================"
	@echo "Running Barotropic Driver"
	@echo "========================================"
	./barotropic_driver 180 180 30 10
	@echo ""
	@echo "========================================"
	@echo "Running Vertical Viscosity Driver"
	@echo "========================================"
	./vert_visc_driver 180 180 75 10
	@echo ""
	@echo "========================================"
	@echo "Running Horizontal Viscosity Driver"
	@echo "========================================"
	./hor_visc_driver 180 180 75 10
	@echo ""
	@echo "========================================"
	@echo "Running RK2 Driver"
	@echo "========================================"
	./rk2_driver 180 180 75 10 30

#==============================================================================
# Scaling tests
#==============================================================================

scale-test: $(DRIVERS)
	@echo "Scaling test: varying grid size"
	@echo ""
	@echo "Continuity:"
	./continuity_driver 90 90 75 5
	./continuity_driver 180 180 75 5
	./continuity_driver 360 360 75 5
	@echo ""
	@echo "Coriolis:"
	./coriolis_driver 90 90 75 5
	./coriolis_driver 180 180 75 5
	./coriolis_driver 360 360 75 5
	@echo ""
	@echo "Barotropic:"
	./barotropic_driver 90 90 30 5
	./barotropic_driver 180 180 30 5
	./barotropic_driver 360 360 30 5
	@echo ""
	@echo "Vertical Viscosity:"
	./vert_visc_driver 90 90 75 5
	./vert_visc_driver 180 180 75 5
	./vert_visc_driver 360 360 75 5
	@echo ""
	@echo "Horizontal Viscosity:"
	./hor_visc_driver 90 90 75 5
	./hor_visc_driver 180 180 75 5
	./hor_visc_driver 360 360 75 5
	@echo ""
	@echo "RK2:"
	./rk2_driver 90 90 75 5 30
	./rk2_driver 180 180 75 5 30
	./rk2_driver 360 360 75 5 30
