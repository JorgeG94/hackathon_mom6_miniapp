# MOM6 Mini-Apps Makefile
#
# Build mini-apps for GPU hackathon benchmarking
#
# Usage:
#   make              - Build all with nvfortran (GPU)
#   make FC=gfortran  - Build with gfortran (CPU)
#   make FC=ifx       - Build with Intel ifx (CPU)
#   make clean        - Remove binaries
#
# For GPU builds (nvfortran):
#   make GPU=yes      - Enable OpenMP target offloading (default)
#   make GPU=no       - Disable GPU, CPU-only OpenMP
#
# Profiling (nvfortran only):
#   make NVTX=yes         - Enable NVTX ranges for Nsight profiling
#   make DISABLE_PROFILER=yes - Disable profiler entirely (zero overhead)
#
# Examples:
#   make FC=nvfortran GPU=yes   # NVIDIA GPU build
#   make FC=gfortran GPU=no     # GNU CPU build
#   make FC=nvfortran GPU=yes NVTX=yes  # GPU build with Nsight profiling

# Default compiler
FC ?= nvfortran

# GPU flag (default: yes for nvfortran)
GPU ?= yes

# NVTX profiling (default: no, set NVTX=yes to enable for NVIDIA)
NVTX ?= no

# Disable profiler entirely (default: no)
DISABLE_PROFILER ?= no

# Vertical viscosity loop ordering variant (jik, ijk, jki, ikj, kji, kij)
# Default: jik for GPU (one thread per column), jki for CPU (stride-1 inner loop)
ifeq ($(GPU),yes)
  VERTVISC_VARIANT ?= jik
else
  VERTVISC_VARIANT ?= jki
endif

# Directories
SRCDIR = src
APPDIR = app
BUILDDIR = build

# Compiler-specific flags (use findstring to match full paths)
ifneq (,$(findstring nvfortran,$(FC)))
  ifeq ($(GPU),yes)
    # NVIDIA GPU offloading with OpenMP target + stdpar
    FFLAGS = -O3 -mp=multicore,gpu -stdpar=multicore,gpu -gpu=cc70 -Minfo=accel -gpu=mem:separate
    LDFLAGS = -mp=multicore,gpu -stdpar=multicore,gpu -gpu=cc70 -cudalib=nvtx
  else
    # CPU-only with OpenMP
    FFLAGS = -O3 -mp -Minfo=opt
    LDFLAGS = -mp
  endif
  MODFLAG = -module
else ifneq (,$(findstring gfortran,$(FC)))
  # GNU Fortran (CPU only)
  FFLAGS = -O3 -fopenmp -fallow-argument-mismatch
  LDFLAGS = -fopenmp
  MODFLAG = -J
else ifneq (,$(findstring ifx,$(FC)))
  # Intel ifx (CPU with OpenMP)
  FFLAGS = -O3 -qopenmp -heap-arrays
  LDFLAGS = -qopenmp
  MODFLAG = -module
else ifneq (,$(findstring flang-new,$(FC)))
  # LLVM Flang (experimental GPU support)
  ifeq ($(GPU),yes)
    FFLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64
    LDFLAGS = -fopenmp -fopenmp-targets=nvptx64
  else
    FFLAGS = -O3 -fopenmp
    LDFLAGS = -fopenmp
  endif
  MODFLAG = -J
else ifneq (,$(findstring lfortran,$(FC)))
  FFLAGS = -O3 --cpp --openmp
  LDFLAGS = 
  MODFLAG = -J
else
  # Default flags
  FFLAGS = -O3
  LDFLAGS =
  MODFLAG = -J
endif

# Profiler preprocessor flags
ifeq ($(NVTX),yes)
  FFLAGS += -DUSE_NVTX
  # NVTX requires cudalib=nvtx for nvfortran
  ifneq (,$(findstring nvfortran,$(FC)))
    FFLAGS += -cudalib=nvtx
    LDFLAGS += -cudalib=nvtx
  endif
endif

ifeq ($(DISABLE_PROFILER),yes)
  FFLAGS += -DDISABLE_PROFILER
endif

# Module include path
MODFLAGS = -I$(BUILDDIR)

# Vertical viscosity submodule source
VERTVISC_SUBMOD_SRC = $(SRCDIR)/mom6_vert_visc_$(VERTVISC_VARIANT).F90

# Module objects (includes vert_visc submodule)
MODULES = $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o \
          $(BUILDDIR)/mom6_continuity.o \
          $(BUILDDIR)/mom6_coriolis.o $(BUILDDIR)/mom6_barotropic.o \
          $(BUILDDIR)/mom6_vert_visc.o $(BUILDDIR)/mom6_vert_visc_sub.o \
          $(BUILDDIR)/mom6_hor_visc.o \
          $(BUILDDIR)/mom6_diag.o

# Driver executables
DRIVERS = continuity_driver coriolis_driver barotropic_driver vert_visc_driver hor_visc_driver rk2_driver

# All vertical viscosity loop ordering variants
VERTVISC_VARIANTS = jik ijk jki ikj kji kij

.PHONY: all clean info run-continuity run-coriolis run-barotropic run-vert-visc run-rk2 run-all \
        vertvisc-all run-vertvisc-all

all: $(BUILDDIR) $(DRIVERS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

#==============================================================================
# Module compilation (order matters due to dependencies)
#==============================================================================

$(BUILDDIR)/mom6_types.o: $(SRCDIR)/mom6_types.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_profiler.o: $(SRCDIR)/mom6_profiler.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_continuity.o: $(SRCDIR)/mom6_continuity.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_coriolis.o: $(SRCDIR)/mom6_coriolis.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_barotropic.o: $(SRCDIR)/mom6_barotropic.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_vert_visc.o: $(SRCDIR)/mom6_vert_visc.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_vert_visc_sub.o: $(VERTVISC_SUBMOD_SRC) $(BUILDDIR)/mom6_vert_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_hor_visc.o: $(SRCDIR)/mom6_hor_visc.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_diag.o: $(SRCDIR)/mom6_diag.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

#==============================================================================
# Driver compilation
#==============================================================================

continuity_driver: $(APPDIR)/continuity_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_continuity.o $(LDFLAGS)

coriolis_driver: $(APPDIR)/coriolis_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o $(LDFLAGS)

barotropic_driver: $(APPDIR)/barotropic_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(LDFLAGS)

vert_visc_driver: $(APPDIR)/vert_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o $(BUILDDIR)/mom6_vert_visc_sub.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o $(BUILDDIR)/mom6_vert_visc_sub.o $(LDFLAGS)

hor_visc_driver: $(APPDIR)/hor_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(LDFLAGS)

rk2_driver: $(APPDIR)/rk2_driver.F90 $(MODULES)
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(MODULES) $(LDFLAGS)

#==============================================================================
# Utility targets
#==============================================================================

clean:
	rm -f $(DRIVERS) $(foreach v,$(VERTVISC_VARIANTS),vert_visc_driver_$(v)) *.o *.mod
	rm -rf $(BUILDDIR)

info:
	@echo "========================================"
	@echo "MOM6 Mini-Apps Build Configuration"
	@echo "========================================"
	@echo "Compiler: $(FC)"
	@echo "GPU:      $(GPU)"
	@echo "NVTX:     $(NVTX)"
	@echo "DISABLE_PROFILER: $(DISABLE_PROFILER)"
	@echo "VERTVISC_VARIANT: $(VERTVISC_VARIANT)"
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

#==============================================================================
# Vertical viscosity variant targets
#==============================================================================

vertvisc-all:
	@echo "========================================"
	@echo "Building all vertical viscosity variants"
	@echo "Compiler: $(FC)  GPU: $(GPU)"
	@echo "========================================"
	@for v in $(VERTVISC_VARIANTS); do \
		echo ""; \
		echo "--- Building variant: $$v ---"; \
		rm -rf $(BUILDDIR) $(DRIVERS); \
		if $(MAKE) FC=$(FC) GPU=$(GPU) VERTVISC_VARIANT=$$v vert_visc_driver; then \
			mv vert_visc_driver vert_visc_driver_$$v; \
			echo "  -> vert_visc_driver_$$v"; \
		else \
			echo "  FAILED to build $$v"; \
		fi; \
	done
	@echo ""
	@echo "========================================"
	@echo "Built executables:"
	@ls -la vert_visc_driver_* 2>/dev/null || echo "  (none)"
	@echo "========================================"

run-vertvisc-all:
	@echo "========================================"
	@echo "Running all vertical viscosity variants"
	@echo "========================================"
	@for v in $(VERTVISC_VARIANTS); do \
		if [ -f vert_visc_driver_$$v ]; then \
			echo ""; \
			echo "=== $$v ==="; \
			./vert_visc_driver_$$v 180 180 75 5; \
		else \
			echo ""; \
			echo "=== $$v === (not built, skipping)"; \
		fi; \
	done

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
