# MOM6 Mini-Apps Makefile
#
# OpenMP target offloading is the default backend (works on AMD, Intel, NVIDIA).
# OpenACC and CUDA Fortran are opt-in (require nvfortran).
#
# Targets:
#   make              - Build OpenMP target single-GPU driver (default)
#   make omp          - Build build/rk2_omp_driver (OpenMP target, single GPU)
#   make mpi-omp      - Build build/rk2_mpi_omp_driver (OpenMP target + MPI)
#   make single-gpu   - Build build/rk2_driver (OpenACC, single GPU, requires nvfortran)
#   make mpi          - Build build/rk2_mpi_driver (OpenACC + MPI, requires nvfortran)
#   make cuda         - Build build/rk2_cuda_driver (CUDA, single GPU, requires nvfortran)
#   make cuda-c       - Build build/rk2_cuda_c_driver (CUDA C kernels + nvfortran driver)
#   make mpi-cuda     - Build build/rk2_mpi_cuda_driver (CUDA + MPI, requires nvfortran)
#   make mpi-cuda-c   - Build build/rk2_mpi_cuda_c_driver (CUDA C kernels + MPI)
#   make modules      - Build all 5 OpenACC module drivers into build/
#   make modules-cuda - Build all 6 CUDA module drivers into build/
#   make all-backends - Build all MPI drivers (OpenACC, CUDA, OpenMP target)
#   make small_scaling - Run scripts/benchmark_scaling.sh
#   make large_scaling - Run scripts/strong_scaling.sh
#   make plots        - Run plotting scripts
#   make clean        - Remove all build artifacts
#   make info         - Print compiler/flags

# Default compiler
FC ?= gfortran

# GPU backend: omp (default) or openacc (requires nvfortran, pass GPU_BACKEND=openacc)
GPU_BACKEND ?= omp

# Directories
SRCDIR_COMMON  = src/common
SRCDIR_OPENACC = src/openacc
SRCDIR_CUDA    = src/cuda
SRCDIR_OPENMP  = src/openmp
SRCDIR_CUDA_C  = src/cuda_c
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
  FFLAGS = -O3 -heap-arrays -fiopenmp -fopenmp-targets=spir64
  LDFLAGS =
  MODFLAG = -module
else ifneq (,$(findstring amdflang,$(FC)))
  # LLVM Flang (OpenMP target offloading for AMD GPUs)
  FFLAGS = -O3 -fopenmp --offload-arch=gfx90a -fopenmp-offload-mandatory -fPIC
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

# Preprocessor flags for GPU backend selection
ifeq ($(GPU_BACKEND),omp)
  CPPFLAGS = -DUSE_OMP_OFFLOAD
else
  CPPFLAGS =
endif

# MPI compiler wrapper (wraps FC)
MPI_FC = mpif90

# CUDA C compiler (for .cu files)
NVCC ?= nvcc
NVCC_FLAGS = -O3 -I $(SRCDIR_CUDA_C)
# Auto-detect GPU architecture if possible; default to sm_80
NVCC_ARCH ?= -arch=sm_70

# Module include path
MODFLAGS = -I$(BUILDDIR)

# =============================================================================
# Object groups
# =============================================================================

# Common objects (always built — types, profiler, diagnostics)
COMMON_MODULES = $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o \
                 $(BUILDDIR)/mom6_diag.o

# OpenACC objects (opt-in, requires nvfortran)
OPENACC_MODULES = $(BUILDDIR)/mom6_continuity.o $(BUILDDIR)/mom6_continuity_adjust.o \
                  $(BUILDDIR)/mom6_coriolis.o $(BUILDDIR)/mom6_barotropic.o \
                  $(BUILDDIR)/mom6_vert_visc.o $(BUILDDIR)/mom6_hor_visc.o

# OpenMP target objects (default backend)
OMP_MODULES = $(BUILDDIR)/mom6_continuity_omp.o $(BUILDDIR)/mom6_continuity_adjust_omp.o \
              $(BUILDDIR)/mom6_coriolis_omp.o $(BUILDDIR)/mom6_barotropic_omp.o \
              $(BUILDDIR)/mom6_vert_visc_omp.o $(BUILDDIR)/mom6_hor_visc_omp.o

# CUDA kernel objects (opt-in, requires nvfortran)
CUDA_MODULES = $(BUILDDIR)/cuda_workspace.o \
               $(BUILDDIR)/mom6_coriolis_cuda.o \
               $(BUILDDIR)/mom6_vert_visc_cuda.o \
               $(BUILDDIR)/mom6_barotropic_cuda.o \
               $(BUILDDIR)/mom6_hor_visc_cuda.o \
               $(BUILDDIR)/mom6_continuity_cuda.o

# CUDA C kernel objects (nvcc compiled .cu + Fortran wrappers)
CUDA_C_COMMON = $(BUILDDIR)/cuda_helpers.o $(BUILDDIR)/mom6_cuda_c_common.o
CUDA_C_MODULES = $(CUDA_C_COMMON) \
                 $(BUILDDIR)/mom6_coriolis_kernels.o $(BUILDDIR)/mom6_coriolis_cuda_c.o \
                 $(BUILDDIR)/mom6_hor_visc_kernels.o $(BUILDDIR)/mom6_hor_visc_cuda_c.o \
                 $(BUILDDIR)/mom6_vert_visc_kernels.o $(BUILDDIR)/mom6_vert_visc_cuda_c.o \
                 $(BUILDDIR)/mom6_barotropic_kernels.o $(BUILDDIR)/mom6_barotropic_cuda_c.o \
                 $(BUILDDIR)/mom6_continuity_kernels.o $(BUILDDIR)/mom6_continuity_cuda_c.o

# MPI module objects
MPI_MODULES = $(BUILDDIR)/mom6_mpi_domain.o $(BUILDDIR)/mom6_mpi_halo.o
MPI_CUDA_MODULES = $(BUILDDIR)/mom6_mpi_domain.o $(BUILDDIR)/mom6_mpi_halo_cuda.o
MPI_OMP_MODULES = $(BUILDDIR)/mom6_mpi_domain.o $(BUILDDIR)/mom6_mpi_halo_omp.o

# OpenACC module drivers (individual kernels)
MODULE_DRIVERS = $(BUILDDIR)/continuity_driver $(BUILDDIR)/coriolis_driver \
                 $(BUILDDIR)/barotropic_driver $(BUILDDIR)/vert_visc_driver \
                 $(BUILDDIR)/hor_visc_driver

# CUDA module drivers (individual kernels)
CUDA_DRIVERS = $(BUILDDIR)/continuity_cuda_driver $(BUILDDIR)/coriolis_cuda_driver \
               $(BUILDDIR)/barotropic_cuda_driver $(BUILDDIR)/vert_visc_cuda_driver \
               $(BUILDDIR)/hor_visc_cuda_driver $(BUILDDIR)/rk2_cuda_driver

.PHONY: all single-gpu cuda cuda-c omp mpi mpi-cuda mpi-cuda-c mpi-omp modules modules-cuda all-backends clean info small_scaling large_scaling plots

# Default: build OpenMP target single-GPU driver
all: omp

# All backends (requires nvfortran for OpenACC and CUDA)
all-backends: mpi mpi-cuda mpi-cuda-c mpi-omp

# Single-GPU targets
omp: $(BUILDDIR)/rk2_omp_driver

single-gpu: $(BUILDDIR)/rk2_driver

cuda: $(BUILDDIR)/rk2_cuda_driver

cuda-c: $(BUILDDIR)/rk2_cuda_c_driver

# MPI targets
mpi-omp: $(BUILDDIR)/rk2_mpi_omp_driver

mpi: $(BUILDDIR)/rk2_mpi_driver

mpi-cuda: $(BUILDDIR)/rk2_mpi_cuda_driver

mpi-cuda-c: $(BUILDDIR)/rk2_mpi_cuda_c_driver

modules: $(MODULE_DRIVERS)

modules-cuda: $(CUDA_DRIVERS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

#==============================================================================
# Common module compilation (src/common — always built)
#==============================================================================

$(BUILDDIR)/mom6_types.o: $(SRCDIR_COMMON)/mom6_types.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) $(CPPFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_profiler.o: $(SRCDIR_COMMON)/mom6_profiler.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_diag.o: $(SRCDIR_COMMON)/mom6_diag.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

#==============================================================================
# OpenMP target module compilation (src/openmp — default backend)
#==============================================================================

$(BUILDDIR)/mom6_continuity_omp.o: $(SRCDIR_OPENMP)/mom6_continuity_omp.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_continuity_adjust_omp.o: $(SRCDIR_OPENMP)/mom6_continuity_adjust_omp.F90 $(BUILDDIR)/mom6_continuity_omp.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_coriolis_omp.o: $(SRCDIR_OPENMP)/mom6_coriolis_omp.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_barotropic_omp.o: $(SRCDIR_OPENMP)/mom6_barotropic_omp.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_vert_visc_omp.o: $(SRCDIR_OPENMP)/mom6_vert_visc_omp.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_hor_visc_omp.o: $(SRCDIR_OPENMP)/mom6_hor_visc_omp.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

#==============================================================================
# OpenMP target driver compilation (default)
#==============================================================================

$(BUILDDIR)/rk2_omp_driver: $(APPDIR)/rk2_omp_driver.F90 $(COMMON_MODULES) $(OMP_MODULES)
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(COMMON_MODULES) $(OMP_MODULES) $(LDFLAGS)

#==============================================================================
# MPI module compilation (requires MPI compiler wrapper)
#==============================================================================

$(BUILDDIR)/mom6_mpi_domain.o: $(SRCDIR_COMMON)/mom6_mpi_domain.F90 $(BUILDDIR)/mom6_types.o
	$(MPI_FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_mpi_halo.o: $(SRCDIR_COMMON)/mom6_mpi_halo.F90 $(BUILDDIR)/mom6_mpi_domain.o
	$(MPI_FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_mpi_halo_omp.o: $(SRCDIR_OPENMP)/mom6_mpi_halo_omp.F90 $(BUILDDIR)/mom6_mpi_domain.o
	$(MPI_FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_mpi_halo_cuda.o: $(SRCDIR_CUDA)/mom6_mpi_halo_cuda.F90 $(BUILDDIR)/mom6_mpi_domain.o
	$(MPI_FC) $(FFLAGS) -cuda $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

#==============================================================================
# MPI + OpenMP target driver compilation
#==============================================================================

$(BUILDDIR)/rk2_mpi_omp_driver: $(APPDIR)/rk2_mpi_omp_driver.F90 $(COMMON_MODULES) $(OMP_MODULES) $(MPI_OMP_MODULES)
	$(MPI_FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(COMMON_MODULES) $(OMP_MODULES) $(MPI_OMP_MODULES) $(LDFLAGS)

#==============================================================================
# OpenACC module compilation (src/openacc — opt-in, requires nvfortran)
#==============================================================================

$(BUILDDIR)/mom6_continuity.o: $(SRCDIR_OPENACC)/mom6_continuity.F90 $(BUILDDIR)/mom6_types.o
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

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

#==============================================================================
# OpenACC driver compilation (requires nvfortran)
#==============================================================================

$(BUILDDIR)/continuity_driver: $(APPDIR)/continuity_driver.F90 $(COMMON_MODULES) $(OPENACC_MODULES)
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(COMMON_MODULES) $(OPENACC_MODULES) $(LDFLAGS)

$(BUILDDIR)/coriolis_driver: $(APPDIR)/coriolis_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_coriolis.o $(LDFLAGS)

$(BUILDDIR)/barotropic_driver: $(APPDIR)/barotropic_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_barotropic.o $(LDFLAGS)

$(BUILDDIR)/vert_visc_driver: $(APPDIR)/vert_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_vert_visc.o $(LDFLAGS)

$(BUILDDIR)/hor_visc_driver: $(APPDIR)/hor_visc_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_hor_visc.o $(LDFLAGS)

$(BUILDDIR)/rk2_driver: $(APPDIR)/rk2_driver.F90 $(COMMON_MODULES) $(OPENACC_MODULES)
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(COMMON_MODULES) $(OPENACC_MODULES) $(LDFLAGS)

#==============================================================================
# MPI + OpenACC driver compilation (requires nvfortran)
#==============================================================================

$(BUILDDIR)/rk2_mpi_driver: $(APPDIR)/rk2_mpi_driver.F90 $(COMMON_MODULES) $(OPENACC_MODULES) $(MPI_MODULES)
	$(MPI_FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(COMMON_MODULES) $(OPENACC_MODULES) $(MPI_MODULES) $(LDFLAGS)

#==============================================================================
# CUDA Fortran kernel compilation (nvfortran only, -cuda flag)
#==============================================================================

$(BUILDDIR)/cuda_workspace.o: $(SRCDIR_CUDA)/cuda_workspace.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -cuda $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

$(BUILDDIR)/mom6_coriolis_cuda.o: $(SRCDIR_CUDA)/mom6_coriolis_cuda.F90 $(BUILDDIR)/cuda_workspace.o | $(BUILDDIR)
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
# CUDA driver compilation (requires nvfortran)
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
# CUDA C kernel compilation (nvcc for .cu, any Fortran for .F90 wrapper)
#==============================================================================

# Shared CUDA C helpers (.cu -> .o via nvcc)
$(BUILDDIR)/cuda_helpers.o: $(SRCDIR_CUDA_C)/cuda_helpers.cu $(SRCDIR_CUDA_C)/cuda_helpers.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@

# Shared Fortran common module (interfaces + alloc helpers)
$(BUILDDIR)/mom6_cuda_c_common.o: $(SRCDIR_CUDA_C)/mom6_cuda_c_common.F90 | $(BUILDDIR)
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

# Coriolis CUDA C kernels
$(BUILDDIR)/mom6_coriolis_kernels.o: $(SRCDIR_CUDA_C)/mom6_coriolis_kernels.cu $(SRCDIR_CUDA_C)/mom6_cuda_common.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@

$(BUILDDIR)/mom6_coriolis_cuda_c.o: $(SRCDIR_CUDA_C)/mom6_coriolis_cuda_c.F90 $(BUILDDIR)/mom6_cuda_c_common.o | $(BUILDDIR)
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

# Horizontal viscosity CUDA C kernels
$(BUILDDIR)/mom6_hor_visc_kernels.o: $(SRCDIR_CUDA_C)/mom6_hor_visc_kernels.cu $(SRCDIR_CUDA_C)/mom6_cuda_common.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@

$(BUILDDIR)/mom6_hor_visc_cuda_c.o: $(SRCDIR_CUDA_C)/mom6_hor_visc_cuda_c.F90 $(BUILDDIR)/mom6_cuda_c_common.o | $(BUILDDIR)
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

# Vertical viscosity CUDA C kernels
$(BUILDDIR)/mom6_vert_visc_kernels.o: $(SRCDIR_CUDA_C)/mom6_vert_visc_kernels.cu $(SRCDIR_CUDA_C)/mom6_cuda_common.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@

$(BUILDDIR)/mom6_vert_visc_cuda_c.o: $(SRCDIR_CUDA_C)/mom6_vert_visc_cuda_c.F90 $(BUILDDIR)/mom6_cuda_c_common.o | $(BUILDDIR)
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

# Barotropic CUDA C kernels
$(BUILDDIR)/mom6_barotropic_kernels.o: $(SRCDIR_CUDA_C)/mom6_barotropic_kernels.cu $(SRCDIR_CUDA_C)/mom6_cuda_common.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@

$(BUILDDIR)/mom6_barotropic_cuda_c.o: $(SRCDIR_CUDA_C)/mom6_barotropic_cuda_c.F90 $(BUILDDIR)/mom6_cuda_c_common.o | $(BUILDDIR)
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

# Continuity CUDA C kernels
$(BUILDDIR)/mom6_continuity_kernels.o: $(SRCDIR_CUDA_C)/mom6_continuity_kernels.cu $(SRCDIR_CUDA_C)/mom6_cuda_common.h | $(BUILDDIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@

$(BUILDDIR)/mom6_continuity_cuda_c.o: $(SRCDIR_CUDA_C)/mom6_continuity_cuda_c.F90 $(BUILDDIR)/mom6_cuda_c_common.o | $(BUILDDIR)
	$(FC) $(FFLAGS) $(MODFLAGS) -c $< -o $@ $(MODFLAG) $(BUILDDIR)

# CUDA C driver: OMP target for data mgmt + CUDA C kernels for all physics
# No OMP physics modules needed — all physics use CUDA C kernels
$(BUILDDIR)/rk2_cuda_c_driver: $(APPDIR)/rk2_cuda_c_driver.F90 $(COMMON_MODULES) $(CUDA_C_MODULES)
	$(FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(COMMON_MODULES) $(CUDA_C_MODULES) $(LDFLAGS) -lstdc++ -lcudart

# MPI + CUDA C driver: OMP target for data mgmt + CUDA C kernels + MPI halo exchange
$(BUILDDIR)/rk2_mpi_cuda_c_driver: $(APPDIR)/rk2_mpi_cuda_c_driver.F90 $(COMMON_MODULES) $(CUDA_C_MODULES) $(MPI_OMP_MODULES)
	$(MPI_FC) $(FFLAGS) $(MODFLAGS) -o $@ $< $(COMMON_MODULES) $(CUDA_C_MODULES) $(MPI_OMP_MODULES) $(LDFLAGS) -lstdc++ -lcudart

#==============================================================================
# MPI + CUDA driver compilation (requires nvfortran)
#==============================================================================

$(BUILDDIR)/rk2_mpi_cuda_driver: $(APPDIR)/rk2_mpi_cuda_driver.F90 $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_hor_visc.o $(CUDA_MODULES) $(MPI_CUDA_MODULES)
	$(MPI_FC) $(FFLAGS) -cuda $(MODFLAGS) -o $@ $< $(BUILDDIR)/mom6_types.o $(BUILDDIR)/mom6_profiler.o $(BUILDDIR)/mom6_barotropic.o $(BUILDDIR)/mom6_hor_visc.o $(CUDA_MODULES) $(MPI_CUDA_MODULES) $(LDFLAGS) -cuda

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
	@echo "Compiler:    $(FC)"
	@echo "GPU backend: $(GPU_BACKEND)"
	@echo "FFLAGS:      $(FFLAGS)"
	@echo "CPPFLAGS:    $(CPPFLAGS)"
	@echo "LDFLAGS:     $(LDFLAGS)"
	@echo "========================================"
	@echo "Default target: omp (OpenMP target offloading)"
	@echo "For OpenACC: make GPU_BACKEND=openacc FC=nvfortran single-gpu"
	@echo "========================================"
