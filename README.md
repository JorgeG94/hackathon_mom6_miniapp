# MOM6 GPU Mini-Apps

Standalone mini-applications extracted from MOM6 for GPU porting and benchmarking. The primary driver (`rk2_driver`) implements the full split RK2 time-stepping algorithm combining all solvers; individual module drivers are available for focused testing.

## Building

### CMake (recommended)

```bash
# CPU (gfortran)
cmake -B build && cmake --build build

# GPU (nvfortran, OpenACC + do concurrent)
cmake -B build -DCMAKE_Fortran_COMPILER=nvfortran && cmake --build build

# GPU with CUDA Fortran kernels
cmake -B build -DCMAKE_Fortran_COMPILER=nvfortran -DMOM6_ENABLE_CUDA_KERNELS=ON && cmake --build build

# MPI + OpenACC (multi-GPU)
cmake -B build -DCMAKE_Fortran_COMPILER=mpif90 -DMOM6_ENABLE_MPI=ON && cmake --build build

# MPI + CUDA Fortran (multi-GPU)
cmake -B build -DCMAKE_Fortran_COMPILER=mpif90 -DMOM6_ENABLE_MPI=ON -DMOM6_ENABLE_CUDA_KERNELS=ON && cmake --build build

# Also build individual module drivers (continuity, coriolis, etc.)
cmake -B build -DMOM6_ENABLE_MODULE_DRIVERS=ON && cmake --build build
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `MOM6_ENABLE_CUDA_KERNELS` | OFF | Build CUDA Fortran kernel variants (requires nvfortran) |
| `MOM6_ENABLE_MPI` | OFF | Build MPI-parallel drivers for multi-GPU/multi-node (requires MPI Fortran) |
| `MOM6_ENABLE_MODULE_DRIVERS` | OFF | Build individual module drivers |

### Make

```bash
# Single GPU
make FC=nvfortran                   # OpenACC
make cuda FC=nvfortran              # CUDA Fortran

# Multi-GPU (MPI) — requires: source env.sh
make mpi FC=nvfortran               # OpenACC + MPI
make mpi-cuda FC=nvfortran          # CUDA Fortran + MPI

# CPU
make FC=gfortran                    # gfortran
make FC=ifx                         # Intel
```

### Fortran Package Manager

```bash
fpm install --prefix . --compiler nvfortran --flag "-O3 -acc=multicore,gpu -gpu=mem:separate"
```

## Running

### RK2 Driver (primary benchmark)

```bash
./rk2_driver [ni] [nj] [nk] [niter] [bt_nsteps] [diag]

# Examples
./rk2_driver 180 180 75 10 30       # Default-ish grid
./rk2_driver 720 720 75 10 50       # Large grid, diagnostics off
./rk2_driver 720 720 75 10 50 1     # Large grid, diagnostics on
```

**Parameters:**
- `ni`, `nj`: Horizontal grid dimensions
- `nk`: Vertical layers
- `niter`: Outer iterations for timing
- `bt_nsteps`: Barotropic substeps per iteration (typically 30–100)
- `diag`: Diagnostics mode (0=disabled, 1=enabled; default 0)

### MPI Drivers (multi-GPU)

```bash
mpirun -np N ./rk2_mpi_driver ni nj nk niter bt_nsteps [npes_x npes_y]
mpirun -np N ./rk2_mpi_cuda_driver ni nj nk niter bt_nsteps [npes_x npes_y]

# Examples
mpirun -np 4 ./rk2_mpi_driver 180 180 75 10 30           # 4 GPUs, auto layout
mpirun -np 4 ./rk2_mpi_cuda_driver 360 360 75 10 30      # CUDA, 4 GPUs
mpirun -np 6 ./rk2_mpi_driver 360 360 75 10 30 3 2       # 3x2 PE layout

# Multi-node
mpirun -np 8 --npernode 4 ./rk2_mpi_cuda_driver 720 720 75 10 30
```

**Additional parameters:**
- `npes_x`, `npes_y`: PE layout (optional). If omitted, `MPI_Dims_create` auto-decomposes.
  `npes_x * npes_y` must equal the number of MPI ranks.

### GPU-Aware MPI

By default, halo exchanges use host staging (GPU→host→MPI→host→GPU). To enable GPU-aware MPI, which passes device pointers directly to MPI and eliminates the intermediate memory copies, set the environment variable:

```bash
export MOM6_GPU_AWARE_MPI=1
mpirun -np 4 ./rk2_mpi_cuda_driver 360 360 75 10 30
```

This requires an MPI library built with CUDA support (e.g., OpenMPI+UCX or MVAPICH2-GDR). A diagnostic message is printed at startup confirming which path is active:
```
[MPI Halo CUDA] GPU-aware MPI: ENABLED
```

### Module Drivers (with `-DMOM6_ENABLE_MODULE_DRIVERS=ON`)

```bash
./continuity_driver [ni] [nj] [nk] [niter]
./coriolis_driver [ni] [nj] [nk] [niter] [scheme]
./barotropic_driver [ni] [nj] [nstep] [niter]
./vert_visc_driver [ni] [nj] [nk] [niter]
./hor_visc_driver [ni] [nj] [nk] [niter] [Kh]
```

### Running GPU Binaries on CPU

```bash
export OMP_TARGET_OFFLOAD=DISABLED    # OpenMP target (used by this miniapp)
./rk2_driver 180 180 75 10 30

export ACC_DEVICE_TYPE=host           # OpenACC do concurrent
export ACC_DEVICE_TYPE=multicore      # OpenACC multicore
```

## Architecture

### Relationship to MOM6

These mini-apps capture the computational kernels from `step_MOM_dyn_split_RK2`:

```
step_MOM_dyn_split_RK2
├── PressureForce          (not included)
├── horizontal_viscosity   ← mom6_hor_visc
├── CorAdCalc              ← mom6_coriolis
├── vertvisc_coef          ← mom6_vert_visc (find_coupling_coef)
├── vertvisc_remnant       ← mom6_vert_visc (BT coupling)
├── continuity             ← mom6_continuity
├── btstep                 ← mom6_barotropic
└── vertvisc               ← mom6_vert_visc (tridiagonal apply)
```

### RK2 Time-Stepping Flow

```
PREDICTOR:
  horizontal_viscosity → CorAdCalc → vertvisc_coef → vertvisc_remnant
  → continuity → btstep → velocity update → vertvisc → continuity

CORRECTOR:
  horizontal_viscosity → CorAdCalc → btstep → velocity update
  → vertvisc → continuity
```

### Call Frequency per Baroclinic Timestep

| Routine | Calls | Substeps | GPU Kernels |
|---------|-------|----------|-------------|
| horizontal_viscosity | 2 | 1 | ~4 |
| continuity | 3 | 1 | ~18 |
| CorAdCalc | 2–3 | 1 | ~20 |
| vertvisc_coef | 2 | 1 | ~6 |
| vertvisc | 4 | 1 | ~8 |
| btstep | 2 | 30–100 each | **~600** |

The barotropic solver dominates GPU workload.

## Solvers

| Solver | Source | Key Pattern | Parallelism |
|--------|--------|-------------|-------------|
| **Continuity** | `MOM_continuity_PPM.F90` | PPM reconstruction, CFL-based flux | 3D (i,j,k) |
| **Coriolis** | `MOM_CoriolisAdv.F90` | Vorticity, PV, Coriolis acceleration | Per-layer 2D |
| **Barotropic** | `MOM_barotropic.F90` | Sub-stepped shallow water equations | 2D (i,j) × nstep |
| **Vertical viscosity** | `MOM_vert_friction.F90` | Tridiagonal solver, coupling coefficients | Column-independent |
| **Horizontal viscosity** | `MOM_hor_visc.F90` | 5-point Laplacian stencil | 3D (i,j,k) |

### Coriolis Schemes

| ID | Scheme | Properties |
|----|--------|------------|
| 1 | SADOURNY75_ENERGY | Energy conserving (default) |
| 2 | ARAKAWA_HSU90 | Energy + local enstrophy |
| 3 | ARAKAWA_LAMB81 | Energy + enstrophy |

## GPU Parallelization Patterns

### 3D stencils (k-j-i ordering)
```fortran
do concurrent (k=1:nk, j=1:nj, i=1:ni)
  h(i,j,k) = hin(i,j,k) - dt * IareaT(i,j) * (uh(i+1,j,k) - uh(i,j,k))
end do
```

### 2D barotropic sub-stepping (data stays on device)
```fortran
do n = 1, nstep
  do concurrent (j=2:nj+1, i=2:ni+1)
    eta(i,j) = eta(i,j) - dtbt * IareaT(i,j) * (uhbt(i,j) - uhbt(i-1,j) + vhbt(i,j) - vhbt(i,j-1))
  end do
end do
```

### Column-independent tridiagonal (vertical viscosity)
```fortran
do concurrent (j=js:je, i=is:ie)
  ! Forward elimination + back substitution in k (sequential)
end do
```

### OpenMP target data management
```fortran
!$omp target enter data map(to: input_arrays)
!$omp target enter data map(alloc: work_arrays)
! ... do concurrent computation ...
!$omp target exit data map(from: output_arrays)
!$omp target exit data map(delete: work_arrays)
```

## Profiler

The built-in profiler reports per-kernel timing. When built with `NVTX=yes`, ranges are visible in Nsight Systems.

```bash
# Profile with Nsight Systems
nsys profile -o report ./rk2_driver 360 360 75 10 30
```

Example output:
```
============================================================
Profiler Report: RK2 Driver
============================================================
  Region                          Time (s)    Calls    %
------------------------------------------------------------
  Barotropic                        15.78     200    34.9
  Coriolis                          14.30     200    31.6
  Continuity                         6.00     200    13.3
  VertVisc                           3.76     200     8.3
  HorVisc                            1.20     200     2.7
  ...
------------------------------------------------------------
  Total:                           45.25
============================================================
```

## Diagnostics

The `rk2_driver` supports a simplified diagnostic system (enabled via the `diag` argument):

| Mode | Value | Description |
|------|-------|-------------|
| `DIAG_NONE` | 0 | Disabled (default) |
| `DIAG_STATS` | 1 | Print min/max/mean/rms |
| `DIAG_FILE` | 2 | Write to binary file |
| `DIAG_BOTH` | 3 | Stats + file output |

## File Structure

```
hackathon_mom6_miniapp/
├── src/
│   ├── common/
│   │   ├── mom6_types.F90              # Grid types and constants
│   │   ├── mom6_profiler.F90           # Portable profiler with NVTX
│   │   ├── mom6_diag.F90              # Simplified diagnostics
│   │   ├── mom6_mpi_domain.F90        # MPI domain decomposition (2D Cartesian)
│   │   └── mom6_mpi_halo.F90         # OpenACC halo exchange (GPU-aware or host-staging)
│   ├── openacc/
│   │   ├── mom6_continuity.F90        # PPM continuity solver
│   │   ├── mom6_continuity_adjust.F90 # Continuity flux adjustment
│   │   ├── mom6_coriolis.F90          # Coriolis acceleration
│   │   ├── mom6_barotropic.F90        # Barotropic solver
│   │   ├── mom6_vert_visc.F90         # Vertical viscosity
│   │   └── mom6_hor_visc.F90          # Horizontal viscosity
│   └── cuda/
│       ├── mom6_continuity_cuda.F90   # CUDA Fortran continuity
│       ├── mom6_coriolis_cuda.F90     # CUDA Fortran Coriolis
│       ├── mom6_barotropic_cuda.F90   # CUDA Fortran barotropic
│       ├── mom6_vert_visc_cuda.F90    # CUDA Fortran vertical viscosity
│       ├── mom6_hor_visc_cuda.F90     # CUDA Fortran horizontal viscosity
│       └── mom6_mpi_halo_cuda.F90    # CUDA halo exchange (GPU-aware or host-staging)
├── app/
│   ├── rk2_driver.F90                 # OpenACC RK2 driver (single GPU)
│   ├── rk2_cuda_driver.F90            # CUDA RK2 driver (single GPU)
│   ├── rk2_mpi_driver.F90            # MPI + OpenACC RK2 driver (multi-GPU)
│   ├── rk2_mpi_cuda_driver.F90       # MPI + CUDA RK2 driver (multi-GPU)
│   └── module_drivers/               # Individual module drivers
├── CMakeLists.txt
├── Makefile
└── fpm.toml
```

## Module Dependencies

```
mom6_types.F90
    ├── mom6_profiler.F90
    ├── mom6_diag.F90
    ├── mom6_continuity.F90
    ├── mom6_continuity_adjust.F90
    ├── mom6_coriolis.F90
    ├── mom6_barotropic.F90
    ├── mom6_vert_visc.F90
    ├── mom6_hor_visc.F90
    └── mom6_mpi_domain.F90          # MPI only
        ├── mom6_mpi_halo.F90        # OpenACC halo exchange
        └── mom6_mpi_halo_cuda.F90   # CUDA halo exchange
```
