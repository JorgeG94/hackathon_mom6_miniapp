# MOM6 Miniapp — Claude Code Guide

## What This Is

A performance-portable miniapp extracting core physics kernels from GFDL's MOM6 ocean model.
Implements a split RK2 time-stepping scheme with 6 physics modules across 3 GPU backends
(OpenMP target offloading, OpenACC, CUDA Fortran) plus MPI for multi-GPU distribution.

## Quick Reference

### Build

```bash
source env.sh                              # nvfortran + MPI on PATH
make FC=nvfortran GPU_BACKEND=openacc single-gpu  # OpenACC single-GPU
make FC=nvfortran GPU_BACKEND=openacc mpi         # OpenACC + MPI
make FC=nvfortran mpi-cuda                        # CUDA + MPI
make FC=nvfortran mpi-cuda-c                      # CUDA C + MPI
make FC=gfortran                                  # OpenMP (default, also works with ifx, amdflang)
make FC=gfortran mpi-omp                          # OpenMP + MPI
make clean                                        # Remove build/
```

The default `make` builds the OpenMP target backend (`-DUSE_OMP_OFFLOAD`).
OpenACC and CUDA require `nvfortran`. OpenMP works with gfortran, ifx, amdflang.

### Run

```bash
./build/rk2_omp_driver              # Single-GPU OpenMP
./build/rk2_driver                  # Single-GPU OpenACC
./build/rk2_cuda_driver             # Single-GPU CUDA
mpirun -np 4 ./build/rk2_mpi_driver # 4-GPU MPI+OpenACC
```

Default grid: 180x180x75, 10 iterations, dt=300s, 30 barotropic substeps.

### Verify

Drivers print mass conservation check at exit. Pass: relative error < 1e-10.
Also prints max |u|, max |v|, max |eta|, total KE per iteration.

## Project Layout

```
src/common/          Shared infrastructure (all backends)
  mom6_types.F90       Grid types, constants, GPU data init (#ifdef USE_OMP_OFFLOAD)
  mom6_profiler.F90    Timing + NVTX integration
  mom6_diag.F90        Diagnostic output system
  mom6_mpi_domain.F90  2D Cartesian MPI decomposition
  mom6_mpi_halo.F90    Halo exchange (OpenACC, host-staging or GPU-aware MPI)

src/openacc/         OpenACC physics (nvfortran only)
  mom6_continuity.F90 + mom6_continuity_adjust.F90  (PPM continuity, submodule at -O1)
  mom6_coriolis.F90        Coriolis + KE gradient
  mom6_barotropic.F90      Barotropic substep solver
  mom6_vert_visc.F90       Vertical viscosity (tridiagonal)
  mom6_hor_visc.F90        Horizontal viscosity (Laplacian/biharmonic/Smagorinsky)

src/openmp/          OpenMP target offloading physics (portable: nvidia/amd/intel)
  Same 6 modules as openacc/ with _omp suffix
  mom6_mpi_halo_omp.F90   Halo exchange with OMP target directives

src/cuda/            CUDA Fortran physics (nvfortran only)
  cuda_workspace.F90       Shared device memory pool
  5 physics modules with _cuda suffix
  mom6_mpi_halo_cuda.F90   Halo exchange with cudafor

app/                 Drivers
  rk2_driver.F90           Single-GPU OpenACC
  rk2_omp_driver.F90       Single-GPU OpenMP target
  rk2_cuda_driver.F90      Single-GPU CUDA
  rk2_mpi_driver.F90       Multi-GPU MPI+OpenACC
  rk2_mpi_omp_driver.F90   Multi-GPU MPI+OpenMP
  rk2_mpi_cuda_driver.F90   Multi-GPU MPI+CUDA
  rk2_mpi_cuda_c_driver.F90 Multi-GPU MPI+CUDA C
  module_drivers/           Per-kernel standalone benchmark drivers

scripts/             Benchmark/scaling scripts + plotting
```

## Architecture & Patterns

### Control Structure (CS) Pattern

Every physics module follows this structure:
1. **`*_CS` derived type** — holds all work arrays, parameters, and `nbytes` for memory tracking
2. **`*_init(CS, G, GV)`** — allocates arrays, copies CS + arrays to GPU
3. **`*_end(CS)`** — deletes from GPU, deallocates
4. **Main compute subroutine** — uses `!$acc data present(...)` or `!$omp target` to operate on GPU-resident data

### GPU Data Lifecycle

```fortran
allocate(array(isd:ied, jsd:jed, nk))
!$acc enter data copyin(array)     ! or !$omp target enter data map(to: array)
! ... use in kernels (present) ...
!$acc exit data delete(array)      ! or !$omp target exit data map(delete: array)
deallocate(array)
```

The `#ifdef USE_OMP_OFFLOAD` preprocessor guard in `mom6_types.F90` switches between OMP and ACC directives for common infrastructure. Physics modules in `src/openacc/` use ACC only; `src/openmp/` use OMP only.

### Grid Indexing (Arakawa C-grid)

- **Data domain**: `isd:ied, jsd:jed` (includes halos)
- **Compute domain**: `isc:iec, jsc:jec`
- **Single-GPU**: `isd=1, ied=ni+2, isc=4, iec=ni-1` (HALO_WIDTH=7 but data domain = ni+2)
- **MPI**: `isd=1, ied=ni_local+2*halo, isc=halo+1, iec=ni_local+halo` (halo=3)
- Variables at h-points (cell centers), u-points (east faces), v-points (north faces), q-points (corners)

### MPI Design

- Physics modules are completely MPI-unaware — all MPI logic is in infrastructure + drivers
- Domain decomposition: `MPI_Cart_create` for 2D Cartesian topology (non-periodic)
- Halo exchange: pack on GPU → MPI send/recv → unpack on GPU (8 directions)
- Barotropic substeps use split API (`btstep_init_state`/`do_step`/`get_output`) with halo_width=1 exchange between substeps
- GPU assignment: `node_rank` from `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`
- Global index reconstruction: `i_global = i + MD%i_offset`

### CUDA Modules

- Use `attributes(global)` kernel subroutines with explicit block/thread indexing
- `cuda_workspace.F90` provides a shared device memory pool to reduce GPU memory footprint
- Thread mapping: `i = (blockIdx%x-1)*blockDim%x + threadIdx%x`, same for j; `k = blockIdx%z`

## Conventions When Modifying Code

### Adding a New Physics Module

1. Implement in `src/openacc/mom6_<name>.F90` following the CS pattern (init/compute/end)
2. Port to `src/openmp/mom6_<name>_omp.F90` (replace `!$acc` → `!$omp target`)
3. Port to `src/cuda/mom6_<name>_cuda.F90` (explicit CUDA kernels)
4. Add build rules to `Makefile` in the appropriate sections
5. Integrate into the RK2 drivers (all 6 variants if it affects the timestep)
6. Add a standalone driver in `app/module_drivers/` for isolated benchmarking

### Adding OpenACC/OpenMP Kernels

- Use `!$acc parallel loop collapse(N)` / `!$omp target teams distribute parallel do collapse(N)`
- Always operate on the compute domain `is:ie, js:je` with stencil reads from data domain
- Ensure all arrays are `present` (already on GPU) before kernel launch
- For 3D arrays, collapse over `(k, j, i)` — k is the outermost loop

### Adding CUDA Kernels

- Define `attributes(global) subroutine` with `value` intent scalars for dimensions
- Use `cuda_workspace` for temporary arrays when possible
- Block size: typically `(16, 16, 1)` for 2D; `(16, 16)` with `blockIdx%z` for 3D
- Keep 1-based Fortran indexing

### Halo Exchange

- After any stencil computation that needs neighbor data, call `halo_exchange_2d/3d`
- Standard physics uses `halo_width` = HALO_WIDTH (7 for single-GPU, 3 for MPI)
- Barotropic substeps use `halo_width=1`
- GPU-aware MPI toggled via `MOM6_GPU_AWARE_MPI=1` environment variable

### Build Rules

- Common modules: compiled with `$(FC)`
- MPI modules: compiled with `$(MPI_FC)` (mpif90 wrapper)
- CUDA modules: need `-cuda` flag
- `mom6_continuity_adjust` (submodule): compiled at `-O1` to avoid nvfortran codegen bug
- Module dependency order: types → profiler → diag → physics → drivers

## Known Quirks

- Single-GPU grid uses asymmetric halo: HALO_WIDTH=7 but data domain is only ni+2 (not ni+14)
- Barotropic `uhbt`/`vhbt` compute ranges are extended by 1 cell on each side (`is-1:ie, js-1:je`) for MPI boundary transport divergence
- No dedicated test suite — verification is via mass conservation + state statistics in drivers
- CUDA driver Makefile links both OpenACC and CUDA `.o` for barotropic/hor_visc (transitive dependency), but the driver uses the `_cuda` modules
