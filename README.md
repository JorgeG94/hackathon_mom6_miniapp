# MOM6 GPU Mini-Apps

Standalone mini-applications extracted from MOM6 for GPU benchmarking and hackathon development.

## Overview

These mini-apps capture the computational patterns and bottlenecks of MOM6's core dynamics:

| Mini-App | Source | Key Patterns | Dimensions |
|----------|--------|--------------|------------|
| `continuity_driver` | `MOM_continuity_PPM.F90` | PPM reconstruction, flux calculation, convergence | 3D (i,j,k) |
| `coriolis_driver` | `MOM_CoriolisAdv.F90` | Vorticity, PV, Coriolis acceleration | 3D (i,j,k) |
| `barotropic_driver` | `MOM_barotropic.F90` | Fast barotropic sub-stepping, pressure gradient, Coriolis | 2D (i,j) × nstep |
| `vert_visc_driver` | `MOM_vert_friction.F90` | Tridiagonal solver, coupling coefficients, BBL/ML viscosity | 3D columns |
| `hor_visc_driver` | `MOM_hor_visc.F90` | Laplacian horizontal viscosity, strain rates | 3D (i,j,k) |
| `rk2_driver` | `MOM_dynamics_split_RK2.F90` | Unified RK2 time-stepping with all solvers | 3D + 2D |

## Building

### Using the Fortran Package Manager (please try it and let me know any qualms) 

```
git clone git@github.com:fortran-lang/fpm.git 
cd fpm 
./install.sh 
export PATH=$PATH:$HOME/.local/bin
```

Then just:

```
fpm install --prefix . --compiler nvfortran --flag "-O3 -stdpar=multicore,gpu -mp=multicore,gpu -Minfo=accel"
```

### NVIDIA GPU (nvfortran)
```bash
make FC=nvfortran GPU=yes
```

### NVIDIA GPU with NVTX profiling (for Nsight Systems)
```bash
make FC=nvfortran GPU=yes NVTX=yes
```

### CPU-only (gfortran)
```bash
make FC=gfortran GPU=no
```

### Intel (ifx)
```bash
make FC=ifx GPU=no
```

### Disable profiler entirely (zero overhead)
```bash
make DISABLE_PROFILER=yes
```

### Build info
```bash
make info
```

### CMake Build
```bash
mkdir build && cd build
cmake -DENABLE_GPU=ON ..
make

# With NVTX:
cmake -DENABLE_GPU=ON -DENABLE_NVTX=ON ..
```



## Running on Host (CPU) Without Recompiling

GPU-compiled binaries can be forced to run on the CPU using environment variables. This is useful for debugging or when a GPU is unavailable.

### OpenMP Target (used by this miniapp)
```bash
export OMP_TARGET_OFFLOAD=DISABLED
./rk2_driver 180 180 75 10 30
```

### Open ACC for doconcurrent
```bash
export ACC_DEVICE_TYPE=host       # Serial execution on host
export ACC_DEVICE_TYPE=multicore  # Multicore CPU
export ACC_DEVICE_TYPE=NVIDIA     # NVIDIA GPU
```

### Values for OMP_TARGET_OFFLOAD
| Value | Behavior |
|-------|----------|
| `DISABLED` | Run on host, ignore target directives |
| `MANDATORY` | Fail if offload device unavailable |
| `DEFAULT` | Offload if possible, fall back to host |

## Running

### Default parameters
```bash
./continuity_driver           # 180x180x75 grid, 10 iterations
./coriolis_driver             # 180x180x75 grid, 10 iterations
./barotropic_driver           # 180x180 grid, 30 substeps, 10 iterations
./vert_visc_driver            # 180x180x75 grid, 10 iterations
./hor_visc_driver             # 180x180x75 grid, 10 iterations
./rk2_driver                  # 180x180x75 grid, 10 iterations, 30 BT substeps
```

### Custom parameters
```bash
./continuity_driver [ni] [nj] [nk] [niter]
./coriolis_driver [ni] [nj] [nk] [niter] [scheme]
./barotropic_driver [ni] [nj] [nstep] [niter]
./vert_visc_driver [ni] [nj] [nk] [niter]
./hor_visc_driver [ni] [nj] [nk] [niter] [Kh]
./rk2_driver [ni] [nj] [nk] [niter] [bt_nsteps] [diag]
```

**Parameters:**
- `ni`, `nj`: Horizontal grid dimensions
- `nk`: Number of vertical layers
- `nstep`: Number of barotropic substeps per outer iteration (barotropic only, typically 30-100 in MOM6)
- `niter`: Number of outer iterations for timing
- `scheme`: Coriolis discretization scheme (coriolis only)
- `Kh`: Laplacian horizontal viscosity coefficient [m²/s] (hor_visc only, default 100)
- `diag`: Diagnostics mode for rk2_driver (0=disabled, 1=enabled; default 0)

**Coriolis schemes:**
- `1` = SADOURNY75_ENERGY (default, energy-conserving)
- `2` = ARAKAWA_HSU90 (energy + local enstrophy)
- `3` = ARAKAWA_LAMB81 (energy + enstrophy)

### Run all tests
```bash
make run-all
```

### Individual tests
```bash
make run-continuity
make run-coriolis
make run-barotropic
make run-vert-visc
make run-hor-visc
make run-rk2
```

### Scaling tests
```bash
make scale-test
```

## Computational Patterns

### Continuity Mini-App

Demonstrates the PPM (Piecewise-Parabolic Method) continuity solver:

1. **PPM Reconstruction** - Compute edge values from cell-centered data with slope limiting
   ```fortran
   do concurrent (k=1:nk, j=1:nj, i=1:ni)
     slp(i,j,k) = 0.5 * (h(i+1,j,k) - h(i-1,j,k))
     ! monotonic limiting
     slp = sign(1.0, slp) * min(abs(slp), 2.0 * min(dMx, dMn))
   end do
   ```

2. **Flux Calculation** - CFL-based PPM interpolation
   ```fortran
   CFL = u * dt * IdxT
   uh = face_area * u * (h_upwind + CFL * (0.5*dh + curv*(CFL - 1.5)))
   ```

3. **Convergence** - Update thickness from flux divergence
   ```fortran
   do concurrent (k=1:nk, j=1:nj, i=1:ni)
     h(i,j,k) = max(h(i,j,k) - dt * IareaT * (uh(i+1) - uh(i)), h_min)
   end do
   ```

### Coriolis Mini-App

Demonstrates vorticity and Coriolis acceleration calculation:

1. **Circulation** - Velocity gradients for vorticity
   ```fortran
   dvdx(i,j) = v(i+1,j)*dy(i+1) - v(i,j)*dy(i)
   dudy(i,j) = u(i,j+1)*dx(j+1) - u(i,j)*dx(j)
   ```

2. **Vorticity** - Relative and absolute vorticity
   ```fortran
   rel_vort = (dvdx - dudy) / area
   abs_vort = f + rel_vort
   q = abs_vort / h_avg  ! potential vorticity
   ```

3. **Coriolis Acceleration** - Multiple schemes available
   ```fortran
   ! Sadourny energy-conserving:
   CAu = 0.25 * (q(i,j)*vh + q(i,j-1)*vh) * Idx - dKE/dx
   CAv = -0.25 * (q(i-1,j)*uh + q(i,j)*uh) * Idy - dKE/dy
   ```

### Vertical Viscosity Mini-App

Demonstrates the implicit vertical viscosity solver from `MOM_vert_friction.F90`:

1. **find_coupling_coef** - Compute viscous coupling coefficients at interfaces
   ```fortran
   ! Accumulate viscosity contributions
   Kv_tot = Kv_interior
   if (z_from_top < Hmix) Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn
   if (z_from_bot < 2*Hbbl) Kv_tot = Kv_tot + (Kv_bbl - Kv) * botfn
   ! botfn = 1 / (1 + 0.09 * z_norm^6)  [MOM6 BBL transition]
   a_cpl = Kv_tot / h_shear
   ```

2. **vertvisc_remnant** - Compute momentum remnant fraction for BT coupling
   ```fortran
   ! Tridiagonal inversion with unit forcing
   do concurrent (j=js:je, i=is:ie)
     ! Forward elimination + back substitution
     ! visc_rem(k) = fraction of BT acceleration retained by layer k
   end do
   ```

3. **Tridiagonal Solver** - Apply implicit viscosity
   ```fortran
   ! Solve: h(k)*u_new = h(k)*u_old + dt*a(k+1)*(u_new(k+1)-u_new(k))
   !                                - dt*a(k)*(u_new(k)-u_new(k-1))
   do concurrent (j=js:je, i=is:ie)
     ! Forward elimination (k=1 to nz)
     ! Back substitution (k=nz-1 to 1)
   end do
   ```

**GPU Pattern:** Each horizontal column (i,j) is independent - parallelize over columns with `collapse(2)`, sequential tridiagonal solve in k.

### Horizontal Viscosity Mini-App

Demonstrates Laplacian horizontal viscosity from `MOM_hor_visc.F90`:

1. **Laplacian Operator** - 5-point stencil for each velocity component
   ```fortran
   ! For u-velocity at Cu points
   do concurrent (k=1:nk, j=js:je, i=is:ie-1)
     d2u_dx2 = (u(i+1,j,k) - 2*u(i,j,k) + u(i-1,j,k)) * Idx2
     d2u_dy2 = (u(i,j+1,k) - 2*u(i,j,k) + u(i,j-1,k)) * Idy2
     diffu(i,j,k) = Kh * (d2u_dx2 + d2u_dy2)
   end do
   ```

2. **Viscous Acceleration** - Applied to momentum equations
   ```fortran
   ! In RK2 time-stepping (before Coriolis)
   up(i,j,k) = u(i,j,k) + dt * (CAu(i,j,k) + diffu(i,j,k))
   ```

**GPU Pattern:** Pure 3D stencil operation with k-j-i ordering. Each point is independent - highly parallelizable.

### Barotropic Mini-App

Demonstrates the fast barotropic (depth-averaged) solver with sub-stepping. This is the **most GPU-intensive** component of MOM6.

**Physics:** Solves the linearized shallow water equations for the depth-averaged (barotropic) mode:
- Free surface height (eta) evolution
- Depth-averaged velocities (ubt, vbt)
- Fast external gravity waves (requires small timestep)

**Algorithm:** Each baroclinic timestep requires 30-100 barotropic substeps:

1. **Eta Predictor** - Estimate free surface height for pressure gradient
   ```fortran
   do concurrent (j=2:nj+1, i=2:ni+1)
     eta_pred(i,j) = eta(i,j) - dtbt * IareaT(i,j) * &
         ((Datu(i,j)*ubt(i,j) - Datu(i-1,j)*ubt(i-1,j)) + &
          (Datv(i,j)*vbt(i,j) - Datv(i,j-1)*vbt(i,j-1)))
   end do
   ```

2. **Pressure Force** - Gradient of eta
   ```fortran
   do concurrent (j=2:nj+1, i=2:ni)
     PFu(i,j) = (eta_pred(i,j)*gtot_E(i,j) - eta_pred(i+1,j)*gtot_W(i+1,j)) * IdxCu(i,j)
   end do
   ```

3. **Coriolis + Velocity Update** - 4-point stencil for f*v and f*u
   ```fortran
   do concurrent (j=2:nj+1, i=2:ni)
     Cor_u(i,j) = (f_4_u(1)*vbt(i,j-1) + f_4_u(2)*vbt(i+1,j-1) + &
                   f_4_u(3)*vbt(i,j) + f_4_u(4)*vbt(i+1,j))
     ubt(i,j) = bt_rem_u(i,j) * (ubt(i,j) + dtbt * (BT_force_u + Cor_u + PFu))
   end do
   ```

4. **Transport + Continuity** - Update eta from divergence
   ```fortran
   do concurrent (j=2:nj+1, i=2:ni+1)
     eta(i,j) = eta(i,j) - dtbt * IareaT(i,j) * &
                ((uhbt(i,j) - uhbt(i-1,j)) + (vhbt(i,j) - vhbt(i,j-1)))
   end do
   ```

**Key features captured:**
- Alternating u/v update order (for rotational symmetry)
- Time filtering of transports (backward Euler weighting)
- Accumulation of time-averaged velocities for baroclinic coupling

## GPU Patterns Demonstrated

### 1. do concurrent with k-j-i ordering (3D kernels)
```fortran
do concurrent (k=1:nk, j=1:nj, i=1:ni)
  ! k outermost for coalesced memory access on GPU
end do
```

### 2. do concurrent for 2D kernels (barotropic)
```fortran
do concurrent (j=2:nj+1, i=2:ni+1)
  ! Pure 2D operations for depth-averaged quantities
end do
```

### 3. OpenMP Target Data Management
```fortran
!$omp target enter data map(to: input_arrays)
!$omp target enter data map(alloc: work_arrays)
! ... computation (data stays on device) ...
!$omp target exit data map(from: output_arrays)
!$omp target exit data map(delete: work_arrays)
```

### 4. Per-Layer Computation (Coriolis pattern)
```fortran
do k = 1, nk
  ! 2D work arrays reused per layer
  do concurrent (j=1:nj, i=1:ni)
    ! 2D computation for layer k
  end do
end do
```

### 5. Time-stepping loop on device (Barotropic pattern)
```fortran
! Data mapped to device before loop
do n = 1, nstep
  ! All substep computation stays on device
  do concurrent (j=2:nj+1, i=2:ni+1)
    ! Update eta, velocities, transports
  end do
end do
! Only final results copied back
```

## Validation

All mini-apps include verification (very ish tho):

- **Continuity**: Checks mass conservation (should be < 1e-10 relative error)
- **Coriolis**: Checks for NaN/Inf and physically reasonable accelerations (< 1 m/s²)
- **Hor Viscosity**: Checks for NaN/Inf and reasonable viscous accelerations
- **Barotropic**: Checks for NaN/Inf, mass conservation, and physically reasonable velocities (< 10 m/s)

## Relationship to Full MOM6

These mini-apps represent the computational kernels called from `step_MOM_dyn_split_RK2`:

```
step_MOM_dyn_split_RK2
├── PressureForce          (not included)
├── horizontal_viscosity   ← hor_visc_driver
├── CorAdCalc              ← coriolis_driver
├── vertvisc_coef          ← vert_visc_driver (find_coupling_coef)
├── vertvisc_remnant       ← vert_visc_driver (for BT coupling)
├── continuity             ← continuity_driver
├── btstep                 ← barotropic_driver
└── vertvisc               ← vert_visc_driver (tridiagonal apply)
```

The `rk2_driver` combines all these solvers in the correct split RK2 sequence.

### Call Frequency per Baroclinic Timestep

| Routine | Calls | Substeps | Total GPU Kernels |
|---------|-------|----------|-------------------|
| horizontal_viscosity | 2 | 1 | ~4 kernels |
| continuity | 3 | 1 | ~18 kernels |
| CorAdCalc | 2-3 | 1 | ~20 kernels |
| vertvisc_coef | 2 | 1 | ~6 kernels |
| vertvisc | 4 | 1 | ~8 kernels |
| btstep | 2 | 30-100 each | **~600 kernels** |

### GPU Parallelization in MOM6

| File | `do concurrent` | `omp target` |
|------|-----------------|--------------|
| MOM_barotropic.F90 | **239** | **93** |
| MOM_continuity_PPM.F90 | 111 | 27 |
| MOM_CoriolisAdv.F90 | 55 | 49 |
| MOM_vert_friction.F90 | 45 | 38 |
| MOM_hor_visc.F90 | 35 | 20 |
| MOM_dynamics_split_RK2.F90 | 21 | 89 |

The barotropic solver is the **dominant GPU workload** because:
1. Most `do concurrent` loops of any MOM6 file

## Diagnostic Module (`mom6_diag.F90`)

Simplified diagnostic system mimicking MOM6's `MOM_diag_mediator.F90` patterns:
(this was a bit awful to design)

### Output Modes
```fortran
DIAG_NONE  = 0   ! Disabled (no computation)
DIAG_STATS = 1   ! Print min/max/mean/rms statistics
DIAG_FILE  = 2   ! Write to binary file
DIAG_BOTH  = 3   ! Stats + file output
```

### Usage
```fortran
use mom6_diag

type(diag_ctrl) :: diag_CS
integer :: id_KE, id_mass

! Initialize
call diag_init(diag_CS, G, GV, output_dir='./', output_freq=1)

! Register diagnostics (returns ID > 0 if active, -1 if disabled)
id_KE = register_diag_field(diag_CS, 'KE', 'Kinetic Energy', 'm2/s2', 3, 'h', DIAG_STATS)
id_mass = register_diag_field(diag_CS, 'mass', 'Total mass', 'm', 2, 'h', DIAG_STATS)

! Post data (skipped if id <= 0)
call post_data_3d(id_KE, KE_array, G, GV, diag_CS)
call post_data_2d(id_mass, mass_array, G, diag_CS)

! MOM6-style product sums (e.g., for momentum budget)
call post_product_sum_u(id, u_a, u_b, G, nz, diag_CS)  ! Vertical sum of u_a * u_b

! Report and cleanup
call diag_report_timing(diag_CS)
call diag_end(diag_CS)
```

### Registered Diagnostics in rk2_driver
| Name | Type | Description |
|------|------|-------------|
| `KE` | 3D h-point | Kinetic energy per layer |
| `diffu_sum` | 2D u-point | Vertically summed u-diffusion |
| `diffv_sum` | 2D v-point | Vertically summed v-diffusion |
| `mass` | 2D h-point | Vertically summed thickness |

---

## Profiler Module (`mom6_profiler.F90`)

Portable profiling wrapper combining NVTX ranges with wall-clock timers.

### Features
- **NVTX ranges** for Nsight Systems GPU timeline (when compiled with `NVTX=yes`)
- **Wall-clock timers** always work (using `system_clock`)
- **Named regions** accumulate time and call counts across iterations
- **Runtime control**: `profiler_enable()` / `profiler_disable()`
- **Zero overhead**: Compile with `DISABLE_PROFILER=yes` for production

### Usage
```fortran
use mom6_profiler

call profiler_init()

do iter = 1, niter
  call profiler_start("RK2_step")

  call profiler_start("Coriolis")
  call CorAdCalc(...)
  call profiler_stop("Coriolis")

  call profiler_start("Continuity")
  call continuity_PPM(...)
  call profiler_stop("Continuity")

  call profiler_stop("RK2_step")
end do

call profiler_report("RK2 Driver")  ! Prints timing summary table
call profiler_end()
```

### Profiled Regions in rk2_driver
| Label | Description | Calls/iter |
|-------|-------------|------------|
| `RK2_step` | Entire iteration | 1 |
| `Transports` | compute_transports | 2 |
| `HorVisc` | Horizontal viscosity | 2 |
| `Coriolis` | CorAdCalc | 2 |
| `VelUpdate_pred` | Predictor velocity update | 1 |
| `VelUpdate_corr` | Corrector velocity update | 1 |
| `VertVisc` | Vertical viscosity | 2 |
| `Barotropic` | btstep | 2 |
| `Continuity` | continuity_PPM | 2 |
| `Diagnostics` | Diagnostic posting (if enabled) | 0-2 |
| `D2H_copy_results` | Device→Host copy of final state | 1 (end) |
| `GPU_dealloc_temps` | Deallocate temporary device arrays | 1 (end) |
| `Finalize_*` | Module cleanup (continuity, coriolis, barotropic, vert_visc, hor_visc, diag, grid) | 1 each (end) |

---

## GPU Data Management

### Current Architecture

The miniapp uses OpenMP target data directives for GPU memory management:

```fortran
! Allocate on device
!$omp target enter data map(alloc: u, v, h, ...)

! Transfer to device
!$omp target update to(u, v, h, ...)

! Computation via do concurrent (offloaded with -stdpar=gpu)
do concurrent (k=1:nk, j=js:je, i=is:ie)
  ...
end do

! Transfer from device
!$omp target update from(u, v, h, ...)

! Deallocate from device
!$omp target exit data map(delete: u, v, h, ...)
```

### `-gpu=mem:separate` Support (NEEDED)

The miniapp fully supports `-gpu=mem:separate` (explicit memory management, recommended for production).

**Data Transfer Pattern** (correct for `-gpu=mem:separate`):

```fortran
! Initialize on HOST with regular loops, then copy to device
do j = ...; do i = ...
  array(i,j) = value
end do; end do
!$omp target update to(array)  ! Host → Device

! Compute on GPU with do concurrent (writes to device memory)
do concurrent (...)
  array(i,j) = computation
end do
! Do NOT use "target update to" here - device already has correct data!

! Before CPU reads, copy from device
!$omp target update from(array)  ! Device → Host
call cpu_routine(array)
```

### Compilation Modes

| Flag | Description | Data Transfers |
|------|-------------|----------------|
| (default) | Managed/unified memory | Automatic |
| `-gpu=mem:separate` | Explicit memory | Manual `!$omp target update` required |
| `-gpu=mem:managed` | Managed memory | Automatic |


---

## Module Dependencies

```
mom6_types.F90          (base types: ocean_grid_type, verticalGrid_type)
    │
    ├── mom6_profiler.F90   (portable profiling, NVTX wrapper)
    │
    ├── mom6_continuity.F90 (PPM continuity solver)
    ├── mom6_coriolis.F90   (Coriolis/momentum advection)
    ├── mom6_barotropic.F90 (fast barotropic solver)
    ├── mom6_vert_visc.F90  (vertical viscosity)
    ├── mom6_hor_visc.F90   (horizontal viscosity)
    │
    └── mom6_diag.F90       (simplified diagnostics)
```

---

## File Structure

```
hackathon_mom6_miniapp/
├── src/
│   ├── mom6_types.F90       # Grid types and constants
│   ├── mom6_profiler.F90    # Portable profiler with NVTX
│   ├── mom6_diag.F90        # Simplified diagnostics
│   ├── mom6_continuity.F90  # PPM continuity solver
│   ├── mom6_coriolis.F90    # Coriolis acceleration
│   ├── mom6_barotropic.F90  # Barotropic solver
│   ├── mom6_vert_visc.F90   # Vertical viscosity
│   └── mom6_hor_visc.F90    # Horizontal viscosity
├── app/
│   ├── rk2_driver.F90       # Unified RK2 driver (main benchmark)
│   ├── continuity_driver.F90
│   ├── coriolis_driver.F90
│   ├── barotropic_driver.F90
│   ├── vert_visc_driver.F90
│   └── hor_visc_driver.F90
├── Makefile
├── CMakeLists.txt
└── README.md
```

---

### Quick Test Commands
```bash
# CPU build and test
make clean && make FC=gfortran GPU=no
./rk2_driver 180 180 75 10 30

# GPU build with NVTX (for Nsight profiling)
make clean && make FC=nvfortran GPU=yes NVTX=yes
nsys profile ./rk2_driver 360 360 75 10 30

# GPU benchmarking (diagnostics disabled for accurate timing)
./rk2_driver 720 720 75 10 50       # Diagnostics OFF (default)
./rk2_driver 720 720 75 10 50 1     # Diagnostics ON

# Check build configuration
make info
```
