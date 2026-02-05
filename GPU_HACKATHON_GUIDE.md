# MOM6 RK2 Dynamics - Hackathon Reference Guide

This document provides a hackathon-focused reference for the MOM6 split RK2 time-stepping driver, with emphasis on the **continuity solver**, **Coriolis force routines**, and **barotropic solver** - all using Fortran `do concurrent` and OpenMP target offloading for GPU parallelization.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [RK2 Time-Stepping Flow](#2-rk2-time-stepping-flow)
3. [Continuity Solver (PPM)](#3-continuity-solver-ppm)
4. [Coriolis Force Routines](#4-coriolis-force-routines)
5. [Vertical Viscosity Solver](#5-vertical-viscosity-solver)
6. [Horizontal Viscosity Solver](#6-horizontal-viscosity-solver)
7. [Barotropic Solver](#7-barotropic-solver)
8. [GPU Parallelization Patterns](#8-gpu-parallelization-patterns)
9. [Key Data Structures](#9-key-data-structures)
10. [Hackathon Entry Points](#10-hackathon-entry-points)

---

## 1. Architecture Overview

### File Structure
```
hackathon_mom6_miniapp/
├── src/
│   ├── mom6_types.F90           # Grid types and initialization
│   ├── mom6_profiler.F90        # Portable profiler with NVTX support
│   ├── mom6_diag.F90            # Simplified diagnostics module
│   ├── mom6_continuity.F90      # PPM continuity solver
│   ├── mom6_coriolis.F90        # Coriolis acceleration
│   ├── mom6_barotropic.F90      # Barotropic solver
│   ├── mom6_vert_visc.F90       # Vertical viscosity (tridiagonal, coupling coef, remnant)
│   └── mom6_hor_visc.F90        # Horizontal viscosity (Laplacian)
├── app/
│   ├── continuity_driver.F90    # Standalone continuity test
│   ├── coriolis_driver.F90      # Standalone Coriolis test
│   ├── barotropic_driver.F90    # Standalone barotropic test
│   ├── vert_visc_driver.F90     # Standalone vertical viscosity test
│   ├── hor_visc_driver.F90      # Standalone horizontal viscosity test
│   └── rk2_driver.F90           # Unified RK2 driver with all solvers
├── Makefile
└── README.md
```

---

## 2. RK2 Time-Stepping Flow

**File:** `src/core/MOM_dynamics_split_RK2.F90`
**Main Subroutine:** `step_MOM_dyn_split_RK2` 

### Algorithm: Split Baroclinic-Barotropic RK2

```
PREDICTOR PHASE:
├──  PressureForce(h, tv, PFu, PFv, ...)           # Pressure gradient - not implemented in miniapp
├──  horizontal_viscosity(u, v, h, ..., diffu, diffv)  # Horizontal viscosity
├──  CorAdCalc(u_av, v_av, h_av, ..., CAu_pred, CAv_pred)  # Coriolis+advection
├──  vertvisc_coef(up, vp, h, ...)                 # Vertical viscosity coeffs (find_coupling_coef)
├──  vertvisc_remnant(visc, visc_rem_u, visc_rem_v, dt, ...)  # BT coupling fractions
├──  continuity(u, v, h, hp, uh, vh, ...)          # First continuity (for BT_cont)
├──  btstep(u, v, eta, dt, ..., u_av, v_av, u_accel_bt, v_accel_bt)  # Barotropic
├──  up = u + dt_pred * (CAu_pred + PFu + diffu + u_accel_bt)  # Predictor update
├──  vertvisc(up, vp, h, ...)                      # Apply vertical viscosity (tridiagonal)
└──  continuity(up, vp, h, hp, uh, vh, ...)        # Main predictor continuity

CORRECTOR PHASE 
├──  horizontal_viscosity(u_av, v_av, h_av, ..., diffu, diffv)  # Horizontal viscosity
├──  CorAdCalc(u_av, v_av, h_av, ..., CAu, CAv)    # Corrector Coriolis
├──  btstep(u, v, eta, dt, ..., u_av, v_av, u_accel_bt, v_accel_bt)  # Barotropic
├──  u = u + dt * (CAu + PFu + diffu + u_accel_bt)  # Corrector update
├──  vertvisc(u, v, h, ...)                         # Final vertical viscosity (tridiagonal)
└──  continuity(u, v, h_tmp, h, uh, vh, ...)        # Final continuity
```

### Key Time-Stepping Parameter
- `dt_pred = dt * CS%be` where `be` (backward extrapolation) is typically **0.5** for RK2 (?)

---

## 3. Continuity Solver (PPM)

**File:** `src/core/MOM_continuity_PPM.F90`
**Main Entry:** `continuity_PPM`

### Algorithm: Directionally-Split PPM

The continuity equation:
```
dh/dt = -div(uh, vh) = -d(uh)/dx - d(vh)/dy
```

#### Call Flow (X-first direction)
```fortran
! 1. PPM reconstruction for zonal edges
call PPM_reconstruction_x(h_in, h_W, h_E, ...)

! 2. Compute zonal mass fluxes
call zonal_mass_flux(u, h_in, uh, ...)       

! 3. Apply zonal convergence
call continuity_zonal_convergence(h_in, h, uh, ...)

! 4. PPM reconstruction for meridional edges (using updated h)
call PPM_reconstruction_y(h, h_S, h_N, ...)              

! 5. Compute meridional mass fluxes
call meridional_mass_flux(v, h, vh, ...)                

! 6. Apply meridional convergence
call continuity_merdional_convergence(h, vh, ...)      
```

### Critical GPU Loops

#### Zonal Convergence 
```fortran
do concurrent (k=1:nz, j=jsh:jeh, i=ish:ieh)
  h(i,j,k) = max( hin(i,j,k) - dt * G%IareaT(i,j) * (uh(I,j,k) - uh(I-1,j,k)), h_min )
enddo
```

#### Meridional Convergence 
```fortran
do concurrent (k=1:nz, j=jsh:jeh, i=ish:ieh)
  h(i,j,k) = max( h(i,j,k) - dt * G%IareaT(i,j) * (vh(i,J,k) - vh(i,J-1,k)), h_min )
enddo
```

#### PPM Flux Element Calculation 
```fortran
! For u > 0 (upwind from left):
CFL = (u * dt) * (G_dy_Cu * G_IareaT)
curv_3 = (h_L + h_R) - 2.0*h
dh = h_L - h_R
uh = tmp * u * (h_R + CFL * (0.5*dh + curv_3*(CFL - 1.5)))
```

### GPU Memory Management 
```fortran
!$omp target enter data map(alloc: h_W, h_E, h_S, h_N)
! ... computational loops ...
!$omp target exit data map(delete: h_W, h_E, h_S, h_N)
```

---

## 4. Coriolis Force Routines

**File:** `src/core/MOM_CoriolisAdv.F90`
**Main Entry:** `CorAdCalc` (lines 125-1043)

### Physics
```
CAu = -(f + zeta) * v_transport/h + d/dx(KE)    [zonal acceleration]
CAv = +(f + zeta) * u_transport/h + d/dy(KE)    [meridional acceleration]
```

### Available Schemes (selectable via `Coriolis_Scheme`)
| Scheme | ID | Properties |
|--------|-----|-----------|
| SADOURNY75_ENERGY | 1 | Energy conserving (default, safest) |
| ARAKAWA_HSU90 | 2 | Energy + local enstrophy |
| ROBUST_ENSTRO | 3 | Enstrophy, robust to vanishing layers |
| SADOURNY75_ENSTRO | 4 | Enstrophy conserving |
| ARAKAWA_LAMB81 | 5 | Energy + enstrophy |
| AL_BLEND | 6 | Adaptive blending |

### Critical GPU Loops

#### Relative Vorticity 
```fortran
do concurrent (J=Jsq:Jeq, I=Isq:Ieq)
  ! Free-slip boundary condition
  rel_vort(I,J) = G%mask2dBu(I,J) * (dvdx(I,J) - dudy(I,J)) * G%IareaBu(I,J)
enddo
```

#### Potential Vorticity 
```fortran
do concurrent (J=Jsq:Jeq, I=Isq:Ieq)
  abs_vort(I,J) = G%CoriolisBu(I,J) + rel_vort(I,J)
  q(I,J) = abs_vort(I,J) * Ih_q(I,J)   ! PV = (f + zeta) / h
enddo
```

#### Coriolis Acceleration - SADOURNY75_ENERGY 
```fortran
do concurrent (k=1:nz, j=js:je, I=Isq:Ieq)
  CAu(I,j,k) = 0.25 * ( (q(I,J) * (vh(i+1,J,k) + vh(i,J,k))) + &
                        (q(I,J-1) * (vh(i,J-1,k) + vh(i+1,J-1,k))) ) * G%IdxCu(I,j)
enddo
```

### GPU Memory Pattern
```fortran
!$omp target enter data map(alloc: dvdx, dudy, hArea_u, hArea_v, rel_vort, ...)
! ... all k-loop computations ...
!$omp target exit data map(delete: dvdx, dudy, hArea_u, hArea_v, rel_vort, ...)
```

---

## 5. Vertical Viscosity Solver

**File:** `src/parameterizations/vertical/MOM_vert_friction.F90`
**Miniapp:** `src/mom6_vert_visc.F90`

### Purpose
Applies implicit vertical viscosity to momentum using a tridiagonal matrix solver. The solver is called multiple times per RK2 timestep.

### Key Functions

#### find_coupling_coef (miniapp: `find_coupling_coef`)
Computes viscous coupling coefficients at each interface based on:
- Interior viscosity (`Kv`)
- Mixed layer enhanced viscosity (`Kv_ml`)
- Bottom boundary layer viscosity (`Kv_bbl`)

```fortran
! Coupling coefficient formula
Kv_tot = Kv_interior
if (z_from_top < Hmix) Kv_tot = Kv_tot + (Kv_ml - Kv) * topfn
if (z_from_bot < 2*Hbbl) Kv_tot = Kv_tot + (Kv_bbl - Kv) * botfn
! MOM6's BBL transition function: botfn = 1 / (1 + 0.09 * z_norm^6)
a_cpl(K) = Kv_tot / h_shear
```

#### vertvisc_remnant (miniapp: `vert_visc_remnant`)
Computes what fraction of barotropic acceleration remains in each layer after viscosity.
Used for barotropic-baroclinic coupling.

```fortran
! Tridiagonal inversion with unit forcing
do concurrent (j=js:je, i=is:ie)
  ! Forward elimination
  b1 = 1.0 / (h(1) + dt * a(2))
  visc_rem(1) = (h(1) + dt * a(2)) * b1
  c1(1) = dt * a(2) * b1

  do k = 2, nz
    b1 = 1.0 / (h(k) + dt * (a(k+1) + a(k) * (1 - c1(k-1))))
    visc_rem(k) = (h(k) + dt * a(k) * visc_rem(k-1)) * b1
    c1(k) = dt * a(k+1) * b1
  end do

  ! Back substitution
  do k = nz-1, 1, -1
    visc_rem(k) = visc_rem(k) + c1(k) * visc_rem(k+1)
  end do
end do
```

Output: `visc_rem(k)` ranges from 0 to 1:
- Near 1.0 = weak viscosity coupling (layer retains most BT acceleration)
- Near 0.0 = strong viscosity coupling (layer smoothed with neighbors)

#### vertvisc / vert_visc_apply (tridiagonal solver)
Solves the implicit diffusion equation:
```
h(k) * u_new(k) = h(k) * u_old(k)
                + dt * a(k+1) * (u_new(k+1) - u_new(k))
                - dt * a(k) * (u_new(k) - u_new(k-1))
```

```fortran
! GPU pattern: parallelize over columns, sequential in k
do concurrent (j=js:je, i=is:ie)
  ! Forward elimination (Thomas algorithm)
  b1 = 1.0 / (h(1) + dt * a(2))
  u(1) = h(1) * u(1) * b1
  c1(1) = dt * a(2) * b1

  do k = 2, nz
    b1 = 1.0 / (h(k) + dt * (a(k+1) + a(k) * (1 - c1(k-1))))
    u(k) = (h(k) * u(k) + dt * a(k) * u(k-1)) * b1
    c1(k) = dt * a(k+1) * b1
  end do

  ! Back substitution
  do k = nz-1, 1, -1
    u(k) = u(k) + c1(k) * u(k+1)
  end do
end do
```

---

## 6. Horizontal Viscosity Solver

**File:** `src/parameterizations/lateral/MOM_hor_visc.F90`
**Miniapp:** `src/mom6_hor_visc.F90`

### Purpose
Applies Laplacian horizontal viscosity to momentum, smoothing horizontal velocity gradients. Called in the corrector phase of RK2 before Coriolis (line 951 in `step_MOM_dyn_split_RK2`).

### Physics
The Laplacian viscosity operator:
```
diffu = Kh * ∇²u = Kh * (∂²u/∂x² + ∂²u/∂y²)
diffv = Kh * ∇²v = Kh * (∂²v/∂x² + ∂²v/∂y²)
```

### Algorithm (Simplified 5-Point Stencil)

```fortran
! Laplacian of u at Cu points (i-1/2, j)
do concurrent (k=1:nz, j=js:je, i=is:ie-1)
  d2u_dx2 = (u(i+1,j,k) - 2.0*u(i,j,k) + u(i-1,j,k)) * Idx2
  d2u_dy2 = (u(i,j+1,k) - 2.0*u(i,j,k) + u(i,j-1,k)) * Idy2
  diffu(i,j,k) = Kh * (d2u_dx2 + d2u_dy2)
end do

! Laplacian of v at Cv points (i, j-1/2)
do concurrent (k=1:nz, j=js:je-1, i=is:ie)
  d2v_dx2 = (v(i+1,j,k) - 2.0*v(i,j,k) + v(i-1,j,k)) * Idx2
  d2v_dy2 = (v(i,j+1,k) - 2.0*v(i,j,k) + v(i,j-1,k)) * Idy2
  diffv(i,j,k) = Kh * (d2v_dx2 + d2v_dy2)
end do
```

### Integration in RK2 Driver

Horizontal viscosity is called before Coriolis in both predictor and corrector phases:

```fortran
! Predictor phase
call hor_visc(u, v, h, diffu, diffv, G, GV, hvisc_CS)
call CorAdCalc(u, v, h, uh, vh, CAu, CAv, G, GV, cor_CS)
up = u + dt * (CAu + diffu)
vp = v + dt * (CAv + diffv)

! Corrector phase
call hor_visc(up, vp, h, diffu, diffv, G, GV, hvisc_CS)
call CorAdCalc(up, vp, h, uh, vh, CAu, CAv, G, GV, cor_CS)
u = u + 0.5*dt * (CAu + diffu)
v = v + 0.5*dt * (CAv + diffv)
```

### GPU Optimization Notes

1. **Pure stencil operation**: Each point is independent - highly parallelizable
2. **Memory bandwidth bound**: Simple arithmetic with 5-point stencil
3. **k-j-i ordering**: NEED to test what the best ordering is
4. **No dependencies**: Unlike tridiagonal solvers, all points can compute in parallel

### Key Parameters
- `Kh`: Laplacian viscosity coefficient [m²/s], typically 10-1000 m²/s
- For uniform grid: `Idx2 = 1/dx²`, `Idy2 = 1/dy²`

---

## 7. Barotropic Solver

**File:** `src/core/MOM_barotropic.F90`
**Miniapp:** `src/mom6_barotropic.F90`
**Main Entry:** `btstep` (large subroutine)

### Purpose
Solves the fast barotropic (depth-averaged) shallow water equations:
- Computes free surface height evolution (eta)
- Provides time-averaged velocities (u_av, v_av) for baroclinic modes
- Calculates barotropic accelerations (u_accel_bt, v_accel_bt)

### Key Outputs Used by RK2 Driver
```fortran
call btstep(u, v, eta, dt, u_bc_accel, v_bc_accel, forces, pbce, eta_PF, &
            u_av, v_av,       &  ! Time-averaged velocities [OUTPUT]
            u_accel_bt, v_accel_bt, &  ! Barotropic accelerations [OUTPUT]
            eta_pred, uhbt, vhbt, ...)  ! Predicted eta and BT fluxes [OUTPUT]
```

### Interaction with Continuity
- `uhbt, vhbt` (barotropic mass fluxes) are passed to continuity solver
- Continuity uses these to constrain layer-by-layer transports
- Mass conservation: `sum(uh) ≈ uhbt`, `sum(vh) ≈ vhbt`

### Supporting Routines
```fortran
call btcalc(h, G, GV, CS)                    ! Calculate BT layer weights
call bt_mass_source(h, eta, .true., G, GV, CS)  ! Mass source/sink terms
```

---

## 8. GPU Parallelization Patterns


### Pattern 1: Simple `do concurrent` with k-j-i ordering
```fortran
! Most common pattern - k outermost for coalesced memory access
do concurrent (k=1:nz, j=jsh:jeh, i=ish:ieh)
  h(i,j,k) = hin(i,j,k) - dt * (flux_out - flux_in)
enddo
```

### Pattern 2: Nested `do concurrent` (j-loop outer)
```fortran
! Used when j-dependent computations needed first
do concurrent (j=jsh:jeh)
  ! j-specific setup
  do concurrent (k=1:nz, i=ish:ieh)
    ! compute
  enddo
enddo
```

### Pattern 3: Conditional `do concurrent`
```fortran
! With mask/boundary condition
do concurrent (k=1:nz, I=ish-1:ieh, OBC%segnum_u(I,j) /= 0)
  ! Only execute where OBC exists
enddo
```

### Pattern 4: OpenMP Target Data Management
```fortran
! Allocate on device
!$omp target enter data map(alloc: array1, array2, array3)

! Transfer to device
!$omp target update to(array1)

! Compute (do concurrent runs on device)
do concurrent (k=1:nz, j=js:je, i=is:ie)
  array2(i,j,k) = f(array1(i,j,k))
enddo

! Transfer from device
!$omp target update from(array2)

! Deallocate
!$omp target exit data map(delete: array1, array2, array3)
```

### Important Performance Note (from MOM_continuity_PPM.F90)
Need to check this!
```fortran
! "poor performance for nvfortran + do concurrent if k is inside loop"
! Prefer: do concurrent (k=1:nz, j=js:je, i=is:ie)  [k outermost]
! Over:   do concurrent (j=js:je, i=is:ie, k=1:nz)  [k innermost]
```

---

## 9. Key Data Structures

### Grid Types
```fortran
type(ocean_grid_type)    :: G   ! Horizontal grid (areas, masks, metrics)
type(verticalGrid_type)  :: GV  ! Vertical grid (layer counts, thickness units)
type(unit_scale_type)    :: US  ! Dimensional unit scaling
```

### Important Grid Fields
```fortran
G%IareaT(i,j)    ! Inverse of tracer cell area [L-2]
G%IdxCu(I,j)     ! Inverse of u-point dx [L-1]
G%IdyCv(i,J)     ! Inverse of v-point dy [L-1]
G%CoriolisBu(I,J) ! Coriolis parameter at vorticity points [T-1]
G%mask2dBu(I,J)  ! Mask at vorticity points
```

### State Variables (3D)
| Variable | Staggering | Units | Description |
|----------|------------|-------|-------------|
| `h(i,j,k)` | h-point (tracer) | H | Layer thickness |
| `u(I,j,k)` | u-point (zonal face) | L T-1 | Zonal velocity |
| `v(i,J,k)` | v-point (merid face) | L T-1 | Meridional velocity |
| `uh(I,j,k)` | u-point | H L2 T-1 | Zonal mass flux |
| `vh(i,J,k)` | v-point | H L2 T-1 | Meridional mass flux |

### Control Structures
```fortran
type(MOM_dyn_split_RK2_CS)  ! RK2 driver state
type(continuity_PPM_CS)      ! Continuity solver config
type(CoriolisAdv_CS)         ! Coriolis scheme selection
type(barotropic_CS)          ! Barotropic solver state
type(hor_visc_CS)            ! Horizontal viscosity state
type(vert_visc_CS)           ! Vertical viscosity state
```

---

## 10. Hackathon Entry Points

### For Continuity Solver Work

**Start here:** `src/core/MOM_continuity_PPM.F90`

| Subroutine | Lines | Focus |
|------------|-------|-------|
| `continuity_PPM` | 88-202 | Main entry, direction splitting |
| `zonal_mass_flux` | 522-764 | Zonal PPM fluxes |
| `meridional_mass_flux` | 768-1010 | Meridional PPM fluxes |
| `continuity_zonal_convergence` | 356-392 | Thickness update (zonal) |
| `continuity_merdional_convergence` | 395-431 | Thickness update (meridional) |
| `PPM_reconstruction_x` | 2600-2740 | Edge value reconstruction |
| `flux_elem` | 1035-1098 | Elemental flux calculation |


### For Coriolis Work

**Start here:** `src/core/MOM_CoriolisAdv.F90`

| Subroutine | Lines | Focus |
|------------|-------|-------|
| `CorAdCalc` | 125-1043 | Main Coriolis+advection |
| `gradKE` | 1047-1150+ | Kinetic energy gradient |

**Key sections within CorAdCalc:**
- Vorticity calculation: lines 506-546
- Arakawa scheme coefficients: lines 570-634
- Zonal acceleration (CAu): lines 693-804
- Meridional acceleration (CAv): lines 815-930


### For Vertical Viscosity Work

**Start here:** `src/parameterizations/vertical/MOM_vert_friction.F90`
**Miniapp:** `src/mom6_vert_visc.F90`

| Subroutine | Lines | Focus |
|------------|-------|-------|
| `vertvisc_coef` | 1297-2107 | Compute coupling coefficients |
| `find_coupling_coef` | 2622-3128 | Per-interface coefficient calculation |
| `find_coupling_coef_k` | 2112-2616 | GPU-optimized (pure) version |
| `vertvisc_remnant` | 1187-1291 | BT coupling remnant fractions |
| `vertvisc` | main | Tridiagonal solver application |


### For Horizontal Viscosity Work

**Start here:** `src/parameterizations/lateral/MOM_hor_visc.F90`
**Miniapp:** `src/mom6_hor_visc.F90`

| Subroutine | Purpose |
|------------|---------|
| `hor_visc_init` | Initialize control structure and pre-compute metrics |
| `hor_visc` | Compute Laplacian viscous accelerations |
| `hor_visc_end` | Cleanup and deallocate |


### For Barotropic Solver Work

**Start here:** `src/core/MOM_barotropic.F90`
**Miniapp:** `src/mom6_barotropic.F90`

| Subroutine | Purpose |
|------------|---------|
| `btstep` | Main barotropic time-stepping |
| `btcalc` | Layer weight calculation |
| `bt_mass_source` | Mass source terms |


---

## Notes for Hackathon

1. **Loop ordering matters**: We really need to test this
2. **Data movement**: Minimize `!$omp target update` by keeping data on device
3. **Elemental functions**: `flux_elem` is designed for vectorization
4. **Direction alternation**: `G%first_direction` alternates x-first vs y-first each step
5. **Mass conservation**: Continuity solver ensures `sum(uh)` matches barotropic `uhbt`
6. **Tridiagonal solvers**: Column-independent - ideal for GPU, but watch register pressure
7. **Vertical viscosity remnant**: Values near 1.0 = weak coupling, near 0.0 = strong coupling
8. **BBL transition**: MOM6 uses `botfn = 1/(1 + 0.09*z^6)` for smooth bottom boundary layer
9. **Horizontal viscosity**: Simple 5-point Laplacian stencil, memory bandwidth bound
10. **Use the profiler**: Run `rk2_driver` first to identify which kernels to optimize
11. **Start with barotropic**: It's 35% of runtime with 239 `do concurrent` loops - biggest impact

## Running the Miniapps

```bash
# Build with gfortran (CPU)
make FC=gfortran GPU=no

# Build with nvfortran (GPU)
make FC=nvfortran GPU=yes

# Build with NVTX for Nsight Systems profiling
make FC=nvfortran GPU=yes NVTX=yes

# Run individual drivers
./continuity_driver 180 180 75 10
./coriolis_driver 180 180 75 10
./barotropic_driver 180 180 30 10
./vert_visc_driver 180 180 75 10
./hor_visc_driver 180 180 75 10
./rk2_driver 180 180 75 10 30

# Run all tests
make run-all

# Scaling tests
make scale-test
```

### Running GPU Code on Host (Without Recompiling)

Use environment variables to run GPU-compiled binaries on the CPU for debugging:

```bash
# OpenMP Target (used by this miniapp)
export OMP_TARGET_OFFLOAD=DISABLED
./rk2_driver 180 180 75 10 30

# OpenACC (for do concurrent with -stdpar)
export ACC_DEVICE_TYPE=host       # Serial on host
export ACC_DEVICE_TYPE=multicore  # Multicore CPU
export ACC_DEVICE_TYPE=NVIDIA     # NVIDIA GPU
```

---

## Profiler Output

The `rk2_driver` includes a built-in profiler that reports timing for each kernel, sorted by percentage:

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
  D2H_copy_results                   1.46       1     3.2
  HorVisc                            1.20     200     2.7
  Transports                         1.07     200     2.4
  VelUpdate_pred                     0.65     100     1.4
  VelUpdate_corr                     0.64     100     1.4
  ...
------------------------------------------------------------
  Total:                           45.25
============================================================
```

### NVTX Integration

When built with `NVTX=yes`, the profiler emits NVTX ranges visible in Nsight Systems:

```bash
make FC=nvfortran GPU=yes NVTX=yes
nsys profile -o profile_report ./rk2_driver 360 360 75 10 30
nsys-ui profile_report.nsys-rep
```

