#!/bin/bash
set -euo pipefail

# =============================================================================
# Strong Scaling Benchmark
#
# Runs host (serial, all cores via default), OpenACC GPU, and CUDA GPU
# across increasing grid sizes.
# =============================================================================

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"

GRIDS=("32 32 100", "64 64 100", "128 128 100" "256 256 100" "512 512 100" "1024 1024 100")
NITER=10
BT_NSTEPS=30
DIAG_MODE=0
CSV_FILE="${CSV_FILE:-larger_scaling_results.csv}"
LOG_DIR="${LOG_DIR:-strong_scaling_logs}"

# --- Validate binaries ---
RK2_DRIVER="$BUILD_DIR/rk2_driver"
RK2_CUDA_DRIVER="$BUILD_DIR/rk2_cuda_driver"

if [ ! -x "$RK2_DRIVER" ]; then
    echo "ERROR: $RK2_DRIVER not found or not executable."
    echo "       Build with: cmake --build build"
    exit 1
fi

HAS_CUDA=true
if [ ! -x "$RK2_CUDA_DRIVER" ]; then
    echo "WARNING: $RK2_CUDA_DRIVER not found, skipping CUDA benchmarks."
    HAS_CUDA=false
fi

# --- Setup ---
mkdir -p "$LOG_DIR"
echo "backend,threads,grid,ni,nj,nk,niter,wall_clock,coriolis,hor_visc,vert_visc,barotropic,continuity,time_per_step" > "$CSV_FILE"

echo "=== Strong Scaling Benchmark ==="
echo "Build dir:     $BUILD_DIR"
echo "Grids:         ${GRIDS[*]}"
echo "Iterations:    $NITER"
echo "BT substeps:   $BT_NSTEPS"
echo "CSV output:    $CSV_FILE"
echo "Log dir:       $LOG_DIR"
echo ""

# --- Helper: run binary, capture output, parse timings ---
run_and_parse() {
    local backend="$1"
    local threads="$2"
    local ni="$3"
    local nj="$4"
    local nk="$5"
    local binary="$6"
    shift 6
    # Remaining args are env var assignments

    local grid_label="${ni}x${nj}x${nk}"
    local log_file="$LOG_DIR/${backend}_${grid_label}.log"

    printf "  %-16s grid=%-16s ... " "$backend" "$grid_label"

    # Build the command
    local output
    local rc=0
    if [ $# -gt 0 ]; then
        output=$(env "$@" "$binary" "$ni" "$nj" "$nk" "$NITER" "$BT_NSTEPS" "$DIAG_MODE" 2>&1) || rc=$?
    else
        output=$("$binary" "$ni" "$nj" "$nk" "$NITER" "$BT_NSTEPS" "$DIAG_MODE" 2>&1) || rc=$?
    fi

    # Save raw output
    echo "$output" > "$log_file"

    if [ $rc -ne 0 ]; then
        echo "FAILED (exit code $rc)"
        echo "$backend,$threads,$grid_label,$ni,$nj,$nk,$NITER,FAILED,,,,,," >> "$CSV_FILE"
        return 0  # Don't crash the script
    fi

    # Parse timings from output
    local wall_clock coriolis hor_visc vert_visc barotropic continuity time_per_step

    wall_clock=$(echo "$output" | grep "Compute (wall clock):" | awk '{print $NF}')
    coriolis=$(echo "$output" | grep "Coriolis:" | awk '{print $2}')
    hor_visc=$(echo "$output" | grep "Hor viscosity:" | awk '{print $3}')
    vert_visc=$(echo "$output" | grep "Vert viscosity:" | awk '{print $3}')
    barotropic=$(echo "$output" | grep "Barotropic:" | awk '{print $2}')
    continuity=$(echo "$output" | grep "Continuity:" | awk '{print $2}')
    time_per_step=$(echo "$output" | grep "Time per RK2 step:" | awk '{print $NF}')

    # Validate we got numbers
    if [ -z "$wall_clock" ] || [ -z "$time_per_step" ]; then
        echo "PARSE ERROR (check $log_file)"
        echo "$backend,$threads,$grid_label,$ni,$nj,$nk,$NITER,PARSE_ERROR,,,,,," >> "$CSV_FILE"
        return 0
    fi

    echo "wall=${wall_clock}s  step=${time_per_step}s"

    echo "$backend,$threads,$grid_label,$ni,$nj,$nk,$NITER,$wall_clock,$coriolis,$hor_visc,$vert_visc,$barotropic,$continuity,$time_per_step" >> "$CSV_FILE"
}

# --- Run all configurations ---
for grid in "${GRIDS[@]}"; do
    read -r ni nj nk <<< "$grid"
    grid_label="${ni}x${nj}x${nk}"

    echo ""
    echo "--- Grid: $grid_label ---"

    # 1. Host (serial, uses all cores by default)
    run_and_parse host 1 "$ni" "$nj" "$nk" "$RK2_DRIVER" \
        ACC_DEVICE_TYPE=host OMP_TARGET_OFFLOAD=disabled

    # 2. OpenACC GPU
    run_and_parse openacc-gpu 1 "$ni" "$nj" "$nk" "$RK2_DRIVER" \
        ACC_DEVICE_TYPE=nvidia OMP_TARGET_OFFLOAD=mandatory

    # 3. CUDA GPU
    if $HAS_CUDA; then
        run_and_parse cuda-gpu 1 "$ni" "$nj" "$nk" "$RK2_CUDA_DRIVER"
    fi
done

# --- Print summary table ---
echo ""
echo ""
echo "=== Strong Scaling Results ==="
echo ""

printf "%-16s %-16s %9s %10s %10s %10s %10s %10s %10s\n" \
    "Grid" "Backend" "Wall(s)" "Per-step" "Coriolis" "HorVisc" "VertVisc" "Barotrop" "Contin"
printf "%-16s %-16s %9s %10s %10s %10s %10s %10s %10s\n" \
    "---------------" "---------------" "--------" "---------" "---------" "---------" "---------" "---------" "---------"

tail -n +2 "$CSV_FILE" | while IFS=',' read -r backend threads grid ni nj nk niter wall_clock coriolis hor_visc vert_visc barotropic continuity time_per_step; do
    if [ "$wall_clock" = "FAILED" ] || [ "$wall_clock" = "PARSE_ERROR" ]; then
        printf "%-16s %-16s %9s\n" "$grid" "$backend" "$wall_clock"
    else
        printf "%-16s %-16s %9.4f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f\n" \
            "$grid" "$backend" \
            "$wall_clock" "$time_per_step" \
            "$coriolis" "$hor_visc" "$vert_visc" "$barotropic" "$continuity"
    fi
done

echo ""
echo "Results saved to: $CSV_FILE"
echo "Run logs saved to: $LOG_DIR/"
