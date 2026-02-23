#!/bin/bash
# Build all 6 vertical viscosity loop ordering variants
#
# Usage:
#   ./build_vertvisc_variants.sh                    # gfortran CPU
#   ./build_vertvisc_variants.sh nvfortran yes      # nvfortran GPU
#   ./build_vertvisc_variants.sh nvfortran no       # nvfortran CPU
#
# Produces: vert_visc_driver_jik, vert_visc_driver_ijk, etc.

FC=${1:-gfortran}
GPU=${2:-no}

VARIANTS="jik ijk jki ikj kji kij"

echo "========================================"
echo "Building all vertical viscosity variants"
echo "Compiler: $FC  GPU: $GPU"
echo "========================================"
echo ""

for v in $VARIANTS; do
    echo "--- Building variant: $v ---"
    make clean -s 2>/dev/null
    make FC=$FC GPU=$GPU VERTVISC_VARIANT=$v vert_visc_driver 2>&1 | tail -1
    if [ $? -eq 0 ]; then
        mv vert_visc_driver vert_visc_driver_$v
        echo "  -> vert_visc_driver_$v"
    else
        echo "  FAILED to build $v"
    fi
    echo ""
done

echo "========================================"
echo "Built executables:"
ls -la vert_visc_driver_* 2>/dev/null
echo ""
echo "Run comparison:"
echo "  for v in $VARIANTS; do echo \"=== \$v ===\"; ./vert_visc_driver_\$v 180 180 75 5; done"
echo "========================================"
