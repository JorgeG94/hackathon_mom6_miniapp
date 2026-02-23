#!/bin/bash
# Run all 6 vertical viscosity variant drivers and collect timing results
#
# Usage:
#   ./run_vertvisc_variants.sh                     # defaults: 1024 1024 75 5
#   ./run_vertvisc_variants.sh 1024 1024 75 10     # custom params
#
# Results saved to vertvisc_results.log
# Grep timing:  grep "Total time" vertvisc_results.log

NI=${1:-1024}
NJ=${2:-1024}
NK=${3:-75}
NITER=${4:-5}

LOGFILE="vertvisc_results.log"
VARIANTS="jik ijk jki ikj kji kij jki_t"

echo "Running vert_visc variants: ni=$NI nj=$NJ nk=$NK niter=$NITER" | tee "$LOGFILE"
echo "Date: $(date)" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

for v in $VARIANTS; do
    exe="./vert_visc_driver_$v"
    if [ ! -f "$exe" ]; then
        echo "SKIP $v (not built)" | tee -a "$LOGFILE"
        continue
    fi
    echo "" | tee -a "$LOGFILE"
    echo "=== $v ===" | tee -a "$LOGFILE"
    $exe $NI $NJ $NK $NITER 2>&1 | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"
echo "Summary:" | tee -a "$LOGFILE"
grep -E "^=== |Total time" "$LOGFILE" | tee -a /dev/null
