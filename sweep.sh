#!/bin/bash
# sweep.sh — Run profiling across models and batch sizes for Phase 1
#
# Each run is a separate process because compile() caches the traced graph,
# so changing batch size requires a fresh process.
#
# Usage:
#   bash sweep.sh                       # all models → results/
#   bash sweep.sh Resnet152             # one model  → results/
#   bash sweep.sh all results_noac/     # all models → results_noac/
#   bash sweep.sh Bert results_ac/      # one model  → results_ac/
#
# If a run OOMs, it prints an error and continues to the next.

set -e

RESNET152_BATCH_SIZES="2 4 8 16 32 64"
BERT_BATCH_SIZES="2 4 8 16 32"

FILTER=${1:-all}
OUTDIR=${2:-results}

mkdir -p "$OUTDIR"

run_sweep() {
    local model=$1
    shift
    local batch_sizes=("$@")

    echo "=========================================="
    echo "  Sweeping $model → $OUTDIR/"
    echo "=========================================="

    for bs in "${batch_sizes[@]}"; do
        echo ""
        echo "--- $model  batch_size=$bs ---"
        if python benchmarks.py "$model" "$bs"; then
            # move generated JSON into output directory
            mv results_${model}_bs${bs}_*.json "$OUTDIR/" 2>/dev/null
            echo "--- $model bs=$bs completed ---"
        else
            echo "--- $model bs=$bs FAILED (likely OOM) ---"
        fi
    done
}

if [ "$FILTER" = "all" ] || [ "$FILTER" = "Resnet152" ]; then
    run_sweep Resnet152 $RESNET152_BATCH_SIZES
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "Bert" ]; then
    run_sweep Bert $BERT_BATCH_SIZES
fi

echo ""
echo "=========================================="
echo "  Sweep complete. Results in $OUTDIR/:"
echo "=========================================="
ls -1 "$OUTDIR"/*.json 2>/dev/null || echo "No result files found."
