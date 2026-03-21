#!/bin/bash
# sweep.sh — Run profiling across models and batch sizes for Phase 1
#
# Each run is a separate process because compile() caches the traced graph,
# so changing batch size requires a fresh process.
#
# Usage:
#   bash sweep.sh              # run all configurations
#   bash sweep.sh Resnet152    # run only ResNet-152
#   bash sweep.sh Bert         # run only BERT
#
# Results are saved as JSON files in the current directory.
# If a run OOMs, it will print an error and continue to the next.

set -e

RESNET152_BATCH_SIZES="2 4 8 16 32"
BERT_BATCH_SIZES="2 4 8 16"

run_sweep() {
    local model=$1
    shift
    local batch_sizes=("$@")

    echo "=========================================="
    echo "  Sweeping $model"
    echo "=========================================="

    for bs in "${batch_sizes[@]}"; do
        echo ""
        echo "--- $model  batch_size=$bs ---"
        if python benchmarks.py "$model" "$bs"; then
            echo "--- $model bs=$bs completed ---"
        else
            echo "--- $model bs=$bs FAILED (likely OOM) ---"
        fi
    done
}

# allow filtering by model name
FILTER=${1:-all}

if [ "$FILTER" = "all" ] || [ "$FILTER" = "Resnet152" ]; then
    run_sweep Resnet152 $RESNET152_BATCH_SIZES
fi

if [ "$FILTER" = "all" ] || [ "$FILTER" = "Bert" ]; then
    run_sweep Bert $BERT_BATCH_SIZES
fi

echo ""
echo "=========================================="
echo "  Sweep complete. JSON results:"
echo "=========================================="
ls -1 results_*.json 2>/dev/null || echo "No result files found."
