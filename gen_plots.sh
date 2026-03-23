#!/bin/bash
# gen_plots.sh — Generate all Phase 1 plots from sweep results
#
# Usage:
#   bash gen_plots.sh                    # results/ → plots/
#   bash gen_plots.sh results_noac/      # custom input dir
#   bash gen_plots.sh results/ my_plots/ # custom input + output dirs

INDIR=${1:-results}
OUTDIR=${2:-plots}

mkdir -p "$OUTDIR/resnet152" "$OUTDIR/bert"

echo "Reading from $INDIR/, writing to $OUTDIR/"
echo ""

# ── 4(a): All single-file plots for one representative batch size ──
echo "--- Generating single-file plots ---"
python plot.py all "$INDIR"/results_Resnet152_bs16_*.json -o "$OUTDIR/resnet152/"
python plot.py all "$INDIR"/results_Bert_bs16_*.json -o "$OUTDIR/bert/"

# ── 4(b): Peak memory vs batch size ──
echo "--- Generating memory vs batch size ---"
python plot.py compare \
    "$INDIR"/results_Resnet152_bs2_*.json "$INDIR"/results_Resnet152_bs4_*.json \
    "$INDIR"/results_Resnet152_bs8_*.json "$INDIR"/results_Resnet152_bs16_*.json \
    "$INDIR"/results_Resnet152_bs32_*.json "$INDIR"/results_Resnet152_bs64_*.json \
    "$INDIR"/results_Resnet152_bs128_*.json "$INDIR"/results_Resnet152_bs256_*.json \
    --metric memory -o "$OUTDIR/resnet152_memory_vs_batch.png"

python plot.py compare \
    "$INDIR"/results_Bert_bs2_*.json "$INDIR"/results_Bert_bs4_*.json \
    "$INDIR"/results_Bert_bs8_*.json "$INDIR"/results_Bert_bs16_*.json \
    "$INDIR"/results_Bert_bs32_*.json "$INDIR"/results_Bert_bs64_*.json \
    "$INDIR"/results_Bert_bs128_*.json "$INDIR"/results_Bert_bs256_*.json \
    --metric memory -o "$OUTDIR/bert_memory_vs_batch.png"

# ── 4(c): Latency vs batch size ──
echo "--- Generating latency vs batch size ---"
python plot.py compare \
    "$INDIR"/results_Resnet152_bs2_*.json "$INDIR"/results_Resnet152_bs4_*.json \
    "$INDIR"/results_Resnet152_bs8_*.json "$INDIR"/results_Resnet152_bs16_*.json \
    "$INDIR"/results_Resnet152_bs32_*.json "$INDIR"/results_Resnet152_bs64_*.json \
    "$INDIR"/results_Resnet152_bs128_*.json "$INDIR"/results_Resnet152_bs256_*.json \
    --metric latency -o "$OUTDIR/resnet152_latency_vs_batch.png"

python plot.py compare \
    "$INDIR"/results_Bert_bs2_*.json "$INDIR"/results_Bert_bs4_*.json \
    "$INDIR"/results_Bert_bs8_*.json "$INDIR"/results_Bert_bs16_*.json \
    "$INDIR"/results_Bert_bs32_*.json "$INDIR"/results_Bert_bs64_*.json \
    "$INDIR"/results_Bert_bs128_*.json "$INDIR"/results_Bert_bs256_*.json \
    --metric latency -o "$OUTDIR/bert_latency_vs_batch.png"

echo ""
echo "Done. Plots:"
ls -1 "$OUTDIR"/*.png "$OUTDIR"/resnet152/*.png "$OUTDIR"/bert/*.png 2>/dev/null
