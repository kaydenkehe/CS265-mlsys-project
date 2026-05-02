#!/usr/bin/env python3
"""
phase1_analysis.py — Compile Phase 1 profiling results into analysis JSONs.

Reads all per-batch-size result files for a model, computes cross-batch-size
statistics and activation analysis, and outputs a single comprehensive JSON.

Usage:
    python phase1_analysis.py results/ results/phase1/
"""

import json
import os
import sys
import glob


def load(path):
    with open(path) as f:
        return json.load(f)


def analyze_model(files):
    """Analyze all batch-size results for a single model."""
    # sort by batch size
    datasets = []
    for f in files:
        d = load(f)
        d["_file"] = os.path.basename(f)
        datasets.append(d)
    datasets.sort(key=lambda d: d["summary"]["batch_size"])

    model_name = datasets[0]["summary"]["model_name"]

    # ── 1. Model overview (from any batch size — graph structure is constant) ──
    ref = datasets[0]
    total_nodes = len(ref["nodes"])
    num_activations = len(ref["activations"])
    type_counts = {}
    for n in ref["nodes"]:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

    # unique operator types
    op_set = set()
    for n in ref["nodes"]:
        op_set.add(n["op"])

    overview = {
        "model_name": model_name,
        "total_graph_nodes": total_nodes,
        "num_activations": num_activations,
        "node_type_counts": type_counts,
        "unique_op_types": len(op_set),
    }

    # ── 2. Per-batch-size summary ──
    batch_summaries = []
    for d in datasets:
        s = d["summary"]
        m = d["memory_breakdown"]
        bs = s["batch_size"]
        total_pos_mem = m["param_bytes"] + m["activation_bytes"] + m["gradient_bytes"] + m["other_bytes"]
        act_pct = (m["activation_bytes"] / total_pos_mem * 100) if total_pos_mem > 0 else 0

        batch_summaries.append({
            "batch_size": bs,
            "forward_time_ms": round(s["forward_time_ms"], 2),
            "backward_time_ms": round(s["backward_time_ms"], 2),
            "total_time_ms": round(s["total_time_ms"], 2),
            "backward_to_forward_ratio": round(s["backward_time_ms"] / s["forward_time_ms"], 2) if s["forward_time_ms"] > 0 else 0,
            "peak_gpu_memory_mb": round(s["peak_gpu_memory_bytes"] / 1024**2, 1),
            "peak_graph_memory_mb": round(s["peak_graph_memory_bytes"] / 1024**2, 1),
            "peak_node": s["peak_graph_memory_node"],
            "activation_memory_mb": round(m["activation_bytes"] / 1024**2, 1),
            "gradient_memory_mb": round(m["gradient_bytes"] / 1024**2, 1),
            "other_memory_mb": round(m["other_bytes"] / 1024**2, 1),
            "activation_pct_of_positive_allocs": round(act_pct, 1),
        })

    # ── 3. Scaling analysis ──
    # memory per batch element (should be roughly constant for activations)
    batch_sizes = [b["batch_size"] for b in batch_summaries]
    gpu_mems = [b["peak_gpu_memory_mb"] for b in batch_summaries]
    act_mems = [b["activation_memory_mb"] for b in batch_summaries]

    act_per_sample = []
    gpu_per_sample = []
    for i, bs in enumerate(batch_sizes):
        act_per_sample.append(round(act_mems[i] / bs, 2))
        gpu_per_sample.append(round(gpu_mems[i] / bs, 2))

    # memory scaling factor: how does doubling batch size affect memory?
    mem_doubling_factors = []
    for i in range(1, len(batch_sizes)):
        if batch_sizes[i] == batch_sizes[i-1] * 2 and gpu_mems[i-1] > 0:
            mem_doubling_factors.append({
                "from_bs": batch_sizes[i-1],
                "to_bs": batch_sizes[i],
                "gpu_memory_factor": round(gpu_mems[i] / gpu_mems[i-1], 2),
                "activation_memory_factor": round(act_mems[i] / act_mems[i-1], 2) if act_mems[i-1] > 0 else None,
            })

    scaling = {
        "activation_memory_per_sample_mb": dict(zip([str(b) for b in batch_sizes], act_per_sample)),
        "gpu_memory_per_sample_mb": dict(zip([str(b) for b in batch_sizes], gpu_per_sample)),
        "memory_doubling_factors": mem_doubling_factors,
    }

    # ── 4. Timing analysis ──
    # forward/backward ratio across batch sizes
    timing = {
        "backward_to_forward_ratio_by_bs": {
            str(b["batch_size"]): b["backward_to_forward_ratio"]
            for b in batch_summaries
        },
    }

    # top 20 slowest ops (from a medium batch size)
    mid_idx = len(datasets) // 2
    mid_data = datasets[mid_idx]
    mid_bs = mid_data["summary"]["batch_size"]
    compute_nodes = [n for n in mid_data["nodes"] if n["op"] != "input"]
    compute_nodes.sort(key=lambda n: n["time_ms"], reverse=True)
    timing["top_20_slowest_ops"] = {
        "batch_size": mid_bs,
        "ops": [
            {
                "name": n["name"],
                "op": n["op"],
                "type": n["type"],
                "time_ms": round(n["time_ms"], 4),
            }
            for n in compute_nodes[:20]
        ],
    }

    # op type time distribution (what fraction of time is spent on each op type)
    op_time = {}
    for n in mid_data["nodes"]:
        op = n["op"]
        op_time[op] = op_time.get(op, 0) + n["time_ms"]
    total_time = sum(op_time.values())
    timing["time_by_op_type"] = {
        "batch_size": mid_bs,
        "distribution": {
            op: {"time_ms": round(t, 2), "pct": round(t / total_time * 100, 1)}
            for op, t in sorted(op_time.items(), key=lambda x: -x[1])
        },
    }

    # ── 5. Activation analysis (graph structure — same for all batch sizes) ──
    acts = ref["activations"]

    # idle gap distribution
    gaps = [a["idle_gap"] for a in acts]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    min_gap = min(gaps) if gaps else 0

    # top 15 activations by idle gap
    acts_by_gap = sorted(acts, key=lambda a: a["idle_gap"], reverse=True)
    top_by_gap = [
        {
            "name": a["name"],
            "idle_gap": a["idle_gap"],
            "last_forward_use": a["last_forward_use"],
            "first_backward_use": a["first_backward_use"],
        }
        for a in acts_by_gap[:15]
    ]

    # top 15 activations by size (from a medium batch size for meaningful sizes)
    mid_acts = mid_data["activations"]
    mid_acts_by_size = sorted(mid_acts, key=lambda a: a["size_bytes"], reverse=True)
    top_by_size = [
        {
            "name": a["name"],
            "size_mb": round(a["size_bytes"] / 1024**2, 3),
            "idle_gap": a["idle_gap"],
        }
        for a in mid_acts_by_size[:15]
    ]

    # activations that are both large AND have long idle gaps — best AC candidates
    # score = size_bytes * idle_gap (heuristic for memory-time product)
    mid_acts_scored = []
    for a in mid_acts:
        if a["size_bytes"] > 0 and a["idle_gap"] > 0:
            mid_acts_scored.append({
                "name": a["name"],
                "size_mb": round(a["size_bytes"] / 1024**2, 3),
                "idle_gap": a["idle_gap"],
                "score": round(a["size_bytes"] / 1024**2 * a["idle_gap"], 1),
            })
    mid_acts_scored.sort(key=lambda a: a["score"], reverse=True)

    # how much activation memory could theoretically be saved
    total_act_mem_mid = sum(a["size_bytes"] for a in mid_acts if a["size_bytes"] > 0) / 1024**2
    mid_gpu_peak = mid_data["summary"]["peak_gpu_memory_bytes"] / 1024**2
    theoretical_savings_pct = round(total_act_mem_mid / mid_gpu_peak * 100, 1) if mid_gpu_peak > 0 else 0

    activation_analysis = {
        "idle_gap_stats": {
            "min": min_gap,
            "max": max_gap,
            "mean": round(avg_gap, 1),
            "median": sorted(gaps)[len(gaps) // 2] if gaps else 0,
        },
        "top_15_by_idle_gap": top_by_gap,
        "top_15_by_size": {
            "batch_size": mid_bs,
            "activations": top_by_size,
        },
        "top_15_ac_candidates": {
            "batch_size": mid_bs,
            "note": "Ranked by size_mb * idle_gap — activations that are both large and idle longest",
            "activations": mid_acts_scored[:15],
        },
        "theoretical_max_savings": {
            "batch_size": mid_bs,
            "total_activation_memory_mb": round(total_act_mem_mid, 1),
            "peak_gpu_memory_mb": round(mid_gpu_peak, 1),
            "activation_pct_of_gpu_peak": theoretical_savings_pct,
            "note": "If all activations were checkpointed (recomputed instead of stored), "
                    "this is the maximum memory reduction as a fraction of peak GPU memory.",
        },
    }

    # ── 6. Key observations ──
    # automatically derive some insights
    observations = []

    # activation scaling
    if len(set(act_per_sample)) > 1:
        spread = max(act_per_sample) - min(act_per_sample)
        mean_aps = sum(act_per_sample) / len(act_per_sample)
        if spread / mean_aps < 0.15:
            observations.append("Activation memory scales linearly with batch size (per-sample activation memory is roughly constant).")
        else:
            observations.append(f"Activation memory per sample varies across batch sizes (range: {min(act_per_sample)}-{max(act_per_sample)} MB).")

    # backward/forward ratio
    ratios = [b["backward_to_forward_ratio"] for b in batch_summaries]
    avg_ratio = sum(ratios) / len(ratios)
    observations.append(f"Backward pass takes ~{avg_ratio:.1f}x longer than forward on average.")

    # activation percentage
    act_pcts = [b["activation_pct_of_positive_allocs"] for b in batch_summaries]
    if max(act_pcts) > 0:
        observations.append(f"Activations account for {min(act_pcts):.0f}-{max(act_pcts):.0f}% of positive memory allocations across batch sizes.")

    observations.append(f"Peak GPU memory at BS={batch_sizes[-1]}: {gpu_mems[-1]:.0f} MB.")
    observations.append(f"Graph has {num_activations} activations with mean idle gap of {avg_gap:.0f} nodes.")

    # ── Assemble ──
    return {
        "overview": overview,
        "batch_size_results": batch_summaries,
        "scaling_analysis": scaling,
        "timing_analysis": timing,
        "activation_analysis": activation_analysis,
        "observations": observations,
    }


def main():
    indir = sys.argv[1] if len(sys.argv) > 1 else "results"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "results/phase1"
    os.makedirs(outdir, exist_ok=True)

    # group files by model
    all_files = glob.glob(os.path.join(indir, "results_*.json"))
    if not all_files:
        print(f"No result files found in {indir}/")
        sys.exit(1)

    models = {}
    for f in all_files:
        d = load(f)
        name = d["summary"]["model_name"]
        if name not in models:
            models[name] = []
        models[name].append(f)

    for model_name, files in models.items():
        print(f"Analyzing {model_name} ({len(files)} batch sizes)...")
        analysis = analyze_model(files)

        outpath = os.path.join(outdir, f"phase1_{model_name}.json")
        with open(outpath, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"  → {outpath}")

    print("Done.")


if __name__ == "__main__":
    main()
