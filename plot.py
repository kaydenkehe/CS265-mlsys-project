#!/usr/bin/env python3
"""
plot.py — Visualization for profiling results

Subcommands:
    breakdown    Peak memory breakdown by tensor type
    waterfall    Cumulative memory over graph execution order
    activations  Activation tensor idle gaps
    top-ops      Slowest operations by execution time
    timing       Forward vs backward pass time
    compare      Compare metrics across multiple runs (w/ optional AC comparison)
    all          Generate all single-file plots at once

Examples:
    python plot.py breakdown results.json
    python plot.py waterfall results.json -o waterfall.png
    python plot.py activations results.json --top 20
    python plot.py top-ops results.json --top 10
    python plot.py timing results.json

    # Compare peak memory across batch sizes (Phase 1, no AC yet):
    python plot.py compare bs4.json bs8.json bs16.json --labels 4 8 16 --metric memory

    # Compare with and without AC (after Phase 2+3):
    python plot.py compare noac_bs4.json noac_bs8.json \\
        --labels 4 8 --metric memory --ac ac_bs4.json ac_bs8.json

    # Generate all single-file plots into a directory:
    python plot.py all results.json -o plots/
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# ── Color Palette ────────────────────────────────────────────────────────
#
#    Cohesive palette based on natural tones.
#    - teal / coral are the primary pair (forward / backward, no-AC / AC)
#    - sky / gold / gray round out the type classification
#    - navy is used for neutral line/fill elements

PAL = {
    "teal":  "#2A9D8F",
    "coral": "#E76F51",
    "navy":  "#264653",
    "gold":  "#E9C46A",
    "sky":   "#5B8FB9",
    "gray":  "#AAAAAA",
    "text":  "#333333",
}

TYPE_COLOR = {
    "PARAM": PAL["sky"],
    "ACT":   PAL["coral"],
    "GRAD":  PAL["gold"],
    "OTHER": PAL["gray"],
}

TYPE_LABEL = {
    "PARAM": "Parameters",
    "ACT":   "Activations",
    "GRAD":  "Gradients",
    "OTHER": "Other",
}


# ── Style ────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#CCCCCC",
        "axes.grid":         True,
        "grid.color":        "#EEEEEE",
        "grid.linewidth":    0.6,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "medium",
        "axes.labelsize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
    })


# ── Helpers ──────────────────────────────────────────────────────────────

def _load(path):
    with open(path) as f:
        return json.load(f)


def _done(fig, out):
    fig.tight_layout()
    if out:
        fig.savefig(out, facecolor=fig.get_facecolor())
        print(f"Saved: {out}")
    else:
        plt.show()
    plt.close(fig)


# ── Plot: breakdown ──────────────────────────────────────────────────────

def _model_tag(data):
    """Build a title suffix like ' — Resnet152 (BS=4)' from JSON metadata."""
    name = data.get("summary", {}).get("model_name")
    bs   = data.get("summary", {}).get("batch_size")
    if name and bs is not None:
        return f" \u2014 {name} (BS={bs})"
    if name:
        return f" \u2014 {name}"
    return ""


def breakdown(data, out=None):
    """Peak memory breakdown by tensor type (horizontal bar chart)."""
    mb = data["memory_breakdown"]
    cats = ["Parameters", "Activations", "Gradients", "Other"]
    vals = [
        mb["param_bytes"]      / 1024**2,
        mb["activation_bytes"] / 1024**2,
        mb["gradient_bytes"]   / 1024**2,
        mb["other_bytes"]      / 1024**2,
    ]
    cols = [PAL["sky"], PAL["coral"], PAL["gold"], PAL["gray"]]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(cats, vals, color=cols, edgecolor="white", height=0.55)

    for bar, v in zip(bars, vals):
        pct = v / total * 100 if total else 0
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {v:.1f} MB  ({pct:.1f}%)",
                va="center", fontsize=10, color=PAL["text"])

    ax.set_xlabel("Memory (MB)")
    ax.set_title("Peak Memory Breakdown" + _model_tag(data))
    ax.set_xlim(0, max(vals) * 1.4 if vals else 1)
    ax.invert_yaxis()
    ax.grid(axis="x", visible=False)
    _done(fig, out)


# ── Plot: waterfall ──────────────────────────────────────────────────────

def waterfall(data, out=None):
    """Cumulative memory over graph execution order (line + fill chart).

    Shows the 'memory mountain' — memory climbs during the forward pass,
    peaks, then declines during backward. This is the core visualization
    that motivates activation checkpointing.
    """
    nodes = data["nodes"]
    cum, running = [], 0
    for n in nodes:
        running += n["mem_delta_bytes"] / 1024**2
        cum.append(running)

    x = np.arange(len(cum))
    fig, ax = plt.subplots(figsize=(12, 5))

    # main curve
    ax.fill_between(x, cum, alpha=0.12, color=PAL["navy"])
    ax.plot(x, cum, color=PAL["navy"], linewidth=1.0)

    # peak marker
    pk = int(np.argmax(cum))
    ax.plot(pk, cum[pk], "o", color=PAL["coral"], markersize=7, zorder=5)
    ax.annotate(
        f'Peak: {cum[pk]:.1f} MB\n({nodes[pk]["name"]})',
        xy=(pk, cum[pk]),
        xytext=(pk + len(x) * 0.06, cum[pk] * 0.95),
        fontsize=9, color=PAL["coral"],
        arrowprops=dict(arrowstyle="->", color=PAL["coral"], lw=1.0),
    )

    # forward / backward boundary markers
    sep_i = sep_b = None
    for i, n in enumerate(nodes):
        if n["op"] == "sep.sep":
            sep_i = i
        elif n["op"] == "sep.sep_backward":
            sep_b = i

    if sep_i is not None and sep_b is not None:
        ax.axvline(sep_i, color=PAL["teal"], ls="--", lw=0.7, alpha=0.5)
        ax.axvline(sep_b, color=PAL["coral"], ls="--", lw=0.7, alpha=0.5)
        y_label = max(cum) * 0.97
        ax.text(sep_i / 2, y_label, "Forward",
                ha="center", fontsize=10, color=PAL["teal"], alpha=0.7)
        ax.text((sep_b + len(nodes)) / 2, y_label, "Backward",
                ha="center", fontsize=10, color=PAL["coral"], alpha=0.7)

    ax.set_xlabel("Execution Order")
    ax.set_ylabel("Cumulative Memory (MB)")
    ax.set_title("Memory Over Graph Execution" + _model_tag(data))
    ax.set_xlim(0, len(x) - 1)
    _done(fig, out)


# ── Plot: activations ────────────────────────────────────────────────────

def activations(data, out=None, top=None):
    """Activation idle gaps (horizontal bar chart).

    Sorted by idle gap (largest first). Color intensity encodes tensor size —
    darker bars are larger tensors, making it easy to spot the biggest
    memory-saving opportunities for activation checkpointing.
    """
    acts = sorted(data["activations"], key=lambda a: a["idle_gap"], reverse=True)
    if not acts:
        print("No activation data.")
        return
    if top:
        acts = acts[:top]

    names = [a["name"] for a in acts]
    gaps  = [a["idle_gap"] for a in acts]
    sizes = [a["size_bytes"] / 1024**2 for a in acts]
    mx_s  = max(sizes) if max(sizes) > 0 else 1

    # color intensity proportional to tensor size (OrRd colormap)
    cols = [plt.cm.OrRd(0.25 + 0.55 * s / mx_s) for s in sizes]

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(names) * 0.32)))
    bars = ax.barh(names, gaps, color=cols, edgecolor="white", height=0.7)

    mx_g = max(gaps) if gaps else 1
    for bar, g, s in zip(bars, gaps, sizes):
        ax.text(bar.get_width() + mx_g * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{g}  ({s:.2f} MB)",
                va="center", fontsize=8, color=PAL["text"])

    ax.set_xlabel("Idle Gap (nodes)")
    ax.set_title("Activation Idle Gaps  (color intensity \u2192 tensor size)" + _model_tag(data))
    ax.set_xlim(0, mx_g * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="x", visible=False)
    _done(fig, out)


# ── Plot: top-ops ────────────────────────────────────────────────────────

def top_ops(data, out=None, top=15):
    """Top-N slowest operations by execution time (horizontal bar chart).

    Bars are colored by node type so you can see whether the bottleneck is
    in forward computation, backward gradients, or optimizer steps.
    """
    ns = [n for n in data["nodes"] if n["op"] != "input"]
    ns = sorted(ns, key=lambda n: n["time_ms"], reverse=True)[:top]
    if not ns:
        print("No ops to plot.")
        return

    labels = [f'{n["name"]}  ({n["op"]})' for n in ns]
    times  = [n["time_ms"] for n in ns]
    cols   = [TYPE_COLOR.get(n["type"], PAL["gray"]) for n in ns]

    fig, ax = plt.subplots(figsize=(10, max(3.5, len(labels) * 0.35)))
    bars = ax.barh(labels, times, color=cols, edgecolor="white", height=0.7)

    mx = max(times)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + mx * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.3f} ms",
                va="center", fontsize=8, color=PAL["text"])

    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Top {len(ns)} Slowest Operations" + _model_tag(data))
    ax.set_xlim(0, mx * 1.2)
    ax.invert_yaxis()
    ax.grid(axis="x", visible=False)

    handles = [Patch(facecolor=c, label=TYPE_LABEL[t]) for t, c in TYPE_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    _done(fig, out)


# ── Plot: timing ─────────────────────────────────────────────────────────

def timing(data, out=None):
    """Forward vs backward pass time (bar chart)."""
    s = data["summary"]
    fwd, bwd = s["forward_time_ms"], s["backward_time_ms"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Forward", "Backward"], [fwd, bwd],
        color=[PAL["teal"], PAL["coral"]],
        edgecolor="white", width=0.45,
    )
    mx = max(fwd, bwd)
    for bar, v in zip(bars, [fwd, bwd]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + mx * 0.02,
                f"{v:.2f} ms",
                ha="center", fontsize=11, color=PAL["text"])

    ax.set_ylabel("Time (ms)")
    ax.set_title("Forward vs Backward Time" + _model_tag(data))
    ax.set_ylim(0, mx * 1.15)
    ax.grid(axis="x", visible=False)
    _done(fig, out)


# ── Plot: compare ────────────────────────────────────────────────────────

def compare(files, labels, metric, ac_files=None, out=None):
    """Compare a metric across runs.

    Without --ac:  simple bar chart (one bar per file).
    With --ac:     grouped bar chart (No AC vs With AC, side by side).

    Handles both deliverable 4(b) (peak memory vs batch size) and
    4(c) (iteration latency vs batch size).

    If labels is None, auto-derives from batch_size in JSON files.
    """
    datasets = [_load(f) for f in files]

    def _val(d):
        if metric == "memory":
            return d["summary"]["peak_gpu_memory_bytes"] / 1024**2
        elif metric == "forward":
            return d["summary"]["forward_time_ms"]
        elif metric == "backward":
            return d["summary"]["backward_time_ms"]
        return d["summary"]["total_time_ms"]

    # auto-derive labels from JSON batch_size if not provided
    if labels is None:
        labels = []
        for d in datasets:
            bs = d["summary"].get("batch_size")
            labels.append(str(bs) if bs is not None else "?")

    vals = [_val(d) for d in datasets]
    ylabel_map = {
        "memory": "Peak Memory (MB)",
        "latency": "Latency (ms)",
        "forward": "Forward Time (ms)",
        "backward": "Backward Time (ms)",
    }
    title_map = {
        "memory": "Peak Memory",
        "latency": "Iteration Latency",
        "forward": "Forward Pass Time",
        "backward": "Backward Pass Time",
    }
    ylabel = ylabel_map[metric]

    # include model name in title if all files share the same model
    model_names = {d["summary"].get("model_name") for d in datasets}
    model_str = model_names.pop() if len(model_names) == 1 and None not in model_names else None
    base_title = title_map[metric]
    title = f"{base_title} vs Batch Size — {model_str}" if model_str else f"{base_title} vs Batch Size"

    if ac_files:
        # ── Grouped bar chart: No AC vs With AC ──
        ac_vals = [_val(_load(f)) for f in ac_files]
        x  = np.arange(len(labels))
        w  = 0.32
        mx = max(max(vals), max(ac_vals))

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 5))
        b1 = ax.bar(x - w / 2, vals,    w, label="No AC",
                     color=PAL["teal"], edgecolor="white")
        b2 = ax.bar(x + w / 2, ac_vals, w, label="With AC",
                     color=PAL["coral"], edgecolor="white")

        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mx * 0.01,
                    f"{bar.get_height():.1f}",
                    ha="center", fontsize=9, color=PAL["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon=False)
    else:
        # ── Simple bar chart ──
        mx = max(vals) if vals else 1
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.2), 5))
        bars = ax.bar(labels, vals,
                      color=PAL["teal"], edgecolor="white", width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mx * 0.01,
                    f"{bar.get_height():.1f}",
                    ha="center", fontsize=9, color=PAL["text"])

    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, mx * 1.15)
    ax.grid(axis="x", visible=False)
    _done(fig, out)


# ── Subcommand: all ──────────────────────────────────────────────────────

def all_plots(data, out_dir=None):
    """Generate all single-file plots at once."""
    plots = {
        "breakdown":   lambda o: breakdown(data, o),
        "waterfall":   lambda o: waterfall(data, o),
        "activations": lambda o: activations(data, o),
        "top_ops":     lambda o: top_ops(data, o),
        "timing":      lambda o: timing(data, o),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, fn in plots.items():
            fn(os.path.join(out_dir, f"{name}.png"))
    else:
        for fn in plots.values():
            fn(None)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    _style()

    p = argparse.ArgumentParser(
        description="Plot profiling results from graph_prof.py JSON output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # single-file subcommands (no extra args)
    for name in ("breakdown", "waterfall", "timing"):
        s = sub.add_parser(name, help=f"Generate {name} plot from a single results file")
        s.add_argument("file", help="JSON results file")
        s.add_argument("-o", "--output", help="Save plot to file (omit to show interactively)")

    # single-file subcommands (with --top)
    for name, default in [("activations", 20), ("top-ops", 15)]:
        s = sub.add_parser(name, help=f"Generate {name} plot from a single results file")
        s.add_argument("file", help="JSON results file")
        s.add_argument("--top", type=int, default=default,
                        help=f"Number of entries to show (default: {default})")
        s.add_argument("-o", "--output", help="Save plot to file")

    # compare subcommand
    s = sub.add_parser("compare",
        help="Compare metrics across multiple runs (e.g. different batch sizes)")
    s.add_argument("files", nargs="+", help="JSON results files (one per configuration)")
    s.add_argument("--labels", nargs="+", default=None,
                    help="Label for each file (e.g. batch sizes: 4 8 16). "
                         "Auto-derived from JSON batch_size if omitted.")
    s.add_argument("--metric", choices=["memory", "latency", "forward", "backward"],
                    required=True, help="Which metric to compare")
    s.add_argument("--ac", nargs="+", metavar="FILE",
                    help="AC result files for grouped No-AC vs With-AC comparison")
    s.add_argument("-o", "--output", help="Save plot to file")

    # all subcommand
    s = sub.add_parser("all", help="Generate all single-file plots at once")
    s.add_argument("file", help="JSON results file")
    s.add_argument("-o", "--output", help="Output directory for PNG files")

    args = p.parse_args()

    if args.cmd in ("breakdown", "waterfall", "timing"):
        fn = {"breakdown": breakdown, "waterfall": waterfall, "timing": timing}[args.cmd]
        fn(_load(args.file), args.output)

    elif args.cmd == "activations":
        activations(_load(args.file), args.output, args.top)

    elif args.cmd == "top-ops":
        top_ops(_load(args.file), args.output, args.top)

    elif args.cmd == "compare":
        if args.labels and len(args.labels) != len(args.files):
            sys.exit(f"Error: got {len(args.files)} files but {len(args.labels)} labels")
        if args.ac and len(args.ac) != len(args.files):
            sys.exit(f"Error: --ac has {len(args.ac)} files but need {len(args.files)}")
        compare(args.files, args.labels, args.metric, args.ac, args.output)

    elif args.cmd == "all":
        all_plots(_load(args.file), args.output)


if __name__ == "__main__":
    main()
