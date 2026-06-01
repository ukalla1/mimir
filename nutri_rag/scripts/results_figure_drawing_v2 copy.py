#!/usr/bin/env python3
"""Per-subtask EmbodiedBench-ALFRED comparison (IEEE wide format).

Reads ``summary.json`` from each subtask directory under
``EmbodiedBench/results/eb_alfred/<run_dir>/<subtask>/results/`` and
``EmbodiedBench_atlasmodified/results/eb_alfred/<run_dir>/<subtask>/results/``
and produces a 2x3 grid of subplots — one per ALFRED subtask.

Each subplot:
    X axis      : model (IQ2_M, Q4_K_S)
    Left  Y     : task success rate (%) — paired bars
                  (EmbodiedBench vs EmbodiedBench_atlasmodified)
    Right Y     : model file size (GB) — scatter marker per model

Usage:
    python nutri_rag/scripts/results_figure_drawing_v2.py
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

RESULTS_ROOTS = {
    "eb":  os.path.join(REPO_ROOT, "EmbodiedBench",              "results", "eb_alfred"),
    "mod": os.path.join(REPO_ROOT, "EmbodiedBench_atlasmodified", "results", "eb_alfred"),
}

SUBTASKS = [
    "base",
    "common_sense",
    "complex_instruction",
    "long_horizon",
    "spatial",
    "visual_appearance",
]

SUBTASK_TITLES = {
    "base":                "Base",
    "common_sense":        "Common Sense",
    "complex_instruction": "Complex Instruction",
    "long_horizon":        "Long Horizon",
    "spatial":             "Spatial",
    "visual_appearance":   "Visual Appearance",
}

# (display label, GGUF size GB, EB run-dir, modified run-dir)
MODELS = [
    ("IQ2_M",  3.4,
     "Qwen3-VL-9B-GGUF_qwen35_iq2m_alf_full",
     "Qwen3-VL-9B-GGUF_qwen35_iq2m_alf_full_memB_v2"),
    ("Q4_K_S", 5.1,
     "Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full",
     "Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full_memB_v2"),
]

COLOR_EB  = "#7a7a7a"   # gray  — EmbodiedBench (baseline)
COLOR_MOD = "#3b6ea8"   # blue  — atlasmodified
SIZE_MARKER_COLOR = "#c0392b"  # red — model size scatter


def load_success(root_key, run_dir, subtask):
    path = os.path.join(RESULTS_ROOTS[root_key], run_dir, subtask, "results", "summary.json")
    if not os.path.isfile(path):
        print(f"  warn: missing {path}", file=sys.stderr)
        return None
    with open(path) as f:
        return float(json.load(f)["task_success"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "nutri_rag", "results", "embodied_per_subtask"),
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharey=False)

    bar_width = 0.35
    x_positions = np.arange(len(MODELS))  # one tick per model

    # Track right-axis max so we can share its limit across all subplots
    size_max = max(m[1] for m in MODELS)

    for ax, subtask in zip(axes.flat, SUBTASKS):
        eb_vals  = []
        mod_vals = []
        sizes    = []
        for _, size_gb, eb_dir, mod_dir in MODELS:
            eb_succ  = load_success("eb",  eb_dir,  subtask)
            mod_succ = load_success("mod", mod_dir, subtask)
            eb_vals.append((eb_succ * 100) if eb_succ is not None else 0.0)
            mod_vals.append((mod_succ * 100) if mod_succ is not None else 0.0)
            sizes.append(size_gb)

        # Bars — paired (EB on left, modified on right) per model
        ax.bar(x_positions - bar_width / 2, eb_vals,  bar_width,
               color=COLOR_EB,  edgecolor="black", linewidth=0.5, zorder=3)
        ax.bar(x_positions + bar_width / 2, mod_vals, bar_width,
               color=COLOR_MOD, edgecolor="black", linewidth=0.5,
               hatch="//", zorder=3)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([m[0] for m in MODELS])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Success rate (%)")
        ax.set_title(SUBTASK_TITLES[subtask])
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        ax.tick_params(direction="in", length=3)

        # Right axis — model size scatter (same point for both EB and mod,
        # since it's the same GGUF file)
        ax2 = ax.twinx()
        ax2.scatter(x_positions, sizes,
                    marker="D", s=36,
                    facecolor=SIZE_MARKER_COLOR, edgecolor="black",
                    linewidth=0.6, zorder=5)
        ax2.set_ylim(0, size_max * 1.6)
        ax2.set_ylabel("Model size (GB)", color=SIZE_MARKER_COLOR)
        ax2.tick_params(axis="y", colors=SIZE_MARKER_COLOR, direction="in", length=3)
        ax2.spines["right"].set_color(SIZE_MARKER_COLOR)

    # Shared legend at the top
    legend_handles = [
        Patch(facecolor=COLOR_EB,  edgecolor="black", linewidth=0.5,
              label="EmbodiedBench"),
        Patch(facecolor=COLOR_MOD, edgecolor="black", linewidth=0.5,
              hatch="//", label="EmbodiedBench (atlasmodified)"),
        Line2D([0], [0], marker="D", linestyle="none",
               markerfacecolor=SIZE_MARKER_COLOR, markeredgecolor="black",
               markeredgewidth=0.6, markersize=6, label="Model size (GB)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.2, h_pad=1.0)

    pdf_path = os.path.join(args.output_dir, "per_subtask_eb_vs_mod.pdf")
    png_path = os.path.join(args.output_dir, "per_subtask_eb_vs_mod.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
