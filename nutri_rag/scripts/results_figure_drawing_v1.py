#!/usr/bin/env python3
"""Plot model size vs. EmbodiedBench-ALFRED success rate (IEEE 1-col).

Reads ``summary.json`` from each subtask directory under
``EmbodiedBench_atlasmodified/results/eb_alfred/<run_dir>/<subtask>/results/``
and produces a scatter plot:

    X = model file size (GB, on disk)
    Y = task_success averaged across the 6 ALFRED subtasks (%)

Usage:
    python nutri_rag/scripts/results_figure_drawing.py
    python nutri_rag/scripts/results_figure_drawing.py --output-dir <dir>
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

RESULTS_ROOT = os.path.join(
    REPO_ROOT, "EmbodiedBench_atlasmodified", "results", "eb_alfred"
)

SUBTASKS = [
    "base",
    "common_sense",
    "complex_instruction",
    "long_horizon",
    "spatial",
    "visual_appearance",
]

# Model run-dir -> (display label, GGUF size on disk in GB, quant family)
# Sizes from /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/*.gguf.
# iq2s -> UD-IQ2_XXS (only IQ2 variant smaller than IQ2_M on disk).
# q4km excluded (smoke run, 5 ep/subtask instead of 50).
MODELS = [
    ("Qwen3-VL-9B-GGUF_qwen35_iq2s_alf_full_memB_v2", "IQ2_XXS", 3.0, "Q2"),
    ("Qwen3-VL-9B-GGUF_qwen35_iq2m_alf_full_memB_v2", "IQ2_M",   3.4, "Q2"),
    ("Qwen3-VL-9B-GGUF_qwen35_q2xl_alf_full_memB_v2", "Q2_K_XL", 3.9, "Q2"),
    ("Qwen3-VL-9B-GGUF_qwen35_q3xl_alf_full_memB_v2", "Q3_K_XL", 4.8, "Q3"),
    ("Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full_memB_v2", "Q4_K_S",  5.1, "Q4"),
    ("Qwen3-VL-9B-GGUF_qwen35_q4xl_alf_full_memB_v2", "Q4_K_XL", 5.6, "Q4"),
]

FAMILY_COLORS = {
    "Q2": "#c97a2d",  # orange
    "Q3": "#4a7d3a",  # green
    "Q4": "#3b6ea8",  # blue
}


def load_mean_success(run_dir):
    successes = []
    for sub in SUBTASKS:
        path = os.path.join(RESULTS_ROOT, run_dir, sub, "results", "summary.json")
        if not os.path.isfile(path):
            print(f"  warn: missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        successes.append(float(data["task_success"]))
    if not successes:
        return None, 0
    return sum(successes) / len(successes), len(successes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "nutri_rag", "results", "embodied_size_vs_success"),
    )
    parser.add_argument("--show", action="store_true", help="open the figure window")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for run_dir, label, size_gb, family in MODELS:
        mean_succ, n = load_mean_success(run_dir)
        if mean_succ is None:
            print(f"  skip {label}: no subtask results")
            continue
        rows.append((label, size_gb, mean_succ, n, family))
        print(f"  {label:8s} [{family}] size={size_gb:.1f} GB  "
              f"success={mean_succ*100:5.1f}%  (n={n} subtasks)")

    if not rows:
        sys.exit("no data found")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    xs = [r[1] for r in rows]
    ys = [r[2] * 100 for r in rows]

    for label, size_gb, succ, n, family in rows:
        ax.scatter(
            size_gb,
            succ * 100,
            s=45,
            marker="o",
            facecolors=FAMILY_COLORS[family],
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(size_gb, succ * 100),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel("Model size (GB)")
    ax.set_ylabel("Avg. task success rate (%)")
    ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4)
    ax.set_ylim(0, max(ys) + 10)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax.tick_params(direction="in", length=3)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=FAMILY_COLORS["Q2"], markeredgecolor="black",
               markeredgewidth=0.6, markersize=6, label="2 bit"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=FAMILY_COLORS["Q3"], markeredgecolor="black",
               markeredgewidth=0.6, markersize=6, label="3 bit"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=FAMILY_COLORS["Q4"], markeredgecolor="black",
               markeredgewidth=0.6, markersize=6, label="4 bit"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.tight_layout(pad=0.4)

    pdf_path = os.path.join(args.output_dir, "size_vs_success.pdf")
    png_path = os.path.join(args.output_dir, "size_vs_success.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
