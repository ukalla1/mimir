#!/usr/bin/env python3
"""Plot speed vs. EmbodiedBench-ALFRED success rate (IEEE 1-col).

Reads ``summary.json`` from each subtask directory under
``EmbodiedBench_atlasmodified/results/eb_alfred/<run_dir>/<subtask>/results/``
and produces a bubble scatter:

    X = speed  (default: env-steps / sec, averaged across the 6 subtasks)
    Y = task_success averaged across the 6 ALFRED subtasks (%)
    marker area  proportional to GGUF file size on disk (GB)

Usage:
    python nutri_rag/scripts/results_figure_drawing.py
    python nutri_rag/scripts/results_figure_drawing.py --speed-metric inv_latency
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

# Model run-dir -> (display label, GGUF size on disk in GB)
# Sizes from /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/*.gguf.
# iq2s -> UD-IQ2_XXS (only IQ2 variant smaller than IQ2_M on disk).
# q3xl excluded (only 4/6 subtasks completed).
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


def load_subtask_summaries(run_dir):
    out = []
    for sub in SUBTASKS:
        path = os.path.join(RESULTS_ROOT, run_dir, sub, "results", "summary.json")
        if not os.path.isfile(path):
            print(f"  warn: missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            out.append(json.load(f))
    return out


def compute_speed(summaries, metric):
    """Per-model speed averaged across subtasks.

    metric values:
      "steps_per_sec"  -> num_steps / episode_elapsed_seconds
      "inv_latency"    -> 60 / episode_elapsed_seconds  (episodes/min)
      "sec_per_call"   -> episode_elapsed_seconds / planner_steps  (lower=faster)
    """
    vals = []
    for s in summaries:
        elapsed = float(s["episode_elapsed_seconds"])
        if metric == "steps_per_sec":
            vals.append(float(s["num_steps"]) / elapsed)
        elif metric == "inv_latency":
            vals.append(60.0 / elapsed)
        elif metric == "sec_per_call":
            vals.append(elapsed / float(s["planner_steps"]))
        else:
            raise ValueError(f"unknown speed metric: {metric}")
    return sum(vals) / len(vals)


def speed_label(metric):
    return {
        "steps_per_sec": "Speed (env steps / s)",
        "inv_latency":   "Throughput (episodes / min)",
        "sec_per_call":  "Avg s per planner call  (← faster)",
    }[metric]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "nutri_rag", "results", "embodied_speed_vs_success"),
    )
    parser.add_argument(
        "--speed-metric",
        choices=["steps_per_sec", "inv_latency", "sec_per_call"],
        default="steps_per_sec",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for run_dir, label, size_gb, family in MODELS:
        summaries = load_subtask_summaries(run_dir)
        if not summaries:
            print(f"  skip {label}: no subtask results")
            continue
        mean_succ = sum(float(s["task_success"]) for s in summaries) / len(summaries)
        speed = compute_speed(summaries, args.speed_metric)
        rows.append((label, size_gb, speed, mean_succ, len(summaries), family))
        print(f"  {label:8s} [{family}] size={size_gb:.1f} GB  speed={speed:6.3f}  "
              f"success={mean_succ*100:5.1f}%  (n={len(summaries)} subtasks)")

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

    # marker area in points^2, scaled so the smallest GGUF is visible
    # and the largest is clearly bigger but doesn't dominate.
    def area(size_gb):
        return 18.0 * size_gb ** 2

    xs = [r[2] for r in rows]
    ys = [r[3] * 100 for r in rows]

    for label, size_gb, speed, succ, _, family in rows:
        ax.scatter(
            speed,
            succ * 100,
            s=area(size_gb),
            marker="o",
            facecolors=FAMILY_COLORS[family],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.75,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(speed, succ * 100),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel(speed_label(args.speed_metric))
    ax.set_ylabel("Avg. task success rate (%)")
    x_span = max(xs) - min(xs) if max(xs) > min(xs) else 1.0
    ax.set_xlim(min(xs) - 0.15 * x_span, max(xs) + 0.30 * x_span)
    ax.set_ylim(0, max(ys) + 16)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
    ax.tick_params(direction="in", length=3)

    # Bubble-size legend (lower-right) at fixed reference sizes. Uses a
    # smaller visual scale than the data points so it fits inside the axes.
    ref_sizes = [3.0, 4.5, 6.0]
    legend_scale = 0.45
    size_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="none",
            markerfacecolor="lightgray", markeredgecolor="black",
            markeredgewidth=0.6, alpha=0.75,
            markersize=((area(s) * legend_scale) ** 0.5),
            label=f"{s:.1f} GB",
        )
        for s in ref_sizes
    ]
    size_legend = ax.legend(
        handles=size_handles, title="Model size",
        loc="lower right", frameon=False,
        labelspacing=1.6, borderpad=0.4, handletextpad=1.2,
    )
    ax.add_artist(size_legend)

    # Quant-family legend (upper-left).
    family_handles = [
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
    ax.legend(handles=family_handles, loc="upper left", frameon=False)

    fig.tight_layout(pad=0.4)

    pdf_path = os.path.join(args.output_dir, "speed_vs_success.pdf")
    png_path = os.path.join(args.output_dir, "speed_vs_success.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
