#!/usr/bin/env python3
"""Per-subtask EmbodiedBench-ALFRED comparison (IEEE 2-col format).

Reads ``summary.json`` from each subtask directory under
``EmbodiedBench/results/eb_alfred/<run_dir>/<subtask>/results/`` and
``EmbodiedBench_atlasmodified/results/eb_alfred/<run_dir>/<subtask>/results/``
and overlays reference open-source baselines on a 2x3 grid (one per subtask).

Each subplot:
    X axis      : models — our quantized Qwen3.5-9B GGUF runs (IQ2_M, Q4_K_S),
                  then 3 open-source baselines stronger than Q4_K_S, then 3
                  open-source baselines weaker.
    Left  Y     : task success rate (%) — bars.
                  Our Qwen3.5-9B runs are paired: EmbodiedBench vs.
                  EmbodiedBench_atlasmodified (same hue, hatched for modified).
                  Reference baselines are single bars.
    Right Y     : model size (GB) — diamond scatter, LOG scale.
                  Our GGUF entries use on-disk size; reference models use the
                  FP16 weight footprint (params * 2 bytes).

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

# Our quantized Qwen3.5-9B GGUF runs:
#   (short label, GGUF on-disk size GB, EB run-dir, modified run-dir)
OUR_MODELS = [
    ("UD-IQ2_M",  3.4,
     "Qwen3-VL-9B-GGUF_qwen35_iq2m_alf_full",
     "Qwen3-VL-9B-GGUF_qwen35_iq2m_alf_full_memB_v2"),
    ("Q4_K_S", 5.1,
     "Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full",
     "Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full_memB_v2"),
]

# Open-source baselines from the EmbodiedBench leaderboard, sorted by
# avg success descending. Sizes are FP16 weight footprint (params * 2 bytes).
REF_MODELS = [
    ("Qwen2.5-72B",     144.0, {"base":50.0,"common_sense":42.0,"complex_instruction":42.0,
                                "long_horizon":34.0,"spatial":34.0,"visual_appearance":36.0}),
    ("InternVL3-78B",   156.0, {"base":38.0,"common_sense":34.0,"complex_instruction":46.0,
                                "long_horizon":36.0,"spatial":38.0,"visual_appearance":42.0}),
    ("InternVL3-38B",    76.0, {"base":42.0,"common_sense":34.0,"complex_instruction":48.0,
                                "long_horizon":44.0,"spatial":30.0,"visual_appearance":30.0}),
    ("InternVL2.5-78B", 156.0, {"base":38.0,"common_sense":34.0,"complex_instruction":42.0,
                                "long_horizon":42.0,"spatial":36.0,"visual_appearance":34.0}),
    ("Gemma3-27B",       54.0, {"base":42.0,"common_sense":40.0,"complex_instruction":48.0,
                                "long_horizon":26.0,"spatial":36.0,"visual_appearance":30.0}),
    ("Qwen2-VL-72B",    144.0, {"base":40.0,"common_sense":30.0,"complex_instruction":40.0,
                                "long_horizon":30.0,"spatial":32.0,"visual_appearance":30.0}),
    ("Llama3.2-90B",    180.0, {"base":38.0,"common_sense":34.0,"complex_instruction":44.0,
                                "long_horizon":16.0,"spatial":32.0,"visual_appearance":28.0}),
    ("Ovis2-34B",        68.0, {"base":34.0,"common_sense":30.0,"complex_instruction":38.0,
                                "long_horizon":24.0,"spatial":18.0,"visual_appearance":28.0}),
    ("Gemma3-12B",       24.0, {"base":32.0,"common_sense":26.0,"complex_instruction":38.0,
                                "long_horizon":12.0,"spatial":20.0,"visual_appearance":26.0}),
    ("InternVL2.5-38B",  76.0, {"base":36.0,"common_sense":30.0,"complex_instruction":36.0,
                                "long_horizon":26.0,"spatial":14.0,"visual_appearance":22.0}),
    ("Ovis2-16B",        32.0, {"base":26.0,"common_sense":16.0,"complex_instruction":24.0,
                                "long_horizon": 4.0,"spatial":16.0,"visual_appearance":12.0}),
    ("Llama3.2-11B",     22.0, {"base":24.0,"common_sense": 8.0,"complex_instruction":16.0,
                                "long_horizon": 6.0,"spatial": 6.0,"visual_appearance":22.0}),
    ("InternVL3-8B",     16.0, {"base":20.0,"common_sense":14.0,"complex_instruction":14.0,
                                "long_horizon": 2.0,"spatial": 0.0,"visual_appearance":12.0}),
    ("Qwen2.5-7B",       14.0, {"base":10.0,"common_sense": 8.0,"complex_instruction": 6.0,
                                "long_horizon": 2.0,"spatial": 0.0,"visual_appearance": 2.0}),
    ("InternVL2.5-8B",   16.0, {"base": 4.0,"common_sense": 6.0,"complex_instruction": 2.0,
                                "long_horizon": 0.0,"spatial": 0.0,"visual_appearance": 0.0}),
    ("Qwen2-VL-7B",      14.0, {"base": 6.0,"common_sense": 0.0,"complex_instruction": 2.0,
                                "long_horizon": 2.0,"spatial": 0.0,"visual_appearance": 0.0}),
]

# SAM-style vivid palette
COLOR_ORIGIN   = "#3A6BA5"   # steel blue   — Qwen3.5-9B original run
COLOR_MEMORY   = "#E76F51"   # warm coral   — Qwen3.5-9B with memory (highlight)
COLOR_BASELINE = "#2A9D8F"   # teal         — open-source baselines
SIZE_MARKER_COLOR = "#D4AC0D"  # gold       — model size diamonds


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
        "font.size": 14,
        "font.weight": "bold",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "axes.linewidth": 1.0,
    })

    fig, axes = plt.subplots(2, 3, figsize=(20.0, 7.0), sharey=False)

    bar_width = 0.30

    n_our    = len(OUR_MODELS)
    n_ref    = len(REF_MODELS)

    our_positions    = np.arange(n_our)                                          # 0, 1
    ref_positions    = np.arange(n_ref)    + n_our + 0.5                         # 2.5 ...
    all_positions = np.concatenate([our_positions, ref_positions])
    all_labels    = ([m[0] for m in OUR_MODELS]
                     + [m[0] for m in REF_MODELS])

    for ax, subtask in zip(axes.flat, SUBTASKS):
        origin_vals = []
        memory_vals = []
        our_sizes   = []
        for _, size_gb, eb_dir, mod_dir in OUR_MODELS:
            origin_succ = load_success("eb",  eb_dir,  subtask)
            memory_succ = load_success("mod", mod_dir, subtask)
            origin_vals.append((origin_succ * 100) if origin_succ is not None else 0.0)
            memory_vals.append((memory_succ * 100) if memory_succ is not None else 0.0)
            our_sizes.append(size_gb)

        ref_vals  = [m[2][subtask] for m in REF_MODELS]
        ref_sizes = [m[1]          for m in REF_MODELS]

        # Paired bars: Origin (steel blue) + With memory (warm coral)
        ax.bar(our_positions - bar_width / 2, origin_vals, bar_width,
               color=COLOR_ORIGIN, edgecolor="black", linewidth=0.5, zorder=3)
        ax.bar(our_positions + bar_width / 2, memory_vals, bar_width,
               color=COLOR_MEMORY, edgecolor="black", linewidth=0.5, zorder=3)

        # Open-source baselines (teal)
        ax.bar(ref_positions, ref_vals, bar_width * 1.2,
               color=COLOR_BASELINE, edgecolor="black", linewidth=0.4, zorder=3)

        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Success rate (%)")
        ax.set_title(SUBTASK_TITLES[subtask])
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        ax.tick_params(direction="in", length=3)

        # Right Y axis — model size on log scale (covers 3-180 GB)
        ax2 = ax.twinx()
        sized_positions = np.concatenate([our_positions, ref_positions])
        sized_values    = our_sizes + ref_sizes
        ax2.scatter(sized_positions, sized_values,
                    marker="D", s=28,
                    facecolor=SIZE_MARKER_COLOR, edgecolor="black",
                    linewidth=0.6, zorder=5)
        ax2.set_yscale("log")
        ax2.set_ylim(1, 300)
        ax2.set_ylabel("Model size (GB, log)", color=SIZE_MARKER_COLOR)
        ax2.tick_params(axis="y", colors=SIZE_MARKER_COLOR, direction="in", length=3)
        ax2.spines["right"].set_color(SIZE_MARKER_COLOR)

    # Shared legend at the top
    legend_handles = [
        Patch(facecolor=COLOR_ORIGIN, edgecolor="black", linewidth=0.5,
              label="Qwen3.5-9B original"),
        Patch(facecolor=COLOR_MEMORY, edgecolor="black", linewidth=0.5,
              label="Nutri-ATLAS"),
        Patch(facecolor=COLOR_BASELINE, edgecolor="black", linewidth=0.5,
              label="Open-source baselines"),
        Line2D([0], [0], marker="D", linestyle="none",
               markerfacecolor=SIZE_MARKER_COLOR, markeredgecolor="black",
               markeredgewidth=0.6, markersize=6, label="Model size (GB)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.6, h_pad=1.6)

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
