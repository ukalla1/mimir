"""
gantt_three_levels.py — Three-level Gantt chart (horizontal layout).

Three subplots side by side (Level 1 | Level 2 | Level 3).
  - Every tool-call block has the same physical width across all levels.
  - Corresponding task rows (row k) are at the same y position in every subplot.
  - Each level's subplot width = its own max_calls × block_w_in.
  - One shared legend + title at the top.

Usage:
    python gantt_three_levels.py [logs_his_dir] [--output output.png]
"""

import re
import os
import textwrap
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl


# Liberation Serif is metric-compatible with Times New Roman
mpl.rcParams["font.family"] = "Liberation Serif"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIG_WIDTH  = 7.0    # total figure width (inches)
FONT_SIZE  = 9      # pt
DPI        = 300

BAR_H_IN   = 0.12                     # bar height in inches (= data units)
LINE_H_IN  = FONT_SIZE / 72 * 1.35   # ~0.168 in per text line at 9pt
PAD_IN     = 0.01   # gap between bar top and request text
SEP_PAD    = -0.08   # ← TUNE: gap from separator line down to request text
BAR_PAD    = 0.07   # ← TUNE: gap from bar bottom up to separator line above next task

# Figure margins / gaps (inches)
L_MARGIN   = 0.40
R_MARGIN   = 0.10
H_GAP      = 0.20   # horizontal gap between subplots
LEGEND_H   = 0.85   # space at top for title + shared legend
BOTTOM_H   = 0.40   # space at bottom for x-axis label

# ---------------------------------------------------------------------------
# Tool color map
# ---------------------------------------------------------------------------
TOOL_COLORS = {
    "get_detected_objects":         "#9FC5E8",
    "get_current_detected_objects": "#B6D7A8",
    "navigate_to_landmark":         "#F9CB9C",
    "spin_robot":                   "#EA9999",
    "move_robot":                   "#F4CCCC",
    "get_lidar_scan":               "#CFE2F3",
    "list_landmarks":               "#FFE599",
}
DEFAULT_COLOR = "#D9D9D9"

ALL_TOOLS = [
    "get_detected_objects",
    "get_current_detected_objects",
    "navigate_to_landmark",
    "spin_robot",
    "list_landmarks",
]

TOOL_CALL_RE = re.compile(r"^\[(\w+)\](.*)")

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log(path: str) -> dict:
    calls, task_name, response = [], None, None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").strip()
            if not line:
                continue
            if task_name is None:
                task_name = line.lstrip("#").strip()
                continue
            if line.lower().startswith("response:"):
                response = line.strip()
                continue
            m = TOOL_CALL_RE.match(line)
            if m:
                calls.append({"tool": m.group(1).strip(),
                               "args": m.group(2).strip(),
                               "index": len(calls)})
    return {"task_name": task_name or os.path.basename(path),
            "response": response, "calls": calls}


def load_logs(logs_dir: str) -> list:
    paths = sorted(
        os.path.join(logs_dir, f)
        for f in os.listdir(logs_dir)
        if f.endswith(".log")
    )
    if not paths:
        raise FileNotFoundError(f"No .log files found in: {logs_dir}")
    return [parse_log(p) for p in paths]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

MAX_LINES = 3   # max rendered lines per Request / Response label


def wrap_truncate(text: str, wrap_chars: int, max_lines: int = MAX_LINES) -> str:
    """Wrap text; if it exceeds max_lines, truncate the last line with '...'."""
    lines = textwrap.wrap(text, wrap_chars)
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # Keep first (max_lines-1) lines, then fit as much of the next line as possible
    kept = lines[:max_lines - 1]
    # Try to fill the last line up to wrap_chars-3 chars, then append '...'
    remaining = " ".join(lines[max_lines - 1:])
    cutoff = wrap_chars - 3
    if cutoff > 0 and len(remaining) > cutoff:
        # break at last space before cutoff
        cut = remaining.rfind(" ", 0, cutoff)
        last = (remaining[:cut] if cut > 0 else remaining[:cutoff]) + "..."
    else:
        last = remaining[:cutoff] + "..." if cutoff > 0 else "..."
    return "\n".join(kept + [last])


def n_lines(text: str, wrap_chars: int) -> int:
    return min(MAX_LINES, len(textwrap.wrap(text, wrap_chars))) if text else 1


def fixed_row_h() -> float:
    """Row height: [separator] SEP_PAD + 3-line text + PAD_IN + bar + BAR_PAD [separator]"""
    return SEP_PAD + MAX_LINES * LINE_H_IN + PAD_IN + BAR_H_IN + BAR_PAD

# ---------------------------------------------------------------------------
# Per-level drawing  (y_centers and axes_h are shared / pre-computed)
# ---------------------------------------------------------------------------

def draw_level(ax, tasks, y_centers, axes_h, max_calls, wrap_chars, row_boundaries):
    ROW_H = fixed_row_h()
    # bar_y: fixed offset from top of row (separator + SEP_PAD + 3-line text + PAD_IN)
    bar_offset_from_top = SEP_PAD + MAX_LINES * LINE_H_IN + PAD_IN + BAR_H_IN / 2

    for row_idx, task in enumerate(tasks):
        y_top = y_centers[row_idx] + ROW_H / 2
        bar_y = y_top - bar_offset_from_top
        for call in task["calls"]:
            color = TOOL_COLORS.get(call["tool"], DEFAULT_COLOR)
            ax.broken_barh(
                [(call["index"], 1)],
                (bar_y - BAR_H_IN / 2, BAR_H_IN),
                facecolors=color, edgecolors="white", linewidth=0.5,
            )

    ax.set_yticks([])
    ax.set_yticklabels([])

    for row_idx, task in enumerate(tasks):
        y_top = y_centers[row_idx] + ROW_H / 2
        bar_y = y_top - bar_offset_from_top
        # request text: bottom of text block sits PAD_IN above bar top
        ax.text(0.05, bar_y + BAR_H_IN / 2 + PAD_IN,
                wrap_truncate(task["task_name"], wrap_chars),
                ha="left", va="bottom",
                fontsize=FONT_SIZE, fontstyle="italic", clip_on=False)


    ax.set_xlim(0, max_calls)
    ax.set_xticks(range(max_calls + 1))
    ax.set_xticklabels([""] * (max_calls + 1))
    ax.set_xticks([i + 0.5 for i in range(max_calls)], minor=True)
    ax.set_xticklabels(range(1, max_calls + 1), minor=True, fontsize=FONT_SIZE)
    ax.tick_params(axis="x", which="major", length=4)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("Tool Call Index", fontsize=FONT_SIZE)

    ax.set_ylim(0, axes_h)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for y_sep in row_boundaries:
        ax.axhline(y_sep, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_three_levels(level_tasks: list, level_names: list, output_path: str):
    level_tasks = [[t for t in tasks if t["calls"]] for tasks in level_tasks]
    n           = len(level_tasks)
    n_rows      = max(len(tasks) for tasks in level_tasks)  # tasks per level (e.g. 5)

    # Per-level max_calls
    max_calls_per = [max(2, max(len(t["calls"]) for t in tasks))
                     for tasks in level_tasks]

    # block_w_in: derived so total figure width = FIG_WIDTH
    #   FIG_WIDTH = L_MARGIN + sum(max_calls_i * block_w) + (n-1)*H_GAP + R_MARGIN
    block_w_in = (FIG_WIDTH - L_MARGIN - R_MARGIN - (n - 1) * H_GAP) / sum(max_calls_per)

    # Per-level subplot widths in inches
    subplot_w = [mc * block_w_in for mc in max_calls_per]

    # -----------------------------------------------------------------------
    # Per-level wrap_chars — tune these three numbers to control line width.
    # Larger value = more chars per line (text fills more of the bar width).
    WRAP_CHARS = [19, 36, 65]   # [Level 1, Level 2, Level 3]
    # -----------------------------------------------------------------------
    wrap_chars_per = WRAP_CHARS

    # All rows same height (assuming MAX_LINES for request & response).
    ROW_H = fixed_row_h()
    global_row_h = [ROW_H] * n_rows

    # Shared y geometry (same for every subplot)
    axes_h    = sum(global_row_h)
    y_centers = []
    y_top     = axes_h
    for h in global_row_h:
        y_centers.append(y_top - h / 2)
        y_top -= h

    # Separator y positions: bottom edge of each row (between tasks)
    row_boundaries = [y_centers[k] - global_row_h[k] / 2
                      for k in range(len(y_centers) - 1)]

    fig_height = BOTTOM_H + axes_h + LEGEND_H
    fig        = plt.figure(figsize=(FIG_WIDTH, fig_height))

    ax_bottom_frac = BOTTOM_H / fig_height
    ax_height_frac = axes_h   / fig_height

    left_in = L_MARGIN
    for i in range(n):
        ax = fig.add_axes([
            left_in / FIG_WIDTH,
            ax_bottom_frac,
            subplot_w[i] / FIG_WIDTH,
            ax_height_frac,
        ])
        draw_level(ax, level_tasks[i], y_centers, axes_h,
                   max_calls_per[i], wrap_chars_per[i], row_boundaries)

        # Level label just above the axes
        ax.text(0.0, 1.02, level_names[i],
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=FONT_SIZE, fontweight="bold")

        left_in += subplot_w[i] + H_GAP

    # Shared legend + title — centered over the plot area (not the figure)
    plot_center_x = (L_MARGIN + (FIG_WIDTH - L_MARGIN - R_MARGIN) / 2) / FIG_WIDTH
    legend_band_bot = (BOTTOM_H + axes_h) / fig_height
    title_y  = legend_band_bot + (LEGEND_H / fig_height) * 0.9
    legend_y = legend_band_bot + (LEGEND_H / fig_height) * 0.83

    fig.text(plot_center_x, title_y, "Tool Calls by Task Across Difficulty Levels",
             ha="center", va="center",
             fontsize=FONT_SIZE + 1, fontweight="bold")

    all_used = {c["tool"] for tasks in level_tasks for t in tasks for c in t["calls"]}
    extra    = sorted(all_used - set(ALL_TOOLS))
    handles  = [
        mpatches.Patch(facecolor=TOOL_COLORS.get(t, DEFAULT_COLOR),
                       edgecolor="grey", linewidth=0.5, label=t)
        for t in ALL_TOOLS + extra
        if t in all_used
    ]
    fig.legend(handles=handles,
               loc="upper center",
               bbox_to_anchor=(plot_center_x, legend_y),
               ncol=3, fontsize=FONT_SIZE - 1, framealpha=0.8)

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {output_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Three-level Gantt chart (7-inch wide, 300 DPI, Liberation Serif)."
    )
    parser.add_argument(
        "logs_his_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "logs_his"),
    )
    parser.add_argument("--output", default="gantt_three_levels.png")
    args = parser.parse_args()

    base        = os.path.abspath(args.logs_his_dir)
    level_dirs  = [os.path.join(base, f"level_{i}") for i in range(1, 4)]
    level_names = ["Simple", "Intermediate", "Chanllenging"]

    level_tasks = []
    for d, name in zip(level_dirs, level_names):
        tasks = load_logs(d)
        print(f"{name}: loaded {len(tasks)} task(s)")
        level_tasks.append(tasks)

    plot_three_levels(level_tasks, level_names, args.output)


if __name__ == "__main__":
    main()
