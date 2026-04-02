"""
gantt_multi_task.py — Compare tool call sequences across multiple tasks.

Each .log file in the logs directory should have a task name on the first line
(plain text, e.g. "Find food in kitchen"), followed by the usual log format:

    Find food in kitchen
    User: i'm hungry...
    [get_detected_objects] called
    [navigate_to_landmark] navigating to: kitchen
    ...

One row per task (log file), colored blocks per tool call.

Usage:
    python gantt_multi_task.py [logs_dir] [--output output.png]
"""

import re
import os
import textwrap
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

mpl.rcParams["font.family"] = "Nimbus Roman"

# ---------------------------------------------------------------------------
# Tool color map
# ---------------------------------------------------------------------------
TOOL_COLORS = {
    "get_detected_objects":         "#9FC5E8",  # soft blue   (Feed Forward)
    "get_current_detected_objects": "#B6D7A8",  # soft green  (Softmax)
    "navigate_to_landmark":         "#F9CB9C",  # soft peach  (Attention)
    "spin_robot":                   "#EA9999",  # soft pink   (Embedding)
    "move_robot":                   "#F4CCCC",  # light pink
    "get_lidar_scan":               "#CFE2F3",  # light blue
    "list_landmarks":               "#FFE599",  # soft yellow (Add & Norm)
}
DEFAULT_COLOR = "#D9D9D9"  # light gray

# Fixed tool order for legend (all known tools from robot_control)
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

def make_label(tool: str, args: str) -> str:
    if tool == "navigate_to_landmark":
        m = re.search(r"navigating to:\s*(\S+)", args)
        dest = m.group(1) if m else args[:8]
        return dest.replace("_landmark", "")
    if tool == "spin_robot":
        m = re.search(r"rotating\s+([\d.]+)", args)
        return f"{m.group(1)}°" if m else ""
    if tool == "move_robot":
        m = re.search(r"moving\s+([\d.]+)", args)
        return f"{m.group(1)}m" if m else ""
    if tool == "get_lidar_scan":
        m = re.search(r"min_dist=([\d.]+)", args)
        return f"{m.group(1)}m" if m else ""
    return ""


def parse_log(path: str) -> dict:
    """Return {task_name, response, calls: [{tool, args, index}]}."""
    calls = []
    task_name = None
    response = None

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
                calls.append({
                    "tool":  m.group(1).strip(),
                    "args":  m.group(2).strip(),
                    "index": len(calls),
                })

    return {"task_name": task_name or os.path.basename(path), "response": response, "calls": calls}


def load_logs(logs_dir: str) -> list:
    """Load all .log files from logs_dir, sorted by filename."""
    paths = sorted(
        os.path.join(logs_dir, f)
        for f in os.listdir(logs_dir)
        if f.endswith(".log")
    )
    if not paths:
        raise FileNotFoundError(f"No .log files found in: {logs_dir}")
    return [parse_log(p) for p in paths]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(tasks: list, output_path: str):
    # Filter out tasks with no tool calls
    tasks = [t for t in tasks if t["calls"]]
    if not tasks:
        print("No tool calls found in any log file.")
        return

    max_calls  = max(2, max(len(t["calls"]) for t in tasks))

    bar_h_in     = 0.12           # bar height in inches (fixed)
    bar_unit_w   = 0.7            # inches per tool slot (keeps bar width fixed)
    axes_frac_w  = 1.2            # axes width as fraction of figure width
    axes_w_in    = max_calls * bar_unit_w
    fig_width    = axes_w_in / axes_frac_w
    chars_per_slot = 15           # chars per tool slot at fontsize 8
    wrap_chars     = max(20, max_calls * chars_per_slot)

    line_h_in  = 8 / 72 * 1.35   # inches per text line at fontsize 8
    pad_in     = 0.06             # gap between bar edge and text
    gap_in     = 0.06             # extra gap between rows

    def n_lines(text):
        return len(textwrap.wrap(text, wrap_chars)) if text else 1

    def task_row_h(task):
        return (n_lines(task["task_name"]) + n_lines(task.get("response") or "")) * line_h_in \
               + bar_h_in + 2 * pad_in + gap_in

    row_heights = [task_row_h(t) for t in tasks]

    # y_centers: top-to-bottom, using inches as data units
    axes_h = sum(row_heights)
    y_centers = []
    y_top = axes_h
    for h in row_heights:
        y_centers.append(y_top - h / 2)
        y_top -= h

    legend_h   = 0.7
    bottom_h   = 0.8
    fig_height = axes_h + legend_h + bottom_h
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_position([
        -0.1,
        bottom_h / fig_height,
        1.2,
        axes_h / fig_height,
    ])

    for row_idx, task in enumerate(tasks):
        y = y_centers[row_idx]

        for call in task["calls"]:
            tool  = call["tool"]
            color = TOOL_COLORS.get(tool, DEFAULT_COLOR)
            x     = call["index"]

            ax.broken_barh(
                [(x, 1)],
                (y - bar_h_in / 2, bar_h_in),
                facecolors=color,
                edgecolors="white",
                linewidth=0.5,
            )

    # Y-axis: remove ticks, write task name above and response below each bar
    ax.set_yticks([])
    ax.set_yticklabels([])
    for row_idx, task in enumerate(tasks):
        y = y_centers[row_idx]
        wrapped_name = textwrap.fill(task["task_name"], width=wrap_chars)
        ax.text(0.05, y + bar_h_in / 2 + pad_in, wrapped_name,
                ha="left", va="bottom", fontsize=8, fontstyle="italic",
                clip_on=False)
        if task.get("response"):
            wrapped = textwrap.fill(task["response"], width=wrap_chars)
            ax.text(0.05, y - bar_h_in / 2 - pad_in, wrapped,
                    ha="left", va="top", fontsize=8, fontstyle="italic",
                    clip_on=False)

    # Print task legend to terminal
    print("Task legend:")
    for i, t in enumerate(tasks):
        print(f"  T{i+1} : {t['task_name']}")

    # X-axis
    ax.set_xlim(0, max_calls)
    ax.set_xticks(range(max_calls + 1))
    ax.set_xticklabels([""] * (max_calls + 1))
    ax.set_xticks([i + 0.5 for i in range(max_calls)], minor=True)
    ax.set_xticklabels(range(1, max_calls + 1), minor=True, fontsize=9)
    ax.tick_params(axis="x", which="major", length=4)
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("Tools Called", fontsize=9)

    ax.set_ylim(0, axes_h)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    axes_top_frac = (bottom_h + axes_h) / fig_height
    top_space     = 1.0 - axes_top_frac

    title_y  = axes_top_frac + top_space * 0.90
    legend_y = axes_top_frac + top_space * 0.75

    fig.text(0.5, title_y, "Tool Calls by Task",
             ha="center", va="center", fontsize=9, fontweight="bold")

    extra_tools = sorted({c["tool"] for t in tasks for c in t["calls"]} - set(ALL_TOOLS))
    handles = [
        mpatches.Patch(
            facecolor=TOOL_COLORS.get(t, DEFAULT_COLOR),
            edgecolor="grey", linewidth=0.5, label=t,
        )
        for t in ALL_TOOLS + extra_tools
    ]
    fig.legend(handles=handles,
               loc="upper center",
               bbox_to_anchor=(0.5, legend_y),
               ncol=3, fontsize=8, framealpha=0.8)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {output_path}")
    # plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gantt chart comparing tool call sequences across tasks."
    )
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "logs"),
        help="Directory containing .log files (default: ../logs)",
    )
    parser.add_argument(
        "--output",
        default="gantt_multi_task.png",
        help="Output PNG path (default: gantt_multi_task.png)",
    )
    args = parser.parse_args()

    tasks = load_logs(os.path.abspath(args.logs_dir))
    print(f"Loaded {len(tasks)} task(s):")
    for t in tasks:
        print(f"  [{len(t['calls'])} calls] {t['task_name']}")
    print()
    plot(tasks, args.output)


if __name__ == "__main__":
    main()
