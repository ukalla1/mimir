#!/usr/bin/env python3
"""
Analyze EmbodiedBench eval results from a model run directory.

Expected directory layout:
    {RUN_DIR}/
      {subset_name}/
        config.txt
        episode_N_step_M.json     ← per-step traces (not used here)
        images/                    ← captured frames (not used here)
        results/
          episode_N_final_res.json ← per-episode final results (this is what we read)

Each `final_res.json` has at minimum these fields:
    task_success, task_progress, reward,
    num_steps, planner_steps, planner_output_error,
    num_invalid_actions, num_invalid_action_ratio,
    empty_plan, episode_elapsed_seconds, instruction

Usage:
    # Default — analyze the Q4_K_M full alf run
    python embench_results_analysis.py

    # Custom run path
    python embench_results_analysis.py --path /path/to/run_dir

    # Save a textual report alongside the live print
    python embench_results_analysis.py --output report.txt

    # Compare two runs (e.g. Q4_K_M vs Q4_K_S)
    python embench_results_analysis.py --compare /path/run_a /path/run_b
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# Default run to inspect — the Q4_K_M full EB-Alfred eval
DEFAULT_RUN = Path(
    "/home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred"
    "/Qwen3-VL-9B-GGUF_qwen35_iq2m_alf_full"
)

# Metrics we aggregate per subset. Order matters for column layout.
KEY_METRICS = [
    "task_success",
    "task_progress",
    "reward",
    "planner_output_error",
    "num_invalid_actions",
    "num_steps",
    "episode_elapsed_seconds",
]

# Failure-mode thresholds (tuned to what we saw in Phase 5 smoke logs)
FAILURE_THRESHOLDS = {
    "high_json_errors": 3,        # planner_output_error > this → "JSON retry loop"
    "very_slow_seconds": 200,     # episode_elapsed_seconds > this → "slow"
    "step_cap": 30,               # num_steps == this → "hit step limit"
}


def load_subset_episodes(subset_dir):
    """Load every episode_*_final_res.json under a subset's results/ dir."""
    results_dir = subset_dir / "results"
    if not results_dir.is_dir():
        return []
    episodes = []
    for f in sorted(results_dir.glob("episode_*_final_res.json")):
        try:
            with open(f) as fp:
                data = json.load(fp)
            data["_file"] = f.name
            episodes.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: could not read {f.name}: {e}", file=sys.stderr)
    return episodes


def stats_for(values):
    """Return mean/median/min/max/stdev/sum/count for a numeric list."""
    if not values:
        return {"count": 0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "sum": sum(values),
        "count": len(values),
    }


def _is_finite_number(x):
    """True iff x is a number AND not NaN/inf. Filters bool out too."""
    if isinstance(x, bool):
        # bools are ints in Python; coerce to numeric explicitly
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    return False


def aggregate(episodes):
    """Compute stats per metric across a list of episodes (NaN/inf filtered out)."""
    agg = {"_count": len(episodes)}
    for m in KEY_METRICS:
        values = [float(e[m]) for e in episodes if m in e and _is_finite_number(e[m])]
        agg[m] = stats_for(values)
    return agg


def find_failures(episodes):
    """Bucket episodes into failure-mode categories."""
    buckets = {
        "high_json_errors": [],
        "step_cap_hit": [],
        "very_slow": [],
        "empty_plan_failed": [],
    }
    for ep in episodes:
        if ep.get("planner_output_error", 0) > FAILURE_THRESHOLDS["high_json_errors"]:
            buckets["high_json_errors"].append(ep)
        if ep.get("num_steps") == FAILURE_THRESHOLDS["step_cap"]:
            buckets["step_cap_hit"].append(ep)
        if ep.get("episode_elapsed_seconds", 0) > FAILURE_THRESHOLDS["very_slow_seconds"]:
            buckets["very_slow"].append(ep)
        if ep.get("empty_plan", 0) > 0 and ep.get("task_success", 0) in (0, 0.0, False):
            buckets["empty_plan_failed"].append(ep)
    return buckets


def analyze_run(run_path):
    """Walk a run directory, return {subset_name: [episodes]}."""
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    out = {}
    for subset_dir in sorted(run_path.iterdir()):
        if subset_dir.is_dir():
            out[subset_dir.name] = load_subset_episodes(subset_dir)
    return out


# --- Reporting ---


def fmt_subset_table(subset_data):
    """Build the main per-subset summary table as a list of lines."""
    col_w = 14
    name_w = 22
    header = "Subset".ljust(name_w) + " | n  | " + " | ".join(
        m[:col_w].rjust(col_w) for m in KEY_METRICS
    )
    sep = "-" * len(header)
    lines = [header, sep]

    all_eps = []
    for subset, episodes in subset_data.items():
        if not episodes:
            lines.append(f"{subset.ljust(name_w)} |   0 | (no episodes)")
            continue
        agg = aggregate(episodes)
        cells = [subset.ljust(name_w), f"{agg['_count']:>3}"]
        for m in KEY_METRICS:
            mean = agg[m].get("mean")
            cells.append(f"{mean:.3f}".rjust(col_w) if mean is not None else " " * col_w)
        lines.append(" | ".join(cells))
        all_eps.extend(episodes)

    # Overall row
    if all_eps:
        lines.append(sep)
        agg_all = aggregate(all_eps)
        cells = ["OVERALL".ljust(name_w), f"{agg_all['_count']:>3}"]
        for m in KEY_METRICS:
            mean = agg_all[m].get("mean")
            cells.append(f"{mean:.3f}".rjust(col_w) if mean is not None else " " * col_w)
        lines.append(" | ".join(cells))
    return lines


def fmt_failure_breakdown(subset_data):
    """Per-subset failure bucket counts."""
    lines = []
    bucket_names = ["high_json_errors", "step_cap_hit", "very_slow", "empty_plan_failed"]
    col_w = 18
    header = "Subset".ljust(22) + " | " + " | ".join(b.rjust(col_w) for b in bucket_names)
    lines.append(header)
    lines.append("-" * len(header))
    for subset, episodes in subset_data.items():
        buckets = find_failures(episodes)
        row = [subset.ljust(22)]
        for b in bucket_names:
            row.append(f"{len(buckets[b])}".rjust(col_w))
        lines.append(" | ".join(row))
    return lines


def fmt_extreme_episodes(subset_data, n=5):
    """List slowest, most-retried, and best episodes."""
    all_eps = [ep for episodes in subset_data.values() for ep in episodes]

    def short(ep):
        instr = ep.get("instruction", "")[:60]
        return (
            f"  task_success={ep.get('task_success'):>4}"
            f"  elapsed={ep.get('episode_elapsed_seconds', 0):6.1f}s"
            f"  steps={ep.get('num_steps'):>3}"
            f"  json_err={ep.get('planner_output_error', 0):>2}"
            f"  | {ep.get('_file')} | {instr!r}"
        )

    lines = [f"\nTop {n} slowest episodes:"]
    for ep in sorted(all_eps, key=lambda e: -e.get("episode_elapsed_seconds", 0))[:n]:
        lines.append(short(ep))

    lines.append(f"\nTop {n} highest JSON-error episodes:")
    for ep in sorted(all_eps, key=lambda e: -e.get("planner_output_error", 0))[:n]:
        lines.append(short(ep))

    lines.append(f"\nSuccessful episodes (task_success > 0):")
    successes = [ep for ep in all_eps if ep.get("task_success", 0) > 0]
    if not successes:
        lines.append("  (none)")
    else:
        for ep in sorted(successes, key=lambda e: -e.get("task_success", 0))[: n * 2]:
            lines.append(short(ep))
    return lines


def write_report(run_path, subset_data, out_lines):
    """Print the full report to stdout AND append to out_lines for optional saving."""

    def emit(s=""):
        out_lines.append(s)
        print(s)

    emit(f"\n{'=' * 80}")
    emit(f"EmbodiedBench Results Analysis")
    emit(f"Run: {run_path}")
    emit(f"{'=' * 80}\n")

    total = sum(len(eps) for eps in subset_data.values())
    emit(f"Subsets found: {len(subset_data)}")
    emit(f"Total episodes: {total}\n")

    emit("## Per-subset means (key metrics)")
    for line in fmt_subset_table(subset_data):
        emit(line)

    emit("\n## Failure-mode breakdown (episode counts per bucket)")
    emit(f"  high_json_errors:  planner_output_error > {FAILURE_THRESHOLDS['high_json_errors']}")
    emit(f"  step_cap_hit:      num_steps == {FAILURE_THRESHOLDS['step_cap']}")
    emit(f"  very_slow:         episode_elapsed_seconds > {FAILURE_THRESHOLDS['very_slow_seconds']}")
    emit(f"  empty_plan_failed: empty_plan>0 AND task_success==0\n")
    for line in fmt_failure_breakdown(subset_data):
        emit(line)

    emit("\n## Extreme episodes (debugging signals)")
    for line in fmt_extreme_episodes(subset_data, n=5):
        emit(line)


# --- Compare mode ---


def write_comparison(run_a, data_a, run_b, data_b):
    """Side-by-side per-subset means for two runs."""
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit(f"\n{'=' * 80}")
    emit(f"EmbodiedBench Comparison")
    emit(f"  A: {run_a}")
    emit(f"  B: {run_b}")
    emit(f"{'=' * 80}\n")

    subsets = sorted(set(data_a.keys()) | set(data_b.keys()))

    # Per-metric tables (cleaner than one huge table)
    for metric in KEY_METRICS:
        emit(f"\n### {metric}")
        emit(f"{'Subset'.ljust(22)} | {'A':>10} | {'B':>10} | {'Δ (B-A)':>10}")
        emit("-" * 60)
        for subset in subsets:
            agg_a = aggregate(data_a.get(subset, []))
            agg_b = aggregate(data_b.get(subset, []))
            va = agg_a.get(metric, {}).get("mean")
            vb = agg_b.get(metric, {}).get("mean")
            cell_a = f"{va:.3f}" if va is not None else "—"
            cell_b = f"{vb:.3f}" if vb is not None else "—"
            if va is not None and vb is not None:
                delta = f"{vb - va:+.3f}"
            else:
                delta = "—"
            emit(f"{subset.ljust(22)} | {cell_a:>10} | {cell_b:>10} | {delta:>10}")
    return lines


# --- Entry point ---


def main():
    parser = argparse.ArgumentParser(
        description="Analyze EmbodiedBench eval results.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_RUN,
        help=f"Run directory to analyze. Default: {DEFAULT_RUN}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, also write the textual report here.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("RUN_A", "RUN_B"),
        default=None,
        help="Compare two runs side-by-side instead of analyzing one.",
    )
    args = parser.parse_args()

    out_lines = []

    if args.compare:
        data_a = analyze_run(args.compare[0])
        data_b = analyze_run(args.compare[1])
        out_lines = write_comparison(args.compare[0], data_a, args.compare[1], data_b)
    else:
        subset_data = analyze_run(args.path)
        write_report(args.path, subset_data, out_lines)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write("\n".join(out_lines) + "\n")
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
