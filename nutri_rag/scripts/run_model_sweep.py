#!/usr/bin/env python3
"""Benchmark all Qwen3.5-9B quantization variants on NutriBench.

For each GGUF model:
  1. Start llama-server with that model
  2. Wait until /health reports ready
  3. Run run_bench.py (mode v3, selected nutrients)
  4. Stop the server
  5. Collect acc / MAE from the result JSON

At the end, print a comparison table and save sweep_summary.json.

Usage:
    # Quick test — 20 samples per model, protein only (~2 min/model)
    python scripts/run_model_sweep.py --limit 20

    # Full run, protein only
    python scripts/run_model_sweep.py

    # Full run, all nutrients
    python scripts/run_model_sweep.py --nutrients carb protein fat energy

    # Specific models only
    python scripts/run_model_sweep.py --models Qwen3.5-9B-UD-Q4_K_XL.gguf Qwen3.5-9B-Q4_K_M.gguf
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import requests

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
NUTRI_RAG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
ATLAS_ROOT     = os.path.abspath(os.path.join(NUTRI_RAG_ROOT, '..', '..'))
LM_EVAL_DIR    = os.path.join(ATLAS_ROOT, 'qwen_test', 'lm-evaluation-harness')
DEFAULT_GGUF_DIR = '/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF'
START_SERVER_SH  = os.path.join(SCRIPT_DIR, 'start_server.sh')


# ──────────────────────────────────────────────────────────────────────────────
# Server lifecycle helpers
# ──────────────────────────────────────────────────────────────────────────────

def start_server(model_path: str, port: int) -> subprocess.Popen:
    """Launch llama-server in the background and return the process handle."""
    cmd = ['bash', START_SERVER_SH, model_path]
    env = os.environ.copy()
    # Override PORT if non-default (start_server.sh currently hardcodes 8080)
    if port != 8080:
        env['PORT'] = str(port)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  # new process group so we can kill the whole tree
    )
    return proc


def wait_for_server(port: int, timeout: int = 180) -> bool:
    """Poll /health until the server responds or timeout expires."""
    url = f'http://localhost:{port}/health'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def stop_server(proc: subprocess.Popen):
    """Kill the whole process group started by start_server."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    # Wait for port to free
    for _ in range(20):
        try:
            requests.get(f'http://localhost:8080/health', timeout=1)
            time.sleep(1)
        except requests.exceptions.RequestException:
            break


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def run_bench(mode: str, nutrient: str, limit, port: int,
              concurrent: int, output_dir: str) -> bool:
    """Run a single run_bench.py call as a subprocess."""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, 'run_bench.py'),
        '--mode',       mode,
        '--nutrient',   nutrient,
        '--port',       str(port),
        '--concurrent', str(concurrent),
        '--output',     output_dir,
    ]
    if limit:
        cmd += ['--limit', str(limit)]

    # Ensure lm_eval (git checkout) and nutri_rag are importable in the subprocess
    env = os.environ.copy()
    extra = f'{LM_EVAL_DIR}{os.pathsep}{NUTRI_RAG_ROOT}'
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{extra}{os.pathsep}{existing}' if existing else extra

    result = subprocess.run(cmd, cwd=NUTRI_RAG_ROOT, env=env)
    return result.returncode == 0


def read_latest_result(output_dir: str, mode: str, nutrient: str) -> dict | None:
    """Find the most recent results JSON for (mode, nutrient) and extract metrics."""
    pattern = os.path.join(output_dir, f'results_{mode}_{nutrient}_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    # results dict: task_name → {acc,mae,...}
    metrics = {}
    for task_metrics in data.get('results', {}).values():
        for k, v in task_metrics.items():
            if k in ('acc', 'mae', 'acc,none', 'mae,none'):
                clean = k.replace(',none', '')
                metrics[clean] = round(float(v), 4)
    return metrics or None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark all GGUF models on NutriBench')
    parser.add_argument('--gguf-dir', default=DEFAULT_GGUF_DIR,
                        help='Directory containing GGUF files')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific GGUF filenames to test. Omit to test all (excl. mmproj).')
    parser.add_argument('--mode', default='v3',
                        choices=['v0', 'v1', 'v2', 'v3'],
                        help='RAG mode (default: v3)')
    parser.add_argument('--nutrients', nargs='+',
                        default=['protein'],
                        choices=['carb', 'protein', 'fat', 'energy'],
                        help='Nutrients to evaluate (default: protein)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Samples per run — use 20 for a quick test, omit for full')
    parser.add_argument('--port', type=int, default=8080,
                        help='llama-server port (default: 8080)')
    parser.add_argument('--concurrent', type=int, default=3,
                        help='Concurrent requests to the server (default: 3)')
    args = parser.parse_args()

    # Discover models
    if args.models:
        model_paths = [os.path.join(args.gguf_dir, m) for m in args.models]
    else:
        all_gguf = sorted(glob.glob(os.path.join(args.gguf_dir, '*.gguf')))
        model_paths = [p for p in all_gguf if 'mmproj' not in os.path.basename(p)]

    if not model_paths:
        print(f'No GGUF models found in {args.gguf_dir}')
        sys.exit(1)

    # Output root for this sweep
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    sweep_dir = os.path.join(NUTRI_RAG_ROOT, 'results', f'model_sweep_{timestamp}')
    os.makedirs(sweep_dir, exist_ok=True)

    print(f'Model sweep — {len(model_paths)} model(s), '
          f'mode={args.mode}, nutrients={args.nutrients}, '
          f'limit={args.limit or "all"}')
    print(f'Results → {sweep_dir}\n')

    # ── Sweep ─────────────────────────────────────────────────────────────────
    summary: list[dict] = []

    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_out   = os.path.join(sweep_dir, model_name)
        os.makedirs(model_out, exist_ok=True)

        print(f'\n{"="*64}')
        print(f'  Model: {model_name}')
        print(f'{"="*64}')

        if not os.path.exists(model_path):
            print(f'  [SKIP] File not found: {model_path}')
            summary.append({'model': model_name, 'status': 'missing'})
            continue

        # 1. Start server
        print(f'  Starting server...', flush=True)
        proc = start_server(model_path, args.port)

        # 2. Wait for ready
        print(f'  Waiting for /health (up to 180s)...', flush=True)
        if not wait_for_server(args.port, timeout=180):
            print(f'  [FAIL] Server did not become ready — skipping model')
            stop_server(proc)
            summary.append({'model': model_name, 'status': 'server_timeout'})
            continue
        print(f'  Server ready.')

        # 3. Run benchmark for each nutrient
        model_entry: dict = {'model': model_name, 'status': 'ok', 'results': {}}
        t_model_start = time.time()

        for nutrient in args.nutrients:
            print(f'  Running: mode={args.mode}, nutrient={nutrient}, '
                  f'limit={args.limit or "all"}', flush=True)
            t0 = time.time()
            success = run_bench(
                mode=args.mode, nutrient=nutrient,
                limit=args.limit, port=args.port,
                concurrent=args.concurrent,
                output_dir=model_out,
            )
            elapsed = round(time.time() - t0, 1)

            if success:
                metrics = read_latest_result(model_out, args.mode, nutrient) or {}
                model_entry['results'][nutrient] = {**metrics, 'elapsed_s': elapsed}
                acc  = metrics.get('acc', '?')
                mae  = metrics.get('mae', '?')
                print(f'    → acc={acc}  mae={mae}  ({elapsed}s)')
            else:
                model_entry['results'][nutrient] = {'status': 'failed', 'elapsed_s': elapsed}
                print(f'    → FAILED ({elapsed}s)')

        model_entry['total_elapsed_s'] = round(time.time() - t_model_start, 1)

        # 4. Stop server
        print(f'  Stopping server...', flush=True)
        stop_server(proc)
        print(f'  Done ({model_entry["total_elapsed_s"]}s total)')

        summary.append(model_entry)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f'\n{"="*64}')
    print('  SWEEP SUMMARY')
    print(f'{"="*64}')

    nutrients = args.nutrients
    col_w = 38

    # Header
    header = f'{"Model":<{col_w}}'
    for n in nutrients:
        header += f'  {"Acc@7.5g":>9}  {"MAE":>7}  ({n})'
    print(header)
    print('-' * (col_w + len(nutrients) * 25))

    for entry in summary:
        if entry.get('status') != 'ok':
            print(f'{entry["model"]:<{col_w}}  {entry["status"]}')
            continue
        row = f'{entry["model"]:<{col_w}}'
        for n in nutrients:
            r = entry['results'].get(n, {})
            acc = f'{r["acc"]:.4f}' if 'acc' in r else '  —  '
            mae = f'{r["mae"]:.2f}'  if 'mae' in r else ' — '
            row += f'  {acc:>9}  {mae:>7}'
        print(row)

    # Save summary JSON
    summary_path = os.path.join(sweep_dir, f'sweep_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump({'timestamp': timestamp, 'mode': args.mode,
                   'nutrients': nutrients, 'limit': args.limit,
                   'models': summary}, f, indent=2)
    print(f'\nSummary saved → {summary_path}')


if __name__ == '__main__':
    main()
