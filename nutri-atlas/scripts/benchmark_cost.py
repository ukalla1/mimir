"""
Cost benchmark for the robot assistant agent.

Measures per-query and per-tool:
  Time costs:
    - LLM inference time  (seconds per turn)
    - Tool execution time (seconds per call)
    - End-to-end task time (seconds per query)
  Token costs:
    - Input tokens per LLM turn  (messages context sent in)
    - Output tokens per LLM turn (response generated)
    - Tool result tokens per call (function result injected back)
  Power costs (operator PC only, not robot side):
    - GPU avg/peak power in watts  (nvidia-smi, no sudo needed)
    - CPU avg utilisation in %     (psutil, proxy for CPU power)

Token counting tries the Qwen transformers tokenizer first; falls back to a
character-based estimate (~4 chars/token) if not available.

Usage (run from nutri-atlas/):
    python scripts/benchmark_cost.py                        # built-in queries
    python scripts/benchmark_cost.py --level easy           # easy only
    python scripts/benchmark_cost.py --out results.json     # save raw records
    python scripts/benchmark_cost.py --queries benchmark_tasks.yaml
    python scripts/benchmark_cost.py --mock                 # stub robot tools
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import yaml

# Tools live in robot_control/ — add it to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'robot_control'))

from qwen_agent.llm import get_chat_model
from tools.navigate_tool import NavigateToLandmark, ListLandmarks
from tools.object_tool import GetDetectedObjects, GetCurrentDetectedObjects
from tools.motion_tool import SpinRobot, MoveRobot
from tools.nutrition_tool import GetMealRecommendation


# ---------------------------------------------------------------------------
# Token counter — tries Qwen tokenizer, falls back to char estimate
# ---------------------------------------------------------------------------
def _build_token_counter():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True
        )
        print('[tokenizer] Qwen2.5 transformers tokenizer loaded')
        return lambda text: len(tok.encode(str(text))), 'exact'
    except Exception:
        print('[tokenizer] transformers unavailable — using char/4 estimate')
        return lambda text: max(1, len(str(text)) // 4), 'estimate'


count_tokens, TOKEN_MODE = _build_token_counter()


# ---------------------------------------------------------------------------
# Power monitor — polls nvidia-smi + psutil in a background thread.
# Start before a query, stop after, then read .result().
# ---------------------------------------------------------------------------
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    print('[power] psutil not installed — CPU% will not be measured')

def _gpu_power_w() -> float | None:
    """Return current GPU power draw in watts via nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        return float(out.decode().strip().split('\n')[0])
    except Exception:
        return None


class PowerMonitor:
    """Polls GPU watts (nvidia-smi) and CPU % (psutil) at a fixed interval."""

    def __init__(self, interval_s: float = 0.5):
        self._interval = interval_s
        self._gpu_samples: list[float] = []
        self._cpu_samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._stop.clear()
        self._gpu_samples.clear()
        self._cpu_samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _poll(self):
        while not self._stop.is_set():
            w = _gpu_power_w()
            if w is not None:
                self._gpu_samples.append(w)
            if _PSUTIL_OK:
                self._cpu_samples.append(psutil.cpu_percent(interval=None))
            self._stop.wait(self._interval)

    def result(self) -> dict:
        gpu_avg  = sum(self._gpu_samples) / len(self._gpu_samples) if self._gpu_samples else None
        gpu_peak = max(self._gpu_samples) if self._gpu_samples else None
        cpu_avg  = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else None
        return {'gpu_avg_w': gpu_avg, 'gpu_peak_w': gpu_peak, 'cpu_avg_pct': cpu_avg}


# ---------------------------------------------------------------------------
# Metric data structures
# ---------------------------------------------------------------------------
@dataclass
class ToolCallRecord:
    tool_name:        str
    args_tokens:      int
    result_tokens:    int
    execution_time_s: float


@dataclass
class LLMTurnRecord:
    input_tokens:     int
    output_tokens:    int
    inference_time_s: float
    tool_calls:       list = field(default_factory=list)   # list[ToolCallRecord]


@dataclass
class QueryRecord:
    query_id:      str
    level:         str
    query:         str
    total_time_s:  float = 0.0
    llm_turns:     list  = field(default_factory=list)     # list[LLMTurnRecord]
    final_reply:   str   = ''
    gpu_avg_w:     float | None = None
    gpu_peak_w:    float | None = None
    cpu_avg_pct:   float | None = None

    # ── Aggregates ──────────────────────────────────────────────────────────
    @property
    def total_input_tokens(self):
        return sum(t.input_tokens for t in self.llm_turns)

    @property
    def total_output_tokens(self):
        return sum(t.output_tokens for t in self.llm_turns)

    @property
    def total_result_tokens(self):
        return sum(tc.result_tokens for t in self.llm_turns for tc in t.tool_calls)

    @property
    def total_tokens(self):
        return self.total_input_tokens + self.total_output_tokens + self.total_result_tokens

    @property
    def total_llm_time_s(self):
        return sum(t.inference_time_s for t in self.llm_turns)

    @property
    def total_tool_time_s(self):
        return sum(tc.execution_time_s for t in self.llm_turns for tc in t.tool_calls)

    @property
    def num_tool_calls(self):
        return sum(len(t.tool_calls) for t in self.llm_turns)

    @property
    def num_llm_turns(self):
        return len(self.llm_turns)


# ---------------------------------------------------------------------------
# LLM + tools setup (mirrors robot_assistant.py)
# ---------------------------------------------------------------------------
LLM = get_chat_model({
    'model':        'unsloth/Qwen3.5-9B-GGUF',
    'model_type':   'qwenvl_oai',
    'model_server': 'http://localhost:8080/v1',
    'api_key':      'EMPTY',
    'generate_cfg': {'thought_in_content': True},
})

TOOL_DEFINITIONS = [
    {'type': 'function', 'function': {
        'name': 'list_landmarks',
        'description': 'List all named landmarks the robot can navigate to.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'navigate_to_landmark',
        'description': 'Navigate the robot to a named landmark or (x, y) coordinates.',
        'parameters': {'type': 'object', 'properties': {
            'landmark_name': {'type': 'string'},
            'x': {'type': 'number'},
            'y': {'type': 'number'},
        }},
    }},
    {'type': 'function', 'function': {
        'name': 'get_detected_objects',
        'description': 'Return the full persistent map of all detected objects.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'get_current_detected_objects',
        'description': 'Return only objects the camera sees right now.',
        'parameters': {'type': 'object', 'properties': {}},
    }},
    {'type': 'function', 'function': {
        'name': 'spin_robot',
        'description': 'Rotate the robot in place by a given angle in degrees.',
        'parameters': {'type': 'object', 'properties': {
            'angle_deg': {'type': 'number'},
        }, 'required': ['angle_deg']},
    }},
    {'type': 'function', 'function': {
        'name': 'get_meal_recommendation',
        'description': 'Recommend what to eat next based on nutritional gap analysis.',
        'parameters': {'type': 'object', 'properties': {
            'eaten_foods':     {'type': 'string'},
            'eaten_meal_type': {'type': 'string'},
            'next_meal':       {'type': 'string'},
        }, 'required': ['eaten_foods']},
    }},
]

TOOLS = {
    'list_landmarks':               ListLandmarks(),
    'navigate_to_landmark':         NavigateToLandmark(),
    'get_detected_objects':         GetDetectedObjects(),
    'get_current_detected_objects': GetCurrentDetectedObjects(),
    'spin_robot':                   SpinRobot(),
    'get_meal_recommendation':      GetMealRecommendation(),
}

# ---------------------------------------------------------------------------
# Mock tool responses — used when --mock flag is set.
# Returns instant realistic responses with no ZMQ/robot connection needed.
# get_meal_recommendation is NOT mocked — it runs the real RAG pipeline.
# ---------------------------------------------------------------------------
def _mock_call(name: str, args: dict) -> str:
    import uuid
    goal_id = str(uuid.uuid4())

    if name == 'list_landmarks':
        # Same format as the real LandmarkLoader.list_all()
        return json.dumps({
            'bathroom':    {'x': 2.86,  'y': -10.68, 'description': 'Bathroom'},
            'hallway':     {'x': 2.40,  'y': -4.64,  'description': 'Hallway'},
            'kitchen':     {'x': -4.88, 'y': -3.46,  'description': 'Kitchen area'},
            'bedroom':     {'x': -3.23, 'y': -8.92,  'description': 'Bedroom'},
            'living_room': {'x': 6.53,  'y': -1.46,  'description': 'Living room'},
        })

    if name == 'navigate_to_landmark':
        target = args.get('landmark_name') or f"({args.get('x', 0):.2f}, {args.get('y', 0):.2f})"
        return json.dumps({'goal_id': goal_id, 'status': 'success',
                           'message': f'Arrived at {target}'})

    if name == 'spin_robot':
        angle = args.get('angle_deg', 0)
        return json.dumps({'goal_id': goal_id, 'status': 'success',
                           'message': f'Rotated {angle:.1f}°'})

    if name == 'move_robot':
        dist = args.get('distance_m', 0)
        return json.dumps({'goal_id': goal_id, 'status': 'success',
                           'message': f'Moved {dist:.2f}m'})

    if name == 'get_detected_objects':
        return json.dumps({'status': 'ok', 'objects': {
            'chair_1': {'px': 1.20, 'py': -0.50, 'conf': 0.91, 'label': 'chair'},
            'table_1': {'px': 2.30, 'py':  1.10, 'conf': 0.87, 'label': 'table'},
            'sofa_1':  {'px': 5.10, 'py': -2.30, 'conf': 0.79, 'label': 'sofa'},
        }})

    if name == 'get_current_detected_objects':
        return json.dumps({'goal_id': goal_id, 'status': 'ok', 'objects': {
            'chair_1': {'px': 1.20, 'py': -0.50, 'conf': 0.91, 'label': 'chair'},
            'cup_1':   {'px': 0.80, 'py':  0.30, 'conf': 0.83, 'label': 'cup'},
        }})

    return json.dumps({'status': 'failed', 'message': f'Unknown mock tool: {name}'})

SYSTEM_MSG = {
    'role': 'system',
    'content': (
        'You are a robot assistant that can navigate and provide nutritional advice. '
        'Use the available tools to fulfil user requests. '
        'Always reply in the same language the user uses.'
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_think(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _parse_tool_calls(text: str) -> list:
    calls = []
    for m in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def _messages_tokens(messages: list) -> int:
    return sum(count_tokens(m.get('content', '') or '') for m in messages)


# ---------------------------------------------------------------------------
# Instrumented turn loop
# ---------------------------------------------------------------------------
def run_turn_measured(messages: list, record: QueryRecord, mock: bool = False) -> str:
    """
    Mirrors robot_assistant._run_turn but records timing and token metrics
    into the given QueryRecord.

    mock=True: ZMQ robot tools return instant stub responses so LLM inference
               time and token costs can be measured without a robot connection.
               get_meal_recommendation always runs the real RAG pipeline.
    """
    while True:
        input_tok = _messages_tokens(messages)

        t0 = time.perf_counter()
        full_content = ''
        for chunks in LLM.chat(messages=messages, functions=TOOL_DEFINITIONS, stream=True):
            for chunk in chunks:
                if chunk.get('role') == 'assistant':
                    full_content = chunk.get('content', '')
        inference_time = time.perf_counter() - t0

        output_tok  = count_tokens(full_content)
        turn_record = LLMTurnRecord(
            input_tokens=input_tok,
            output_tokens=output_tok,
            inference_time_s=inference_time,
        )

        clean      = _strip_think(full_content)
        tool_calls = _parse_tool_calls(clean)
        messages.append({'role': 'assistant', 'content': clean or ' '})

        if not tool_calls:
            if not clean.strip():
                messages.append({'role': 'user', 'content': 'Continue.'})
                record.llm_turns.append(turn_record)
                continue
            record.llm_turns.append(turn_record)
            return clean

        _REAL_TOOLS = {'get_meal_recommendation'}   # always run real pipeline

        for call in tool_calls:
            name   = call.get('name', '')
            args   = call.get('arguments', call.get('parameters', {}))
            args_s = json.dumps(args)

            t1 = time.perf_counter()
            if mock and name not in _REAL_TOOLS:
                result = _mock_call(name, args)
            else:
                tool   = TOOLS.get(name)
                result = tool.call(args_s) if tool else json.dumps({'error': f'Unknown tool: {name}'})
            exec_t = time.perf_counter() - t1

            turn_record.tool_calls.append(ToolCallRecord(
                tool_name=name,
                args_tokens=count_tokens(args_s),
                result_tokens=count_tokens(result),
                execution_time_s=exec_t,
            ))
            messages.append({'role': 'function', 'name': name, 'content': result})

        record.llm_turns.append(turn_record)


# ---------------------------------------------------------------------------
# Run a single query and return its record
# ---------------------------------------------------------------------------
def run_query(query_id: str, level: str, query: str, mock: bool = False) -> QueryRecord:
    record   = QueryRecord(query_id=query_id, level=level, query=query)
    messages = [SYSTEM_MSG, {'role': 'user', 'content': query}]

    print(f'\n[{query_id}] {query}')
    monitor = PowerMonitor(interval_s=0.5)
    monitor.start()
    t0 = time.perf_counter()
    reply = run_turn_measured(messages, record, mock=mock)
    record.total_time_s = time.perf_counter() - t0
    monitor.stop()

    power = monitor.result()
    record.gpu_avg_w   = power['gpu_avg_w']
    record.gpu_peak_w  = power['gpu_peak_w']
    record.cpu_avg_pct = power['cpu_avg_pct']
    record.final_reply = reply

    _print_query_result(record)
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_power(r: QueryRecord) -> str:
    parts = []
    if r.gpu_avg_w is not None:
        parts.append(f'GPU avg {r.gpu_avg_w:.0f}W  peak {r.gpu_peak_w:.0f}W')
    if r.cpu_avg_pct is not None:
        parts.append(f'CPU avg {r.cpu_avg_pct:.0f}%')
    return '  |  '.join(parts) if parts else 'n/a'


def _print_query_result(r: QueryRecord):
    print(f'  Time   : {r.total_time_s:.2f}s total  '
          f'(LLM {r.total_llm_time_s:.2f}s  |  tools {r.total_tool_time_s:.2f}s)')
    print(f'  Tokens : {r.total_tokens} total  '
          f'(in {r.total_input_tokens}  out {r.total_output_tokens}  results {r.total_result_tokens})')
    print(f'  Power  : {_fmt_power(r)}')
    print(f'  Turns/Calls: {r.num_llm_turns} LLM turns, {r.num_tool_calls} tool calls')
    for i, turn in enumerate(r.llm_turns, 1):
        print(f'  Turn {i}: {turn.input_tokens} in / {turn.output_tokens} out tok  '
              f'{turn.inference_time_s:.2f}s')
        for tc in turn.tool_calls:
            print(f'    → {tc.tool_name:<32} {tc.execution_time_s:.3f}s  '
                  f'result={tc.result_tokens} tok')


def print_summary(records: list):
    # ── Per-tool aggregates ────────────────────────────────────────────────
    tool_stats: dict = {}
    for r in records:
        for turn in r.llm_turns:
            for tc in turn.tool_calls:
                s = tool_stats.setdefault(tc.tool_name, {
                    'count': 0, 'total_exec_s': 0.0, 'total_result_tok': 0
                })
                s['count']            += 1
                s['total_exec_s']     += tc.execution_time_s
                s['total_result_tok'] += tc.result_tokens

    print(f'\n{"═"*70}')
    print(f'SUMMARY  (token counts are {TOKEN_MODE})')
    print(f'{"═"*70}')

    print(f'\n{"─"*70}')
    print('Per-Tool Averages')
    print(f'{"Tool":<34} {"Calls":>5} {"Avg exec(s)":>12} {"Avg result tok":>15}')
    print(f'{"─"*70}')
    for name, s in sorted(tool_stats.items()):
        avg_t = s['total_exec_s']     / s['count']
        avg_r = s['total_result_tok'] / s['count']
        print(f'{name:<34} {s["count"]:>5} {avg_t:>12.3f} {avg_r:>15.0f}')

    print(f'\n{"─"*90}')
    print('Per-Query')
    print(f'{"ID":<4} {"Level":<12} {"Query":<28} {"Time(s)":>8} {"Tokens":>8} {"Calls":>6} {"GPU avg W":>10} {"GPU peak W":>11} {"CPU avg%":>9}')
    print(f'{"─"*90}')
    for r in records:
        q       = (r.query[:26] + '..') if len(r.query) > 28 else r.query
        gpu_avg  = f'{r.gpu_avg_w:.0f}'  if r.gpu_avg_w  is not None else 'n/a'
        gpu_peak = f'{r.gpu_peak_w:.0f}' if r.gpu_peak_w is not None else 'n/a'
        cpu_avg  = f'{r.cpu_avg_pct:.0f}' if r.cpu_avg_pct is not None else 'n/a'
        print(f'{r.query_id:<4} {r.level:<12} {q:<28} '
              f'{r.total_time_s:>8.2f} {r.total_tokens:>8} {r.num_tool_calls:>6} '
              f'{gpu_avg:>10} {gpu_peak:>11} {cpu_avg:>9}')

    n = len(records)
    if n:
        avg_time     = sum(r.total_time_s   for r in records) / n
        avg_tok      = sum(r.total_tokens   for r in records) / n
        avg_call     = sum(r.num_tool_calls for r in records) / n
        gpu_avgs     = [r.gpu_avg_w  for r in records if r.gpu_avg_w  is not None]
        gpu_peaks    = [r.gpu_peak_w for r in records if r.gpu_peak_w is not None]
        cpu_avgs     = [r.cpu_avg_pct for r in records if r.cpu_avg_pct is not None]
        avg_gpu_avg  = f'{sum(gpu_avgs)/len(gpu_avgs):.0f}'   if gpu_avgs  else 'n/a'
        avg_gpu_peak = f'{sum(gpu_peaks)/len(gpu_peaks):.0f}' if gpu_peaks else 'n/a'
        avg_cpu      = f'{sum(cpu_avgs)/len(cpu_avgs):.0f}'   if cpu_avgs  else 'n/a'
        print(f'{"─"*90}')
        print(f'{"Average":<45} {avg_time:>8.2f} {avg_tok:>8.0f} {avg_call:>6.1f} '
              f'{avg_gpu_avg:>10} {avg_gpu_peak:>11} {avg_cpu:>9}')


# ---------------------------------------------------------------------------
# Built-in test queries
# ---------------------------------------------------------------------------
DEFAULT_QUERIES = [
    {'id': 'T1', 'level': 'easy',   'prompt': 'List all available landmarks.'},
    {'id': 'T2', 'level': 'easy',   'prompt': 'What can you see right now?'},
    {'id': 'T3', 'level': 'easy',   'prompt': 'What objects have been detected so far?'},
    {'id': 'T4', 'level': 'easy',   'prompt': 'Go to the kitchen.'},
    {'id': 'T5', 'level': 'easy',   'prompt': 'Spin 90 degrees to the right.'},
    {'id': 'T6', 'level': 'easy',   'prompt': 'I had an apple and milk for breakfast. What should I eat for lunch?'},
    {'id': 'T7', 'level': 'medium', 'prompt': 'Go to the bedroom and tell me what you see.'},
    {'id': 'T8', 'level': 'medium', 'prompt': 'Do a full 360-degree scan and list every object you find.'},
]


def load_queries(path: str, level_filter: str | None) -> list:
    with open(path) as f:
        data = yaml.safe_load(f)
    queries = [
        {'id': t['id'], 'level': t['level'], 'prompt': t['prompt']}
        for t in data.get('tasks', [])
    ]
    if level_filter:
        queries = [q for q in queries if q['level'] == level_filter]
    return queries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Benchmark agent cost (time + tokens)')
    parser.add_argument('--queries', type=str, default=None,
                        help='YAML file with tasks (default: built-in queries)')
    parser.add_argument('--level', type=str, default=None,
                        choices=['easy', 'medium', 'challenging'],
                        help='Filter by difficulty level')
    parser.add_argument('--out', type=str, default=None,
                        help='Save raw records as JSON to this path')
    parser.add_argument('--mock', action='store_true',
                        help='Replace ZMQ robot tools with instant stubs (no robot needed). '
                             'get_meal_recommendation still runs the real RAG pipeline.')
    # Robot IP is read from the ROBOT_IP env var by each tool at import time.
    # Default in the tool files is 10.203.168.250 (neither real robot nor localhost).
    #
    # Real robot  : export ROBOT_IP=192.168.0.114  (shared WiFi)
    # Local sim   : export ROBOT_IP=127.0.0.1      (zmq_bridge_node.py on this machine)
    # --mock mode : ROBOT_IP is irrelevant — ZMQ is never called.
    args = parser.parse_args()

    if args.queries:
        queries = load_queries(args.queries, args.level)
    else:
        queries = DEFAULT_QUERIES
        if args.level:
            queries = [q for q in queries if q['level'] == args.level]

    print(f'Running {len(queries)} queries  (token mode: {TOKEN_MODE})'
          + ('  [MOCK — robot tools stubbed]' if args.mock else ''))
    print('Make sure the LLM server is running on http://localhost:8080/v1\n')

    records = []
    for q in queries:
        records.append(run_query(q['id'], q['level'], q['prompt'], mock=args.mock))

    print_summary(records)

    if args.out:
        import dataclasses
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_to_dict(i) for i in obj]
            return obj

        with open(args.out, 'w') as f:
            json.dump([_to_dict(r) for r in records], f, indent=2)
        print(f'\nRaw results saved to {args.out}')


if __name__ == '__main__':
    main()
