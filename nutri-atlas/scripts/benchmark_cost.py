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

Token counting tries the Qwen transformers tokenizer first; falls back to a
character-based estimate (~4 chars/token) if not available.

Usage (run from nutri-atlas/):
    python scripts/benchmark_cost.py                        # built-in queries
    python scripts/benchmark_cost.py --level easy           # easy only
    python scripts/benchmark_cost.py --out results.json     # save raw records
    python scripts/benchmark_cost.py --queries benchmark_tasks.yaml
"""
import argparse
import json
import os
import re
import sys
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
    query_id:     str
    level:        str
    query:        str
    total_time_s: float = 0.0
    llm_turns:    list  = field(default_factory=list)      # list[LLMTurnRecord]
    final_reply:  str   = ''

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
def run_turn_measured(messages: list, record: QueryRecord) -> str:
    """
    Mirrors robot_assistant._run_turn but records timing and token metrics
    into the given QueryRecord.
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

        for call in tool_calls:
            name   = call.get('name', '')
            args   = call.get('arguments', call.get('parameters', {}))
            args_s = json.dumps(args)
            tool   = TOOLS.get(name)

            t1     = time.perf_counter()
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
def run_query(query_id: str, level: str, query: str) -> QueryRecord:
    record   = QueryRecord(query_id=query_id, level=level, query=query)
    messages = [SYSTEM_MSG, {'role': 'user', 'content': query}]

    print(f'\n[{query_id}] {query}')
    t0 = time.perf_counter()
    reply = run_turn_measured(messages, record)
    record.total_time_s = time.perf_counter() - t0
    record.final_reply  = reply

    _print_query_result(record)
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_query_result(r: QueryRecord):
    print(f'  Time   : {r.total_time_s:.2f}s total  '
          f'(LLM {r.total_llm_time_s:.2f}s  |  tools {r.total_tool_time_s:.2f}s)')
    print(f'  Tokens : {r.total_tokens} total  '
          f'(in {r.total_input_tokens}  out {r.total_output_tokens}  results {r.total_result_tokens})')
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

    print(f'\n{"─"*70}')
    print('Per-Query')
    print(f'{"ID":<4} {"Level":<12} {"Query":<30} {"Time(s)":>8} {"Tokens":>8} {"Calls":>6}')
    print(f'{"─"*70}')
    for r in records:
        q = (r.query[:28] + '..') if len(r.query) > 30 else r.query
        print(f'{r.query_id:<4} {r.level:<12} {q:<30} '
              f'{r.total_time_s:>8.2f} {r.total_tokens:>8} {r.num_tool_calls:>6}')

    n = len(records)
    if n:
        avg_time = sum(r.total_time_s   for r in records) / n
        avg_tok  = sum(r.total_tokens   for r in records) / n
        avg_call = sum(r.num_tool_calls for r in records) / n
        print(f'{"─"*70}')
        print(f'{"Average":<47} {avg_time:>8.2f} {avg_tok:>8.0f} {avg_call:>6.1f}')


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
    args = parser.parse_args()

    if args.queries:
        queries = load_queries(args.queries, args.level)
    else:
        queries = DEFAULT_QUERIES
        if args.level:
            queries = [q for q in queries if q['level'] == args.level]

    print(f'Running {len(queries)} queries  (token mode: {TOKEN_MODE})')
    print('Make sure the LLM server is running on http://localhost:8080/v1\n')

    records = []
    for q in queries:
        records.append(run_query(q['id'], q['level'], q['prompt']))

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
