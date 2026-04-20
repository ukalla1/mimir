# Agent Cost Benchmark — Results Analysis

**Date:** 2026-04-05  
**Model:** Qwen3.5-9B-GGUF (llama-server, port 8080)  
**Token counting:** exact (Qwen2.5 transformers tokenizer)  
**Robot connection:** none — ZMQ servers not running (all robot tool calls timed out)

---

## Important Caveat: All ZMQ Calls Timed Out

The robot-side ZMQ servers were not running during this benchmark. As a result, all tools that communicate with the robot hit their configured timeouts:

| Tool | Timeout (s) | Cause |
|------|------------|-------|
| `navigate_to_landmark` | 60.0 | ZMQ REQ `RCVTIMEO` in `zmq_client.py` |
| `spin_robot` | 60.0 | same |
| `get_detected_objects` | 3.0 | separate object server (port 5556) |
| `get_current_detected_objects` | 10.0 | ZMQ bridge (port 5555) |

The execution times for these tools are entirely timeout overhead, **not actual robot latency**. Only two tools reflect real computation:
- `list_landmarks` — pure local YAML lookup, ~0 ms
- `get_meal_recommendation` — real RAG pipeline, ~11.8 s

---

## 1. Tool Execution Time

| Tool | Calls | Avg exec (s) | Nature |
|------|-------|-------------|--------|
| `list_landmarks` | 3 | 0.000 | Local YAML lookup |
| `get_meal_recommendation` | 1 | 11.799 | RAG + LLM (nutri-atlas pipeline) |
| `get_detected_objects` | 4 | 3.003 | ZMQ timeout (3 s) |
| `get_current_detected_objects` | 5 | 10.011 | ZMQ timeout (10 s) |
| `navigate_to_landmark` | 6 | 60.050 | ZMQ timeout (60 s) |
| `spin_robot` | 4 | 60.053 | ZMQ timeout (60 s) |

**Key observations:**

- `list_landmarks` is essentially free (36 µs). It is the only tool that requires no network hop and is backed by an in-memory YAML structure.
- `get_meal_recommendation` at 11.8 s is the most computationally expensive real tool. This is the full nutri-rag pipeline: text embedding → GAT retrieval → LLM nutritional reasoning.
- The 60 s timeout for navigate/spin is very long. In a real deployment where the robot is connected, actual navigation times will vary (10–120 s depending on distance), but the timeout sets the worst-case waiting time the LLM thread must block.
- The 3 s timeout for `get_detected_objects` is the least disruptive. Consider reducing `get_current_detected_objects` from 10 s to 3–5 s for faster failure recovery.

---

## 2. Token Cost

### Per-tool result token size

| Tool | Avg result tokens | Comment |
|------|------------------|---------|
| `get_detected_objects` | 19 | Timeout error message (short) |
| `spin_robot` | 56 | Timeout error message |
| `navigate_to_landmark` | 58 | Timeout error message |
| `get_current_detected_objects` | 57 | Timeout error message |
| `list_landmarks` | 162 | Full landmark list with coords |
| `get_meal_recommendation` | 680 | Detailed nutritional recommendation |

**Key observations:**

- `get_meal_recommendation` produces 680 tokens per call — 4× larger than any other tool result. This is injected back into the LLM context, substantially increasing the cost of the final reply turn.
- `list_landmarks` (162 tokens) is disproportionately large for what it does. It returns full coordinates for all 5 landmarks every time, even when only one is needed. A lighter response format could reduce this.
- Timeout error messages (~19–58 tokens) are cheap but cause the LLM to re-plan, leading to additional turns and higher cumulative token cost.

### Context growth (input tokens over turns)

The LLM context accumulates with each turn. T4 ("Go to the kitchen") shows the worst case:

| Turn | Input tokens | What was added |
|------|-------------|----------------|
| 1 | 37 | System + user query |
| 2 | 120 | + navigate timeout result |
| 3 | 222 | + second navigate timeout |
| 4 | 440 | + list_landmarks result (162 tok) |
| 5 | 562 | + third navigate with coords |
| 6 | 665 | + spin timeout result |

Context grew **18× from turn 1 to turn 6** (37 → 665 tokens), driven by repeated failure results being appended. In a real session with a connected robot and successful tool calls, growth would be similar — each result adds to the permanent context.

---

## 3. LLM Inference Time

| Query | LLM turns | Total LLM time (s) | Avg per turn (s) |
|-------|-----------|-------------------|-----------------|
| T1 — list landmarks | 2 | 1.93 | 0.96 |
| T2 — what can you see | 4 | 3.46 | 0.87 |
| T3 — detected objects | 4 | 2.38 | 0.59 |
| T4 — go to kitchen | 6 | 4.32 | 0.72 |
| T5 — spin 90° | 2 | 1.37 | 0.69 |
| T6 — meal recommendation | 2 | 4.90 | 2.45 |
| T7 — go to bedroom | 6 | 4.63 | 0.77 |
| T8 — 360° scan | 5 | 4.40 | 0.88 |

**Key observations:**

- LLM inference per turn is fast and consistent: **0.6–1.1 s** for navigation/object queries (short outputs). The outlier is T6 turn 2: **4.18 s** because the model generated a 341-token meal recommendation reply.
- LLM inference time scales with **output length**, not input length — context size has negligible effect on generation speed at this scale.
- Total LLM time across all queries is **27.4 s**, vs **695 s** total tool execution time. The LLM itself accounts for only **~4%** of total wall time — all bottlenecks are in tool execution.

---

## 4. Per-Query Summary

| ID | Query | Total (s) | LLM (s) | Tools (s) | Tool calls | Tokens |
|----|-------|-----------|---------|-----------|-----------|--------|
| T1 | List landmarks | 1.93 | 1.93 | 0.00 | 1 | 542 |
| T6 | Meal recommendation | 16.71 | 4.90 | 11.80 | 1 | 1891 |
| T3 | Detected objects (persistent) | 18.40 | 2.38 | 16.02 | 3 | 757 |
| T2 | Current detected objects | 26.49 | 3.46 | 23.02 | 3 | 1085 |
| T5 | Spin 90° | 61.41 | 1.37 | 60.03 | 1 | 286 |
| T8 | 360° scan | 137.54 | 4.40 | 133.13 | 4 | 1939 |
| T7 | Go to bedroom + look | 194.68 | 4.63 | 190.09 | 5 | 2831 |
| T4 | Go to kitchen | 244.76 | 4.32 | 240.43 | 5 | 2765 |

**Cheapest query:** T1 (`list_landmarks` only) — 1.93 s, 542 tokens, 1 tool call.  
**Most expensive query:** T4 (`navigate` × 3 + `list_landmarks` + `spin`) — 244 s, 2765 tokens. All dominated by 60 s timeouts.

---

## 5. Agent Re-planning Behaviour Under Failure

Even without a connected robot, the results reveal how the LLM re-plans under repeated failure:

**T4 / T7 — Navigation failures (navigate → navigate → list_landmarks → navigate with coords → spin):**
The LLM attempted `navigate_to_landmark` by name, failed twice, then called `list_landmarks` to verify the coordinates, then retried with explicit (x, y). When that also timed out, it tried `spin_robot` as a fallback diagnostic. This 5-call chain is correct recovery logic but expensive: **240 s and 2700+ tokens** spent entirely on timeouts.

**T2 / T3 — Object detection failures (get_current → get_current → get_detected):**
The LLM escalated from live camera (`get_current_detected_objects`, 10 s timeout) to the persistent map (`get_detected_objects`, 3 s timeout) after two live failures. Sensible escalation, but still adds 23 s of wait time for what should be a 0.1 s lookup.

**Implication:** The agent's retry/recovery logic is functionally correct but generates significant cost when tools are slow or unavailable. Reducing timeouts or adding a failure fast-path in the system prompt would materially reduce cost in degraded conditions.

---

## 6. Key Findings

1. **Tool execution time dominates total cost.** LLM inference is only ~4% of total wall time. Optimising the model has negligible impact on end-to-end latency — the bottleneck is always tool execution.

2. **`get_meal_recommendation` is the token-heaviest tool** (680 result tokens). This directly inflates the context of the final LLM reply turn, which is why T6 has both the largest output token count (341) and the slowest single LLM turn (4.18 s).

3. **Navigate and spin timeouts (60 s) are the primary cost driver.** For any query requiring movement, one timeout alone makes the query >60 s. The 60 s ZMQ timeout in `zmq_client.py` should be tuned to match actual robot responsiveness in deployment.

4. **Context bloat from repeated failures is significant.** Failed tool results are permanently appended to the conversation context. A query requiring 5 tool calls with failures can grow input context from 37 to 665 tokens, increasing per-turn inference cost and eventually approaching context window limits in long sessions.

5. **`list_landmarks` (162 tokens) returns more than needed.** Each call injects all 5 landmarks with full coordinates. A targeted lookup would reduce context growth in navigation-heavy tasks.

---

## 7. Recommendations

| Issue | Recommendation |
|-------|---------------|
| 60 s navigate/spin timeouts | Reduce to 10–15 s for bench; tune to actual robot latency in deployment |
| 10 s `get_current_objects` timeout | Reduce to 3–5 s to match `get_detected_objects` |
| `get_meal_recommendation` 680-token result | Truncate or summarise the RAG result before returning to LLM |
| `list_landmarks` 162-token result | Return only the requested landmark if a name is known; full list only when needed |
| Context bloat from failures | Add a retry limit (max 2) in the system prompt to prevent unbounded re-planning loops |
| Re-run benchmark with robot connected | Current results only measure timeout/failure paths; real execution times are needed for navigate and spin |
