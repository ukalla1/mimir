# EB-Alfred Task Subset Preview

Three example tasks per subset, pulled from [splits.json](../EmbodiedBench/embodiedbench/envs/eb_alfred/data/splits/splits.json) — the same file `EBAlfEnv` loads at runtime. Each subset contains exactly **50 tasks**; the examples below are indexes 0, 24, and 49 to span the dataset.

> Regenerate at any time: `python EmbodiedBench/preview.py` (defaults to 3 examples per subset). Pass `--n 5` for more, `--subsets base spatial` to filter, or `--indexes 0 10 20` for explicit task picks.

---

## Benchmark mechanics: inputs, outputs, success criteria

What the model sees, what it must return, and how success is decided. Anchored to the running example *"Rinse off a ladle and move it to the table."* (`base[0]`, FloorPlan4).

### 1. Inputs to the model (per step)

One **RGB image + one large text prompt**, packed as a single OpenAI-compatible chat message.

**Image**: 500×500 RGB rendered from the AI2-THOR agent's head camera, base64-encoded into a `data:image/png;base64,…` URL inside an `image_url` content block.

**Text prompt**, constructed by [`vlm_planner.py:process_prompt()`](../EmbodiedBench/embodiedbench/planner/vlm_planner.py#L51) and `get_message()`:

| Section | Content |
|---|---|
| System prompt | Task framing + output format spec |
| In-context examples | `n_shots=10` example trajectories from [`alfred_prompt_examples.json`](../EmbodiedBench/embodiedbench/envs/eb_alfred/data/alfred_prompt_examples.json) |
| Available actions | All ~210 EB-Alfred actions enumerated as `action id 0: find a Apple, action id 1: pick up the Apple, …` |
| User instruction | The task line from `splits.json` (e.g. *"Rinse off a ladle and move it to the table."*) |
| Action history | What's been tried so far this episode, with env feedback |
| **Persistent memory** *(Direction B addition)* | `## Previously observed objects (across earlier steps in this episode):` list with first/last seen step |
| Output instruction | "Output in json. Describe the visual state, reason, then output the executable plan with action IDs from the available actions." |

### 2. What the model must output

A **JSON object with 4 required fields**, schema-constrained by llama-server (see [`vlm_generation_guide`](../EmbodiedBench/embodiedbench/planner/planner_config/generation_guide.py#L1)):

```json
{
  "visual_state_description": "I see a kitchen sink with a ladle and faucet visible…",
  "reasoning_and_reflection": "The instruction asks to rinse and move a ladle. I should pick it up, take it to the faucet, clean it, then carry it to the dining table.",
  "language_plan": "1. Find the ladle. 2. Pick it up. 3. Find the sink. 4. Clean the ladle. 5. Find the dining table. 6. Put the ladle on the table.",
  "executable_plan": [
    {"action_id": 109, "action_name": "find a Ladle"},
    {"action_id": 18,  "action_name": "pick up the Ladle"},
    {"action_id": 142, "action_name": "find a SinkBasin"},
    {"action_id": 87,  "action_name": "clean the Ladle"},
    {"action_id": 124, "action_name": "find a DiningTable"},
    {"action_id": 56,  "action_name": "put down the Ladle"}
  ]
}
```

Only `executable_plan` is **operationally consumed** — [`vlm_planner.py:json_to_action()`](../EmbodiedBench/embodiedbench/planner/vlm_planner.py#L153) extracts `[x["action_id"] for x in json_object["executable_plan"]]`. The other three fields are for logging / debugging and they help the model reason chain-of-thought style.

### 3. How an episode plays out

The planner can emit **multiple actions per call** (a plan, not a single action). The evaluator loop:

1. Planner → returns a list of action IDs (e.g. 6 actions for the ladle example)
2. Env executes them **one by one**:
   - If an action succeeds → execute next planned action
   - If an action fails (object not visible, invalid receptacle, etc.) → stop executing the plan, call planner again with updated state + env feedback
3. Episode ends when one of:
   - `task_success == 1` (all goal predicates satisfied)
   - `num_steps >= 30` (step cap → failure)
   - `num_invalid_actions >= max_invalid` (too many invalid actions → failure)

The `Invalid action or task complete. If invalid then Replanning.` line that scrolls during eval is the evaluator transitioning between plan-execution and re-planning calls.

### 4. How success is evaluated

Three metrics, all derived from AI2-THOR's ground-truth task module. From [`EBAlfEnv.step()` line 313-317](../EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py#L313-L317):

```python
reward, done    = self.env.get_transition_reward()      # shaped reward
subgoal_met     = self.env.get_goal_conditions_met()    # (met_count, total)
info['task_success']  = float(self.env.get_goal_satisfied())   # 0 or 1
info['task_progress'] = subgoal_met[0] / subgoal_met[1]        # 0.0–1.0
```

| Metric | Range | What it means |
|---|---|---|
| `task_success` | 0.0 or 1.0 | **Binary win/loss.** All goal predicates from the task's `pddl_params` are satisfied. The headline number in `final_results`. |
| `task_progress` | 0.0–1.0 | **Partial credit.** Fraction of subgoals met. E.g. if the task is "pick up ladle, clean it, place on table" and the agent did the first two, `task_progress = 0.667`. Distinguishes "got close" from "completely lost". |
| `reward` | float, usually negative | **Shaped per-step signal.** Positive for subgoal completion, negative for invalid actions / step-cost. Episode-final value is the cumulative sum. Useful for diagnosing how confused the planner got. |

Goal predicates are task-specific and live inside the trial's `traj_data.json`. For *"Rinse off a ladle and move it to the table"* the predicates are roughly:

- `is_clean(Ladle)` ✓
- `on(Ladle, DiningTable)` ✓
- Both must be true → `task_success = 1`

**No human judgment, no LLM-as-judge** — every metric is computed by AI2-THOR comparing the current scene predicate state to the task's ground-truth goal state.

### TL;DR

- **Input**: 1 image + ~10k-token text prompt (system + 10 ICL examples + actions list + instruction + history + memory block)
- **Output**: JSON with `executable_plan: [{action_id: int, action_name: str}, …]`
- **Evaluated by**: AI2-THOR's `get_goal_satisfied()` (binary) and `get_goal_conditions_met()` (partial credit) checking ground-truth scene predicates

---

## Big-picture observation about subset design

The first three subsets — **base**, **common_sense**, **complex_instruction** — are **the same 50 underlying tasks rephrased three different ways**. They test how robust the model is to instruction style on the same scene/goal. Note how indexes 0, 24, and 49 reference identical task families across these three subsets (e.g. `pick_clean_then_place_in_recep-Ladle-None-DiningTable-4` appears in all three).

The other three subsets — **spatial**, **visual_appearance**, **long_horizon** — are **distinct task families** chosen to stress specific capabilities (spatial relations, visual identification by appearance, multi-step planning).

| Subset | What it tests | Shares tasks with |
|---|---|---|
| `base` | Plain, literal instructions | common_sense, complex_instruction |
| `common_sense` | Inferring objects from descriptions ("something for serving soup" = ladle) | base, complex_instruction |
| `complex_instruction` | Following instructions with verbose / distractor phrasing | base, common_sense |
| `spatial` | Spatial-relation reasoning ("in the drawer under the sink") | — (its own 50 tasks) |
| `visual_appearance` | Object identification by color/shape ("the red rag") | — (its own 50 tasks) |
| `long_horizon` | Multi-step plans (heat → slice → place, etc.) | — (its own 50 tasks) |

---

## `base` — plain instructions

50 tasks. Direct, literal phrasing — names the object and the target receptacle.

| # | Scene | Instruction |
|---|---|---|
| 0 | FloorPlan4 | "Rinse off a ladle and move it to the table." |
| 24 | FloorPlan229 | "move the book to the couch" |
| 49 | FloorPlan8 | "Put a clean ice cream ladle on the counter." |

Task families: `pick_clean_then_place_in_recep-Ladle-None-DiningTable-4`, `pick_and_place_simple-Book-None-Sofa-229`, `pick_clean_then_place_in_recep-Ladle-None-CounterTop-8`.

---

## `common_sense` — inferred object from description

50 tasks. Same underlying tasks as `base`, but the target object is described by **function / category** rather than named. Model must apply common-sense knowledge to map description → object.

| # | Scene | Instruction |
|---|---|---|
| 0 | FloorPlan4 | "Rinse off something for serving soup and move it to the table." *(soup-serving thing = ladle)* |
| 24 | FloorPlan23 | "Position a transparent drinking vessel with a cutting tool for butter in it on the table." *(cup + butter knife)* |
| 49 | FloorPlan326 | "Place a timekeeping device in a round container on the shelf." *(watch in a bowl)* |

---

## `complex_instruction` — verbose / distractor phrasing

50 tasks. Same underlying tasks as `base`, with added pleasantries, context, or **explicit distractor objects** that should be ignored. Tests whether the model can extract the actual goal from noisy instruction text.

| # | Scene | Instruction |
|---|---|---|
| 0 | FloorPlan4 | "Although there's a bowl in the cupboard, rinse off a ladle and move it to the table." *(distractor: "bowl in the cupboard")* |
| 24 | FloorPlan229 | "When organizing reading spaces, kindly move the book to the couch. It's perfect for when the next chapter awaits." *(verbose framing)* |
| 49 | FloorPlan8 | "After sprucing up the kitchen, place a clean ice cream ladle on the counter. It's all set for those delightful dessert moments." *(verbose framing)* |

---

## `spatial` — spatial-relation reasoning

50 tasks. Distinct task family. Instructions reference **spatial relationships** between objects ("in the drawer under the sink", "next to the toilet", "back of the toilet") that require the model to understand 3D scene layout.

| # | Scene | Instruction |
|---|---|---|
| 0 | FloorPlan27 | "wash the ladle and put it back on the table" |
| 24 | FloorPlan423 | "Put two rolls of toilet paper in a drawer under the sink." *(specific drawer: "under the sink")* |
| 49 | FloorPlan415 | "knock an empty toilet paper roll off of the back of the toilet while placing a spray bottle next to it" *("back of", "next to")* |

This is one of the two subsets Direction B's persistent-memory port is predicted to help most. Q4_K_M baseline: 0.32 task_success.

---

## `visual_appearance` — identification by appearance

50 tasks. Distinct task family. Objects are described by **color, shape, or visual property** rather than name. Model must do visual identification from the rendered RGB observation.

| # | Scene | Instruction |
|---|---|---|
| 0 | FloorPlan429 | "Place two yellow bottles on top of a toilet" *(color)* |
| 24 | FloorPlan405 | "Put the red rag into the bathtub" *(color)* |
| 49 | FloorPlan418 | "To move two bars of soap to the black bin." *(color of container)* |

Q4_K_M baseline showed this as a relative strength (~0.6 in smoke) — model leverages its vision stack well when given color cues.

---

## `long_horizon` — multi-step plans

50 tasks. Distinct task family. Each instruction requires **multiple sub-skills chained together** (heat + slice + place, clean + cool + transfer, etc.). Step budget is the same 30-step cap as other subsets, so the planner has less slack.

| # | Scene | Instruction |
|---|---|---|
| 0 | FloorPlan11 | "Warming up an apple slice and put it in a garbagecan." *(slice → microwave → place)* |
| 24 | FloorPlan12 | "Put a cooked slice of tomato on a kitchen counter." *(slice → cook → place)* |
| 49 | FloorPlan1 | "Put two heads of lettuce into the fridge and use a knife to slice them." *(two objects + slice + place)* |

The other subset Direction B's memory port is predicted to help most. Q4_K_M baseline: 0.18 task_success — the weakest of all six. Smoke run (memory ON) at ratio=0.1 hit 0.4 task_success (2/5), though n=5 is noisy.

---

## Reference

- Source: [splits.json](../EmbodiedBench/embodiedbench/envs/eb_alfred/data/splits/splits.json)
- Extraction script: [EmbodiedBench/preview.py](../EmbodiedBench/preview.py)
- Per-task trial JSONs live under [data/json_2.1.0/{family}/trial_*/traj_data.json](../EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0/) — these contain the AI2-THOR scene state, ground-truth action sequence, and turker annotations referenced by each split entry.
- Evaluator (loads splits at runtime): [EBAlfEnv.py:214-221](../EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py#L214-L221)
