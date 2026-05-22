# Integrating nutri-atlas with EmbodiedBench — Feasibility Analysis

## Context

The current EmbodiedBench Phase 5 evaluation (see [plans/qwen35embodiedbench_test.md](qwen35embodiedbench_test.md)) tests the **base Qwen3.5-9B-Q4_K_M model** on EB-Alfred and EB-Habitat. It does NOT yet evaluate any nutri-atlas-specific pipeline contributions.

The question this doc explores: can nutri-atlas's robot-control patterns (specifically the persistent landmark memory model from [nutri-atlas/robot_control/robot_assistant.py](../nutri-atlas/robot_control/robot_assistant.py)) be tested on EmbodiedBench? If so, how — and what's the effort vs. value tradeoff?

## What nutri-atlas does vs. what EmbodiedBench expects

| Aspect | nutri-atlas / `robot_assistant.py` | EmbodiedBench |
|--------|-----------------------------------|--------------|
| LLM output | Free-form tool calls via `qwen-agent` (`navigate_to_landmark`, `scan_objects`, etc.) | JSON action sequence with **fixed integer IDs** from a per-env action vocabulary (~70-200 actions) |
| Action set | Open-ended, defined by tool registry | Fixed per scene — e.g. EB-Alfred has `find a Ladle = 109`, `pick up the Ladle = 18`, etc. |
| Memory | Persistent across calls (TF frames, `detected_objects.json`) | Per-episode only; `chat_history=True` keeps in-episode history but discarded at episode boundary |
| Simulator | Real Go2 + ROS2, or sim via ZMQ | AI2-THOR (eb-alf) or Habitat (eb-hab), called from Python directly |
| Agent loop | Multi-turn tool-calling, LLM decides when to act | Single-shot per step: VLM gets observation, returns action sequence |
| Observation format | RGB-D + ZMQ-bridged ROS2 messages | RGB image at fixed resolution (500x500 default) |
| Domain | Real apartments / lab spaces; nutrition tasks | Synthetic AI2-THOR rooms / ReplicaCAD apartments; household tasks |

The action-space mismatch is the biggest structural blocker — EmbodiedBench's evaluator expects integer action IDs, not tool names.

## Three possible directions

### Direction A — Treat EmbodiedBench as a way to *evaluate the planner LLM only*

What you're already doing in Phase 5. The LLM under test is Qwen3.5-9B-Q4_K_M (the same model nutri-atlas uses as its planner brain), so EmbodiedBench already gives you signal on how well that base model can plan embodied tasks.

- **Effort:** none beyond Phase 5
- **Value:** establishes a baseline for what the Qwen3.5 base model can do without nutri-atlas's memory/tool augmentation
- **What it doesn't test:** any nutri-atlas-specific contribution

**Verdict:** ✅ Already in progress. Label results clearly as "Qwen3.5-9B baseline planner — no memory, no nutri-atlas tooling".

### Direction B — Port nutri-atlas's persistent-memory pattern into EmbodiedBench's planner

The most interesting research angle. Hypothesis: long_horizon and spatial tasks fail partly because the model forgets objects seen earlier in the episode. nutri-atlas avoids this with `register_objects` storing TF landmarks across calls.

Implementation sketch — modify `EmbodiedBench/embodiedbench/planner/remote_model.py` (or `vlm_planner.py`) to:
1. Maintain a "seen objects" memory inside the planner: `dict[object_name → (last_observed_step, scene_relative_position)]`
2. Inject this memory into the system prompt at each step (e.g. as a "Previously observed:" section)
3. Update the memory when the simulator returns new observations
4. Optionally persist across episodes within the same scene (matches nutri-atlas's `detected_objects.json` model)

**Effort:** ~half day of code. Localized to the planner layer. Doesn't touch the simulator side.

**Value:** directly tests whether nutri-atlas's memory pattern improves performance on the **two weakest capabilities** for Qwen3.5-Q4_K_M from Phase 5 smoke results:
- spatial: 0.2 (EB-Alfred) / 0.0 (EB-Habitat)
- long_horizon: not yet measured

A clean A/B compares Phase 5 "baseline" numbers vs. "baseline + memory injection" on the same dataset.

**Verdict:** ✅ Best value-to-effort ratio. Recommended as the primary research extension.

### Direction C — Plug nutri-atlas's full tool-calling agent into EmbodiedBench as the planner

The heavy lift. Replace EmbodiedBench's `VLMPlanner` with a thin wrapper around nutri-atlas's `_run_turn` loop.

Would require a translation layer that:
1. Maps EmbodiedBench's per-scene action vocabulary onto nutri-atlas tool calls — e.g. `navigate_to_landmark("Ladle")` → emit action ID `109` ("find a Ladle") into EmbodiedBench's expected output format
2. Replaces ZMQ + ROS2 with direct calls into AI2-THOR / habitat-sim
3. Reconciles the chat-history (multi-turn tool calling) vs single-shot-per-step paradigms
4. Handles the open-ended tool registry vs fixed action vocabulary mismatch (some nutri-atlas tools like `get_meal_recommendation` have no EB equivalent and should no-op)

**Effort:** several days to a couple of weeks. Lots of plumbing.

**Value:** lets you publish numbers like "full nutri-atlas pipeline on standard EmbodiedBench" — but the integration cost dominates the research signal.

**Verdict:** ⚠️ Defer unless the research goal is specifically "evaluate the full nutri-atlas pipeline on a standard benchmark". For now, Direction B captures the most novel contribution (persistent memory) at far lower cost.

## Recommendation

Sequence:
1. **Now:** finish the Phase 5 full EB-Alfred and EB-Habitat runs → establishes the Direction A baseline.
2. **Next:** implement Direction B — add persistent memory to `remote_model.py`. Re-run **only** the `long_horizon` and `spatial` subsets (where memory should matter most) with `down_sample_ratio=0.1` first to validate the implementation, then full runs.
3. **Compare:** baseline vs. memory-augmented on the same subsets. The delta is the contribution of nutri-atlas's memory pattern.
4. **Skip Direction C** until/unless there's a specific publication-style claim that requires the full integration.

## EmbodiedBench customization flexibility (tiered)

Aside from the three integration directions above, EmbodiedBench itself is fairly modular. If we want to extend the benchmark (rather than just plug nutri-atlas into it), here are the customization tiers ordered by effort.

### Tier 1 — Add new task instances using existing action vocabulary (easiest)

Add more JSONs (EB-Alfred) or append to the pickle (EB-Habitat) using the existing action/scene/object vocabularies.

- **EB-Alfred**: tasks live as JSONs in `embodiedbench/envs/eb_alfred/data/json_2.1.0/{task_name}/...`. Each task has: AI2-THOR scene ID, initial state, target object, instruction, ground-truth action sequence. New tasks must use the existing 200+ action set (find / pick up / put down / open / close / slice / etc.) and reference AI2-THOR's existing FloorPlans.
- **EB-Habitat**: tasks live in `embodiedbench/envs/eb_habitat/datasets/{subset}.pickle`. Each episode references a ReplicaCAD scene + object configuration + instruction. New tasks must use the ~70-action set (navigate / pick up / place / open / close).

**Effort**: small — write the dataset file, drop it in.
**Limitation**: must fit existing action / scene / object vocabularies.

### Tier 2 — Create a new "eval subset" (medium-easy)

Group related custom tasks into a new capability category alongside `base`, `common_sense`, etc. (e.g. `nutrition_aware`).

Steps:
1. Create the new dataset file (Tier 1)
2. Register the subset name in the env config so it's picked up
3. Update aggregate-results code if you want the new category displayed

**Effort**: ~1 day. Mostly plumbing.

### Tier 3 — Modify success criteria / reward functions (medium)

Reward and `task_success` checks live inside the environment classes (`EBAlfEnv.py`; for EB-Habitat it's task-specific logic via Habitat's `RearrangePredicateTask`).

Useful for:
- Partial credit for picking up the right object even if placement fails
- Custom success metrics (e.g. "task succeeded if any nutrient-rich object was selected")
- Different penalty schemes for invalid actions

**Effort**: 1-3 days. Some risk of breaking comparability with the EmbodiedBench paper's published numbers.

### Tier 4 — Add new actions to the action space (medium-hard)

Each environment has a fixed action vocabulary mapped to simulator calls. Adding `inspect_for_freshness` or `read_nutrition_label` (nutri-atlas-like actions) requires:

1. Extending the action ID enumeration
2. Implementing the simulator-side behavior (AI2-THOR API or Habitat API)
3. Updating the prompt template so the LLM knows the new action exists

**Effort**: several days. Most of the work is on the simulator side. AI2-THOR has rich object metadata you could query; Habitat is more limited.

### Tier 5 — Add a new environment entirely (hard)

E.g. plug nutri-atlas-style real-world scenes into EmbodiedBench's evaluation framework.

Needs:
- Gym-style wrapper exposing `reset` / `step` / `observe`
- Action vocabulary mapping
- Task dataset
- Success criteria
- Integration with the main evaluator

**Effort**: weeks. This is what the EmbodiedBench paper did for its 4 envs on top of AI2-THOR, Habitat, CoppeliaSim/PyRep.

### Cross-reference: which tier does each direction need?

| Direction | Customization tier needed |
|-----------|--------------------------|
| **A** — baseline LLM eval | None (already done) |
| **B** — add persistent memory | None — modifies the planner, not the environment |
| **C** — full nutri-atlas in EB | Tier 4-5 (action extension + planner replacement) |
| **New** — nutrition-aware tasks | Tier 1-2 (custom task instances + new subset) |

### A new candidate direction worth considering

**Direction D — Add a nutrition-aware eval subset (Tier 1-2)**

Create tasks like *"Find an apple AND a piece of meat AND place them on the table"* using only EB-Alfred's existing actions, but design them so a nutrition-knowledge-augmented planner has an advantage. Then prepend the nutri-rag knowledge base context to the prompt and measure whether it improves task_success.

- **Tests**: does access to nutri-rag's knowledge base actually help on embodied tasks?
- **Effort**: similar to Direction B (~half-day to ~2 days)
- **Plays well with Direction B**: persistent memory + nutrition KB could be evaluated together

### Verification commands (run before committing to a tier)

```bash
# 1. See the task JSON format for EB-Alfred
ls EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0/ | head -5
cat EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0/$(ls EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0 | head -1)/*.json 2>/dev/null | head -50

# 2. See the action vocabulary in EB-Alfred env
grep -A 5 "language_skill" EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py | head -30

# 3. See how EB-Habitat tasks are structured
python3 -c "
import pickle
with open('EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/long_horizon.pickle', 'rb') as f:
    data = pickle.load(f)
print('Type:', type(data))
print('Length:', len(data) if hasattr(data, '__len__') else 'N/A')
print('Sample:', data[0] if isinstance(data, list) else list(data.items())[0])
" 2>&1 | head -20
```

## What this doc does NOT cover

- Specific code changes to `remote_model.py` for Direction B — would be its own implementation plan once Phase 5 results land
- Integration testing of nutri-atlas's other tools (`get_meal_recommendation`, `spin_robot`) — these have no analog in EmbodiedBench's action space
- Using EmbodiedBench tasks to fine-tune Qwen3.5 — that's a separate research direction (data curation, not architecture)
- Concrete Tier-1 task design for Direction D — would be its own plan if we pursue that

## Related files

- Current EmbodiedBench plan: `plans/embodiedbench_docker_setup.md`
- Qwen3.5 test plan + results: `plans/qwen35embodiedbench_test.md`
- nutri-atlas robot assistant: `nutri-atlas/robot_control/robot_assistant.py`
- EmbodiedBench planner (Direction B target): `EmbodiedBench/embodiedbench/planner/remote_model.py`, `vlm_planner.py`
