# Direction B — Port nutri-atlas's Persistent-Memory Pattern into a Forked EmbodiedBench

## Context

Phase 5 of the Qwen3.5 EmbodiedBench eval established the **baseline**: Qwen3.5-9B-Q4_K_M on EB-Alfred lands at ~33% task_success overall, with `long_horizon` (success 0.18, task_progress 0.43, num_steps 24.1, reward −0.14) and `spatial` (success 0.32, planner_output_error 0.24) clearly weakest. The analysis showed those subsets fail not because the model can't pick the right action *now* — but because it forgets objects it observed several steps ago, then either times out searching for them again or chooses the wrong container.

[plans/atlas_integrated_emben_testing.md](../../work/atlas/mimir/plans/atlas_integrated_emben_testing.md) Direction B proposed porting nutri-atlas's `detected_objects.json` pattern — a per-session record of `{label, position, timestamp}` for every object the agent has seen — into the EmbodiedBench planner so the model gets a running "Previously observed:" list across steps within an episode. This plan implements that port, scoped to EB-Alfred v1, and runs a clean A/B against the Q4_K_M baseline.

**Per user request, the upstream `EmbodiedBench/` directory stays untouched.** All edits land in a sibling copy `EmbodiedBench_atlasmodified/`. The two trees run side-by-side so you can compare baseline vs. memory-augmented at any time without swapping git branches.

The hypothesis: if we hand the planner the names of objects it has already seen in this episode, `long_horizon` and `spatial` task_success should rise without any model-side change.

## What we keep / drop from nutri-atlas's pattern

Reading [nutri-atlas/robot_control/robot_side/zmq_bridge_real/zmq_bridge_node_working_v2.py](../../work/atlas/mimir/nutri-atlas/robot_control/robot_side/zmq_bridge_real/zmq_bridge_node_working_v2.py) and the tool layer:

| nutri-atlas mechanism | EmbodiedBench port |
|---|---|
| `detected_objects.json` keyed by `detected_{label}_{idx}` storing `{label, px, py, timestamp}` | In-process dict on `VLMPlanner` storing `{label → {first_seen_step, last_seen_step}}`. Per-episode only; cleared on `reset()`. No disk file — EmbodiedBench evaluates episodes serially. |
| Two-stage `scan_objects` (temp) → `register_objects` (persist) | Single-stage: every `update_info(info)` call accumulates whatever `info['object_states']['visible_objs']` reported. No explicit "register" action because the AI2-THOR `visible` flag is already ground truth. |
| Spatial dedup at 0.5 m threshold | Dedup by `label` string. Same objectType seen across frames just updates `last_seen_step`. |
| Read via `get_detected_objects` tool, injected as tool result JSON | Read inside `process_prompt()`, injected as a `## Previously observed objects:` text block before action history. Single-shot prompts can't do tool calls, so direct prompt-side injection is the only option. |
| Reset only on startup / explicit clear | Reset on every episode (`VLMPlanner.reset()` already runs at episode start — line 136). |

The deliberate omission: **3D position**. AI2-THOR exposes `position` in `last_event.metadata['objects']`, but encoding world-frame x/y/z in the prompt is noisy for a VLM and risks confusing the model more than it helps. The user flagged this as an "we'll think more later" design point — v1 captures the cheapest signal (which objects, when) and we revisit position in v2 if v1 shows a positive delta. See [Open design points](#open-design-points-deferred-to-v2) below.

## Layout: two trees side-by-side

```
/home/boxun/work/atlas/mimir/
├── EmbodiedBench/                       ← UPSTREAM, unchanged (baseline source of truth)
│   ├── embodiedbench/                   ← read-only from this plan's perspective
│   └── results/
│       └── eb_alfred/.../qwen35_q4km_alf_full/   ← Phase 5 baseline lives here
└── EmbodiedBench_atlasmodified/         ← NEW: full copy + edits for Direction B
    ├── embodiedbench/                   ← edited: vlm_planner.py, main.py, configs/eb-alf.yaml
    ├── embodiedbench/envs/eb_alfred/data → symlink to upstream's json_2.1.0 (avoid duplicating ~GB of data)
    ├── Docker/Dockerfile.alfhab-atlasmodified   ← NEW: thin Dockerfile that bakes the edits in
    └── results/                         ← memory-variant results land here, fully separate

Docker images:
  embench:alfhab                 ← active baseline image (existing, unchanged)
  embench:alfhab-v3              ← backup of the baseline (existing, unchanged)
  embench:alfhab-atlasmodified   ← NEW: FROM alfhab, copies the modified embodiedbench/ in
```

Heavy data directories (`eb_alfred/data/json_2.1.0/`, `eb_habitat/datasets/` if present) are **symlinked** from upstream to avoid duplicating GBs. Python source is **copied** so edits don't leak back. The two Docker images give us a clean provenance: baseline tests run against the baseline image, modified tests run against the modified image — no bind-mount juggling at runtime.

## Critical files in the modified copy

All file paths below refer to the new tree `EmbodiedBench_atlasmodified/`.

- **[EmbodiedBench_atlasmodified/embodiedbench/planner/vlm_planner.py]** — adds the memory store, the injection into `process_prompt`, and the update in `update_info`. Specific line anchors (same as upstream, since the copy starts identical):
  - line 14–36: `__init__` — accept new `persistent_memory` flag, init `self.observed_objects = {}`
  - line 51–95: `process_prompt` — inject the "Previously observed objects" block when flag is on
  - line 136–141: `reset` — also clear `self.observed_objects`
  - line 254–259: `update_info` — extract `info['object_states']['visible_objs']` and update the dict
- **[EmbodiedBench_atlasmodified/embodiedbench/main.py]** — pass the new flag from yaml → `VLMPlanner` constructor (look at how `chat_history` is threaded through; mirror it exactly).
- **[EmbodiedBench_atlasmodified/embodiedbench/configs/eb-alf.yaml]** — add `persistent_memory: True` as the **default** for this fork (the whole point of the fork is memory-on). Keep the flag rather than hardcoding so we can do quick ablations with `persistent_memory=False`.

Files we explicitly do NOT touch in v1:
- `EBAlfEnv.py` — `info['object_states']['visible_objs']` already exists at line 336; we just consume it.
- `remote_model.py` — no change to the LLM call path; the prompt-string change is all server-side.
- `eb_alfred_evaluator.py` — calls `planner.update_info(info)` at line 130 / 145 already, and the existing `info` dict is exactly what we need.
- EB-Habitat configs/env — out of scope for v1 (Habitat's info dict has no equivalent visible-objects list).

## Step-by-step plan

### Step B.0 — Create the modified copy

```bash
cd /home/boxun/work/atlas/mimir

# 1. Copy the tree (excludes ignored content like __pycache__ via rsync filter; cp -r also works).
#    Use rsync so we can exclude __pycache__ and the heavy data/results dirs cleanly.
rsync -a \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='results/' \
    --exclude='running/' \
    --exclude='embodiedbench/envs/eb_alfred/data/json_2.1.0' \
    --exclude='embodiedbench/envs/eb_habitat/datasets/*.pickle' \
    --exclude='.git/' \
    EmbodiedBench/ EmbodiedBench_atlasmodified/

# 2. Symlink the heavy data dirs back to the upstream location so we don't duplicate them.
mkdir -p EmbodiedBench_atlasmodified/embodiedbench/envs/eb_alfred/data
ln -s /home/boxun/work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0 \
      EmbodiedBench_atlasmodified/embodiedbench/envs/eb_alfred/data/json_2.1.0

# (Only if EB-Habitat data exists; for v1 we don't need it, but symlink anyway for completeness.)
ls EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/*.pickle 2>/dev/null | while read f; do
    ln -s "$f" "EmbodiedBench_atlasmodified/embodiedbench/envs/eb_habitat/datasets/$(basename $f)"
done

# 3. Create a fresh, empty results dir for the fork.
mkdir -p EmbodiedBench_atlasmodified/results

# 4. Sanity check: diff the two trees, expect only excluded dirs to differ.
diff -rq EmbodiedBench EmbodiedBench_atlasmodified | head -20
```

**Sanity check criterion**: after the copy and before any edits, running the same eval CLI against `EmbodiedBench_atlasmodified/` should reproduce baseline numbers. We won't actually run this — but the absence of diffs in the Python source confirms it.

### Step B.1 — Code: add the memory store to `VLMPlanner`

Three localized edits to `EmbodiedBench_atlasmodified/embodiedbench/planner/vlm_planner.py`:

**Edit 1 — `__init__` (line 14–36)**: add `persistent_memory=False` kwarg, init `self.observed_objects = {}`.

**Edit 2 — `reset()` (line 136–141)**: add `self.observed_objects = {}` so memory clears per episode (matching how `episode_messages` and `episode_act_feedback` clear).

**Edit 3 — `update_info(info)` (line 254–259)**: after the existing `self.episode_act_feedback.append(...)`, also:
```python
if self.persistent_memory and 'object_states' in info:
    step = self.planner_steps  # already incremented in act()
    for obj_name in info['object_states'].get('visible_objs', []):
        if obj_name not in self.observed_objects:
            self.observed_objects[obj_name] = {'first_seen': step, 'last_seen': step}
        else:
            self.observed_objects[obj_name]['last_seen'] = step
```

**Edit 4 — `process_prompt()` (line 51–95)**: in the two prompt branches that emit `prev_act_feedback` (the `chat_history=True` branch at line 65 and the standard branch at line 78), inject a section right before "The action history:" — only when `self.persistent_memory and self.observed_objects`. Format:

```python
if self.persistent_memory and self.observed_objects:
    prompt += '\n\n## Previously observed objects (across earlier steps in this episode):'
    for name, info in sorted(self.observed_objects.items()):
        prompt += f"\n- {name} (first seen at step {info['first_seen']}, last seen at step {info['last_seen']})"
```

Place this block AFTER the `## Now the human instruction is: ...` line and BEFORE the action history. This way the model reads the memory in context with the task, then the action history, then the current image.

### Step B.2 — Code: thread the flag through `main.py` and the yaml

`EmbodiedBench_atlasmodified/embodiedbench/configs/eb-alf.yaml` — add `persistent_memory: True` at the bottom (default ON in the fork; can override to False for ablations).

`EmbodiedBench_atlasmodified/embodiedbench/main.py` — grep for `chat_history` and add `persistent_memory` to the exact same plumbing path. Expected to be a one-line addition when constructing `VLMPlanner(...)`.

### Step B.3 — Local sanity check (no GPU)

Before touching the container, run on the host:
```bash
cd /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified
python -c "from embodiedbench.planner.vlm_planner import VLMPlanner; p = VLMPlanner('dummy', 'remote', ['action0'], 'sys', [], persistent_memory=True); print(p.observed_objects, p.persistent_memory)"
```

Expected: prints `{} True` and no exception. Catches typos in 5 seconds vs. 5 minutes of container start-up.

### Step B.4 — Build the modified-version Docker image

We build a **new image** `embench:alfhab-atlasmodified` that inherits everything from `embench:alfhab` (all the Phase 4/5/6 X11, NVIDIA, fonts, LFS, modesetting, runtime fixes) and just bakes our modified `embodiedbench/` directory in on top. The original install was editable (`pip install -e .` at line 62 of `Dockerfile.alfhab`), so overwriting the files under `/opt/embodiedbench/embodiedbench/` is enough — pip's editable registration still points at that path, and Python loads our edited files.

**Create `EmbodiedBench_atlasmodified/Docker/Dockerfile.alfhab-atlasmodified`**:

```dockerfile
FROM embench:alfhab

# Bake the modified planner / configs / main in on top of the baseline install.
# /opt/embodiedbench was an editable install, so this is all we need.
COPY embodiedbench /opt/embodiedbench/embodiedbench

# Smoke-test that the import still works at build time (catches typos before runtime).
RUN /opt/conda/envs/embench/bin/python -c "from embodiedbench.planner.vlm_planner import VLMPlanner; print('import ok')"
```

**Build the image**:

```bash
cd /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified
docker build -t embench:alfhab-atlasmodified -f Docker/Dockerfile.alfhab-atlasmodified .
```

Expected build time: ~10 seconds (only one COPY + one short Python smoke). Image size: ~same as the parent plus a few MB.

**Confirm both images coexist**:

```bash
docker images | grep alfhab
# Expected after the build:
# embench   alfhab-atlasmodified   ...   (newly built — what this plan creates)
# embench   alfhab                 ...   (existing — active baseline, unchanged)
# embench   alfhab-v3              ...   (existing — backup of the baseline, unchanged)
```

### Step B.5 — Run the modified container

```bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -it --rm \
  --name embench-eval-memB \
  --add-host=host.docker.internal:host-gateway \
  --device=/dev/tty0 --device=/dev/tty1 --device=/dev/tty2 --device=/dev/tty3 \
  --device=/dev/tty4 --device=/dev/tty5 --device=/dev/tty6 --device=/dev/tty7 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0:/opt/embodiedbench/embodiedbench/envs/eb_alfred/data/json_2.1.0 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results:/opt/embodiedbench/results \
  --shm-size=8g \
  embench:alfhab-atlasmodified
```

**Key differences vs. the Phase 5 baseline docker run:**
- Image is `embench:alfhab-atlasmodified` — the modified code is baked into the image, no source bind-mount needed.
- Results bind targets `EmbodiedBench_atlasmodified/results` so memory-variant outputs stay separate from baseline.
- Container name `embench-eval-memB` to avoid clashing with a still-running baseline container.
- The json_2.1.0 dataset is still bind-mounted from the upstream tree (the in-tree symlink can't resolve inside the container because the host path it points to doesn't exist there; the bind overrides it cleanly).

(Optional dev-iteration trick: during code iteration, you can additionally pass `-v /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/embodiedbench:/opt/embodiedbench/embodiedbench` to override the baked-in code with whatever's on disk. For the actual evaluation runs, leave this off so results are tied to a specific image SHA.)

After `docker run`, inside the container apply the Phase 5 in-container fixes (modesetting symlink, `running` symlink, env vars). Reference: Step 5.10 (the "fresh-restart" block) of `plans/qwen35embodiedbench_test.md`.

**Quick verification inside the container:**
```bash
grep persistent_memory /opt/embodiedbench/embodiedbench/planner/vlm_planner.py
grep persistent_memory /opt/embodiedbench/embodiedbench/configs/eb-alf.yaml
```
Both should print hits — confirms the modified code is in the running container.

### Step B.6 — Smoke test (memory ON) on all 6 EB-Alfred subsets

Inside the container, with llama-server already running on the host (same setup as Phase 5):

```bash
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    exp_name='qwen35_q4km_alf_smoke_memB'
```

Note: we don't need `persistent_memory=True` on the CLI because the fork's `eb-alf.yaml` defaults to True. To run an ablation (memory off) inside the fork, append `persistent_memory=False`.

Watch the llama-server log to confirm prompts now contain `## Previously observed objects:` lines after the first step or two. If the block doesn't appear, the flag isn't threaded — stop and fix before burning more GPU.

### Step B.7 — Smoke comparison

```bash
python nutri-atlas/scripts/embench_results_analysis.py --compare \
    /home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4km_alf_smoke \
    /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4km_alf_smoke_memB
```

Pass criterion: no subset shows a *catastrophic* drop (e.g. base going from ~0.4 → 0.0). A small noisy delta on a 0.1 subsample is acceptable since `down_sample_ratio=0.1` means ~5 episodes per subset.

### Step B.8 — Full run (memory ON) on all 6 EB-Alfred subsets

```bash
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    exp_name='qwen35_q4km_alf_full_memB'
```

Expected wall-clock: similar to the Q4_K_M full baseline (~few hours, GPU-bound by Qwen3.5 inference rather than env).

### Step B.9 — Full A/B comparison

```bash
# Confirm the exact baseline directory name first
ls /home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred/ | grep q4km_alf_full

python nutri-atlas/scripts/embench_results_analysis.py --compare \
    /home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4km_alf_full \
    /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4km_alf_full_memB \
    --output /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results/comparisons/q4km_baseline_vs_memB.txt
```

The analysis script's default `DEFAULT_RUN` path points at the IQ2_M run; the `--compare` flags above explicitly pass the right Q4_K_M baseline path.

## Verification

| Step | Pass criterion |
|------|----------------|
| B.0 | `diff -rq EmbodiedBench EmbodiedBench_atlasmodified` shows only excluded paths differ; symlink `ls -l .../json_2.1.0` shows it points back to upstream |
| B.1 | `grep persistent_memory EmbodiedBench_atlasmodified/embodiedbench/planner/vlm_planner.py` shows 3+ hits; `python -c "import ast; ast.parse(...)"` succeeds |
| B.2 | `grep persistent_memory eb-alf.yaml main.py` shows hits |
| B.3 | Host import smoke succeeds |
| B.4 | `docker images` shows both `embench:alfhab` (baseline) and `embench:alfhab-atlasmodified` (new); build's inline `import VLMPlanner` smoke prints `import ok` |
| B.5 | In-container `grep persistent_memory /opt/embodiedbench/...` shows hits — confirms the image was built from the modified tree |
| B.6 | llama-server log shows prompts containing `## Previously observed objects:` after step 1 |
| B.7 | No subset's smoke success drops >25 pp vs. Q4_K_M smoke |
| B.8 | 6 subsets × ~50 episodes = 300 episodes in `EmbodiedBench_atlasmodified/results/eb_alfred/.../qwen35_q4km_alf_full_memB/{subset}/results/` |
| B.9 | Comparison report shows the per-metric delta table; positive task_success delta on `long_horizon` and `spatial` is the success signal we're looking for |

## Open design points (deferred to v2)

These came up while scoping v1 and the user wants to revisit:

1. **Object position**: nutri-atlas stores `(label, px, py)`. AI2-THOR exposes per-object world-frame `position` via `last_event.metadata['objects'][i]['position']`. v1 records only `label + steps`. If v1 shows a positive delta, v2 should A/B `label-only` vs. `label + relative position to agent` to see whether spatial info helps the `spatial` subset specifically.
2. **Forgetting / staleness**: nutri-atlas never forgets within a session. EmbodiedBench episodes are short (≤30 steps), so v1 doesn't forget either. If we extend to longer episodes or cross-episode persistence, add a `max_age_steps` knob.
3. **Object container relationships**: AI2-THOR also exposes `parentReceptacles` — e.g. "the Apple is inside the Fridge". Could be a very strong cue for tasks like *"find the apple in the fridge"*. Pure win candidate for v2 if v1 lands.
4. **Confidence / frequency**: count of frames each object was visible in. Useful if `visible_objs` turns out to be noisy (it shouldn't be — AI2-THOR's flag is deterministic).

## Risks and fallbacks

| Risk | Detection | Fallback |
|------|-----------|----------|
| COPY in the new Dockerfile doesn't replace the parent image's `embodiedbench/` directory cleanly (e.g. leftover files from the parent layer) | Modified `vlm_planner.py` lives in the image but old `__pycache__` files mask the changes | Add `RUN find /opt/embodiedbench -name '__pycache__' -prune -exec rm -rf {} +` before the COPY; alternatively the inline import smoke test at build time should catch a regression |
| Build-time import smoke fails (the COPY put a typo'd file in place) | `docker build` fails on the inline `python -c "from embodiedbench.planner.vlm_planner import VLMPlanner"` line | Fix the source file in the modified tree, rerun `docker build`. The smoke step is cheap insurance against shipping a broken image |
| Symlink to upstream's `json_2.1.0` inside the source tree is unreachable from the container | Eval errors with "task file not found" | Already addressed: the docker run explicitly bind-mounts the upstream json_2.1.0 path to the same in-container path, overriding the symlink |
| Adding the memory block pushes some prompts past `--ctx-size 65536` | llama-server log shows context-truncation warnings | Lower `n_shot` (currently 10) for memory-augmented runs only, or cap the observed-objects list at the most-recent 20 |
| `info['object_states']['visible_objs']` returns object IDs instead of types in some scenes | Memory block contains gibberish like `Apple\|+01.23\|+...` | EBAlfEnv.py line 336 extracts `objectType` not `objectId`, so this is unlikely — but if it appears, confirm and adjust the extraction |
| Memory list grows large and dominates the prompt visually, hurting model attention | Eval shows REGRESSION not improvement | Truncate to last N objects by `last_seen` (e.g. N=15); or only inject objects whose `last_seen` is within K=5 steps of current |
| `persistent_memory=True` accidentally interacts with `chat_history=True` (double injection) | Both paths add prompt content | We only modify the prompt construction once per `process_prompt` call regardless of `chat_history`; flag-gating is independent |

## After Direction B v1 — what's next

- If positive delta on `long_horizon` / `spatial`: write up the result alongside the Phase 5 baseline and proceed to v2 (add position, or container relationships) — all v2 work continues to land in `EmbodiedBench_atlasmodified/`.
- If neutral: try v2 with richer object metadata (parentReceptacles is the most promising single addition).
- If negative: most likely the prompt-injection format is confusing the model — try moving the memory block from "before action history" to "after the image, framed as part of the visual context".
- The upstream `EmbodiedBench/` directory remains the source of truth for baseline numbers and stays clean throughout.

## Related references

- Baseline plan + results: [plans/qwen35embodiedbench_test.md](../../work/atlas/mimir/plans/qwen35embodiedbench_test.md)
- Integration feasibility analysis (this plan implements its Direction B): [plans/atlas_integrated_emben_testing.md](../../work/atlas/mimir/plans/atlas_integrated_emben_testing.md)
- Nutri-atlas memory implementation (source of the pattern): [nutri-atlas/robot_control/robot_side/zmq_bridge_real/zmq_bridge_node_working_v2.py](../../work/atlas/mimir/nutri-atlas/robot_control/robot_side/zmq_bridge_real/zmq_bridge_node_working_v2.py), [nutri-atlas/robot_control/tools/object_tool.py](../../work/atlas/mimir/nutri-atlas/robot_control/tools/object_tool.py)
- Analysis script for A/B comparison: [nutri-atlas/scripts/embench_results_analysis.py](../../work/atlas/mimir/nutri-atlas/scripts/embench_results_analysis.py)

---

# Direction B v2 — Trajectory memory + RAG retrieval

## v2 Context

v1 stored only `{name → {first_seen, last_seen}}` and dumped the entire memory dict into every prompt. v2 makes two changes the user requested:

1. **Richer memory**: store **every sighting** of each object as a `(step, x, y, z)` tuple instead of just first/last step. Gives the model a movement trajectory rather than a presence flag.
2. **RAG over memory**: instead of dumping the whole dict, embed each memory entry with `Qwen/Qwen3-Embedding-0.6B` (the same model nutri-atlas uses for its RAG), embed the user instruction, and inject only the top-10 most similar entries. Falls back to "show all" when memory has ≤10 entries.

**Object-identity decision (user-confirmed):** key by `objectType` (e.g., `"Apple"`); multiple instances at the same step are kept as separate sightings in the same list and disambiguated by position. We do NOT use the full `objectId` because its format embeds the position (`Apple|+01.23|+0.93|-00.45`), so the same instance's ID changes when it moves — making cross-frame instance tracking harder than helpful for EB-Alfred's type-level tasks.

**Embedding decision (user-confirmed):** reuse nutri-atlas's exact recipe — `Qwen/Qwen3-Embedding-0.6B` loaded via HuggingFace `transformers` AutoModel/AutoTokenizer with last-token pooling + L2-normalization. Runs locally on CPU (model is 0.6B, ~50 short strings + 1 query per step ≈ <200ms). No second llama-server needed.

**Top-K decision (user-confirmed):** K=10 with passthrough fallback when memory has ≤10 entries.

## v2 Critical files

All paths under `EmbodiedBench_atlasmodified/`. v1's three files are still modified; v2 modifies one more and adds one new module.

| File | What v2 changes |
|------|-----------------|
| `embodiedbench/envs/eb_alfred/EBAlfEnv.py` | **NEW edit**: line 336 — change `visible_objs` to emit `{name, x, y, z}` dicts instead of bare name strings. (This is the world-frame position addition we sketched earlier; v2 wires it through.) |
| `embodiedbench/planner/memory_embedder.py` | **NEW file**: a `TextEmbedder` class — exact port of [nutri_rag/embedding.py:43-98](../../work/atlas/mimir/nutri_rag/nutri_rag/embedding.py#L43-L98) without nutri-atlas's `FoodVectorIndex` (we don't precompute, embed on the fly). |
| `embodiedbench/planner/vlm_planner.py` | **EDIT**: (a) memory store shape: `{name → list[{step, x, y, z}]}` instead of `{name → {first, last}}`. (b) `update_info` appends a sighting per visible object. (c) `_format_observed_objects_block` does RAG-then-render. (d) `__init__` lazily creates the embedder when `persistent_memory=True`. |
| `embodiedbench/configs/eb-alf.yaml` | **NEW key**: `memory_top_k: 10`. |
| `Docker/Dockerfile.alfhab-atlasmodified` | **EDIT**: add EBAlfEnv.py and memory_embedder.py to the surgical COPY list. Also `RUN pip install --upgrade 'transformers>=4.51.0'` (container has 4.46.2 which predates Qwen3). Pre-download the embedding model in the build so first run isn't a 600MB HF download. |

## v2 Data shapes

**Per-episode memory** (in-process dict on the planner):
```python
self.observed_objects = {
    "Ladle":     [{"step": 0, "x": 1.23, "y": 0.91, "z": -0.45},
                  {"step": 1, "x": 1.23, "y": 0.91, "z": -0.45},  # stationary
                  {"step": 5, "x": 1.50, "y": 1.10, "z": -0.20}], # agent picked it up & moved
    "SinkBasin": [{"step": 1, "x": 2.10, "y": 0.83, "z": -0.45},
                  {"step": 2, "x": 2.10, "y": 0.83, "z": -0.45}],
    # Two apples seen at step 5 ─ kept as separate sightings, position differentiates
    "Apple":     [{"step": 5, "x": 1.20, "y": 0.92, "z":  0.50},
                  {"step": 5, "x": 2.05, "y": 0.92, "z": -0.30}],
}
```

**Per-entry rendering for embedding** (one short string per object, summarizing its trajectory):
```
"Ladle observed at positions (1.23, 0.91, -0.45) at steps 0, 1; (1.50, 1.10, -0.20) at step 5"
"SinkBasin observed at position (2.10, 0.83, -0.45) at steps 1, 2"
"Apple observed at positions (1.20, 0.92, 0.50) at step 5; (2.05, 0.92, -0.30) at step 5"
```

**Top-K retrieval**: cosine similarity between each entry string's embedding and the instruction's embedding. Top-10 strings (or all of them if fewer than 10 exist) get joined into the `## Previously observed objects:` block.

## Step-by-step plan

### Step v2.1 — Memory shape change in `update_info`

Edit `vlm_planner.py:update_info`. The block changes from "name → {first,last}" to "name → list of sightings", and uses the new `info['object_states']['visible_objs']` dict shape (which we also change — see v2.2):

```python
if self.persistent_memory and 'object_states' in info:
    step = self.planner_steps
    for obj in info['object_states'].get('visible_objs', []):
        # obj is now a dict {name, x, y, z} (v2 shape), not a bare string
        name = obj['name']
        self.observed_objects.setdefault(name, []).append({
            'step': step, 'x': obj['x'], 'y': obj['y'], 'z': obj['z'],
        })
```

No dedup of repeated stationary sightings — keep the trajectory verbatim. RAG truncation handles bloat downstream.

### Step v2.2 — Env emits positions

Edit `EBAlfEnv.py:336` (same as the position change we sketched earlier; v2 just commits to it):

```python
"visible_objs": [
    {
        'name': obj['objectType'],
        'x': round(obj['position']['x'], 2),
        'y': round(obj['position']['y'], 2),
        'z': round(obj['position']['z'], 2),
    }
    for obj in self.env.last_event.metadata['objects']
    if obj['visible']
]
```

### Step v2.3 — New module `memory_embedder.py`

Port nutri-atlas's [TextEmbedder](../../work/atlas/mimir/nutri_rag/nutri_rag/embedding.py#L43-L98). Self-contained ~50 lines: `__init__` loads `AutoTokenizer.from_pretrained(model_name)` + `AutoModel.from_pretrained(model_name)`, `encode(texts: list[str]) -> Tensor` tokenizes → forward pass → `_last_token_pool` → L2-normalize.

We don't replicate `FoodVectorIndex`; we just keep both `entry_texts: list[str]` and `entry_embeddings: Tensor` on the planner and recompute on the fly. The memory is tiny (≤50 entries), so building an actual index would be overkill.

### Step v2.4 — RAG retrieval in `_format_observed_objects_block`

Helper changes from "render dict" to "render top-K closest to instruction":

```python
def _format_observed_objects_block(self, user_instruction: str):
    if not (self.persistent_memory and self.observed_objects):
        return ''
    # 1. Render each object's trajectory as a single short string.
    entries = [self._render_one_object(name, sightings)
               for name, sightings in sorted(self.observed_objects.items())]
    # 2. If we have ≤K entries, skip retrieval — just show all.
    if len(entries) <= self.memory_top_k:
        return '\n\n## Previously observed objects:\n' + '\n'.join('- ' + e for e in entries)
    # 3. Otherwise: embed entries + instruction, take top-K by cosine.
    entry_emb = self.embedder.encode(entries)              # [N, 1024]
    query_emb = self.embedder.encode([user_instruction])   # [1, 1024]
    sims = (entry_emb @ query_emb.T).squeeze(-1)           # cosine since both L2-normalized
    topk = sims.topk(self.memory_top_k).indices.tolist()
    selected = [entries[i] for i in topk]
    return ('\n\n## Previously observed objects (top {} most relevant of {}):\n'
            .format(self.memory_top_k, len(entries))
            + '\n'.join('- ' + e for e in selected))
```

`_render_one_object` collapses consecutive same-position sightings:
```python
def _render_one_object(self, name, sightings):
    # Group consecutive sightings at the same position into "steps X-Y" ranges
    # Renders as: "Ladle at (1.23, 0.91, -0.45) at steps 0-1; at (1.50, 1.10, -0.20) at step 5"
    ...
```

The `process_prompt` call sites change from `self._format_observed_objects_block()` to `self._format_observed_objects_block(user_instruction)`.

### Step v2.5 — Wire the embedder + config

`__init__` accepts `memory_top_k=10` and lazily instantiates the embedder when `persistent_memory=True`:

```python
if self.persistent_memory:
    from embodiedbench.planner.memory_embedder import TextEmbedder
    self.embedder = TextEmbedder('Qwen/Qwen3-Embedding-0.6B', device='cpu')
else:
    self.embedder = None
```

Add `memory_top_k: 10` to `eb-alf.yaml`. Plumb through `eb_alfred_evaluator.py` to the constructor.

### Step v2.6 — Dockerfile changes

Append to `Dockerfile.alfhab-atlasmodified`:

```dockerfile
# v2: bump transformers for Qwen3 support and pre-download the embedding model
RUN /opt/conda/envs/embench/bin/pip install --upgrade 'transformers>=4.51.0'

RUN /opt/conda/envs/embench/bin/python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B'); \
AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B'); \
print('Qwen3-Embedding cached in image')"

# v2: surgical COPY additions
COPY embodiedbench/envs/eb_alfred/EBAlfEnv.py        /opt/embodiedbench/embodiedbench/envs/eb_alfred/EBAlfEnv.py
COPY embodiedbench/planner/memory_embedder.py        /opt/embodiedbench/embodiedbench/planner/memory_embedder.py
```

Pre-downloading saves ~600MB worth of first-run HF download per container start. Image grows by ~700MB (model weights + transformers upgrade).

### Step v2.7 — Host sanity check before rebuilding

```bash
cd /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified
python3 -c "import ast; [ast.parse(open(f).read()) for f in [
    'embodiedbench/envs/eb_alfred/EBAlfEnv.py',
    'embodiedbench/planner/vlm_planner.py',
    'embodiedbench/planner/memory_embedder.py',
]]; print('syntax ok')"
```

### Step v2.8 — Rebuild image

```bash
docker build -t embench:alfhab-atlasmodified -f Docker/Dockerfile.alfhab-atlasmodified .
docker images | grep alfhab
```

Expect: longer build than v1 (the pip upgrade + HF model download). ~3-5 minutes. Image grows ~700MB.

### Step v2.9 — Smoke test (memory-with-RAG ON) at ratio=0.1

Use a new `exp_name` so v1 and v2 smoke results coexist:

```bash
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    exp_name='qwen35_q4km_alf_smoke_memB_v2'
```

Watch the llama-server tmux for `## Previously observed objects (top 10 most relevant of N)` once an episode passes step 10. If you see `## Previously observed objects:` (without the "top 10" qualifier) on episodes where memory has >10 entries, retrieval isn't engaging.

### Step v2.10 — Three-way A/B

Once we have baseline + v1 + v2:

```bash
python /home/boxun/work/atlas/mimir/nutri-atlas/scripts/embench_results_analysis.py --compare \
    /home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4km_alf_full \
    /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4km_alf_full_memB_v2
```

The interesting deltas:
- v1 → v2 on `spatial`: did adding positions help?
- v1 → v2 on `long_horizon`: does RAG retrieval focus the model better?
- v2 vs baseline overall: is there a net win?

## Verification

| Step | Pass criterion |
|------|----------------|
| v2.1–v2.5 | `grep -n "observed_objects\|Qwen3-Embedding\|memory_top_k" vlm_planner.py memory_embedder.py eb-alf.yaml eb_alfred_evaluator.py` shows all new wiring |
| v2.6 build | Docker build's inline smokes both pass (`import VLMPlanner` and `Qwen3-Embedding cached in image`) |
| v2.7 syntax | All three files parse |
| v2.9 runtime | llama-server log shows the `top K most relevant` qualifier on prompts past step 10 |

## v2 Risks

| Risk | Detection | Fallback |
|------|-----------|----------|
| `transformers>=4.51.0` upgrade breaks the conda env's other deps | Build fails with pip resolver errors | Pin to a known-good version (e.g., 4.51.3 if 4.51.0 doesn't pin transitively); or use a venv inside conda |
| Qwen3-Embedding model download fails during build (HF rate-limit / network) | Build fails on the pre-download step | Re-run; if persistent, set `HF_HUB_OFFLINE=0` and increase retries; worst case ship without pre-download and accept first-episode latency hit |
| Embedding compute on CPU adds noticeable per-step latency | per-step latency >1s on top of Qwen3.5 inference | Currently estimating ~200ms — if larger, batch encode all entries + query in one forward pass (already doing); fall back to `all-MiniLM-L6-v2` (10x smaller) |
| RAG drops a memory entry the model actually needed | Eval REGRESSES vs v1 on some subset | Lower K to 5 to see if filtering is too loose, or raise K to 15 (closer to dump-all); ultimately if retrieval-based memory underperforms dump-all, that itself is a finding worth reporting |
| Trajectory bloat in long episodes pushes prompt past ctx-size | llama-server context warnings | Collapse consecutive same-position sightings in `_render_one_object` (already in the design); if still over, cap to last 20 sightings per object |
| Embedding model's last-token pooling produces poor embeddings for our short strings | top-K picks irrelevant entries | Replace pooling with mean-pool; or switch to `all-MiniLM-L6-v2` which uses standard CLS pooling |
