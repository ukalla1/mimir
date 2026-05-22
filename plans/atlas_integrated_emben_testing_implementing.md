# Direction B — Implementation Status & Reproducible Recipe

This doc tracks the **implementation** of Direction B (persistent-memory port into EmbodiedBench's planner). For the *design rationale*, see [atlas_integrated_emben_testing.md](atlas_integrated_emben_testing.md). For the original Q4_K_M baseline numbers we're A/B-ing against, see [qwen35embodiedbench_test.md](qwen35embodiedbench_test.md).

**Status (as of 2026-05-21):**

| Step | What it does | Status |
|------|--------------|--------|
| B.0 | Create `EmbodiedBench_atlasmodified/` as a sibling tree of upstream | ✅ done |
| B.1 | Edit `vlm_planner.py` — add memory store, helper, prompt injection | ✅ done |
| B.2 | Thread `persistent_memory` through evaluator + yaml | ✅ done |
| B.3 | Host-side syntax + AST sanity check | ✅ done |
| B.4 | Build `embench:alfhab-atlasmodified` Docker image | ✅ done |
| B.5 | Launch modified container with in-container fixes | ✅ done |
| B.6 | Smoke test (memory ON) at `down_sample_ratio=0.1`, all 6 EB-Alfred subsets | 🟡 running |
| B.7 | Smoke A/B comparison vs. Q4_K_M baseline smoke | ⏳ pending |
| B.8 | Full run (memory ON) all 6 subsets | ⏳ pending |
| B.9 | Full A/B comparison vs. Q4_K_M baseline full | ⏳ pending |

---

## Artifacts produced

### Source tree

```
/home/boxun/work/atlas/mimir/
├── EmbodiedBench/                          ← upstream, untouched (baseline source of truth)
│   └── results/eb_alfred/.../qwen35_q4ks_alf_full/   ← Phase 5 baseline
└── EmbodiedBench_atlasmodified/            ← Direction B fork
    ├── embodiedbench/                      ← edited files live here
    │   ├── planner/vlm_planner.py          ← MODIFIED (memory store + injection)
    │   ├── configs/eb-alf.yaml             ← MODIFIED (persistent_memory: True)
    │   ├── evaluator/eb_alfred_evaluator.py ← MODIFIED (passes flag to VLMPlanner)
    │   └── envs/eb_alfred/data/json_2.1.0 → symlink to upstream's dataset
    ├── Docker/Dockerfile.alfhab-atlasmodified  ← NEW: surgical 3-file overlay on embench:alfhab
    └── results/                            ← memory-variant eval outputs land here
```

### Docker images

```
embench:alfhab                  ← active baseline (existing, unchanged)
embench:alfhab-v3               ← backup of baseline (existing, unchanged)
embench:alfhab-atlasmodified    ← NEW: FROM embench:alfhab + 3 surgical file overlays
```

### Exact code changes in `vlm_planner.py`

1. **`__init__`**: new kwarg `persistent_memory=False`; `self.persistent_memory = persistent_memory` and `self.observed_objects = {}` added to the body.
2. **`_format_observed_objects_block()`** (new helper method): returns the `## Previously observed objects (across earlier steps in this episode):` text block, or empty string when disabled / dict empty.
3. **`process_prompt()`**: both action-history branches (`chat_history=True` and the standard `else`) now call `self._format_observed_objects_block()` and prepend it before "The action history:".
4. **`reset()`**: clears `self.observed_objects = {}` per episode (alongside the existing `episode_messages` / `episode_act_feedback` clear).
5. **`update_info(info)`**: after the existing `episode_act_feedback` append, also iterates `info['object_states']['visible_objs']` (already emitted by `EBAlfEnv.step()` at line 336 of upstream) and updates `observed_objects[label] = {'first_seen', 'last_seen'}`.

### Exact change in `eb_alfred_evaluator.py`

Added `persistent_memory=self.config.get('persistent_memory', False)` to the `VLMPlanner(...)` constructor call. The `.get(..., False)` keeps backward compatibility with configs that don't define the key.

### Exact change in `configs/eb-alf.yaml`

Appended `persistent_memory: True` at the bottom (default ON in the fork). Pass `persistent_memory=False` on the CLI to ablate.

---

## Implementation gotcha (worth remembering)

**The broad-COPY-overwrites-baked-patches bug.** First version of the Dockerfile did `COPY embodiedbench /opt/embodiedbench/embodiedbench` — i.e. dumped the whole modified tree on top of `/opt/embodiedbench/embodiedbench/` inside the baseline image. This silently **overwrote** the baseline image's `sed`-applied patch to `startx.py` (the Phase 4 `Option "UseDisplayDevice" "none"` line that tells the NVIDIA driver not to scan for connected displays).

Symptom inside the modified container: Xorg dies with `(EE) NVIDIA(0): Failure to construct a valid MetaMode list: no MetaModes remaining` because the driver tries to use the host's actual connected displays and discards them all as too large for the 1024x768 virtual screen.

**Fix**: make the Dockerfile COPY surgical — copy only the three files we changed, leave everything else from the baseline image alone. The current Dockerfile also runs a `grep -q 'UseDisplayDevice'` check at build time so we'd catch regression of this exact pattern immediately.

If you ever modify another file in the modified tree, **add it to the Dockerfile's COPY list explicitly** — do not switch back to broad copy.

---

## Fresh-restart recipe (run after a server reboot / new shell)

This is the full sequence to go from "nothing running" → "memory-augmented eval producing JSONs". Assumes you're on `vega` at the standard `/home/boxun/work/atlas/mimir` checkout.

### Part A — Host: start llama-server (in tmux, persists across SSH disconnect)

```bash
tmux new -s qwen-server

# Inside tmux:
~/softwares/llama.cpp/build/bin/llama-server \
    --model /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
    --mmproj /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/mmproj-BF16.gguf \
    --port 8080 --host 0.0.0.0 \
    --ctx-size 65536 --n-gpu-layers 999 --parallel 1 \
    --chat-template-kwargs '{"enable_thinking":false}'

# Detach with Ctrl-B D
```

### Part B — Host: rebuild the modified image (one-time; only after source changes)

**Skip this for normal restarts.** The image persists in Docker's local store across container exit, daemon restart, host reboot, and SSH disconnect. You only need to rebuild when one of these is true:

- You edited one of the three modified source files (`vlm_planner.py`, `configs/eb-alf.yaml`, `eb_alfred_evaluator.py`)
- You added a new file to the Dockerfile's COPY list
- You changed `Dockerfile.alfhab-atlasmodified` itself
- Someone explicitly removed the image (`docker rmi embench:alfhab-atlasmodified`)

Quick check before deciding — does the image still exist?
```bash
docker images | grep alfhab-atlasmodified
# If a row is printed: skip to Part C.
# If empty: run the build below.
```

Rebuild command (when needed):
```bash
cd /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified
docker build -t embench:alfhab-atlasmodified -f Docker/Dockerfile.alfhab-atlasmodified .
docker images | grep alfhab   # confirm embench:alfhab-atlasmodified is present
```

### Part C — Host: launch the modified container

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

### Part D — Container: in-container fixes (same Phase 5 dance)

```bash
# 1. Disable broken modesetting driver (Phase 5 Issue 5.1)
mv /usr/lib/xorg/modules/drivers/modesetting_drv.so \
   /usr/lib/xorg/modules/drivers/modesetting_drv.so.disabled

# 2. Symlink running/ → results/ so episode JSONs land in the bind-mounted dir (Phase 5 Issue 5.6)
[ -d /opt/embodiedbench/running ] && mv /opt/embodiedbench/running /opt/embodiedbench/running.old
ln -s /opt/embodiedbench/results /opt/embodiedbench/running

# 3. Patch startx.py to add -ac so X allows connections without auth (Phase 5 Issue 5.4)
sed -i 's|Xorg -noreset|Xorg -noreset -ac|' \
    /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py

# 4. Conda + env vars
source /opt/conda/etc/profile.d/conda.sh
conda activate embench
export remote_url=http://host.docker.internal:8080/v1
export OPENAI_API_KEY=EMPTY

# 5. Verify nothing got clobbered (catches the broad-COPY regression)
grep persistent_memory /opt/embodiedbench/embodiedbench/planner/vlm_planner.py | head -3
grep persistent_memory /opt/embodiedbench/embodiedbench/configs/eb-alf.yaml
grep "UseDisplayDevice" /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py
# Expected: 3 hits on persistent_memory, 1 hit on UseDisplayDevice "none"

# 6. Start headless X
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
sleep 3
export DISPLAY=:1
# Expected: no (EE) lines after "Using system config directory"
```

### Part E — Container: run the eval

```bash
# Smoke (10% subsample, ~5 episodes per subset, ~30 total, ~30 min)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    exp_name='qwen35_q4ks_alf_smoke_memB'

# Full (matches Phase 5 baseline scope, ~few hours, 300 episodes)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    exp_name='qwen35_q4ks_alf_full_memB'

# Ablation (memory OFF, same fork) — useful if you suspect prompt-format issues are confounding
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    persistent_memory=False \
    exp_name='qwen35_q4ks_alf_full_memB_ablation'
```

**Note on `eval_sets`:** the modified `eb-alf.yaml` has `eval_sets: []` and the evaluator falls back to `ValidEvalSets` (all 6 subsets, in canonical order: `base`, `common_sense`, `complex_instruction`, `spatial`, `visual_appearance`, `long_horizon`) — see [eb_alfred_evaluator.py:44-47](../../work/atlas/mimir/EmbodiedBench/embodiedbench/evaluator/eb_alfred_evaluator.py#L44-L47). So we don't need to pass `eval_sets=[...]` on the CLI; passing it explicitly is risky because typoing a subset name (e.g. `spatial_relationship`) trips an `AssertionError` mid-run after several subsets already finished. Only pass it when you genuinely want a single subset, e.g. `eval_sets=[spatial]` to fill in a missing one.

`persistent_memory=True` is the default in the modified `eb-alf.yaml`, so the standard runs don't need to pass it on the CLI. Override on the CLI to ablate.

**Note on overwriting:** EmbodiedBench writes to `running/eb_alfred/{model}_{exp_name}/{subset}/` and silently overwrites existing files at the same path ([EBAlfEnv.py:388-389](../../work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py#L388-L389) only `os.makedirs` if missing, no fail-if-exists). So:
- Re-running with the same `exp_name` and overlapping subsets **clobbers** the prior run's results.
- Re-running with the same `exp_name` but a different subset (e.g. `eval_sets=[spatial]` to fill in a missing one) leaves the other subsets untouched — only the named subset's directory is overwritten.
- Use a new `exp_name` (or `mv` the old dir to `..._exp_name.v1`) before re-running if you want to preserve the previous results.

### Part F — Verify the memory injection is actually happening

While the eval is running, in a separate **host** shell, attach to the llama-server tmux and watch for `## Previously observed objects` in the prompt bodies:

```bash
tmux attach -t qwen-server
# Look for prompts containing "## Previously observed objects (across earlier steps in this episode):"
# These should appear from step 2 onward of each episode (step 1 has no prior observations).
# Detach with Ctrl-B D.
```

If the line never appears, something's wrong with the flag plumbing — stop the eval and re-verify Part D step 5.

---

## A/B comparison commands

Once a full memory-on run lives in `EmbodiedBench_atlasmodified/results/eb_alfred/...`, compare it to the Q4_K_M baseline:

```bash
# Smoke vs. smoke
python /home/boxun/work/atlas/mimir/nutri-atlas/scripts/embench_results_analysis.py --compare \
    /home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_smoke \
    /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_smoke_memB

# Full vs. full (the real comparison)
python /home/boxun/work/atlas/mimir/nutri-atlas/scripts/embench_results_analysis.py --compare \
    /home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full \
    /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results/eb_alfred/Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full_memB \
    --output /home/boxun/work/atlas/mimir/EmbodiedBench_atlasmodified/results/comparisons/q4ks_baseline_vs_memB.txt
```

Confirm the exact baseline directory name first with `ls /home/boxun/work/atlas/mimir/EmbodiedBench/results/eb_alfred/` — the analysis script's default `DEFAULT_RUN` points at the IQ2_M run, not Q4_K_M.

**Success signal**: positive `task_success` delta on `long_horizon` and `spatial` subsets (the two where the integration doc predicted memory should help most).

---

## What's left after B.6 lands

- **B.7**: smoke A/B comparison. Pass criterion: no subset's smoke success drops >25 pp vs. Q4_K_M smoke (a 0.1 subsample is noisy, we're just looking for "didn't break anything catastrophically").
- **B.8**: full run, all 6 subsets, no `down_sample_ratio`. ~300 episodes total.
- **B.9**: full A/B comparison. The actual research deliverable.

If B.6 surfaces a flag-threading bug (no `## Previously observed objects` in prompts), pause B.7+ until fixed.

---

## Open design points (deferred to v2)

These are intentional v1 omissions — record them so we don't re-relitigate.

1. **3D position** — AI2-THOR exposes per-object `position` in `last_event.metadata['objects']`. v1 stores only `(label, first_seen_step, last_seen_step)`. If v1 shows a positive delta, v2 should A/B `label-only` vs. `label + relative position to agent` on the `spatial` subset specifically.
2. **`parentReceptacles`** — AI2-THOR also exposes container relationships (e.g. "Apple is inside Fridge"). Strong candidate for v2.
3. **Forgetting / staleness** — EmbodiedBench episodes max at 30 steps, so v1 never forgets. Add `max_age_steps` only if we extend to longer episodes.
4. **Confidence / frequency** — count of frames each object was visible in. Useful only if AI2-THOR's `visible` flag turns out to be noisier than we think (it shouldn't be).

---

## Reference

- Original design doc: [atlas_integrated_emben_testing.md](atlas_integrated_emben_testing.md)
- Baseline (Q4_K_M) eval plan + results: [qwen35embodiedbench_test.md](qwen35embodiedbench_test.md)
- Docker / container setup history: [embodiedbench_docker_setup.md](embodiedbench_docker_setup.md)
- Analysis script: [nutri-atlas/scripts/embench_results_analysis.py](../nutri-atlas/scripts/embench_results_analysis.py)
- Approved plan file (this session's working plan): `/home/boxun/.claude/plans/validated-coalescing-whisper.md`
