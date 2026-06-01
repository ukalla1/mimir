# Direction B — Implementation Status & Reproducible Recipe

This doc tracks the **implementation** of Direction B (persistent-memory port into EmbodiedBench's planner). For the *design rationale*, see [atlas_integrated_emben_testing.md](atlas_integrated_emben_testing.md). For the original Q4_K_M baseline numbers we're A/B-ing against, see [qwen35embodiedbench_test.md](qwen35embodiedbench_test.md).

**Status (as of 2026-05-29):**

| Step | What it does | Status |
|------|--------------|--------|
| B.0–B.5 (v1) | Tree setup, v1 edits, image build, container launch | ✅ done |
| B.6 (v1) | v1 smoke (name-only memory, full dump) | ✅ superseded by v2 |
| v2.1–v2.8 | Trajectory memory + Qwen3-Embedding RAG retrieval; code, build, run | ✅ done |
| v2.10 (Q4_K_S) | v2 full A/B vs Q4_K_S baseline | ✅ done — see results below |
| v3.1–v3.3 | Switchable `memory_render` flag; preserves v2 as default, adds `last_only` mode | ✅ done |
| v3.4 | Rebuild image with v3 edits (~30s, cached layers) | ✅ done |
| v3.5–v3.6 | Run `complex_instruction` at full with `+memory_render=last_only` (~1.5h) | 🟡 running |
| v3.7 | Three-way comparison (baseline / v2 / v3) on `complex_instruction` | ⏳ pending |

**v2 vs Q4_K_S baseline (full run, ~300 episodes):**

| Subset | Baseline | v2 | Δ |
|---|---|---|---|
| base | 0.34 | 0.46 | **+0.12** |
| common_sense | 0.38 | 0.36 | -0.02 |
| complex_instruction | **0.52** | 0.44 | **-0.08** |
| long_horizon | 0.32 | 0.32 | 0.00 |
| spatial | 0.34 | 0.38 | +0.04 |
| visual_appearance | 0.34 | 0.36 | +0.02 |

4 subsets win/tied, 2 regressed. The `complex_instruction` regression is what v3 targets.

**Current architecture: v2 + v3 switchable render mode in one image.** Yaml default = `trajectory` (v2 behavior, byte-identical). CLI override `+memory_render=last_only` enables v3 (drop trajectory, show only most recent sighting per object). Same image, switch at eval time.

---

## Artifacts produced

### Source tree

```
/home/boxun/work/atlas/mimir/
├── EmbodiedBench/                          ← upstream, untouched (baseline source of truth)
│   └── results/eb_alfred/.../qwen35_q4ks_alf_full/   ← baseline
└── EmbodiedBench_atlasmodified/            ← Direction B fork
    ├── embodiedbench/                      ← edited/new files
    │   ├── planner/vlm_planner.py          ← MODIFIED (v2 trajectory memory + RAG)
    │   ├── planner/memory_embedder.py      ← NEW (Qwen3-Embedding-0.6B wrapper, ported from nutri-atlas)
    │   ├── envs/eb_alfred/EBAlfEnv.py      ← MODIFIED (v2: emit {name,x,y,z} dicts for visible objects)
    │   ├── configs/eb-alf.yaml             ← MODIFIED (persistent_memory: True, memory_top_k: 10)
    │   ├── evaluator/eb_alfred_evaluator.py ← MODIFIED (passes both kwargs to VLMPlanner)
    │   └── envs/eb_alfred/data/json_2.1.0 → symlink to upstream's dataset
    ├── Docker/Dockerfile.alfhab-atlasmodified  ← surgical overlay; 5 COPYs + transformers upgrade + Qwen3-Embedding pre-download
    └── results/                            ← memory-variant eval outputs land here
```

### Docker images

```
embench:alfhab                  ← active baseline (existing, unchanged)
embench:alfhab-v3               ← backup of baseline (existing, unchanged)
embench:alfhab-atlasmodified    ← v2: FROM embench:alfhab + transformers≥4.51 + Qwen3-Embedding-0.6B cached + 5 surgical file overlays (~27.1GB, +1.4GB vs baseline)
```

### What v2 stores per object (memory shape)

```python
self.observed_objects = {
    "Ladle":     [{"step": 0, "x": 1.23, "y": 0.91, "z": -0.45},
                  {"step": 1, "x": 1.23, "y": 0.91, "z": -0.45},   # stationary
                  {"step": 5, "x": 1.50, "y": 1.10, "z": -0.20}],  # moved
    "SinkBasin": [{"step": 1, "x": 2.10, "y": 0.83, "z": -0.45}, ...],
}
```

Same `objectType` seen at multiple positions in the same step → both kept as separate sightings, position differentiates. Per-episode reset.

### What gets injected into the prompt

Each object's trajectory is rendered as one short string (consecutive same-position sightings collapsed):
```
- Ladle observed at positions (1.23, 0.91, -0.45) at steps 0-1; (1.50, 1.10, -0.20) at step 5
- SinkBasin observed at positions (2.10, 0.83, -0.45) at steps 1-2
```

**RAG**: if memory has >10 entries, embed each string + the user instruction with `Qwen/Qwen3-Embedding-0.6B`, take top-10 by cosine similarity. Prompt header changes to `## Previously observed objects (top 10 most relevant of N):`. If ≤10 entries, dump them all with the unqualified header.

### Code change inventory

| File | What changed |
|---|---|
| `vlm_planner.py` `__init__` | new kwargs `persistent_memory=False`, `memory_top_k=10`, `memory_render='trajectory'` (v3); init `self.observed_objects = {}`; lazy-load `TextEmbedder` from `Qwen3-Embedding-0.6B` (CPU) when memory is on |
| `vlm_planner.py` `reset()` | clear `self.observed_objects` (per-episode) |
| `vlm_planner.py` `update_info(info)` | for each visible obj dict, append `{step, x, y, z}` to `observed_objects[name]` |
| `vlm_planner.py` `_render_one_object()` | **v3:** flag-gated dispatch. `memory_render='trajectory'` → v2 behavior (collapse consecutive same-position runs into "steps X-Y"). `memory_render='last_only'` → return only the most recent sighting per object |
| `vlm_planner.py` `_format_observed_objects_block(user_instruction)` | render every entry, then either dump all (≤K) or top-K via embedding similarity |
| `vlm_planner.py` `process_prompt()` | both injection sites now pass `user_instruction` so RAG can query against it |
| `memory_embedder.py` | NEW file — `TextEmbedder` class, ~50 lines, ported from [nutri_rag/embedding.py](../nutri_rag/nutri_rag/embedding.py) (last-token pooling + L2-norm, batch encode) |
| `EBAlfEnv.py` line 336 | `visible_objs` now emits `{name, x, y, z}` dicts (rounded to 2 decimals) instead of bare strings |
| `eb_alfred_evaluator.py` | passes `persistent_memory`, `memory_top_k`, and `memory_render` (v3) from config dict to `VLMPlanner(...)` |
| `eb-alf.yaml` | new keys: `persistent_memory: True`, `memory_top_k: 10`, `memory_render: trajectory` (v3 default = v2 behavior) |
| `Dockerfile.alfhab-atlasmodified` | `pip install --upgrade 'transformers>=4.51.0'`; pre-download Qwen3-Embedding-0.6B into HF cache; 5 surgical COPYs (vlm_planner, memory_embedder, eb-alf.yaml, eb_alfred_evaluator, EBAlfEnv); build-time smoke imports both VLMPlanner and TextEmbedder and asserts both new kwargs are present |

### v3 in one render diff

When the same memory dict (Ladle stationary then picked up) is rendered:

**`memory_render: trajectory`** (yaml default, v2 behavior):
```
- Ladle observed at positions (1.23, 0.91, -0.45) at steps 0-1; (1.50, 1.10, -0.20) at step 5
```

**`memory_render: last_only`** (v3, opt-in via CLI):
```
- Ladle last observed at (1.50, 1.10, -0.20) at step 5
```

The writer side (`update_info`) is unchanged — still records every sighting — so switching renders requires no rebuild, just a CLI flag.

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

# 5. Verify v2 + v3 wiring + that nothing got clobbered (catches the broad-COPY regression)
echo "--- v2 wiring ---"
grep memory_top_k /opt/embodiedbench/embodiedbench/configs/eb-alf.yaml
grep -c "observed_objects\|memory_top_k\|memory_embedder" /opt/embodiedbench/embodiedbench/planner/vlm_planner.py
grep "'name':" /opt/embodiedbench/embodiedbench/envs/eb_alfred/EBAlfEnv.py | head -1
ls -la /opt/embodiedbench/embodiedbench/planner/memory_embedder.py
echo "--- v3 switchable-render wiring ---"
grep "memory_render" /opt/embodiedbench/embodiedbench/configs/eb-alf.yaml
grep "last observed at" /opt/embodiedbench/embodiedbench/planner/vlm_planner.py
grep "observed at positions" /opt/embodiedbench/embodiedbench/planner/vlm_planner.py
echo "--- Phase 4 patch still present ---"
grep "UseDisplayDevice" /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py
echo "--- embedder load smoke (~5-10s first time) ---"
python3 -c "from embodiedbench.planner.memory_embedder import TextEmbedder; e = TextEmbedder(device='cpu'); v = e.encode(['hello']); print('embedder ok, vec shape:', v.shape)"
# Expected: memory_top_k: 10; grep count >5; 'name': line shown; memory_embedder.py exists;
#           memory_render: trajectory in yaml; both render branches in vlm_planner.py;
#           UseDisplayDevice patch present; vec shape (1, 1024).

# 6. Start headless X
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
sleep 3
export DISPLAY=:1
# Expected: no (EE) lines after "Using system config directory"
```

### Part E — Container: run the eval

Naming convention: `qwen35_{quant}_alf_{scope}_{variant}` where `variant ∈ {memB, memB_v2, memB_ablation, …}`. **The quant in the exp_name should match the GGUF the host's llama-server is currently serving** — verify with `python3 -c "import urllib.request,json; print(json.loads(urllib.request.urlopen('http://host.docker.internal:8080/v1/models').read())['data'][0]['id'])"` from inside the container.

```bash
# v2 smoke (10% subsample, ~5 episodes per subset, ~30 total, ~30 min)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    exp_name='qwen35_q4ks_alf_smoke_memB_v2'

# v2 full (matches baseline scope, ~few hours, 300 episodes)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    exp_name='qwen35_q4ks_alf_full_memB_v2'

# v3: same image, but switch to last_only rendering via CLI flag (note the `+` prefix — see Hydra gotcha below)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    eval_sets=[complex_instruction] \
    +memory_render=last_only \
    exp_name='qwen35_q4ks_alf_complexinstr_memB_v3_lastonly'

# v3 full all-subset run (use after v3 single-subset clears the regression)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    +memory_render=last_only \
    exp_name='qwen35_q4ks_alf_full_memB_v3_lastonly'

# Ablation 1: memory OFF entirely (note: `+` prefix needed if Hydra rejects bare override)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    +persistent_memory=False \
    exp_name='qwen35_q4ks_alf_full_memB_v2_ablation_off'

# Ablation 2: memory ON but RAG disabled by setting top_k absurdly high (effectively dump all)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    +memory_top_k=999 \
    exp_name='qwen35_q4ks_alf_full_memB_v2_ablation_dumpall'
```

**Hydra gotcha: use `+` prefix for memory-related overrides.** Hydra's struct mode rejects plain `key=value` overrides for some keys even when they're defined in the yaml, with the error `Could not override 'memory_render'. To append to your config use +memory_render=last_only`. The `+` prefix is the canonical Hydra escape hatch — it bypasses struct-mode validation and forces the override. Apply it to `memory_render`, `memory_top_k`, `persistent_memory` whenever overriding from the CLI. The yaml defaults still work without the `+` because they're loaded before struct mode kicks in.

**Note on `eval_sets`:** the modified `eb-alf.yaml` has `eval_sets: []` and the evaluator falls back to `ValidEvalSets` (all 6 subsets, in canonical order: `base`, `common_sense`, `complex_instruction`, `spatial`, `visual_appearance`, `long_horizon`) — see [eb_alfred_evaluator.py:44-47](../../work/atlas/mimir/EmbodiedBench/embodiedbench/evaluator/eb_alfred_evaluator.py#L44-L47). So we don't need to pass `eval_sets=[...]` on the CLI; passing it explicitly is risky because typoing a subset name (e.g. `spatial_relationship`) trips an `AssertionError` mid-run after several subsets already finished. Only pass it when you genuinely want a single subset, e.g. `eval_sets=[spatial]` to fill in a missing one.

`persistent_memory=True` is the default in the modified `eb-alf.yaml`, so the standard runs don't need to pass it on the CLI. Override on the CLI to ablate.

**Note on overwriting:** EmbodiedBench writes to `running/eb_alfred/{model}_{exp_name}/{subset}/` and silently overwrites existing files at the same path ([EBAlfEnv.py:388-389](../../work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/EBAlfEnv.py#L388-L389) only `os.makedirs` if missing, no fail-if-exists). So:
- Re-running with the same `exp_name` and overlapping subsets **clobbers** the prior run's results.
- Re-running with the same `exp_name` but a different subset (e.g. `eval_sets=[spatial]` to fill in a missing one) leaves the other subsets untouched — only the named subset's directory is overwritten.
- Use a new `exp_name` (or `mv` the old dir to `..._exp_name.v1`) before re-running if you want to preserve the previous results.

### Part F — Verify the memory injection + RAG + render mode are actually engaging

While the eval is running, from a separate **host** shell, grep the llama-server tmux scrollback. Non-interactive (recommended):

```bash
# How many times has memory been injected?
tmux capture-pane -pS -10000 -t qwen-server | grep -c "Previously observed objects"

# Which render mode is actually being used?
tmux capture-pane -pS -10000 -t qwen-server | grep -c "last observed at"        # v3 last_only
tmux capture-pane -pS -10000 -t qwen-server | grep -c "observed at positions"   # v2 trajectory

# See an actual example
tmux capture-pane -pS -10000 -t qwen-server | grep -B1 -A5 "Previously observed objects" | tail -40
```

Interactive (alternative): `tmux attach -t qwen-server`, then `Ctrl-B [` to enter copy mode, `/` to search forward, `?` to search backward; `Ctrl-B D` to detach without killing.

**Prompt header variants** (RAG layer):
- `## Previously observed objects (across earlier steps in this episode):` — appears when memory has ≤10 entries (RAG fallback to dump-all).
- `## Previously observed objects (top 10 most relevant of N):` — appears once memory has >10 entries; confirms RAG retrieval is engaging.

**Per-entry render variants** (v2/v3 switch):
- `memory_render: trajectory` (v2 default): `- Ladle observed at positions (1.23, 0.91, -0.45) at steps 0-1; (1.50, 1.10, -0.20) at step 5`
- `memory_render: last_only` (v3): `- Ladle last observed at (1.50, 1.10, -0.20) at step 5`

**Pass criteria:**
| Result | Meaning |
|---|---|
| `last observed at` count > 0 AND `observed at positions` count = 0 | ✅ v3 mode active |
| `observed at positions` count > 0 AND `last observed at` count = 0 | ✅ v2 mode active (default) |
| Both > 0 | ⚠️ Mixed — you started one mode, killed it, restarted with the other. Look at the most-recent prompts only |
| Both = 0 | ⚠️ Eval not started yet OR `persistent_memory` flag is broken |

If neither header appears, the flag isn't threading. If only the first appears even on long episodes, RAG isn't engaging (check `memory_top_k` config). Either case → stop the eval and debug.

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

## What's left after v2 smoke lands

- **v2.10 (smoke A/B)**: compare v2 smoke against baseline smoke. Pass criterion: no subset drops >25 pp vs. baseline (a 0.1 subsample is noisy, just looking for "didn't break anything catastrophically").
- **v2 full run**: all 6 subsets, no `down_sample_ratio`. ~300 episodes total. Same `exp_name` template but drop the `_smoke` suffix.
- **v2 full A/B**: the actual research deliverable. Compare against the Q4_K_M baseline full run.
- **Optional ablations** (Part E commands above): memory-off run + memory-on-but-dump-all run. Lets us decompose the v2 delta into "did memory help?" vs "did RAG filtering help on top of memory?".

If v2 smoke surfaces a flag-threading bug or RAG never engages, pause full runs and debug.

---

## Open design points (resolution by version)

| Topic | v1 | v2 | v3 |
|---|---|---|---|
| 3D position | not captured | captured | captured (still) |
| RAG over memory | not done | ✅ Qwen3-Embedding-0.6B + top-10 cosine | unchanged |
| **Stale-position confusion** (model thinks an object is still at where it last saw it after picking it up) | n/a | **culprit in `complex_instruction` -0.08 regression** | ✅ addressed by `memory_render=last_only` (only most-recent sighting per object) |
| `parentReceptacles` (container relationships) | not captured | not captured | not captured — v4 candidate |
| Forgetting / staleness (`max_age_steps`) | not needed (≤30 step episodes) | not needed | not needed |
| Agent-relative coords (vs world-frame) | not captured | not captured | not captured — v4 candidate if VLM still struggles with world-frame numbers |
| **Runtime switchability** between render modes | n/a | n/a | ✅ `memory_render` flag in yaml + CLI |

---

## Reference

- Original design doc: [atlas_integrated_emben_testing.md](atlas_integrated_emben_testing.md)
- Baseline (Q4_K_M) eval plan + results: [qwen35embodiedbench_test.md](qwen35embodiedbench_test.md)
- Docker / container setup history: [embodiedbench_docker_setup.md](embodiedbench_docker_setup.md)
- Analysis script: [nutri-atlas/scripts/embench_results_analysis.py](../nutri-atlas/scripts/embench_results_analysis.py)
- Approved plan file (this session's working plan): `/home/boxun/.claude/plans/validated-coalescing-whisper.md`
