# Phase 5 — Run Qwen3.5-9B Evaluation on EmbodiedBench

## Status

🟡 **In progress** — initial EB-Alfred smoke test on 5-episode `base` subset succeeded with 40% task_success and **zero JSON parse errors**. Full multi-subset eval running.

## Context

Phase 4 + Phase 6 of the EmbodiedBench Docker setup are complete: both EB-Alfred and EB-Habitat verified working inside `embench:alfhab`. Now we run actual benchmark evaluations against the user's locally-built Qwen3.5-9B-Q4_K_M GGUF model (with `mmproj-BF16.gguf` for vision).

**Why this is non-trivial:**
- The model is a GGUF (llama.cpp format), not HuggingFace transformers — can't use `model_type=local` (vLLM).
- The model is **outside the container**; the container must reach it over the network.
- EmbodiedBench's `remote_model.py` dispatches based on string matching the model name — `"Qwen3.5"` won't match any existing branch.

## Architecture (revised)

```
┌─────────────────────────┐                      ┌─────────────────────────────┐
│  HOST                   │                      │  CONTAINER (embench:alfhab)  │
│                         │                      │                              │
│  llama-server :8080  ◄──┼─ HTTP via ───────────┼─ EmbodiedBench python      │
│  + Qwen3.5-9B-Q4_K_M    │   host.docker.internal│   model_type=remote        │
│  + mmproj-BF16          │                      │   remote_url=                │
│  (GPU 0)                │                      │   http://host.docker.internal│
│                         │                      │             :8080/v1         │
└─────────────────────────┘                      └─────────────────────────────┘
       Bridge network +  --add-host=host.docker.internal:host-gateway
```

**Note:** Original plan was to use `--network=host`, but this caused X11 client connection failures inside the container (AI2-THOR Unity couldn't get a display). Switched to bridge network with `--add-host` — Unity now reaches the X server normally, and the container reaches the host's llama-server via `host.docker.internal`.

## Critical files identified

- **llama-server binary**: `/home/boxun/softwares/llama.cpp/build/bin/llama-server` (CUDA-built)
- **Reference start script** (parallel=1, for single sequential client): `/home/boxun/work/atlas/mimir/nutri_rag/scripts/start_server.sh`
- **Reference start script** (parallel=3, for benchmark workloads): `/home/boxun/work/atlas/qwen_test/start_server.sh`
- **Model files**: `/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf` + `mmproj-BF16.gguf`
- **EmbodiedBench dispatch**: `EmbodiedBench/embodiedbench/planner/remote_model.py` lines 18 (`remote_url` env), 54-63 (Qwen3-VL → `OpenAI(base_url=remote_url)`), 230-236 (`_call_qwen7b` uses `response_format=json_schema`)
- **Working container snapshot**: `embench:alfhab-v3`

## Key design decision — how to make EmbodiedBench dispatch to the OpenAI path

`remote_model.py` line 54-63 routes to a vanilla OpenAI client only if `"Qwen3-VL"`, `"Qwen2-VL"`, or `"Qwen2.5-VL"` appears in `model_name`. Our actual model is Qwen3.5 — no match.

**Recommended approach (no code change):** pass `model_name="Qwen3-VL-9B-GGUF"`. EmbodiedBench dispatches via OpenAI client; llama-server ignores the model-name string in the API request (it only has one model loaded anyway). The actual chat template applied is Qwen3.5's — comes from the GGUF metadata, applied by llama-server itself.

Alternative (cleaner but requires editing the repo): add a `"Qwen3.5"` branch to `remote_model.py` that mirrors the `"Qwen3-VL"` branch. Skip unless the name-mismatch causes issues at runtime.

## Step-by-step plan

### Step 5.1 — Wait for the GPU to be free

```bash
watch -n 5 nvidia-smi
```

Stop when GPU-Util is low and memory used is under ~5GB. Q4_K_M + mmproj + KV cache for ctx=65536 will need roughly 10-15GB.

### Step 5.2 — Start llama-server on the host (tmux for persistence)

```bash
tmux new -s qwen-server

# Inside tmux:
~/softwares/llama.cpp/build/bin/llama-server \
    --model /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
    --mmproj /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/mmproj-BF16.gguf \
    --port 8080 \
    --host 0.0.0.0 \
    --ctx-size 65536 \
    --n-gpu-layers 999 \
    --parallel 1 \
    --chat-template-kwargs '{"enable_thinking":false}'
```

Detach with `Ctrl-B D`. Re-attach later with `tmux attach -t qwen-server`.

Flags match `nutri_rag/scripts/start_server.sh`. We use `Q4_K_M.gguf` (user-chosen) instead of the script's default `UD-Q4_K_XL.gguf`. `--parallel 1` is correct for EmbodiedBench (sequential single-client eval).

### Step 5.3 — Verify the server is reachable on the host

In a separate shell:
```bash
# Should print loaded model metadata
curl -s http://localhost:8080/v1/models | head -20

# Quick text-only smoke test
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"x","messages":[{"role":"user","content":"reply with one word: ready"}],"max_tokens":10}'
```

Expected: JSON response with one word ("ready" or similar). If this fails, stop and debug — no point starting the container until the server is good.

### Step 5.4 — Start the EmbodiedBench container with bridge networking

```bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -it --rm \
  --name embench-eval \
  --add-host=host.docker.internal:host-gateway \
  --device=/dev/tty0 --device=/dev/tty1 --device=/dev/tty2 --device=/dev/tty3 \
  --device=/dev/tty4 --device=/dev/tty5 --device=/dev/tty6 --device=/dev/tty7 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0:/opt/embodiedbench/embodiedbench/envs/eb_alfred/data/json_2.1.0 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/results:/opt/embodiedbench/results \
  --shm-size=8g \
  embench:alfhab
```

Key change from earlier plan: replaced `--network=host` with `--add-host=host.docker.internal:host-gateway`. See Issue 5.2 below for why.

### Step 5.5 — Inside the container: setup + verify

```bash
# A. Apply manual patches (until Phase 7 bakes these in)
mv /usr/lib/xorg/modules/drivers/modesetting_drv.so \
   /usr/lib/xorg/modules/drivers/modesetting_drv.so.disabled

sed -i 's|Xorg -noreset|Xorg -noreset -ac|' \
    /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py

# B. Symlink running/ → results/ so eval output persists to host
# Without this, EmbodiedBench writes to /opt/embodiedbench/running/ which is
# NOT the mounted volume, so results die when the container exits.
# See Issue 5.6.
[ -d /opt/embodiedbench/running ] && mv /opt/embodiedbench/running /opt/embodiedbench/running.old
ln -s /opt/embodiedbench/results /opt/embodiedbench/running

# C. Activate env
source /opt/conda/etc/profile.d/conda.sh
conda activate embench

# D. Confirm container can reach the host's llama-server (no curl in image; use python)
python3 -c "
import urllib.request, json
r = urllib.request.urlopen('http://host.docker.internal:8080/v1/models')
print('Model:', json.loads(r.read())['data'][0]['id'])
"

# E. Set env vars
export remote_url=http://host.docker.internal:8080/v1
export OPENAI_API_KEY=EMPTY   # OpenAI Python client requires a non-empty key

# F. For EB-Alfred only: start headless X on display :1
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
sleep 3
ls /tmp/.X11-unix/   # should show X1
export DISPLAY=:1
```

### Step 5.6 — Run a small-scale EB-Alfred evaluation first

Start with `down_sample_ratio=0.1` (10% of dataset, ~few episodes) to confirm end-to-end before launching a long run.

```bash
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    exp_name='qwen35_q4km_alf_smoke'
```

Note: `model_name=Qwen3-VL-9B-GGUF` is the **dispatch trick** explained earlier — the substring `Qwen3-VL` makes `remote_model.py` route through the OpenAI client. The actual loaded model is Qwen3.5-9B.

Watch the llama-server tmux window for incoming requests. Watch the eval terminal for episode-level output.

**EB-Alfred smoke-test results (5 of 6 subsets completed; long_horizon cancelled to start EB-Habitat smoke earlier):**

| Subset | `task_success` | `task_progress` | `planner_output_error` (avg) | Wall time | Notes |
|--------|----------------|-----------------|------------------------------|-----------|-------|
| `base` | 0.4 (2/5) | 0.567 | 0 | ~2:30 | "Books on desk" + "Towel to bathtub" succeeded |
| `common_sense` | 0.4 (2/5) | 0.533 | 1.4 | ~8:30 | One outlier ep had 7 JSON retries → 340s |
| `complex_instruction` | 0.4 (2/5) | 0.467 | 0 | ~5:00 | Distractor phrases didn't seem to hurt |
| `spatial` | **0.2 (1/5)** | 0.267 | 0.8 | ~23:00 | Weakest subset; one ep took 12+ min on retries |
| `visual_appearance` | **0.6 (3/5)** | 0.6 | 1.8 | ~27:00 | **Best subset**; one ep had 9 JSON retries → 23 min |
| `long_horizon` | not run | — | — | — | Cancelled to switch to EB-Habitat smoke |

**Pipeline health: confirmed.** Qwen3.5 chat template applied correctly by llama-server, JSON schema constrained generation works, OpenAI client → llama-server connection stable.

**Capability profile observation (Qwen3.5-9B-Q4_K_M):**
- **Strong**: visual perception (60%), basic instruction following (40%)
- **Weak**: spatial reasoning (20%)
- **Failure mode**: occasional JSON retry loops on harder prompts — wastes ~5-20 min per stuck episode

### Step 5.7 — EB-Habitat smoke test

```bash
python -m embodiedbench.main \
    env=eb-hab \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    exp_name='qwen35_q4km_hab_smoke'
```

EB-Habitat doesn't need DISPLAY (uses EGL off-screen). Same dispatch logic.

**EB-Habitat smoke-test results (only `spatial_relationship` subset completed before stopping):**

| Subset | `task_success` | `task_progress` | `subgoal_reward` | `planner_output_error` | Wall time |
|--------|----------------|-----------------|-------------------|------------------------|-----------|
| `spatial_relationship` | **0.0 (0/5)** | 0.467 | 0.0 | 0 | ~17:00 |

All 5 episodes failed task_success, but model made ~50% partial progress on each — picked up correct object, attempted navigation, failed at final placement at "right of sink" / "right receptacle of left counter".

**Why EB-Habitat is harder than EB-Alfred for this model:**
- Objects identified by visual description ("a small red object with green top") instead of by name
- Multi-step navigation in 3D apartment scenes
- Spatial relationship terms ("right receptacle of the left counter") are inherently ambiguous

**Pipeline confirmed working on EB-Habitat** (zero JSON errors, full episodes completing, GPU rendering active via NVIDIA OpenGL 4.6). Score is a model-capability finding, not a setup issue.

The EB-Habitat smoke was stopped after `spatial_relationship` to skip ahead to full EB-Alfred runs (Step 5.8). The other 5 EB-Habitat subsets remain to be evaluated separately.

### Step 5.8 — Scale up to full runs (once smoke tests pass)

Drop `down_sample_ratio` (defaults to `1.0`, full dataset):
```bash
python -m embodiedbench.main env=eb-alf model_name=Qwen3-VL-9B-GGUF model_type=remote exp_name='qwen35_q4km_alf_full'
python -m embodiedbench.main env=eb-hab model_name=Qwen3-VL-9B-GGUF model_type=remote exp_name='qwen35_q4km_hab_full'
```

### Step 5.9 — Switch to a different quantization (Q4_K_S, Q3_K_M, etc.)

After getting one quantization's results, swap the model on llama-server and re-run. The container stays running — no Docker restart needed.

Available quants in `/home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/`:
```
Qwen3.5-9B-Q4_K_M.gguf       ← current "default" smoke/full run
Qwen3.5-9B-Q4_K_S.gguf       ← slightly smaller, faster
Qwen3.5-9B-Q3_K_M.gguf       ← noticeable quality drop, faster
Qwen3.5-9B-Q3_K_S.gguf
Qwen3.5-9B-UD-Q4_K_XL.gguf   ← larger Q4, better quality
Qwen3.5-9B-UD-Q3_K_XL.gguf
Qwen3.5-9B-UD-Q2_K_XL.gguf
Qwen3.5-9B-UD-IQ2_M.gguf     ← smallest, lowest quality
Qwen3.5-9B-UD-IQ2_XXS.gguf
```
The mmproj file (`mmproj-BF16.gguf`) is shared across all quants — same vision adapter regardless of base-model quant.

#### Process (example: switching to Q4_K_S)

**1. Stop the current eval** (if running) — Ctrl+C in the container shell. Env vars, X server, symlink all persist.

**2. Stop the current llama-server** — attach to its tmux on the host:
```bash
tmux attach -t qwen-server
# Inside: Ctrl+C to stop. Then Ctrl+B D to detach (or exit).
```

Verify it's down:
```bash
ss -tlnp 2>/dev/null | grep 8080   # should be empty
```

**3. Start llama-server with the new quant**:
```bash
tmux attach -t qwen-server || tmux new -s qwen-server
```
Inside tmux:
```bash
~/softwares/llama.cpp/build/bin/llama-server \
    --model /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_S.gguf \
    --mmproj /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/mmproj-BF16.gguf \
    --port 8080 \
    --host 0.0.0.0 \
    --ctx-size 65536 \
    --n-gpu-layers 999 \
    --parallel 1 \
    --chat-template-kwargs '{"enable_thinking":false}'
```
Only one line changed vs. Q4_K_M: `Q4_K_M.gguf` → `Q4_K_S.gguf`.

Detach with Ctrl+B D once it's "main loop"-ready.

**4. Verify the new model is loaded** (from host):
```bash
python3 -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8080/v1/models')
print('Loaded:', json.loads(r.read())['data'][0]['id'])
"
# Expect: Loaded: Qwen3.5-9B-Q4_K_S.gguf
```

**5. Re-run EB-Alfred from inside the container** with a **new `exp_name`** so results don't overwrite the previous quant:

Quick smoke first (recommended):
```bash
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    eval_sets='[base]' \
    exp_name='qwen35_q4ks_alf_smoke' \
    2>&1 | tee /opt/embodiedbench/results/alf_smoke_q4ks.log
```
~5-15 min. Confirms the new model produces valid JSON before committing to the full run.

Then the full run:
```bash
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    exp_name='qwen35_q4ks_alf_full' \
    2>&1 | tee /opt/embodiedbench/results/alf_full_q4ks.log
```

#### What changes vs. what stays the same

| Component | Q4_K_M run | Q4_K_S run |
|-----------|------------|------------|
| llama-server `--model` flag | `Q4_K_M.gguf` | **`Q4_K_S.gguf`** |
| llama-server port | 8080 | 8080 (same) |
| Docker container | running | running (no restart) |
| In-container env vars + X server + symlink | as-is | as-is (same) |
| `model_name=` argument to eval | `Qwen3-VL-9B-GGUF` | `Qwen3-VL-9B-GGUF` (same — just a dispatch hint) |
| `exp_name=` argument | `qwen35_q4km_alf_full` | **`qwen35_q4ks_alf_full`** |
| Log path | `alf_full.log` | `alf_full_q4ks.log` |

`model_name` does NOT need to change because it's just the substring-match dispatch hint for `remote_model.py`. The actual loaded model is whatever the host's llama-server is serving — controlled entirely by its `--model` flag.

#### Result location after multi-quant runs

```
/home/boxun/work/atlas/mimir/EmbodiedBench/results/
├── eb_alfred/
│   ├── Qwen3-VL-9B-GGUF_qwen35_q4km_alf_full/    ← Q4_K_M
│   ├── Qwen3-VL-9B-GGUF_qwen35_q4ks_alf_full/    ← Q4_K_S
│   └── Qwen3-VL-9B-GGUF_qwen35_q3km_alf_full/    ← Q3_K_M (if you do more)
├── alf_full.log               ← Q4_K_M log
├── alf_full_q4ks.log          ← Q4_K_S log
└── alf_full_q3km.log
```

A/B / multi-way comparison across quantizations is then a matter of reading the per-subset `final_results` JSONs from each directory.

#### Suggested naming convention for `exp_name`

`qwen35_{quant}_{env}_{scope}` where:
- `quant` = `q4km`, `q4ks`, `q3km`, `q3ks`, `udq4kxl`, `udiq2m`, etc.
- `env` = `alf` or `hab`
- `scope` = `smoke`, `full`, or `subset_<name>`

Examples: `qwen35_q4ks_alf_smoke`, `qwen35_udq4kxl_hab_full`, `qwen35_iq2m_alf_subset_spatial`.

### Step 5.10 — Fresh restart from scratch (after exiting the container)

This is the **copy-paste reference** for coming back to this work after `exit`-ing the container. It consolidates Steps 5.1 → 5.5 into one block so you don't have to scroll between sections. Assumes the `embench:alfhab` image and the host's llama-server build are still present (they are, since both live outside the container).

Pick which quantization you want for this run by editing the `MODEL_FILE` variable at the top.

#### A) On the HOST — start llama-server in tmux

```bash
# 1. Check GPU is free (or at least has ~10-15 GB free)
nvidia-smi

# 2. Start llama-server in tmux (replace MODEL_FILE for the quant you want)
MODEL_FILE=Qwen3.5-9B-Q4_K_M.gguf    # or Q4_K_S, UD-IQ2_M, etc.

tmux new -s qwen-server -d \
    "~/softwares/llama.cpp/build/bin/llama-server \
        --model /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/$MODEL_FILE \
        --mmproj /home/boxun/work/atlas/unsloth/Qwen3.5-9B-GGUF/mmproj-BF16.gguf \
        --port 8080 --host 0.0.0.0 \
        --ctx-size 65536 --n-gpu-layers 999 --parallel 1 \
        --chat-template-kwargs '{\"enable_thinking\":false}'"

# 3. Wait ~10 seconds, then verify server is up
sleep 10
python3 -c "
import urllib.request, json
print('Loaded:', json.loads(urllib.request.urlopen('http://localhost:8080/v1/models').read())['data'][0]['id'])
"
# Expected: Loaded: Qwen3.5-9B-Q4_K_M.gguf (or whichever you set)
```

#### B) On the HOST — start the container

```bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -it --rm \
  --name embench-eval \
  --add-host=host.docker.internal:host-gateway \
  --device=/dev/tty0 --device=/dev/tty1 --device=/dev/tty2 --device=/dev/tty3 \
  --device=/dev/tty4 --device=/dev/tty5 --device=/dev/tty6 --device=/dev/tty7 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0:/opt/embodiedbench/embodiedbench/envs/eb_alfred/data/json_2.1.0 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/results:/opt/embodiedbench/results \
  --shm-size=8g \
  embench:alfhab
```

#### C) Inside the CONTAINER — full setup (one paste)

```bash
# 1. Manual patches (until Phase 7 bakes them into the image)
mv /usr/lib/xorg/modules/drivers/modesetting_drv.so \
   /usr/lib/xorg/modules/drivers/modesetting_drv.so.disabled

sed -i 's|Xorg -noreset|Xorg -noreset -ac|' \
    /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py

[ -d /opt/embodiedbench/running ] && mv /opt/embodiedbench/running /opt/embodiedbench/running.old
ln -s /opt/embodiedbench/results /opt/embodiedbench/running

# 2. Conda env + env vars
source /opt/conda/etc/profile.d/conda.sh
conda activate embench
export remote_url=http://host.docker.internal:8080/v1
export OPENAI_API_KEY=EMPTY

# 3. Confirm network reachability to llama-server
python3 -c "
import urllib.request, json
print('Container sees:', json.loads(urllib.request.urlopen('http://host.docker.internal:8080/v1/models').read())['data'][0]['id'])
"

# 4. Start headless X server (only needed for EB-Alfred; EB-Habitat doesn't need it)
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
sleep 3
ls /tmp/.X11-unix/   # expect: X1
export DISPLAY=:1
```

#### D) Inside the CONTAINER — run the eval

Pick one based on what you want to test, and **edit `exp_name`** to match the actual quant + env + scope you chose:

```bash
# EB-Alfred smoke (~10-15 min, 5 eps base subset)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    eval_sets='[base]' \
    exp_name='qwen35_q4km_alf_smoke' \
    2>&1 | tee /opt/embodiedbench/results/alf_smoke_q4km.log

# EB-Alfred full (~5-10 hours, all subsets)
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    exp_name='qwen35_q4km_alf_full' \
    2>&1 | tee /opt/embodiedbench/results/alf_full_q4km.log

# EB-Habitat smoke (no DISPLAY needed; can run alongside or instead of alf)
python -m embodiedbench.main \
    env=eb-hab \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    down_sample_ratio=0.1 \
    eval_sets='[base]' \
    exp_name='qwen35_q4km_hab_smoke' \
    2>&1 | tee /opt/embodiedbench/results/hab_smoke_q4km.log
```

#### Quick mental model

| Scenario | What to do |
|----------|------------|
| Coming back after exit, do new run | Sections A → B → C → D (this whole step) |
| Container alive, swap quant only | Step 5.9 (just A again + restart eval in container) |
| Server died, container alive | Re-run section A inside the existing tmux, then re-run D in the container |
| Just exiting / wrapping up | Nothing — `exit` the container, optionally Ctrl+C the llama-server tmux |

#### What survives across container exits (already on host)

- All eval results: `/home/boxun/work/atlas/mimir/EmbodiedBench/results/`
- All logs: `/home/.../results/*.log`
- The `embench:alfhab` image (Phase 6 rebuild)
- llama.cpp build + all Qwen3.5 GGUF files
- This plan doc + other plan docs

#### What you have to redo each container start

- The 4 manual patches in section C step 1 (modesetting, -ac, symlink, conda activate)
- The 3 env vars (`remote_url`, `OPENAI_API_KEY`, `DISPLAY`)
- The X server startup

These are baked into section C as a single paste-able block, so it's a 30-second redo. **Phase 7** (in `plans/embodiedbench_docker_setup.md`) would eliminate the first 3 of those by baking them into the image — do it whenever the manual repetition starts feeling tedious.

## Issues encountered during Phase 5 setup (resolved)

**Issue 5.1 — Xorg crashes in modesetting driver's glamor init**

Symptom: `Xorg: dixRegisterPrivateKey: Assertion 'global_keys[type].created' failed.` in `libglamoregl.so` + `modesetting_drv.so`. Xorg tried to load Mesa's modesetting driver alongside NVIDIA on the same DRM device.

Fix: rename modesetting_drv.so out of the way (see Step 5.5 setup commands).

**Issue 5.2 — `--network=host` breaks AI2-THOR X11 connection**

With `--network=host`, AI2-THOR's Unity process couldn't acquire a display — kept printing `Authorization required, but no authorization protocol specified` and never got to the "Display 0 '0': 1024x768" line. Tried `-ac` Xorg flag and explicit Xauthority cookies; neither helped. Root cause likely the shared abstract Unix socket namespace.

Fix: use `--add-host=host.docker.internal:host-gateway` (bridge network) and have the container reach the host's llama-server at `host.docker.internal:8080` instead of `localhost:8080`.

**Issue 5.3 — OpenAI client requires API key even for local servers**

`remote_model.py` does `OpenAI(base_url=remote_url)` without an `api_key`. The `openai` library refuses to initialize without one: `OpenAIError: The api_key client option must be set...`.

Fix: `export OPENAI_API_KEY=EMPTY` before running the eval (llama-server doesn't validate the key value).

**Issue 5.4 — `-ac` flag needed on Xorg for client connections**

Even after switching off `--network=host`, X11 sometimes rejected client connections. Adding `-ac` (disable host-based access control) to the Xorg command makes it consistent. See Step 5.5 setup `sed` command.

**Issue 5.5 — Display port conflict (only when using `--network=host`)**

Other users on the shared host have X servers on `:1`, `:2`, `:12`, `:13`, `:14`. With `--network=host`, the container shares the host's port space — TCP port `6000 + display_num` collisions. Solved by switching off `--network=host` (then we can safely use `:1` again since the container has its own loopback).

**Issue 5.6 — Eval results written to `running/` not the mounted `results/`**

EmbodiedBench's `main.py` writes per-episode JSONs and logs to `running/{env}/{model}_{exp_name}/{subset}/results/...` relative to the container's working dir `/opt/embodiedbench`. **This is not the mounted volume** — `results/` is. With `--rm`, the smoke results from `running/` died with the container.

Symptom: `ls /home/.../EmbodiedBench/results/` on the host returned an empty directory after completing the smoke runs.

Fix: symlink `running/` to point at the mount:
```bash
# Inside container, before launching the eval:
[ -d /opt/embodiedbench/running ] && mv /opt/embodiedbench/running /opt/embodiedbench/running.old
ln -s /opt/embodiedbench/results /opt/embodiedbench/running
```

To recover the existing smoke results before container exit:
```bash
mkdir -p /opt/embodiedbench/results/smoke_runs
cp -r /opt/embodiedbench/running.old/* /opt/embodiedbench/results/smoke_runs/
```

The Phase 5 smoke runs were rescued this way — results are now at
`/home/boxun/work/atlas/mimir/EmbodiedBench/results/smoke_runs/eb_alfred/...` and
`/home/.../results/smoke_runs/eb_habitat/...`.

This issue will be resolved permanently in Phase 7 (Dockerfile bake-in).

## Verification — what success looks like

| Step | Pass criterion | Status |
|------|----------------|--------|
| 5.3 | `curl /v1/models` returns JSON; chat completion returns a coherent word | ✅ Done |
| 5.5 | Python check from inside container reaches `host.docker.internal:8080`; X server up on `:1` | ✅ Done |
| 5.6 | EB-Alfred episodes run, model responses parsed, per-subset metrics printed | ✅ 5 of 6 subsets done (40% avg) |
| 5.7 | Same as 5.6 for EB-Habitat | ✅ 1 of 6 subsets done; pipeline confirmed working |
| 5.8 | Full-dataset metrics appear in `/home/boxun/work/atlas/mimir/EmbodiedBench/results/` (mounted on host) | ▶ Next — EB-Alfred full run |

## Known risks and fallbacks

| Risk | Detection | Status / Fallback |
|------|-----------|--------------------|
| `response_format=json_schema` not supported by this llama.cpp build | Eval crashes with 400 from server, or invalid-JSON parse error | ✅ Confirmed working — `planner_output_error: 0` |
| Qwen3.5 prompt format mismatch (using "Qwen3-VL" dispatch path) | Garbled / refusal outputs from the model | ✅ Confirmed working — model returns valid actions |
| Context length exceeded for long episodes | `ctx-size` exceeded errors in llama-server log | Watching during full runs; bump `--ctx-size` if needed |
| llama-server hits OOM | CUDA OOM in server tmux | Switch to a smaller quant (`UD-IQ2_M` or `Q3_K_S`), or lower `--ctx-size` |

## After Phase 5 — what's still left

1. **Full EB-Alfred run** (Step 5.8) — drop `down_sample_ratio`, do all 6 subsets on the entire dataset. Expected runtime: several hours (overnight). The smoke runs already confirmed everything works.
2. **Full EB-Habitat run** — same plan, separate session. Possibly overnight.
3. **Phase 7 (planned)** — bake Issues 5.1 and 5.4 fixes into `Dockerfile.alfhab` so containers don't require the manual `mv` + `sed` steps. See `plans/embodiedbench_docker_setup.md` Phase 7.
4. **Append final results** to `plans/embodiedbench_docker_setup.md` once full runs complete.

## Recommendation for full EB-Alfred run

Given the smoke-test observations:
- Average per-episode time ≈ 30-300 seconds (high variance due to JSON retry loops)
- 6 subsets × ~50 episodes each (rough estimate of full dataset) ≈ 300 episodes
- At ~100 seconds avg per episode ≈ **5-10 hours**

**Best to run in tmux** so a connection drop doesn't kill it. Also worth keeping the llama-server tmux window visible — if it spikes to high memory or hangs, you can intervene.

```bash
# Inside the container (env vars + X server already set up):
python -m embodiedbench.main \
    env=eb-alf \
    model_name=Qwen3-VL-9B-GGUF \
    model_type=remote \
    exp_name='qwen35_q4km_alf_full' \
    2>&1 | tee /opt/embodiedbench/results/alf_full.log
```

Note: `down_sample_ratio` is omitted → defaults to `1.0` (full dataset). Output piped to `tee` so you have a saved log even if you lose the terminal.
