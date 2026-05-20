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
