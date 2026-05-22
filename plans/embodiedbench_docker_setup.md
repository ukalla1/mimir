# EmbodiedBench Docker Setup — EB-Alfred + EB-Habitat

Setting up EmbodiedBench in Docker on the shared `vega` server to test Qwen 3.5 models. Only EB-Alfred and EB-Habitat are needed (not EB-Navigation or EB-Manipulation).

**Why Docker instead of native install:** the native setup requires editing `/etc/X11/Xwrapper.config` to allow non-console users to run Xorg with `needs_root_rights=yes`. Server admin advised against this on a shared box. Docker isolates the Xorg + GPU setup entirely.

---

## Status

- ✅ Phase 0 — Prerequisites verified
- ✅ Phase 1 — Stripped-down `Dockerfile.alfhab` + `.dockerignore`
- ✅ Phase 2 — Image built (`embench:alfhab`, SHA `144f343add3d`)
- ✅ Phase 3 — EB-ALFRED dataset downloaded (527MB, 2156 scenarios)
- ✅ Phase 4 — Both **EB-Alfred** and **EB-Habitat** verified working in container (snapshot: `embench:alfhab-v3`, retained as backup)
- 🟡 Phase 5 — Qwen3.5-9B smoke tests on both EB-Alfred (5/6 subsets, avg 40% success) and EB-Habitat (1/6 subsets, pipeline confirmed). Capability profile: strong at visual (60%), weak at spatial (20% / 0%). Full plan + findings in `plans/qwen35embodiedbench_test.md`. Next: full EB-Alfred run.
- ✅ Phase 6 — `Dockerfile.alfhab` updated, image rebuilt as clean `embench:alfhab` (SHA `2e4f1cb7fe07b6`), end-to-end verified
- ⬜ Phase 7 — Bake **Phase 5** additional patches into `Dockerfile.alfhab` (modesetting removal, `-ac` flag in startx.py)

---

## Phase 0 — Prerequisites

### What was already present

- Docker v29.4.3 installed
- User `boxun` already in the `docker` group (`docker ps` worked without sudo)
- NVIDIA driver 580.82.07 on host
- `/etc/docker/daemon.json` already had the `nvidia` runtime declared:
  ```json
  { "runtimes": { "nvidia": { "args": [], "path": "nvidia-container-runtime" } } }
  ```

### What was missing

- The `nvidia-container-runtime` binary did not exist on the system. `docker info` showed `nvidia` as a registered runtime, but the binary was never installed. The `daemon.json` was a "stale pointer."
- Symptom: `docker run --gpus all ... nvidia-smi` failed with `executable file not found in $PATH`.

### Fix — install `nvidia-container-toolkit`

```bash
# 1. Pre-flight: backup daemon.json + confirm no running containers
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%Y%m%d-%H%M)
docker ps -a
sudo docker ps

# 2. Add NVIDIA's GPG key + apt source
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Installed packages: `libnvidia-container1`, `libnvidia-container-tools`, `nvidia-container-toolkit-base`, `nvidia-container-toolkit` (all v1.19.0-1).

### Did NOT need to restart Docker

Since `daemon.json` was already correct (relative-path lookup: `"path": "nvidia-container-runtime"`), Docker found the new binary on `$PATH` immediately. **No `systemctl restart docker` needed** — saved disrupting other users on the shared server.

### Verification

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Printed the RTX A6000 with driver 580.82.07. GPU passthrough confirmed working.

---

## Phase 1 — Build files

### `.dockerignore`

Created `EmbodiedBench/.dockerignore` to exclude:
- `.git/` (~161MB)
- `docs/` (~13MB of images)
- Python caches, egg-info
- `embodiedbench/envs/eb_alfred/data/json_2.1.0/` (dataset — mounted at run time instead)
- `embodiedbench/envs/eb_habitat/data/` (downloaded inside the image)

Build context went from 402MB → ~228MB.

### `Docker/Dockerfile.alfhab`

Stripped-down version of `Docker/Dockerfile` with these changes:

| Change | Why |
|--------|-----|
| `git clone` (required GitHub token) → `COPY . /opt/embodiedbench` | Repo is now public; using local copy avoids token setup |
| `COPY scripts/install_nvidia.sh` → `COPY Docker/scripts/install_nvidia.sh` | Build context moved from `Docker/` to repo root |
| `install_nvidia.sh` script → inlined `wget` using `NVIDIA_VERSION` build arg | The script reads `/proc/driver/nvidia/version` which isn't available during `docker build` |
| Removed EB-Navigation section | Not needed |
| Removed EB-Manipulation section (CoppeliaSim, PyRep) | Not needed; saves ~10 min build + a few GB |
| `ubuntu18.04` → `ubuntu20.04` (CUDA 11.4.2 → 11.8.0) | The 18.04 base image was removed from Docker Hub (Ubuntu 18.04 EOL) |
| Added `conda tos accept` for `pkgs/main` and `pkgs/r` | New Anaconda policy (2024) requires explicit TOS acceptance |

EB-ALFRED dataset is **not baked into the image** — to be mounted as a volume at run time (Phase 4).

---

## Phase 2 — Image build

### Build command

```bash
cd /home/boxun/work/atlas/mimir/EmbodiedBench
docker build \
  -f Docker/Dockerfile.alfhab \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg NVIDIA_VERSION=580.82.07 \
  -t embench:alfhab \
  --progress=plain \
  . 2>&1 | tee /tmp/embench-build.log
```

Recommended to run inside `tmux` so a connection drop doesn't kill the build.

### Errors encountered + fixes

1. **`nvidia/cuda:11.4.2-devel-ubuntu18.04: not found`**
   Fix: switched base to `nvidia/cuda:11.8.0-devel-ubuntu20.04`. Verified existence via `docker manifest inspect` before pulling.

2. **`CondaToSNonInteractiveError: Terms of Service have not been accepted`**
   Fix: added these calls before `conda env create`:
   ```bash
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   ```

### Final result

- Image: `embench:alfhab`
- SHA: `144f343add3d`
- Contents:
  - Ubuntu 20.04 + CUDA 11.8 dev tools
  - NVIDIA userspace driver 580.82.07 (matches host)
  - Miniconda
  - `embench` conda env (EmbodiedBench + dependencies)
  - habitat-sim 0.3.0, habitat-lab v0.3.0, habitat-baselines
  - YCB + ReplicaCAD datasets (baked in)

### Build performance notes

- First build: ~30-40 min (mostly conda solve + habitat-sim install + habitat dataset download)
- Rebuilds: Docker layer cache skips everything before the changed step

---

## Phase 3 — Download EB-ALFRED dataset

Done on the **host**, mounted into the container at run time:

```bash
cd /home/boxun/work/atlas/mimir/EmbodiedBench
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0
```

Result: 527MB total, 2156 task scenario directories. Git LFS pulled content automatically.

---

## Phase 4 — Run container + verify environments

This phase required **several iterative fixes** for X server + NVIDIA driver issues that the original Dockerfile didn't account for. All fixes need to be baked into `Dockerfile.alfhab` (Phase 6).

### Initial run command

```bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -it --rm \
  --name embench-test \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0:/opt/embodiedbench/embodiedbench/envs/eb_alfred/data/json_2.1.0 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/results:/opt/embodiedbench/results \
  --shm-size=8g \
  embench:alfhab
```

Inside the container, `nvidia-smi` worked and `import embodiedbench` succeeded.

### Issues encountered + fixes

**Issue 4.1 — Xorg server not installed**

`startx` failed with `FileNotFoundError: 'Xorg'`. The `.run` driver installer only provided NVIDIA's Xorg modules, not the Xorg server itself.

Fix (run inside container):
```bash
apt-get update
apt-get install -y --no-install-recommends \
    xserver-xorg-core xinit \
    libxext6 libxrender1 libxtst6 libxi6
```

Note: `xserver-xorg-video-nvidia` does NOT exist as an unversioned apt package in Ubuntu 20.04 — NVIDIA's `.run` installer already provided that.

**Issue 4.2 — NVIDIA Xorg modules in wrong path**

The NVIDIA `.run` installer warned: `nvidia-installer was forced to guess the X library path '/usr/lib64'`. Result: modules installed at `/usr/lib64/xorg/modules/` but Xorg on Ubuntu 20.04 looks at `/usr/lib/xorg/modules/`.

The installer also placed copies at `/usr/lib/x86_64-linux-gnu/nvidia/xorg/` (Ubuntu-style path). Fix is to symlink from there:

```bash
ln -sf /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so \
       /usr/lib/xorg/modules/drivers/nvidia_drv.so
ln -sf /usr/lib/x86_64-linux-gnu/nvidia/xorg/libglxserver_nvidia.so \
       /usr/lib/xorg/modules/extensions/libglxserver_nvidia.so
```

**Issue 4.3 — `/dev/tty0` not available**

`parse_vt_settings: Cannot open /dev/tty0 (No such file or directory)`. Containers don't have TTY devices by default.

Adding `-novtswitch -sharevts` flags to `Xorg` command did NOT help (Xorg 1.20 opens `/dev/tty0` before parsing flags).

Fix: container restart with `--device` mounts.

**Issue 4.4 — Xorg picks VT 2, also missing**

After mounting only `/dev/tty0`: `xf86OpenConsole: Cannot open virtual console 2`. Xorg picks the next available VT, needs the device file for it.

Fix: mount multiple TTY devices.

**Issue 4.5 — Failed to acquire modesetting permission**

Error: `(EE) NVIDIA(GPU-0): Failed to acquire modesetting permission. (EE) NVIDIA(0): Failing initialization of X screen`. Another process on the host (likely another user's X session) holds DRM master on the GPU.

Fix: tell NVIDIA we don't need a real display — patch `startx.py` to add `Option "UseDisplayDevice" "none"` to the Screen section:

```bash
python3 -c "
path = '/opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py'
with open(path) as f: c = f.read()
c = c.replace(
    'Option         \"AllowEmptyInitialConfiguration\" \"True\"',
    'Option         \"AllowEmptyInitialConfiguration\" \"True\"\n    Option         \"UseDisplayDevice\" \"none\"'
)
with open(path, 'w') as f: f.write(c)
"
```

**Note — is this fix always required?** Strictly speaking, it's only needed when *something* on the host holds DRM master on the GPU (another user's X server, an active display manager, the boot console framebuffer, etc.). On a fully idle headless box you could skip it.

In practice, **keep this patch always**:
- The shared `vega` server almost always has multiple users running X sessions, so contention is the norm
- `UseDisplayDevice "none"` is a no-op when not needed — performance is identical, AI2-THOR/Habitat still get GPU rendering via the off-screen framebuffer
- AI2-THOR's own Docker recipes hard-code this option for the same reason
- Without it, the same image will work on one host but mysteriously fail on another

**Issue 4.6 — Missing Ubuntu font**

After X server came up, `EBAlfEnv` initialized AI2-THOR successfully (Unity engine running, GPU rendering at 1024x768) but crashed loading `/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf` — package `fonts-ubuntu` not in image.

Fix (run inside container):
```bash
apt-get install -y fonts-ubuntu
```

After this, EBAlfEnv ran successfully, loading the scene and entering the interactive `action id:` prompt with 207+ available actions (find/pick up/open/close/turn on/slice/etc.).

ALSA error messages about `cannot find card '0'` printed during AI2-THOR startup are **benign** — just no audio device in the container. Doesn't affect simulation.

**Issue 4.7 — EB-Habitat: Git LFS pointers instead of actual mesh data**

EBHabEnv loaded the dataset and initialized the simulator (OpenGL 4.6 via NVIDIA driver), then crashed at:
```
Utility::Json: unexpected v at .../Stage_v3_sc3_staging.glb:1:1
Trade::GltfImporter::openData(): invalid JSON
AssertionError: ESP_CHECK failed: Error loading general mesh data from file 'data/replica_cad/configs/stages/../../stages/Stage_v3_sc3_staging.glb'
```

Diagnosis: `head -c 300 Stage_v3_sc3_staging.glb` showed `version https://git-lfs.github.com/spec/v1 / oid sha256:... / size 7884212` — the file was a 132-byte LFS pointer, not the 7.8MB binary `.glb` mesh.

Root cause: the Dockerfile installs `git-lfs` via conda, but never runs `git lfs install` to wire up the smudge filter, so `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets` cloned the data repos with pointer files only.

Three Habitat data dirs are affected (all under `versioned_data/`):
- `replica_cad_dataset/` — scene meshes
- `ycb/` — YCB object dataset
- `hab_fetch/` — robot URDFs and meshes

Fix (run inside container):
```bash
for dir in /opt/embodiedbench/embodiedbench/envs/eb_habitat/data/versioned_data/replica_cad_dataset \
           /opt/embodiedbench/embodiedbench/envs/eb_habitat/data/versioned_data/ycb \
           /opt/embodiedbench/embodiedbench/envs/eb_habitat/data/versioned_data/hab_fetch; do
    cd "$dir"
    git lfs install --local
    git lfs pull
done
```

Total ~1-3 GB downloaded. After this, EBHabEnv ran successfully, showing 50 episodes and the interactive `action id:` prompt with ~70 actions (navigate / pick up / place / open / close).

Remaining warnings (all benign):
- `navmesh_instances` not found → optional navmesh files; sim works without them
- `MeshTools::compile(): ignoring TextureCoordinates 1` → rendering hint, doesn't affect output
- `Articulated Object: No Glob path result` → URDF still resolves via `versioned_data/hab_fetch`

### Final working `docker run` command

```bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -it --rm \
  --name embench-test \
  --device=/dev/tty0 --device=/dev/tty1 --device=/dev/tty2 --device=/dev/tty3 \
  --device=/dev/tty4 --device=/dev/tty5 --device=/dev/tty6 --device=/dev/tty7 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/embodiedbench/envs/eb_alfred/data/json_2.1.0:/opt/embodiedbench/embodiedbench/envs/eb_alfred/data/json_2.1.0 \
  -v /home/boxun/work/atlas/mimir/EmbodiedBench/results:/opt/embodiedbench/results \
  --shm-size=8g \
  embench:alfhab-v2
```

Where `embench:alfhab-v2` is a snapshot committed from the running container after applying issues 4.1, 4.2, 4.5 fixes:
```bash
docker commit embench-test embench:alfhab-v2
```

### Inside the container — verification

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate embench

# Start headless X
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
sleep 3
ls /tmp/.X11-unix/        # should show X1
export DISPLAY=:1

# Run env tests
python -m embodiedbench.envs.eb_alfred.EBAlfEnv      # AI2-THOR auto-downloads on first run (~500MB)
python -m embodiedbench.envs.eb_habitat.EBHabEnv
```

---

## Phase 5 — Run Qwen evaluation (in progress)

See `plans/qwen35embodiedbench_test.md` for the detailed step-by-step plan. Below is a summary of the additional issues discovered during Phase 5 setup that the Phase 6 image didn't anticipate.

### Architecture decision — Container ↔ host networking

llama-server runs on **the host** (outside the container). Two options to bridge:

| Option | Command flag | X11 side effects | Verdict |
|--------|--------------|------------------|---------|
| `--network=host` | shares host network namespace | Triggers X11 abstract-socket auth issues (see Issue 5.1) | ❌ Don't use |
| `--add-host=host.docker.internal:host-gateway` | bridge network + DNS to host | None | ✅ Use this |

Container reaches llama-server at `http://host.docker.internal:8080/v1`.

### Phase 5 issues encountered + fixes

**Issue 5.1 — Xorg crashes in modesetting driver's glamor init**

After running `python -m embodiedbench.envs.eb_alfred.scripts.startx 1`, Xorg crashed with:
```
Xorg: dixRegisterPrivateKey: Assertion `!global_keys[type].created' failed.
... libglamoregl.so (glamor_init+0xcf) ...
... modesetting_drv.so ...
```

Xorg was trying to auto-load both NVIDIA and Mesa's modesetting driver, with glamor being initialized twice on the same DRM device. Fix: disable modesetting by renaming its driver module:
```bash
mv /usr/lib/xorg/modules/drivers/modesetting_drv.so \
   /usr/lib/xorg/modules/drivers/modesetting_drv.so.disabled
```

**Issue 5.2 — X11 client connections rejected ("Authorization required")**

With `--network=host`, AI2-THOR Unity couldn't connect to the X server — got "Authorization required, but no authorization protocol specified" repeated, and Unity never reported its `Display 0 ...` line. Tried `-ac` Xorg flag and explicit Xauthority cookies; neither helped.

Root cause: `--network=host` shares the host's abstract Unix socket namespace, complicating X auth.

Fix: switch off `--network=host`, use `--add-host=host.docker.internal:host-gateway` instead.

Side benefit: avoids port conflicts on display numbers already used on host (e.g., other users' X servers on `:1`, `:2`).

**Issue 5.3 — OpenAI client requires API key even for local servers**

`remote_model.py` calls `OpenAI(base_url=remote_url)` without an `api_key` argument. The `openai` Python library refuses to initialize without one — raises `OpenAIError: The api_key client option must be set...`.

Fix: set a dummy API key (llama-server doesn't validate it):
```bash
export OPENAI_API_KEY=EMPTY
```

**Issue 5.4 — `-ac` flag still needed on Xorg for reliable client connection**

Even without `--network=host`, X11 sometimes rejected connections. Adding `-ac` (disable host-based access control) to the Xorg command made it consistent:
```bash
sed -i 's|Xorg -noreset|Xorg -noreset -ac|' \
    /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py
```

**Issue 5.5 — Eval results written to `running/` instead of mounted `results/`**

EmbodiedBench's `main.py` writes per-episode JSON files and logs to `running/{env}/{model}_{exp_name}/{subset}/...` relative to the container's working directory `/opt/embodiedbench`. **This is NOT the mounted volume.** With `--rm` set, those results die with the container.

Symptom: after running smoke tests successfully, `ls /home/.../EmbodiedBench/results/` on the host shows an empty directory.

Fix: replace `/opt/embodiedbench/running` with a symlink to `/opt/embodiedbench/results` (which IS the bind-mounted host directory):
```bash
# Inside container, BEFORE running the eval:
mkdir -p /opt/embodiedbench/results
# If results dir somehow already exists in `running/`, move it first
[ -d /opt/embodiedbench/running ] && mv /opt/embodiedbench/running /opt/embodiedbench/running.old
ln -s /opt/embodiedbench/results /opt/embodiedbench/running
```

After this, all eval output (per-episode JSONs, logs, intermediate files) lands on the host at `/home/boxun/work/atlas/mimir/EmbodiedBench/results/...` and survives `--rm`.

**Better permanent fix (deferred to Phase 7):** bake this symlink into the Dockerfile, OR change the bind mount to map host's `results/` → `/opt/embodiedbench/running/` instead.

### Final working `docker run` for Phase 5

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

Inside, the manual setup needed each container start (until Phase 7 bakes them into the image):
```bash
# 1. Disable Mesa modesetting driver (Issue 5.1)
mv /usr/lib/xorg/modules/drivers/modesetting_drv.so \
   /usr/lib/xorg/modules/drivers/modesetting_drv.so.disabled

# 2. Add -ac flag to startx (Issue 5.4)
sed -i 's|Xorg -noreset|Xorg -noreset -ac|' \
    /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py

# 3. Symlink running/ → results/ so eval outputs persist to host (Issue 5.5)
[ -d /opt/embodiedbench/running ] && mv /opt/embodiedbench/running /opt/embodiedbench/running.old
ln -s /opt/embodiedbench/results /opt/embodiedbench/running

# 4. Activate env + start X
source /opt/conda/etc/profile.d/conda.sh && conda activate embench
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
sleep 3
export DISPLAY=:1

# 5. Env vars (Issue 5.3 + remote_url for llama-server)
export remote_url=http://host.docker.internal:8080/v1
export OPENAI_API_KEY=EMPTY
```

### Smoke-test results summary (down_sample_ratio=0.1, 5 episodes per subset)

**EB-Alfred** (5 of 6 subsets completed before stopping early to run EB-Habitat):

| Subset | `task_success` | Wall time | Notable |
|--------|----------------|-----------|---------|
| `base` | 0.4 | ~2:30 | |
| `common_sense` | 0.4 | ~8:30 | One ep had 7 JSON retries → 340s |
| `complex_instruction` | 0.4 | ~5:00 | Handles distractor phrases OK |
| `spatial` | **0.2** | ~23:00 | Weakest; one ep took 12+ min on retries |
| `visual_appearance` | **0.6** | ~27:00 | **Best**; one ep had 9 JSON retries → 23 min |
| `long_horizon` | not run | — | Cancelled to switch to EB-Habitat smoke |

**EB-Habitat** (1 of 6 subsets completed):

| Subset | `task_success` | `task_progress` | Wall time |
|--------|----------------|-----------------|-----------|
| `spatial_relationship` | 0.0 | 0.467 | ~17:00 |

### Key findings

- **Pipeline is healthy**: zero JSON parse errors on most subsets, llama-server stable, network bridge solid
- **Capability profile (Qwen3.5-9B-Q4_K_M)**: strong on visual perception (60%), weak on spatial reasoning (20% on EB-Alfred / 0% on EB-Habitat). Pattern is consistent across both envs.
- **EB-Habitat is harder than EB-Alfred for this model**: generic visual descriptions ("a small red object with green top") + 3D apartment navigation + ambiguous spatial terms ("right receptacle of the left counter")
- **Failure mode to watch**: occasional JSON retry loops on harder prompts can waste 5-20 minutes per stuck episode. Doesn't crash, just slows down.

Successful EB-Alfred examples: "Put the books on the desk", "Carry a towel to the bath tub", "Despite avoiding water spills, carry a towel to the bathtub" (complex_instruction), "Move a pencil from a small black bin to a wooden desk" (visual_appearance).

The dispatch trick `model_name=Qwen3-VL-9B-GGUF` correctly routes to OpenAI client; llama-server applies Qwen3.5's chat template from the GGUF metadata.

---

## Phase 6 — Bake Phase 4 fixes into `Dockerfile.alfhab` (completed)

All in-container patches from Phase 4 have been baked into `Dockerfile.alfhab` and verified by a clean rebuild. Required additions to `Dockerfile.alfhab`:

### Add after the apt install + NVIDIA driver install

```dockerfile
# Install Xorg server packages (NVIDIA .run provided modules but not the server)
# Plus libopengl0/libegl1/libgl1 (xserver-xorg-core pulls in libglvnd0 but NOT
# libopengl0, and NVIDIA's installer then skips its own GLVND stack — leaving
# libOpenGL.so.0 missing and breaking habitat_sim import. Be explicit.)
# Plus fonts-ubuntu (EBAlfEnv hardcodes a TTF path that requires it).
RUN apt-get update && apt-get install -y --no-install-recommends \
    xserver-xorg-core xinit \
    libxext6 libxrender1 libxtst6 libxi6 \
    libopengl0 libegl1 libgl1 \
    fonts-ubuntu \
 && rm -rf /var/lib/apt/lists/*

# Symlink NVIDIA's Xorg modules from where its installer placed them
# to where Ubuntu's Xorg actually looks
RUN ln -sf /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so \
           /usr/lib/xorg/modules/drivers/nvidia_drv.so \
 && ln -sf /usr/lib/x86_64-linux-gnu/nvidia/xorg/libglxserver_nvidia.so \
           /usr/lib/xorg/modules/extensions/libglxserver_nvidia.so
```

### Modify the existing repo COPY / pip install section

```dockerfile
# After `COPY . /opt/embodiedbench` and before pip install:
# Patch startx.py to skip modesetting (host's display server holds DRM master)
RUN sed -i 's|Option         "AllowEmptyInitialConfiguration" "True"|&\n    Option         "UseDisplayDevice" "none"|' \
    /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py
```

### Modify the existing Habitat dataset download section

Replace:
```dockerfile
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh && \
             conda activate embench && \
             cd /opt/embodiedbench/embodiedbench/envs/eb_habitat && \
             conda install -y -c conda-forge git-lfs && \
             python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets"
```

with:
```dockerfile
RUN bash -c "source /opt/conda/etc/profile.d/conda.sh && \
             conda activate embench && \
             cd /opt/embodiedbench/embodiedbench/envs/eb_habitat && \
             conda install -y -c conda-forge git-lfs && \
             git lfs install && \
             python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets && \
             for dir in data/versioned_data/replica_cad_dataset data/versioned_data/ycb data/versioned_data/hab_fetch; do \
                 cd \$dir && git lfs install --local && git lfs pull && cd -; \
             done"
```

The `git lfs install` (global) ensures the LFS smudge filter is active for the clones that follow. The post-clone loop is a belt-and-suspenders pass in case any individual repo wasn't smudged at clone time.

### Container run command unchanged

After rebuild, the `docker run` command still needs `--device=/dev/tty0 ... /dev/tty7` because the TTY devices are a host-side concern, not something the image can fix.

### Rebuild error encountered (resolved)

First Phase 6 rebuild failed at the Habitat dataset download step with:
```
ImportError: libOpenGL.so.0: cannot open shared object file: No such file or directory
```

This was a regression introduced by adding `xserver-xorg-core` to apt: it pulls in `libglvnd0` (via `libegl1`) but NOT `libopengl0` (the package that actually provides `libOpenGL.so.0`). When NVIDIA's `.run` installer ran later, it detected `libglvnd0` already present and skipped installing its own GLVND libraries — leaving `libOpenGL.so.0` missing.

The original `alfhab` build didn't hit this because `xserver-xorg-core` wasn't installed; the NVIDIA installer set up the full GLVND stack itself.

**Fix:** explicitly add `libopengl0 libegl1 libgl1` to the apt install line. After this, the rebuild succeeded.

### Snapshot images

| Tag | Status | Notes |
|-----|--------|-------|
| `embench:alfhab` | **Current source of truth** (SHA `2e4f1cb7fe07b6`) | Clean rebuild from updated Dockerfile; verified end-to-end |
| `embench:alfhab-v2` | Removable | Intermediate; superseded by `-v3` and clean `alfhab` |
| `embench:alfhab-v3` | **Retained as backup** | Last known-good snapshot before Phase 6 rebuild; user-requested to keep |

`-v2` can be deleted any time (`docker rmi embench:alfhab-v2`); `-v3` is intentionally retained as a fallback.

---

## Phase 7 — Bake Phase 5 patches into `Dockerfile.alfhab` (planned)

Three patches from Phase 5 issues (5.1, 5.4, 5.5) need to be baked into the Dockerfile so future containers don't require manual fixes. Add after the NVIDIA driver install + symlink section:

```dockerfile
# Disable Mesa's modesetting driver — Xorg auto-loads it alongside nvidia
# and crashes in glamor_init when both try to claim the same DRM device.
# See Issue 5.1.
RUN mv /usr/lib/xorg/modules/drivers/modesetting_drv.so \
       /usr/lib/xorg/modules/drivers/modesetting_drv.so.disabled

# Patch startx.py to also pass -ac (disable host-based access control).
# Combined with the existing UseDisplayDevice patch from Phase 6.
# See Issue 5.4.
RUN sed -i 's|Xorg -noreset|Xorg -noreset -ac|' \
    /opt/embodiedbench/embodiedbench/envs/eb_alfred/scripts/startx.py

# Make eval output (`running/`) write through to the bind-mounted `results/`
# directory, so per-episode JSONs survive container exit.
# See Issue 5.5.
RUN ln -s /opt/embodiedbench/results /opt/embodiedbench/running
```

Note for Issue 5.5: the symlink approach assumes `results/` is the canonical mount point. Alternative is to change the bind mount to map host's `results/` → `/opt/embodiedbench/running/` directly (no symlink needed). Either way works; the symlink is more discoverable.

After Phase 7 rebuild, the only setup needed in the container will be:
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate embench
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 &
sleep 3
export DISPLAY=:1
export remote_url=http://host.docker.internal:8080/v1
export OPENAI_API_KEY=EMPTY
```

Run command remains the same (still needs `--device=/dev/tty0..7` + `--add-host=host.docker.internal:host-gateway`).

Deferred until Phase 5 evaluation runs complete — no rush, current `embench:alfhab` works with the manual steps.

---

## Key references

- Stripped Dockerfile: `EmbodiedBench/Docker/Dockerfile.alfhab`
- Original Dockerfile (kept for reference): `EmbodiedBench/Docker/Dockerfile`
- Build context excludes: `EmbodiedBench/.dockerignore`
- daemon.json backup: `/etc/docker/daemon.json.bak.20260519-1159`
- Build log: `/tmp/embench-build.log` (last build run)
- Current source-of-truth image (clean rebuild from Dockerfile): `embench:alfhab` (SHA `2e4f1cb7fe07b6`)
- Backup image (Phase 4 commit, retained): `embench:alfhab-v3`
- Phase 5 plan: `plans/qwen35embodiedbench_test.md`
