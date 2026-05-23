# OSWorld Rollout

Collect trajectories via HTTP rollout in the OSWorld GUI environment, then run Qwen-VL GRPO training with slime + Megatron. Entry script: `examples/osworld/run_osworld_vlm.sh`.

---

## 1. Prepare the Dataset

Convert the OSWorld dataset into the format slime expects. Each line in the final dataset is the path to one task setup file.

### Generate `osworld_tasks.jsonl`

```bash
export OSWORLD_ROOT=<OSWORLD_REPO>
python3 <SLIME_REPO>/examples/osworld/build_task_dataset.py \
  --osworld-root "${OSWORLD_ROOT}"
# Output: ${OSWORLD_ROOT}/datasets/osworld_tasks.jsonl
```

---

## 2. Launch (Docker + Training)

### 2.1 Start Docker

Replace the three `-v` paths below with paths on your machine:

| Placeholder | Mount in container | Description |
|-------------|-------------------|-------------|
| `<SLIME_REPO>` | `/root/slime` | This repo (includes `run_osworld_vlm.sh`) |
| `<MODEL_DIR>` | `/root/models/Qwen3-VL-2B-Instruct` | Downloaded Qwen3-VL-2B-Instruct weights (must match `SLIME_SCRIPT_MODEL_NAME`) |
| `<OSWORLD_REPO>` | `/root/OSWorld` | Custom [OSWorld repo](https://github.com/WU-James/OSWorld) root (includes `evaluation_examples`, `datasets/`) |

```bash
docker run --rm --name yongjinwu_slime_dev \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --add-host=host.docker.internal:host-gateway \
  -v <SLIME_REPO>:/root/slime \
  -v <MODEL_DIR>:/root/models/Qwen3-VL-2B-Instruct \
  -v <OSWORLD_REPO>:/root/OSWorld \
  -w /root/slime \
  -it slimerl/slime:latest /bin/bash
```

### 2.2 Fix the environment inside the container

You may need to install `torch_memory_saver` (not bundled in the image; required for colocated training):

```bash
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 \
  --no-cache-dir --force-reinstall
```

### 2.3 Start the OSWorld Rollout Server

https://github.com/WU-James/OSWorld

### 2.4 Start training inside the container

Set `CUDA_VISIBLE_DEVICES` to your GPU IDs (the number of visible GPUs must match `SLIME_SCRIPT_NUM_GPUS`, default 4):

```bash
CUDA_VISIBLE_DEVICES=0,1,4,7 \
OSWORLD_ROOT=/root/OSWorld \
bash /root/slime/examples/osworld/run_osworld_vlm.sh
```

Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SLIME_SCRIPT_MODEL_NAME` | `Qwen3-VL-2B-Instruct` | Model name (must be in the script allowlist) |
| `SLIME_SCRIPT_NUM_GPUS` | `4` | GPUs for Ray / training / colocate |
| `OSWORLD_ROLLOUT_URL` | `http://127.0.0.1:18081/rollout` | OSWorld HTTP endpoint |
