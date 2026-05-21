# OSWorld + slime VLM GRPO

Train Qwen-VL with slime while rollout runs on the **OSWorld HTTP server** (GUI env + SGLang). Each slime rollout step:

1. Read task paths from `--prompt-data` jsonl  
2. `POST /rollout` with `file_list` (relative `examples/<domain>/<id>.json`)  
3. OSWorld runs env + SGLang, writes `traj/<rollout_idx>/*.json`  
4. `traj_to_sample()` → `Sample` (tokens, loss_mask, reward, `multimodal_train_inputs`)  
5. Megatron GRPO training  

## Path convention (important)

| Where | Path format |
|-------|-------------|
| **slime dataset** (`example_file`) | Prefer **relative**: `examples/libreoffice_calc/<uuid>.json` |
| **OSWorld `file_list`** | Same relative paths (required by server) |
| **OSWorld resolves** | `join(test_config_base_dir, path)` → `evaluation_examples/examples/...` |

Absolute paths in the dataset are OK: `osworld_rollout` strips the `evaluation_examples` prefix and sends relative paths to `/rollout`.

Do **not** put only a basename in the dataset; OSWorld expects `examples/<domain>/<id>.json`.

## Files

| File | Role |
|------|------|
| `build_task_dataset.py` | Scan `evaluation_examples/examples/**/*.json` → jsonl |
| `traj_to_sample.py` | traj JSON → `Sample` |
| `smoke_test_traj_to_sample.py` | Validate traj / adapter |
| `run_osworld_vlm.sh` | Launch slime training (like `geo3k_vlm`) |
| `slime/rollout/osworld_rollout.py` | Custom `--rollout-function-path` |

## Setup

### Docker (`--network=host`) + OSWorld on host

With `--network=host`, container and host share `127.0.0.1` — no `host.docker.internal` needed.

**Recommended `docker run` extra mounts** (in addition to slime repo):

```bash
-v /home/yongjinwu/work/git/OSWorld:/root/OSWorld \
-v /mnt/nfs/yongjinwu/models/Qwen3-VL-2B-Instruct:/root/models/Qwen3-VL-2B-Instruct \
```

**Ports**

| Service | URL | Who starts it |
|---------|-----|----------------|
| SGLang OpenAI API | `http://127.0.0.1:30000/v1` | slime `train.py` (`--sglang-router-port 30000`) |
| OSWorld `/rollout` | `http://127.0.0.1:18081/rollout` | host conda (`run_qwen3vl_rollout_server.sh`) |

**Start order**

1. **Container**: `docker run ...` → `cd /root/slime/examples/osworld` → `./run_osworld_vlm.sh`  
   Wait until logs show `Router launched at 127.0.0.1:30000`.
2. **Host**: `cd $OSWORLD_ROOT && bash scripts/bash/run_qwen3vl_rollout_server.sh`  
   Keep `export OPENAI_BASE_URL="http://127.0.0.1:30000/v1"` (or `localhost`).

**Env inside container** (also passed via `run_osworld_vlm.sh` ray runtime):

```bash
export OSWORLD_ROOT=/root/OSWorld
export OSWORLD_ROLLOUT_URL=http://127.0.0.1:18081/rollout
export SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-2B-Instruct
```

Do **not** start OSWorld before slime’s first `pkill sglang` in `run_osworld_vlm.sh` unless you accept SGLang being killed; normal flow is slime first, then OSWorld server.

### All on host (no Docker)

**Terminal A – slime** (`run_osworld_vlm.sh` → Ray + SGLang :30000 + train)

**Terminal B – OSWorld server** (after SGLang is up):

```bash
cd /home/yongjinwu/work/git/OSWorld
export OPENAI_BASE_URL="http://127.0.0.1:30000/v1"
bash scripts/bash/run_qwen3vl_rollout_server.sh
```

Build dataset only:

```bash
python build_task_dataset.py --osworld-root "$OSWORLD_ROOT"
# -> $OSWORLD_ROOT/datasets/osworld_tasks.jsonl
```

Smoke test existing traj:

```bash
export HF_CHECKPOINT=/mnt/nfs/yongjinwu/models/Qwen3-VL-2B-Instruct
python smoke_test_traj_to_sample.py --osworld-root "$OSWORLD_ROOT" \
  "$OSWORLD_ROOT/traj/0/"*.json
```

## Environment variables (slime OSWorld rollout)

| Variable | Default | Meaning |
|----------|---------|---------|
| `OSWORLD_ROOT` | (required) | OSWorld repo root (`traj/`, `traj_images/`) |
| `OSWORLD_ROLLOUT_URL` | `http://127.0.0.1:18081/rollout` | HTTP endpoint |
| `OSWORLD_TEST_CONFIG_BASE_DIR` | `$OSWORLD_ROOT/evaluation_examples` | Prefix for `file_list` paths |
| `OSWORLD_ROLLOUT_TIMEOUT` | `7200` | HTTP timeout (seconds) |
| `OSWORLD_ROLLOUT_MAX_RETRIES` | `3` | Retries on 409 / connection errors |

Slime CLI: `--rollout-function-path slime.rollout.osworld_rollout.generate_rollout`, `--prompt-data`, `--input-key example_file`.  
`--rollout-batch-size` should be ≤ OSWorld `--num_envs` (server handles one request at a time).  
Set `--global-batch-size` to `rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout` (default: 4×1÷4=1). Use `--num-steps-per-rollout 4` so Megatron runs `compute_log_prob` before GRPO advantages (OSWorld traj has no SGLang `rollout_log_probs`). OSWorld trajs are long (~5k tokens with vision); use `--max-tokens-per-gpu 2048` (not 8192) and `--global-batch-size 1` on 24GB colocate to avoid train OOM. Lower `--sglang-mem-fraction-static` (e.g. 0.45) to leave headroom after rollout wake-up.

Reward comes from traj (`reward` field); no `--rm-type` needed for rollout.

## `multimodal_train_inputs`

Traj already has expanded vision `tokens`. To rebuild `pixel_values`, the adapter **collapses** vision spans in token space, runs processor once, and checks `len(input_ids) == len(traj["tokens"])`. See comments in `traj_to_sample.py`.
