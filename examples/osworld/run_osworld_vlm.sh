#!/bin/bash
# Qwen3-VL GRPO on OSWorld GUI tasks (OSWorld HTTP rollout + slime Megatron train)
#
# Prerequisites (run in separate terminals):
#   1) SGLang router for the VLM (OSWorld agent calls OPENAI_BASE_URL, e.g. :30000)
#   2) OSWorld rollout server from OSWorld repo:
#        cd $OSWORLD_ROOT && bash scripts/bash/run_qwen3vl_rollout_server.sh
#      Server cwd must be OSWorld root; test_config_base_dir=evaluation_examples
#
# Usage:
#   OSWORLD_ROOT=/path/to/OSWorld SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-2B-Instruct ./run_osworld_vlm.sh
#
# Use specific physical GPUs (colocate train + SGLang rollout on the same 4 cards):
#   export CUDA_VISIBLE_DEVICES=0,1,6,7
#   export SLIME_SCRIPT_NUM_GPUS=4
#   ./run_osworld_vlm.sh

set -euo pipefail

TRAIN_BACKEND="megatron"
MODEL_NAME=${SLIME_SCRIPT_MODEL_NAME:-"Qwen3-VL-2B-Instruct"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-4}

# Must be set before `ray start` so Ray/slime only see the selected GPUs.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
   echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} logical GPUs for slime)"
fi

OSWORLD_ROOT=${OSWORLD_ROOT:-"/home/yongjinwu/work/git/OSWorld"}
OSWORLD_TASKS_JSONL="${OSWORLD_ROOT}/datasets/osworld_tasks.jsonl"
OSWORLD_ROLLOUT_URL=${OSWORLD_ROLLOUT_URL:-"http://127.0.0.1:18081/rollout"}

# Validate MODEL_NAME
VALID_MODELS="
  Qwen2.5-VL-3B-Instruct
  Qwen2.5-VL-7B-Instruct
  Qwen3-VL-2B-Instruct
  Qwen3-VL-4B-Instruct
  Qwen3-VL-8B-Instruct
  Qwen3-VL-30B-A3B-Instruct
"
if ! echo "$VALID_MODELS" | grep -qw "$MODEL_NAME"; then
   echo "Error: MODEL_NAME must be one of: $VALID_MODELS"
   exit 1
fi

MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

if [ -z "${SLIME_SCRIPT_EXTERNAL_RAY:-}" ] || [ "${SLIME_SCRIPT_EXTERNAL_RAY}" = "0" ]; then
   USE_EXTERNAL_RAY=0
else
   USE_EXTERNAL_RAY=1
fi

pkill -9 sglang || true
sleep 3
if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   ray stop --force || true
   pkill -9 ray || true
fi
pkill -9 slime || true
sleep 3

set -ex
export PYTHONBUFFERED=16

SLIME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)"

# Build task list jsonl (relative paths: examples/<domain>/<id>.json)
if [ ! -f "${OSWORLD_TASKS_JSONL}" ]; then
   python3 "${SLIME_DIR}/examples/osworld/build_task_dataset.py" \
      --osworld-root "${OSWORLD_ROOT}"
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi

mkdir -p /root/models
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
   hf download "Qwen/${MODEL_NAME}" --local-dir "/root/models/${MODEL_NAME}"
fi

CKPT_ARGS=(
   --hf-checkpoint "/root/models/${MODEL_NAME}"
   --rotary-base 5000000
)

# Dataset: each line {"example_file": "examples/libreoffice_calc/<uuid>.json"}
# Use RELATIVE paths — OSWorld joins them with evaluation_examples (test_config_base_dir).
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.osworld_rollout.generate_rollout
   --prompt-data "${OSWORLD_TASKS_JSONL}"
   --input-key example_file
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 2048
   --rollout-max-prompt-len 2048
   # One sample per Megatron step: OSWorld trajs are ~5k+ tokens (vision); packing 2+ per step OOMs on 24GB.
   # num_steps=4 still runs compute_log_prob before advantages (no SGLang rollout_log_probs / can_reuse).
   --num-steps-per-rollout 4
   --global-batch-size 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# SGLang still required: OSWorld agent calls it via OPENAI_BASE_URL during env rollout.
# Only set --sglang-router-port (not --sglang-router-ip): if ip is preset, slime skips launching the router.
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-router-port 30000
)

MISC_ARGS=(
   --colocate
   --num-gpus-per-node "${NUM_GPUS}"
)

BACKEND_ARGS=(
   --train-backend megatron
   --load "/root/models/${MODEL_NAME}"
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)

MODEL_ARGS_FILE=$(echo "$MODEL_NAME" | sed 's/-Instruct//g; s/-Thinking//g; s/Qwen3-VL-/qwen3-/g; s/-2B/-1.7B/g')
MODEL_ARGS_ROTARY_BASE=5000000 source "${SLIME_DIR}/scripts/models/${MODEL_ARGS_FILE}.sh"

if [ "$USE_EXTERNAL_RAY" = "0" ]; then
   export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SLIME_DIR}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"OSWORLD_ROOT\": \"${OSWORLD_ROOT}\",
    \"OSWORLD_ROLLOUT_URL\": \"${OSWORLD_ROLLOUT_URL}\",
    \"SLIME_HOST_IP\": \"127.0.0.1\"
  }
}"

# Ray job cwd defaults to where you invoke this script; train.py lives in repo root.
ray job submit --address="http://127.0.0.1:8265" \
   --working-dir "${SLIME_DIR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}
