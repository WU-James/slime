# OSWorld Rollout

在 OSWorld GUI 环境上通过 HTTP rollout 采集轨迹，再用 slime + Megatron 做 Qwen-VL 的 GRPO 训练。入口脚本：`examples/osworld/run_osworld_vlm.sh`。

---

## 1. 准备 Dataset

将OSWorld仓库中Dataset准备为Slime训练需要版本。最终Dataset中每条数据为一个Task Setup的路径。

### 生成 `osworld_tasks.jsonl`


```bash
export OSWORLD_ROOT=<OSWORLD_REPO>
python3 <SLIME_REPO>/examples/osworld/build_task_dataset.py \
  --osworld-root "${OSWORLD_ROOT}"
# 输出: ${OSWORLD_ROOT}/datasets/osworld_tasks.jsonl
```
---

## 2. 启动方式（Docker + 训练）

### 2.1 启动 Docker


将下面三条 `-v` 换成你机器上的实际路径：

| 占位符 | 挂载到容器内 | 说明 |
|--------|----------------|------|
| `<SLIME_REPO>` | `/root/slime` | 本仓库（含 `run_osworld_vlm.sh`） |
| `<MODEL_DIR>` | `/root/models/Qwen3-VL-2B-Instruct` | 已下载的 Qwen3-VL-2B-Instruct 权重目录（与 `SLIME_SCRIPT_MODEL_NAME` 一致） |
| `<OSWORLD_REPO>` | `/root/OSWorld` | 定制版 [OSWorld 仓库](https://github.com/WU-James/OSWorld) 根目录（含 `evaluation_examples`、`datasets/`） |

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

### 2.2 容器内修复环境

进入容器后可能需要安装 `torch_memory_saver`（当前镜像未自带，训练 colocate 时需要）：

```bash
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 \
  --no-cache-dir --force-reinstall
```

### 2.3 启动 OSWorld Rollout Server

https://github.com/WU-James/OSWorld

### 2.4 容器内启动训练

按实际 GPU 编号设置 `CUDA_VISIBLE_DEVICES`（逻辑 GPU 数量需与 `SLIME_SCRIPT_NUM_GPUS` 一致，默认 4）：

```bash
CUDA_VISIBLE_DEVICES=0,1,4,7 \
OSWORLD_ROOT=/root/OSWorld \
bash /root/slime/examples/osworld/run_osworld_vlm.sh
```

可选环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `SLIME_SCRIPT_MODEL_NAME` | `Qwen3-VL-2B-Instruct` | 模型名（须在脚本白名单内） |
| `SLIME_SCRIPT_NUM_GPUS` | `4` | Ray / 训练 / colocate 使用的 GPU 数 |
| `OSWORLD_ROLLOUT_URL` | `http://127.0.0.1:18081/rollout` | OSWorld HTTP 端点 |


