# MAS Orchestra Offline Pipeline

This directory contains the user-facing offline replay entrypoints:

- `convert_tree_samples_to_offline_replay.py`
- `prepare_offline_replay_dataset.py`
- `train_offline_mas_orchestra.py`
- `train_offline.sh`

Compatibility wrappers are also kept at:

- `examples/mas_orchestra/convert_tree_samples_to_offline_replay.py`
- `examples/mas_orchestra/prepare_offline_replay_dataset.py`
- `examples/mas_orchestra/train_offline_mas_orchestra.py`
- `examples/mas_orchestra/train_offline.sh`

## Current Retained Training Data

The retained local data artifacts are:

- `data/run_20260328_132306/mas_orchestra_offline_replay_all_shards`

The main training input is the shard directory:

- `784` trajectories total
- `249` correct
- `535` incorrect
- train only orchestra `main` steps
- `step_reward = 0`
- `trajectory_reward = binary_correct`
- `delegate_local = 0` in this run

`prepare_offline_replay_dataset.py` can read either a single file or a shard directory.

All commands below assume the current working directory is the repo root `rllm/`.

## Register

```bash
RLLM_HOME=.rllm \
.venv/bin/python -m examples.mas_orchestra.offline_pipeline.prepare_offline_replay_dataset \
  --dataset-name sgi_reasoning_mas_orchestra_offline_replay \
  --train-file data/run_20260328_132306/mas_orchestra_offline_replay_all_shards
```

## Cleanup

If you want to remove the local registered training copy under `.rllm` and free disk space:

```bash
rm -rf .rllm/datasets/sgi_reasoning_mas_orchestra_offline_replay
```

That only removes the registered cache. To train again later, rerun the register command above.

## Train

Recommended single-GPU example using physical GPU `1`:

```bash
RLLM_HOME=.rllm \
CUDA_VISIBLE_DEVICES=1 \
DATASET_NAME=sgi_reasoning_mas_orchestra_offline_replay \
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
./examples/mas_orchestra/offline_pipeline/train_offline.sh
```

Equivalent explicit version:

```bash
RLLM_HOME=.rllm \
CUDA_VISIBLE_DEVICES=1 \
N_GPUS_PER_NODE=1 \
TP_SIZE=1 \
DATASET_NAME=sgi_reasoning_mas_orchestra_offline_replay \
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
./examples/mas_orchestra/offline_pipeline/train_offline.sh
```

General form:

```bash
RLLM_HOME=.rllm \
DATASET_NAME=sgi_reasoning_mas_orchestra_offline_replay \
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
./examples/mas_orchestra/offline_pipeline/train_offline.sh
```

Old wrapper path still works:

```bash
RLLM_HOME=.rllm \
CUDA_VISIBLE_DEVICES=1 \
DATASET_NAME=sgi_reasoning_mas_orchestra_offline_replay \
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct \
./examples/mas_orchestra/train_offline.sh
```
