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

## Data Shape

The offline replay format here is **stepwise replay**, not one cumulative chat transcript.

- One row = one trajectory.
- `steps` = ordered decision points inside that trajectory.
- Each step carries its own `messages` prompt context, and the workflow appends `response` as the assistant turn during training.
- `convert_tree_samples_to_offline_replay.py` reconstructs one trainable `main` step per orchestra attempt, so converted tree-search data is usually **multi-step** when a task needed multiple attempts.
- The step prompts are not raw full-history chat dumps. Earlier attempts are summarized into the next prompt through the reconstructed memory text.

One easy point to miss: the current offline training entrypoint defaults to **`reinforce`**, not `grpo`.

- Offline default: `algorithm.adv_estimator=reinforce`
- Offline default: `rllm.stepwise_advantage.mode=per_step`
- Offline default: `actor_rollout_ref.rollout.n=1`
- Online `examples/mas_orchestra/train.sh` is the example path that defaults to `grpo`

## Minimal Single-Turn Data

If you want a **single-turn** offline sample, keep `steps` length equal to `1`.

The important convention is:

- Put the prompt context before generation in `steps[0].messages`
- Put the model output in `steps[0].response`
- Do not include the final assistant response inside `messages`

Example `.json` payload with two single-turn rows:

```json
[
  {
    "task_id": "single-turn-correct-1",
    "data_source": "manual_single_turn",
    "is_correct": true,
    "trajectory_reward": 1.0,
    "steps": [
      {
        "messages": [
          {
            "role": "system",
            "content": "You are a strict orchestration controller. Output JSON only."
          },
          {
            "role": "user",
            "content": "Question: What is 2 + 2?\nOptions: A. 3 B. 4 C. 5 D. 6"
          }
        ],
        "response": "{\"action\":\"submit\",\"reasoning\":\"2 + 2 = 4.\",\"submit_reason\":\"high confidence\"}",
        "step_reward": 0.0,
        "trainable": true,
        "step_type": "main",
        "model": "local-policy",
        "metadata": {
          "attempt_index": 1
        }
      }
    ]
  },
  {
    "task_id": "single-turn-incorrect-1",
    "data_source": "manual_single_turn",
    "is_correct": false,
    "trajectory_reward": 0.0,
    "steps": [
      {
        "messages": [
          {
            "role": "system",
            "content": "You are a strict orchestration controller. Output JSON only."
          },
          {
            "role": "user",
            "content": "Question: What is the capital of France?\nOptions: A. Berlin B. Madrid C. Paris D. Rome"
          }
        ],
        "response": "{\"action\":\"submit\",\"reasoning\":\"I think it is Berlin.\",\"submit_reason\":\"best guess\"}",
        "step_reward": 0.0,
        "trainable": true,
        "step_type": "main",
        "model": "local-policy",
        "metadata": {
          "attempt_index": 1
        }
      }
    ]
  }
]
```

Notes:

- `steps` only needs to be non-empty, but at least one step must have `trainable=true`
- For a single-turn row, the full training reward lands on that one trainable step
- With the default `TRAJECTORY_BONUS_WEIGHT=1.0`, the effective reward is `step_reward + trajectory_reward`
- If you want multiple turns, keep appending additional step objects to `steps`

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
