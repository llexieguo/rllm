#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

RLLM_HOME="${REPO_ROOT}/.rllm"
CUDA_VISIBLE_DEVICES="0"
N_GPUS_PER_NODE="1"
TP_SIZE="1"
GPU_MEMORY_UTILIZATION="0.6"

MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
TRUST_REMOTE_CODE="true"

TOTAL_EPOCHS="1"
TRAIN_BATCH_SIZE="2"
PPO_MINI_BATCH_SIZE="2"
PPO_MICRO_BATCH_SIZE_PER_GPU="1"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="1"
MAX_PROMPT_LENGTH="16384"
MAX_RESPONSE_LENGTH="1024"

PROJECT_NAME="mas_orchestra_offline"
EXPERIMENT_NAME="qwen3_vl_2b_offline_reinforce"
LOGGER_BACKENDS="[console]"
SAVE_FREQ="-1"
CKPT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
WANDB_API_KEY=""
WANDB_MODE=""

DATASET_NAME="sgi_reasoning_mas_orchestra_offline_replay"
DATASET_TRAIN_FILE="data/run_20260328_132306/mas_orchestra_offline_replay_all_shards"
DATASET_VAL_FILE=""
DATASET_TEST_FILE=""
TRAJECTORY_BONUS_WEIGHT="1.0"

export CUDA_VISIBLE_DEVICES
export RLLM_HOME
export MAS_ORCHESTRA_OFFLINE_DATASET_NAME="${DATASET_NAME}"
export MAS_ORCHESTRA_TRAJECTORY_BONUS_WEIGHT="${TRAJECTORY_BONUS_WEIGHT}"
export WANDB_API_KEY
export WANDB_MODE

echo "Offline dataset name: ${DATASET_NAME}"
if [[ -n "${DATASET_TRAIN_FILE}" ]]; then
  echo "Offline dataset source: ${DATASET_TRAIN_FILE}"
fi
echo "Model path: ${MODEL_PATH}"
echo "Loggers: ${LOGGER_BACKENDS}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Checkpoint save freq: ${SAVE_FREQ}"

if [[ -n "${DATASET_TRAIN_FILE}" ]]; then
  register_action=(
    "${PYTHON_BIN}" -m examples.mas_orchestra.offline_pipeline.prepare_offline_replay_dataset
    --dataset-name "${DATASET_NAME}"
    --train-file "${DATASET_TRAIN_FILE}"
  )
  if [[ -n "${DATASET_VAL_FILE}" ]]; then
    register_action+=(--val-file "${DATASET_VAL_FILE}")
  fi
  if [[ -n "${DATASET_TEST_FILE}" ]]; then
    register_action+=(--test-file "${DATASET_TEST_FILE}")
  fi

  echo "Registering offline replay dataset into ${RLLM_HOME}"
  "${register_action[@]}"
fi

action=(
  "${PYTHON_BIN}" -m examples.mas_orchestra.offline_pipeline.train_offline_mas_orchestra
  trainer.nnodes=1
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.logger=${LOGGER_BACKENDS}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size=null"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size=null"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size=null"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=${TRUST_REMOTE_CODE}"
  "critic.model.path=${MODEL_PATH}"
  "critic.model.trust_remote_code=${TRUST_REMOTE_CODE}"
)

"${action[@]}" "$@"
