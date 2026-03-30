#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TP_SIZE="${TP_SIZE:-1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
PROJECT_NAME="${PROJECT_NAME:-mas_orchestra_offline}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_vl_2b_offline_reinforce}"
DATASET_NAME="${DATASET_NAME:-sgi_reasoning_mas_orchestra_offline_replay}"
TRAJECTORY_BONUS_WEIGHT="${TRAJECTORY_BONUS_WEIGHT:-1.0}"

export CUDA_VISIBLE_DEVICES
export RLLM_HOME="${RLLM_HOME:-${REPO_ROOT}/.rllm}"
export MAS_ORCHESTRA_OFFLINE_DATASET_NAME="${DATASET_NAME}"
export MAS_ORCHESTRA_TRAJECTORY_BONUS_WEIGHT="${TRAJECTORY_BONUS_WEIGHT}"

action=(
  "${PYTHON_BIN}" -m examples.mas_orchestra.offline_pipeline.train_offline_mas_orchestra
  trainer.nnodes=1
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  'trainer.logger=[console]'
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
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
