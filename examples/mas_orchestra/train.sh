#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

CUDA_VISIBLE_DEVICES=0
N_GPUS_PER_NODE=1
TP_SIZE=1
MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct
TRUST_REMOTE_CODE=true
TOTAL_EPOCHS=1
TRAIN_BATCH_SIZE=2
PPO_MINI_BATCH_SIZE=2
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
GPU_MEMORY_UTILIZATION=0.6
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=1024
PROJECT_NAME=mas_orchestra
EXPERIMENT_NAME=qwen3_vl_2b_grpo
STEP_LOG_ROOT=logs/mas_orchestra
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
API_MODEL=""
API_COST_BUDGET=1.0
API_INPUT_COST_PER_1M=""
API_OUTPUT_COST_PER_1M=""
SUB_MODELS=(
  "gpt-4.1"
  "gpt-4.1-mini"
  "gpt-4o-mini"
  "o3"
  "o3-mini"
  "gpt-5"
  "gpt-5-mini"
  "gemini-2.5-flash"
  "gemini-2.5-pro"
  "gemini-3-flash-preview"
  "gemini-3-pro-preview"
  "claude-sonnet-4-20250514"
  "claude-sonnet-4-5-20250929"
  "claude-haiku-4-5-20251001"
)

export CUDA_VISIBLE_DEVICES
export MAS_ORCHESTRA_STEP_LOG_ROOT="${STEP_LOG_ROOT}"
export MAS_ORCHESTRA_RUN_ID="${RUN_TIMESTAMP}"
export MAS_ORCHESTRA_API_COST_BUDGET="${API_COST_BUDGET}"
export MAS_ORCHESTRA_SUB_MODELS="$(printf '%s
' "${SUB_MODELS[@]}")"
if [ -n "${API_MODEL}" ]; then
  export MAS_ORCHESTRA_API_MODEL="${API_MODEL}"
fi
if [ -n "${API_INPUT_COST_PER_1M}" ]; then
  export MAS_ORCHESTRA_API_INPUT_COST_PER_1M="${API_INPUT_COST_PER_1M}"
fi
if [ -n "${API_OUTPUT_COST_PER_1M}" ]; then
  export MAS_ORCHESTRA_API_OUTPUT_COST_PER_1M="${API_OUTPUT_COST_PER_1M}"
fi

echo "Step logs: ${STEP_LOG_ROOT}/${RUN_TIMESTAMP}"
echo "API cost budget: ${API_COST_BUDGET}"
echo "Sub-model pool: ${SUB_MODELS[*]}"

action=(
  "${PYTHON_BIN}" -m examples.mas_orchestra.train_mas_orchestra
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
