#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" -m pytest   -o cache_dir=/tmp/rllm_mas_orchestra_pytest   examples/mas_orchestra/test_workflow.py   examples/mas_orchestra/test_prepare_dataset.py   examples/mas_orchestra/test_prepare_offline_replay_dataset.py   examples/mas_orchestra/test_offline_replay.py   examples/mas_orchestra/test_training.py   -q
