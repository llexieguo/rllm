#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

conda run -n orchestra python -m pytest \
  -o cache_dir=/tmp/rllm_mas_orchestra_pytest \
  examples/mas_orchestra/test_workflow.py \
  examples/mas_orchestra/test_prepare_dataset.py \
  examples/mas_orchestra/test_training.py \
  -q
