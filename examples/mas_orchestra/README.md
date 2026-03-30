# MAS Orchestra

This directory contains two paths:

- Online workflow: `workflow.py`, `train_mas_orchestra.py`, `train.sh`
- Offline replay pipeline: `offline_pipeline/`

For the current local setup, the relevant entry is [offline_pipeline/README.md](offline_pipeline/README.md).

## Preserved Run Scripts

The runnable scripts are intentionally kept in both places:

- New offline entrypoints:
  - `examples/mas_orchestra/offline_pipeline/train_offline.sh`
  - `examples/mas_orchestra/offline_pipeline/prepare_offline_replay_dataset.py`
  - `examples/mas_orchestra/offline_pipeline/convert_tree_samples_to_offline_replay.py`
- Compatibility wrappers kept at the old paths:
  - `examples/mas_orchestra/train_offline.sh`
  - `examples/mas_orchestra/train_offline_mas_orchestra.py`
  - `examples/mas_orchestra/prepare_offline_replay_dataset.py`
  - `examples/mas_orchestra/convert_tree_samples_to_offline_replay.py`

Other files here are support modules used by those two paths:

- `offline_replay.py`: canonical offline replay schema validation
- `offline_workflow.py`: fixed-trajectory replay workflow
- `offline_reward.py`: per-step effective reward construction
- `prepare_dataset.py`: base dataset registration helper for the online path
- `test_*.py`: example tests
