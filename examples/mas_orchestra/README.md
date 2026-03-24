# MAS Orchestra Workflow Example

This example ports the MAS Orchestra control loop into a native `rllm` workflow so it can be trained with the built-in VeRL backend.

## What is included

- `workflow.py`: native `Workflow` implementation for the orchestra policy
- `prepare_dataset.py`: converts `InternScience/SGI-Reasoning` into an `rllm` dataset
- `train_mas_orchestra.py`: VeRL training entrypoint using `AgentTrainer`
- `subclients.py`: mocked external sub-model client used by tests and local smoke runs
- `test.sh`: runs the example test suite in the `orchestra` conda environment

External delegate models are mocked in this example by default. No real API calls are made unless you later replace the mock client yourself.

## Prepare the dataset

Run from the repository root:

```bash
python -m examples.mas_orchestra.prepare_dataset
```

This example assumes the source dataset only exposes a `test` split. The script shuffles that split with `seed=42`, reserves `145` examples as the held-out `test` split, and uses the remainder as `train`.

Useful overrides:

```bash
python -m examples.mas_orchestra.prepare_dataset   --dataset-name sgi_reasoning_mas_orchestra_small   --train-limit 32   --test-size 145   --seed 42
```

If you must create a validation split, carve it out of the held-out test pool:

```bash
python -m examples.mas_orchestra.prepare_dataset --val-size 16
```

That registers `train` and `test` in `DatasetRegistry`, and registers `val` only when `--val-size` is greater than zero.

## Run tests

Install the test dependencies first:

```bash
uv pip install -e ".[dev]"
```

Then run:

```bash
./examples/mas_orchestra/test.sh
```

The suite is fully mock-based. If `torch` or `verl` are not installed in the environment, the VeRL rollout smoke test is skipped and the pure workflow tests still run.

## Train with VeRL

The default dataset name is `sgi_reasoning_mas_orchestra`. Prepare that dataset first, then launch training:

```bash
./examples/mas_orchestra/train.sh
```

When no `val` split exists, this example disables validation automatically instead of reusing `test`. The example also defaults to `algorithm.adv_estimator=grpo`.

The training script centralizes the common knobs:

- `CUDA_VISIBLE_DEVICES`: which GPUs to use, default `0,1`
- `N_GPUS_PER_NODE`: how many GPUs Ray should reserve, defaults to the number of visible GPUs
- `TP_SIZE`: tensor parallel size for vLLM rollout, default `1`
- `MODEL_PATH`: local path or Hugging Face repo id, default `Qwen/Qwen3-VL-2B-Instruct`
- `TRAIN_BATCH_SIZE` and `PPO_MINI_BATCH_SIZE`: common batch settings, default `64`

Example overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1 MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct ./examples/mas_orchestra/train.sh
```

```bash
CUDA_VISIBLE_DEVICES=1 N_GPUS_PER_NODE=1 TP_SIZE=1 MODEL_PATH=/abs/path/to/local-model ./examples/mas_orchestra/train.sh
```

For a dry run that only prints the final command:

```bash
DRY_RUN=1 ./examples/mas_orchestra/train.sh
```

This example always trains the local `main_model` policy. If the workflow delegates to `main_model`, that branch is treated as self-think and is also recorded as trainable policy behavior.

## Key workflow defaults

- `main_model="local-policy"`
- `sub_models=["remote-submodel", "local-policy"]`
- `max_attempts=3`
- `submit_confidence_threshold=0.75`
- `mock_external_submodels=True`

You can override these through `workflow_args` when constructing the trainer in Python.
