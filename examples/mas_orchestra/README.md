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
conda run -n orchestra python examples/mas_orchestra/prepare_dataset.py
```

Useful overrides:

```bash
conda run -n orchestra python examples/mas_orchestra/prepare_dataset.py \
  --dataset-name sgi_reasoning_mas_orchestra_small \
  --train-limit 32 \
  --val-limit 16
```

This registers `train` and `test` splits in `DatasetRegistry` and also creates the companion VeRL parquet files.

## Run tests

```bash
./examples/mas_orchestra/test.sh
```

The suite is fully mock-based. If `torch` or `verl` are not installed in the `orchestra` environment, the VeRL rollout smoke test is skipped and the pure workflow tests still run.

## Train with VeRL

The default dataset name is `sgi_reasoning_mas_orchestra`. Prepare that dataset first, then launch training:

```bash
conda run -n orchestra python examples/mas_orchestra/train_mas_orchestra.py
```

Example with a few common Hydra overrides:

```bash
conda run -n orchestra python examples/mas_orchestra/train_mas_orchestra.py \
  trainer.total_epochs=1 \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  model.path=<your-local-policy-model>
```

This example always trains the local `main_model` policy. If the workflow delegates to `main_model`, that branch is treated as self-think and is also recorded as trainable policy behavior.

## Key workflow defaults

- `main_model="local-policy"`
- `sub_models=["remote-submodel", "local-policy"]`
- `max_attempts=3`
- `submit_confidence_threshold=0.75`
- `mock_external_submodels=True`

You can override these through `workflow_args` when constructing the trainer in Python.
