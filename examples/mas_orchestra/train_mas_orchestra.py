from __future__ import annotations

import hydra

from examples.mas_orchestra.workflow import DEFAULT_DATASET_NAME, DEFAULT_WORKFLOW_ARGS, MasOrchestraWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


def default_workflow_args() -> dict:
    return dict(DEFAULT_WORKFLOW_ARGS)


def build_trainer(config, *, dataset_name: str = DEFAULT_DATASET_NAME, workflow_args: dict | None = None, trainer_cls=AgentTrainer):
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = DatasetRegistry.load_dataset(dataset_name, "test")
    if train_dataset is None or val_dataset is None:
        raise RuntimeError(
            f"Dataset '{dataset_name}' not found. "
            "Run `python examples/mas_orchestra/prepare_dataset.py` first."
        )

    resolved_workflow_args = default_workflow_args()
    if workflow_args:
        resolved_workflow_args.update(workflow_args)
    if getattr(config, "rllm", None) is not None:
        if getattr(config.rllm, "stepwise_advantage", None) is not None:
            config.rllm.stepwise_advantage.enable = True
        if getattr(config.rllm, "workflow", None) is not None:
            config.rllm.workflow.use_workflow = True

    return trainer_cls(
        workflow_class=MasOrchestraWorkflow,
        workflow_args=resolved_workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        backend="verl",
    )


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
