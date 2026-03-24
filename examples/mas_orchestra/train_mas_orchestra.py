from __future__ import annotations

import os

import hydra

from examples.mas_orchestra.workflow import DEFAULT_DATASET_NAME, DEFAULT_WORKFLOW_ARGS, MasOrchestraWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


def default_workflow_args() -> dict:
    return dict(DEFAULT_WORKFLOW_ARGS)


def _disable_validation(config, train_dataset=None) -> None:
    if getattr(config, "trainer", None) is not None:
        config.trainer.val_before_train = False
        config.trainer.val_only = False
        config.trainer.test_freq = -1
    if getattr(config, "data", None) is not None and hasattr(config.data, "val_files"):
        fallback_val = train_dataset.get_verl_data_path() if train_dataset is not None else None
        config.data.val_files = fallback_val


def _set_example_algorithm_defaults(config) -> None:
    if getattr(config, "algorithm", None) is None:
        return

    current = getattr(config.algorithm, "adv_estimator", None)
    if current is None or str(current).lower() == "gae":
        config.algorithm.adv_estimator = "grpo"


def _expand_model_paths(config) -> None:
    actor_model = getattr(getattr(config, "actor_rollout_ref", None), "model", None)
    if actor_model is not None and getattr(actor_model, "path", None):
        actor_model.path = os.path.expanduser(actor_model.path)

    critic_model = getattr(getattr(config, "critic", None), "model", None)
    if critic_model is not None and getattr(critic_model, "path", None):
        critic_model.path = os.path.expanduser(critic_model.path)



def _tune_small_dataset_defaults(config, train_dataset) -> None:
    if getattr(config, "data", None) is None or train_dataset is None:
        return

    train_size = len(train_dataset)
    if train_size <= 0:
        return

    # The upstream PPO config defaults to train_batch_size=1024, which makes
    # small example datasets produce an empty dataloader with drop_last=True.
    if getattr(config.data, "train_batch_size", None) is None or config.data.train_batch_size > train_size:
        config.data.train_batch_size = train_size

    actor_cfg = getattr(getattr(config, "actor_rollout_ref", None), "actor", None)
    if actor_cfg is not None:
        current_mini_batch = getattr(actor_cfg, "ppo_mini_batch_size", None)
        if current_mini_batch is None or current_mini_batch > config.data.train_batch_size:
            actor_cfg.ppo_mini_batch_size = config.data.train_batch_size

def build_trainer(config, *, dataset_name: str = DEFAULT_DATASET_NAME, workflow_args: dict | None = None, trainer_cls=AgentTrainer):
    train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
    val_dataset = DatasetRegistry.load_dataset(dataset_name, "val")
    if train_dataset is None:
        raise RuntimeError(
            f"Dataset '{dataset_name}' train split not found. "
            "Run `python -m examples.mas_orchestra.prepare_dataset` first."
        )

    resolved_workflow_args = default_workflow_args()
    if workflow_args:
        resolved_workflow_args.update(workflow_args)
    if getattr(config, "rllm", None) is not None:
        if getattr(config.rllm, "stepwise_advantage", None) is not None:
            config.rllm.stepwise_advantage.enable = True
        if getattr(config.rllm, "workflow", None) is not None:
            config.rllm.workflow.use_workflow = True

    _set_example_algorithm_defaults(config)
    _expand_model_paths(config)
    _tune_small_dataset_defaults(config, train_dataset)

    if val_dataset is None:
        _disable_validation(config, train_dataset=train_dataset)

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
