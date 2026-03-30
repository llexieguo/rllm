from __future__ import annotations

import os

import hydra

from examples.mas_orchestra.offline_replay import DEFAULT_OFFLINE_DATASET_NAME
from examples.mas_orchestra.offline_workflow import OfflineMasOrchestraReplayWorkflow
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

DEFAULT_OFFLINE_WORKFLOW_ARGS = {
    "trajectory_bonus_weight": 1.0,
}


def default_workflow_args() -> dict:
    return dict(DEFAULT_OFFLINE_WORKFLOW_ARGS)


def _resolve_dataset_name(dataset_name: str | None = None) -> str:
    return dataset_name or os.environ.get("MAS_ORCHESTRA_OFFLINE_DATASET_NAME", DEFAULT_OFFLINE_DATASET_NAME)


def _resolve_bonus_weight() -> float | None:
    raw = os.environ.get("MAS_ORCHESTRA_TRAJECTORY_BONUS_WEIGHT")
    if not raw:
        return None
    return float(raw)


def _disable_validation(config) -> None:
    if getattr(config, "trainer", None) is not None:
        config.trainer.val_before_train = False
        config.trainer.val_only = False
        config.trainer.test_freq = -1
    if getattr(config, "data", None) is not None and hasattr(config.data, "val_files"):
        config.data.val_files = None


def _set_offline_algorithm_defaults(config) -> None:
    if getattr(config, "algorithm", None) is None:
        return

    current = getattr(config.algorithm, "adv_estimator", None)
    if current is None or str(current).lower() == "gae":
        config.algorithm.adv_estimator = "reinforce"


def _configure_offline_rllm_defaults(config) -> None:
    if getattr(config, "rllm", None) is not None:
        if getattr(config.rllm, "stepwise_advantage", None) is not None:
            config.rllm.stepwise_advantage.enable = True
            config.rllm.stepwise_advantage.mode = "per_step"
        if getattr(config.rllm, "workflow", None) is not None:
            config.rllm.workflow.use_workflow = True

    rollout_cfg = getattr(getattr(config, "actor_rollout_ref", None), "rollout", None)
    if rollout_cfg is not None:
        rollout_cfg.n = 1
        if getattr(rollout_cfg, "val_kwargs", None) is not None:
            rollout_cfg.val_kwargs.n = 1


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

    if getattr(config.data, "train_batch_size", None) is None or config.data.train_batch_size > train_size:
        config.data.train_batch_size = train_size

    actor_cfg = getattr(getattr(config, "actor_rollout_ref", None), "actor", None)
    if actor_cfg is not None:
        current_mini_batch = getattr(actor_cfg, "ppo_mini_batch_size", None)
        if current_mini_batch is None or current_mini_batch > config.data.train_batch_size:
            actor_cfg.ppo_mini_batch_size = config.data.train_batch_size


def build_trainer(
    config,
    *,
    dataset_name: str | None = None,
    workflow_args: dict | None = None,
    trainer_cls=AgentTrainer,
):
    resolved_dataset_name = _resolve_dataset_name(dataset_name)
    train_dataset = DatasetRegistry.load_dataset(resolved_dataset_name, "train")
    if train_dataset is None:
        raise RuntimeError(
            f"Dataset '{resolved_dataset_name}' train split not found. "
            "Run `python -m examples.mas_orchestra.offline_pipeline.prepare_offline_replay_dataset --train-file ...` first."
        )

    resolved_workflow_args = default_workflow_args()
    if workflow_args:
        resolved_workflow_args.update(workflow_args)

    bonus_weight = _resolve_bonus_weight()
    if bonus_weight is not None:
        resolved_workflow_args["trajectory_bonus_weight"] = bonus_weight

    _disable_validation(config)
    _set_offline_algorithm_defaults(config)
    _configure_offline_rllm_defaults(config)
    _expand_model_paths(config)
    _tune_small_dataset_defaults(config, train_dataset)

    # VeRL still constructs a validation dataloader even when validation is disabled.
    # Reuse the train split as a placeholder so val_files is never left as None.
    placeholder_val_dataset = train_dataset

    return trainer_cls(
        workflow_class=OfflineMasOrchestraReplayWorkflow,
        workflow_args=resolved_workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=placeholder_val_dataset,
        backend="verl",
    )


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
