from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from omegaconf import OmegaConf

from examples.mas_orchestra.offline_workflow import OfflineMasOrchestraReplayWorkflow
from examples.mas_orchestra.test_workflow import FakeRolloutEngine, build_task, make_output
from examples.mas_orchestra.train_offline_mas_orchestra import build_trainer as build_offline_trainer
from examples.mas_orchestra.train_mas_orchestra import build_trainer
from examples.mas_orchestra.workflow import MasOrchestraWorkflow
from rllm.data import Dataset


class FakeAgentTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_trainer_constructs_verl_workflow_trainer(monkeypatch):
    fake_train_dataset = Dataset([{"task_id": "t1"}], name="mock", split="train")
    fake_val_dataset = Dataset([{"task_id": "t2"}], name="mock", split="val")

    def fake_load_dataset(name: str, split: str):
        if split == "train":
            return fake_train_dataset
        if split == "val":
            return fake_val_dataset
        return None

    monkeypatch.setattr("examples.mas_orchestra.train_mas_orchestra.DatasetRegistry.load_dataset", fake_load_dataset)

    config = OmegaConf.create({"trainer": {"project_name": "demo", "test_freq": 5}, "data": {"val_files": "dummy"}, "algorithm": {"adv_estimator": "gae"}, "actor_rollout_ref": {"model": {"path": "~/actor-model"}}, "critic": {"model": {"path": "~/critic-model"}}})
    trainer = build_trainer(
        config,
        dataset_name="mock-dataset",
        workflow_args={"main_model": "trained-policy"},
        trainer_cls=FakeAgentTrainer,
    )

    assert trainer.kwargs["backend"] == "verl"
    assert trainer.kwargs["workflow_class"] is MasOrchestraWorkflow
    assert trainer.kwargs["train_dataset"] is fake_train_dataset
    assert trainer.kwargs["val_dataset"] is fake_val_dataset
    assert trainer.kwargs["workflow_args"]["main_model"] == "trained-policy"
    assert trainer.kwargs["workflow_args"]["mock_external_submodels"] is True
    assert config.algorithm.adv_estimator == "grpo"
    assert config.actor_rollout_ref.model.path.endswith("/actor-model")
    assert config.critic.model.path.endswith("/critic-model")
    assert config.trainer.test_freq == 5


def test_build_trainer_tunes_batch_sizes_for_small_datasets(monkeypatch):
    fake_train_dataset = Dataset([{"task_id": f"t{i}"} for i in range(10)], name="mock", split="train")

    def fake_load_dataset(name: str, split: str):
        if split == "train":
            return fake_train_dataset
        return None

    monkeypatch.setattr("examples.mas_orchestra.train_mas_orchestra.DatasetRegistry.load_dataset", fake_load_dataset)

    config = OmegaConf.create(
        {
            "trainer": {"project_name": "demo", "val_before_train": False, "val_only": False, "test_freq": -1},
            "data": {"train_batch_size": 1024, "val_files": None},
            "algorithm": {"adv_estimator": "gae"},
            "actor_rollout_ref": {"actor": {"ppo_mini_batch_size": 256}, "model": {"path": "~/actor-model"}},
            "critic": {"model": {"path": "~/critic-model"}},
        }
    )
    trainer = build_trainer(
        config,
        dataset_name="mock-dataset",
        trainer_cls=FakeAgentTrainer,
    )

    assert trainer.kwargs["train_dataset"] is fake_train_dataset
    assert config.data.train_batch_size == 10
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 10



def test_build_trainer_preserves_explicit_adv_estimator_override(monkeypatch):
    fake_train_dataset = Dataset([{"task_id": "t1"}], name="mock", split="train")

    def fake_load_dataset(name: str, split: str):
        if split == "train":
            return fake_train_dataset
        return None

    monkeypatch.setattr("examples.mas_orchestra.train_mas_orchestra.DatasetRegistry.load_dataset", fake_load_dataset)

    config = OmegaConf.create(
        {
            "trainer": {"project_name": "demo", "val_before_train": False, "val_only": False, "test_freq": -1},
            "data": {"val_files": None},
            "algorithm": {"adv_estimator": "reinforce"},
        }
    )
    trainer = build_trainer(
        config,
        dataset_name="mock-dataset",
        trainer_cls=FakeAgentTrainer,
    )

    assert trainer.kwargs["train_dataset"] is fake_train_dataset
    assert config.algorithm.adv_estimator == "reinforce"


def test_build_trainer_disables_validation_when_val_split_is_missing(monkeypatch):
    fake_train_dataset = Dataset([{"task_id": "t1"}], name="mock", split="train")

    def fake_load_dataset(name: str, split: str):
        if split == "train":
            return fake_train_dataset
        return None

    monkeypatch.setattr("examples.mas_orchestra.train_mas_orchestra.DatasetRegistry.load_dataset", fake_load_dataset)

    config = OmegaConf.create(
        {
            "trainer": {"project_name": "demo", "val_before_train": True, "val_only": True, "test_freq": 5},
            "data": {"val_files": "dummy"},
        }
    )
    trainer = build_trainer(
        config,
        dataset_name="mock-dataset",
        trainer_cls=FakeAgentTrainer,
    )

    assert trainer.kwargs["train_dataset"] is fake_train_dataset
    assert trainer.kwargs["val_dataset"] is None
    assert config.trainer.val_before_train is False
    assert config.trainer.val_only is False
    assert config.trainer.test_freq == -1
    assert config.data.val_files == fake_train_dataset.get_verl_data_path()


def test_build_offline_trainer_configures_reinforce_replay_defaults(monkeypatch):
    fake_train_dataset = Dataset([{"task_id": "t1"}], name="mock", split="train")
    fake_val_dataset = Dataset([{"task_id": "t2"}], name="mock", split="val")

    def fake_load_dataset(name: str, split: str):
        if split == "train":
            return fake_train_dataset
        if split == "val":
            return fake_val_dataset
        return None

    monkeypatch.setattr("examples.mas_orchestra.train_offline_mas_orchestra.DatasetRegistry.load_dataset", fake_load_dataset)

    config = OmegaConf.create(
        {
            "trainer": {"project_name": "demo", "val_before_train": True, "val_only": True, "test_freq": 5},
            "data": {"train_batch_size": 1024, "val_files": "dummy", "custom_cls": {"path": None, "name": None}},
            "algorithm": {"adv_estimator": "gae"},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 256},
                "rollout": {"n": 8, "val_kwargs": {"n": 4}},
                "model": {"path": "~/actor-model"},
            },
            "critic": {"model": {"path": "~/critic-model"}},
            "rllm": {
                "workflow": {"use_workflow": False},
                "stepwise_advantage": {"enable": False, "mode": "broadcast"},
            },
        }
    )
    trainer = build_offline_trainer(
        config,
        dataset_name="mock-dataset",
        workflow_args={"trajectory_bonus_weight": 0.25},
        trainer_cls=FakeAgentTrainer,
    )

    assert trainer.kwargs["backend"] == "verl"
    assert trainer.kwargs["workflow_class"] is OfflineMasOrchestraReplayWorkflow
    assert trainer.kwargs["train_dataset"] is fake_train_dataset
    assert trainer.kwargs["val_dataset"] is None
    assert trainer.kwargs["workflow_args"]["trajectory_bonus_weight"] == 0.25
    assert config.algorithm.adv_estimator == "reinforce"
    assert config.actor_rollout_ref.model.path.endswith("/actor-model")
    assert config.critic.model.path.endswith("/critic-model")
    assert config.rllm.workflow.use_workflow is True
    assert config.rllm.stepwise_advantage.enable is True
    assert config.rllm.stepwise_advantage.mode == "per_step"
    assert config.actor_rollout_ref.rollout.n == 1
    assert config.actor_rollout_ref.rollout.val_kwargs.n == 1
    assert config.data.custom_cls.path == "pkg://rllm.trainer.verl.local_parquet_rl_dataset"
    assert config.data.custom_cls.name == "LocalParquetRLHFDataset"
    assert config.trainer.val_before_train is False
    assert config.trainer.val_only is False
    assert config.trainer.test_freq == -1
    assert config.data.val_files is None
    assert config.data.train_batch_size == 1
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 1


@dataclass
class FakeTokenizer:
    pad_token_id: int = 0


class FakeRolloutEngineForBatch(FakeRolloutEngine):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.tokenizer = FakeTokenizer()


@pytest.mark.asyncio
async def test_execute_tasks_verl_builds_trainable_batch():
    torch = pytest.importorskip("torch")
    DataProto = pytest.importorskip("verl").DataProto
    from rllm.engine.agent_workflow_engine import AgentWorkflowEngine

    config = OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 16,
                "max_response_length": 16,
            },
            "rllm": {
                "workflow": {"n_parallel_tasks": 1, "retry_limit": 1},
                "stepwise_advantage": {"enable": True},
                "compact_filtering": {
                    "enable": False,
                    "mask_max_prompt_length_exceeded": True,
                    "mask_max_response_length_exceeded": True,
                    "mask_env_done": False,
                    "mask_max_turns_exceeded": True,
                    "mask_timeout": True,
                    "mask_unknown": False,
                    "mask_error": True,
                },
            },
        }
    )

    rollout_engine = FakeRolloutEngineForBatch(
        [
            make_output(
                '{"action":"delegate_task","reasoning":"need help","model":"remote-sub","instruction":"check carefully"}',
                prompt_start=10,
                response_start=20,
            ),
            make_output(
                '{"action":"submit","reasoning":"done","submit_reason":"ready"}',
                prompt_start=30,
                response_start=40,
            ),
        ]
    )
    engine = AgentWorkflowEngine(
        workflow_cls=MasOrchestraWorkflow,
        workflow_args={
            "main_model": "local-policy",
            "sub_models": ["remote-sub", "local-policy"],
            "mock_delegate_responses": {
                "remote-sub": '{"reasoning":"checked","final_answer":"\\boxed{A}","confidence":0.94}'
            },
        },
        rollout_engine=rollout_engine,
        config=config,
        n_parallel_tasks=1,
        retry_limit=1,
    )

    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        }
    )
    batch.non_tensor_batch["extra_info"] = np.array([build_task()], dtype=object)
    batch.non_tensor_batch["task_ids"] = np.array(["task-1"], dtype=object)
    batch.meta_info["validate"] = False

    result = await engine.execute_tasks_verl(batch)

    assert rollout_engine.wake_calls == 1
    assert rollout_engine.sleep_calls == 1
    assert result.meta_info["repeat_counts"] == [2]
    assert result.non_tensor_batch["is_correct"].tolist() == [True, True]
    assert result.non_tensor_batch["step_nums"].tolist() == [2, 2]
    assert result.batch["responses"].shape[0] == 2
