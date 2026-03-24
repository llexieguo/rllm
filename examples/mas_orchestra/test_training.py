from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from omegaconf import OmegaConf

from examples.mas_orchestra.test_workflow import FakeRolloutEngine, build_task, make_output
from examples.mas_orchestra.train_mas_orchestra import build_trainer
from examples.mas_orchestra.workflow import MasOrchestraWorkflow
from rllm.data import Dataset


class FakeAgentTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_trainer_constructs_verl_workflow_trainer(monkeypatch):
    fake_train_dataset = Dataset([{"task_id": "t1"}], name="mock", split="train")
    fake_val_dataset = Dataset([{"task_id": "t2"}], name="mock", split="test")

    def fake_load_dataset(name: str, split: str):
        return fake_train_dataset if split == "train" else fake_val_dataset

    monkeypatch.setattr("examples.mas_orchestra.train_mas_orchestra.DatasetRegistry.load_dataset", fake_load_dataset)

    trainer = build_trainer(
        OmegaConf.create({"trainer": {"project_name": "demo"}}),
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
                "remote-sub": '{"reasoning":"checked","final_answer":"\\\\boxed{A}","confidence":0.94}'
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
    assert result.tensors["responses"].shape[0] == 2
