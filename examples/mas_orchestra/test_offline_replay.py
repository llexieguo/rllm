from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from omegaconf import OmegaConf

from examples.mas_orchestra.offline_reward import compute_effective_step_rewards
from examples.mas_orchestra.offline_workflow import OfflineMasOrchestraReplayWorkflow
from examples.mas_orchestra.schema import OfflineReplayStepSpec


def build_offline_task() -> dict:
    return {
        "task_id": "offline-1",
        "data_source": "mas_orchestra_offline",
        "is_correct": True,
        "trajectory_reward": 4.0,
        "steps": [
            {
                "messages": [{"role": "user", "content": "First question"}],
                "response": "first answer",
                "step_reward": 0.5,
                "trainable": True,
                "step_type": "main",
                "model": "local-policy",
                "metadata": {"attempt": 1},
            },
            {
                "messages": [{"role": "user", "content": "External question"}],
                "response": "external answer",
                "step_reward": 7.0,
                "trainable": False,
                "step_type": "delegate_external",
                "model": "remote-submodel",
                "metadata": {"attempt": 1},
            },
            {
                "messages": [{"role": "user", "content": "Final question"}],
                "response": "final answer",
                "step_reward": -0.25,
                "trainable": True,
                "step_type": "main",
                "model": "local-policy",
                "metadata": {"attempt": 2},
            },
        ],
    }


def test_compute_effective_step_rewards_only_uses_trainable_steps():
    steps = [
        OfflineReplayStepSpec(messages=[{"role": "user", "content": "q1"}], response="a1", step_reward=1.0, trainable=True),
        OfflineReplayStepSpec(messages=[{"role": "user", "content": "q2"}], response="a2", step_reward=99.0, trainable=False),
        OfflineReplayStepSpec(messages=[{"role": "user", "content": "q3"}], response="a3", step_reward=2.0, trainable=True),
    ]

    effective_steps, bonus_per_step = compute_effective_step_rewards(
        steps,
        trajectory_reward=6.0,
        trajectory_bonus_weight=0.5,
    )

    assert bonus_per_step == 1.5
    assert len(effective_steps) == 2
    assert [reward for _, reward in effective_steps] == [2.5, 3.5]


@dataclass
class DummyRolloutEngine:
    tokenizer: object | None = None


@pytest.mark.asyncio
async def test_offline_replay_workflow_filters_external_steps_and_applies_bonus():
    workflow = OfflineMasOrchestraReplayWorkflow(
        rollout_engine=DummyRolloutEngine(),
        trajectory_bonus_weight=0.5,
    )

    episode = await workflow.run(build_offline_task(), "uid-offline-1")

    assert len(episode.trajectories) == 1
    assert len(episode.trajectories[0].steps) == 2
    assert episode.trajectories[0].steps[0].chat_completions[-1]["content"] == "first answer"
    assert episode.trajectories[0].steps[1].chat_completions[-1]["content"] == "final answer"
    assert episode.trajectories[0].steps[0].reward == 1.5
    assert episode.trajectories[0].steps[1].reward == 0.75
    assert episode.trajectories[0].reward == 2.25
    assert episode.metrics["raw_trajectory_reward"] == 4.0
    assert episode.metrics["bonus_per_step"] == 1.0
    assert episode.metrics["num_trainable_steps"] == 2
    assert episode.is_correct is True
    assert episode.metadata["num_filtered_steps"] == 1


class FakeChatParser:
    def tokenize_and_mask(self, chat_completions):
        torch = pytest.importorskip("torch")

        prompt_tokens = [idx + 1 for idx, _ in enumerate(chat_completions[:-1] or chat_completions)]
        response_tokens = [100 + len(str(chat_completions[-1].get("content", ""))), 200 + len(chat_completions)]
        prompt = torch.tensor(prompt_tokens or [1], dtype=torch.long)
        response = torch.tensor(response_tokens, dtype=torch.long)
        mask = torch.ones_like(response, dtype=torch.long)
        return prompt, response, mask

    def tokenize_and_mask_cumulative(self, chat_completions):
        return self.tokenize_and_mask(chat_completions)


@dataclass
class FakeTokenizer:
    pad_token_id: int = 0


class FakeReplayRolloutEngine:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.chat_parser = FakeChatParser()
        self.processor = None
        self.validate = False
        self.wake_calls = 0
        self.sleep_calls = 0

    async def wake_up(self):
        self.wake_calls += 1

    async def sleep(self):
        self.sleep_calls += 1


@pytest.mark.asyncio
async def test_offline_replay_workflow_builds_per_step_verl_batch():
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
                "stepwise_advantage": {"enable": True, "mode": "per_step"},
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

    rollout_engine = FakeReplayRolloutEngine()
    engine = AgentWorkflowEngine(
        workflow_cls=OfflineMasOrchestraReplayWorkflow,
        workflow_args={"trajectory_bonus_weight": 0.5},
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
    batch.non_tensor_batch["extra_info"] = np.array([build_offline_task()], dtype=object)
    batch.non_tensor_batch["task_ids"] = np.array(["task-1"], dtype=object)
    batch.meta_info["validate"] = False

    result = await engine.execute_tasks_verl(batch)

    assert rollout_engine.wake_calls == 1
    assert rollout_engine.sleep_calls == 1
    assert result.meta_info["repeat_counts"] == [2]
    assert result.non_tensor_batch["is_correct"].tolist() == [True, True]
    assert result.non_tensor_batch["step_nums"].tolist() == [2, 2]
    step_reward_sums = result.batch["step_rewards"].sum(dim=-1).tolist()
    traj_reward_sums = result.batch["traj_rewards"].sum(dim=-1).tolist()
    assert step_reward_sums == [1.5, 0.75]
    assert traj_reward_sums == [2.25, 2.25]
