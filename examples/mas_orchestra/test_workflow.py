from __future__ import annotations

from dataclasses import dataclass

import pytest

from examples.mas_orchestra.subclients import MockSubModelClient
from examples.mas_orchestra.workflow import MasOrchestraWorkflow
from rllm.engine import ModelOutput


def build_task() -> dict:
    return {
        "task_id": "sample-1",
        "question": "Which option is correct?",
        "options": ["A", "B", "C", "D"],
        "answer_index": 0,
        "steps": ["inspect the evidence"],
        "discipline": "physics",
        "images": [],
        "data_source": "sgi_reasoning",
    }


def make_output(text: str, *, prompt_start: int, response_start: int, finish_reason: str | None = "stop") -> ModelOutput:
    return ModelOutput(
        text=text,
        content=text,
        reasoning="mock reasoning",
        prompt_ids=[prompt_start, prompt_start + 1],
        completion_ids=[response_start, response_start + 1],
        logprobs=[-0.1, -0.2],
        prompt_length=2,
        completion_length=2,
        finish_reason=finish_reason,
    )


@dataclass
class FakeTokenizer:
    pad_token_id: int = 0


class FakeRolloutEngine:
    def __init__(self, outputs: list[ModelOutput]):
        self.outputs = list(outputs)
        self.tokenizer = FakeTokenizer()
        self.processor = None
        self.validate = False
        self.wake_calls = 0
        self.sleep_calls = 0

    async def get_model_response(self, messages, **kwargs) -> ModelOutput:
        assert self.outputs, "No more scripted policy outputs"
        return self.outputs.pop(0)

    async def wake_up(self):
        self.wake_calls += 1

    async def sleep(self):
        self.sleep_calls += 1


@pytest.mark.asyncio
async def test_external_delegate_only_tracks_main_policy_steps():
    engine = FakeRolloutEngine(
        [
            make_output(
                '{"action":"delegate_task","reasoning":"need help","model":"remote-sub","instruction":"check carefully"}',
                prompt_start=10,
                response_start=20,
            ),
            make_output(
                '{"action":"submit","reasoning":"enough evidence","submit_reason":"ready"}',
                prompt_start=30,
                response_start=40,
            ),
        ]
    )
    subclient = MockSubModelClient(
        {
            "remote-sub": '{"reasoning":"checked","final_answer":"\\\\boxed{A}","confidence":0.93}'
        }
    )

    workflow = MasOrchestraWorkflow(
        rollout_engine=engine,
        main_model="local-policy",
        sub_models=["remote-sub", "local-policy"],
        sub_model_client=subclient,
    )

    episode = await workflow.run(build_task(), "uid-1")

    assert len(episode.trajectories) == 1
    assert len(episode.trajectories[0].steps) == 2
    assert episode.trajectories[0].reward == 1.0
    assert episode.metrics["main_policy_calls"] == 2
    assert episode.metrics["delegate_policy_calls"] == 0
    assert episode.metrics["external_delegate_calls"] == 1
    assert episode.metrics["mca"] == 1.0
    assert episode.artifacts["boxed_letter"] == "A"
    assert len(subclient.call_log) == 1


@pytest.mark.asyncio
async def test_self_think_delegate_adds_trainable_policy_step():
    engine = FakeRolloutEngine(
        [
            make_output(
                '{"action":"delegate_task","reasoning":"self think","model":"local-policy","instruction":"solve it yourself"}',
                prompt_start=10,
                response_start=20,
            ),
            make_output(
                '{"reasoning":"checked","final_answer":"\\\\boxed{A}","confidence":0.97}',
                prompt_start=30,
                response_start=40,
            ),
            make_output(
                '{"action":"submit","reasoning":"done","submit_reason":"ready"}',
                prompt_start=50,
                response_start=60,
            ),
        ]
    )
    subclient = MockSubModelClient()
    workflow = MasOrchestraWorkflow(
        rollout_engine=engine,
        main_model="local-policy",
        sub_models=["remote-sub", "local-policy"],
        sub_model_client=subclient,
    )

    episode = await workflow.run(build_task(), "uid-2")

    assert len(episode.trajectories[0].steps) == 3
    assert episode.trajectories[0].reward == 1.0
    assert episode.metrics["main_policy_calls"] == 2
    assert episode.metrics["delegate_policy_calls"] == 1
    assert episode.metrics["external_delegate_calls"] == 0
    assert subclient.call_log == []
