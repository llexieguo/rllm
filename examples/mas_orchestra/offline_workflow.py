from __future__ import annotations

from typing import Any

from examples.mas_orchestra.offline_replay import validate_offline_replay_row
from examples.mas_orchestra.offline_reward import compute_effective_step_rewards
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine import RolloutEngine
from rllm.workflows.workflow import TerminationReason, Workflow


class OfflineMasOrchestraReplayWorkflow(Workflow):
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        trajectory_bonus_weight: float = 1.0,
        trajectory_name: str = "orchestra_policy",
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.trajectory_bonus_weight = trajectory_bonus_weight
        self.trajectory_name = trajectory_name

    async def run(self, task: dict[str, Any], uid: str, **kwargs) -> Episode:
        self.reset(task, uid)
        sample = validate_offline_replay_row(task)
        effective_steps, bonus_per_step = compute_effective_step_rewards(
            sample.steps,
            sample.trajectory_reward,
            trajectory_bonus_weight=self.trajectory_bonus_weight,
        )

        trajectory = Trajectory(name=self.trajectory_name, task=task)
        filtered_steps = []
        for step_spec in sample.steps:
            if not step_spec.trainable:
                filtered_steps.append(
                    {
                        "step_type": step_spec.step_type,
                        "model": step_spec.model,
                        "step_reward": step_spec.step_reward,
                        "metadata": step_spec.metadata,
                    }
                )

        for step_spec, effective_reward in effective_steps:
            assistant_message = {"role": "assistant", "content": step_spec.response}
            chat_completions = list(step_spec.messages) + [assistant_message]
            trajectory.steps.append(
                Step(
                    chat_completions=chat_completions,
                    action=Action(action=step_spec.response),
                    model_response=step_spec.response,
                    reward=effective_reward,
                    metadata={
                        "step_type": step_spec.step_type,
                        "model": step_spec.model,
                        "raw_step_reward": step_spec.step_reward,
                        "effective_step_reward": effective_reward,
                        "trainable": True,
                        **step_spec.metadata,
                    },
                )
            )

        if trajectory.steps:
            trajectory.steps[-1].done = True

        episode = Episode(
            task=task,
            trajectories=[trajectory],
            metrics={},
            metadata={},
        )
        episode = self.postprocess_episode(episode, TerminationReason.ENV_DONE)

        if sample.is_correct is not None:
            episode.is_correct = sample.is_correct

        episode.metrics.update(
            {
                "raw_trajectory_reward": sample.trajectory_reward,
                "bonus_per_step": bonus_per_step,
                "num_trainable_steps": len(effective_steps),
            }
        )
        episode.metadata.update(
            {
                "source_task_id": sample.task_id,
                "data_source": sample.data_source,
                "raw_trajectory_reward": sample.trajectory_reward,
                "bonus_per_step": bonus_per_step,
                "num_trainable_steps": len(effective_steps),
                "num_filtered_steps": len(filtered_steps),
                "filtered_non_trainable_steps": filtered_steps,
                "trajectory_bonus_weight": self.trajectory_bonus_weight,
            }
        )
        episode.trajectories[0].metadata = {
            "source_task_id": sample.task_id,
            "raw_trajectory_reward": sample.trajectory_reward,
            "bonus_per_step": bonus_per_step,
            "num_trainable_steps": len(effective_steps),
            "num_filtered_steps": len(filtered_steps),
        }
        return episode
