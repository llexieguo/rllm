from __future__ import annotations

from examples.mas_orchestra.schema import OfflineReplayStepSpec


def compute_effective_step_rewards(
    steps: list[OfflineReplayStepSpec],
    trajectory_reward: float,
    trajectory_bonus_weight: float = 1.0,
) -> tuple[list[tuple[OfflineReplayStepSpec, float]], float]:
    trainable_steps = [step for step in steps if step.trainable]
    if not trainable_steps:
        raise ValueError("Offline replay samples must contain at least one trainable step")

    bonus_per_step = (trajectory_bonus_weight * trajectory_reward) / len(trainable_steps)
    effective_rewards = [(step, step.step_reward + bonus_per_step) for step in trainable_steps]
    return effective_rewards, bonus_per_step
