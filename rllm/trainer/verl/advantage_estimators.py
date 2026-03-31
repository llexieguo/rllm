from __future__ import annotations

import torch
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("reinforce")
def compute_reinforce_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute baseline-free REINFORCE advantages for outcome-style rewards.

    rLLM workflows encode scalar step/trajectory rewards on the final response token.
    For policy-gradient training we broadcast that scalar reward to every valid
    response token so the whole sampled action sequence receives the same credit.
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1, keepdim=True)
        returns = scores * response_mask
        advantages = returns
    return advantages, returns
