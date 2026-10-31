import torch
from typing import Literal


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return on-policy losses of shape (batch_size, sequence_length).

    Rewards/advantages are treated as fixed weights. Masking and aggregation
    over response tokens and sequences belong in the training loop.
    """
    if importance_reweighting_method != "none":
        raise NotImplementedError("Only on-policy loss (method='none') is supported")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length)")

    batch_size = policy_log_probs.shape[0]
    if raw_rewards_or_advantages.shape == (batch_size,):
        advantages = raw_rewards_or_advantages.unsqueeze(-1)
    elif raw_rewards_or_advantages.shape == (batch_size, 1):
        advantages = raw_rewards_or_advantages
    else:
        raise ValueError(
            "raw_rewards_or_advantages must have shape (batch_size,) or (batch_size, 1)"
        )

    # Negate the objective so gradient descent increases rewarded log-probabilities.
    per_token_loss = -advantages.detach() * policy_log_probs
    return per_token_loss, {}

