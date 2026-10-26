from typing import Literal
import torch


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages for consecutive groups of responses to each prompt."""
    if raw_rewards.ndim != 1 or raw_rewards.numel() == 0:
        raise ValueError("raw_rewards must be a nonempty one-dimensional tensor")
    if group_size <= 0 or raw_rewards.numel() % group_size != 0:
        raise ValueError("group_size must be positive and divide the reward count")
    if baseline not in ("mean", "none"):
        raise ValueError(f"Unsupported baseline: {baseline}")
    if advantage_normalizer not in ("std", "none", "mean"):
        raise ValueError(f"Unsupported advantage normalizer: {advantage_normalizer}")

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_mean = grouped_rewards.mean(dim=1, keepdim=True)
    advantages = grouped_rewards - group_mean if baseline == "mean" else grouped_rewards

    if advantage_normalizer == "std":
        # Use sample std, with zero variance for singleton groups.
        group_std = grouped_rewards.std(
            dim=1, keepdim=True, correction=1 if group_size > 1 else 0
        )
        advantages = advantages / (group_std + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (group_mean + advantage_eps)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std(correction=0).item(),
        "reward_min": raw_rewards.min().item(),
        "reward_max": raw_rewards.max().item(),
    }
    return advantages.reshape(-1), metadata
