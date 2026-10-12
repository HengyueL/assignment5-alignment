import torch
from typing import Callable

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:

    assert len(rollout_responses) == len(repeated_ground_truths), "Length of rollouts and groundtruth must be the same."
    n = len(rollout_responses)

    raw_reward_list = []
    format_reward_list = []
    answer_reward_list = []

    for i in range(n):
        reward = reward_fn(rollout_responses[i], repeated_ground_truths[i])
        raw_reward_list.append(reward["reward"])
        format_reward_list.append(reward["format_reward"])
        answer_reward_list.append(reward["answer_reward"])
    

    raw_reward_tensor = torch.tensor(raw_reward_list, dtype=torch.float32)
    raw_reward_mean = raw_reward_tensor.mean().item()

    format_reward_tensor = torch.tensor(format_reward_list, dtype=torch.float32)
    format_reward_mean = format_reward_tensor.mean().item()

    answer_reward_tensor = torch.tensor(answer_reward_list, dtype=torch.float32)
    answer_reward_mean = answer_reward_tensor.mean().item()

    return raw_reward_tensor, {"reward_mean": raw_reward_mean, "format_reward_mean": format_reward_mean}

