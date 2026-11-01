import torch
from typing import Literal


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    if loss_normalization == "constant":
        assert normalization_constant is not None, "Constant cannot be none for constant normalization"

    masked_loss = mask.detach() * per_token_policy_gradient_loss
    per_seq_count = mask.sum(dim=1, keepdim=False).detach()

    if loss_normalization == "sequence":
        masked_loss = (masked_loss.sum(dim=1, keepdim=False) / per_seq_count).mean(dim=0, keepdim=False)
    elif loss_normalization == "constant":
        masked_loss = masked_loss.sum() / normalization_constant
    else:
        raise NotImplementedError

    return masked_loss

