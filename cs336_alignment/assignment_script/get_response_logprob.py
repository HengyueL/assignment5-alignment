from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy of next-token distribution (over vocab dim).

    Args:
        logits: (batch_size, sequence_length, vocab_size)

    Returns:
        entropies: (batch_size, sequence_length)
    """
    # if logits.ndim != 3:
    #     raise ValueError(f"logits must have shape (B, T, V), got {tuple(logits.shape)}")
    if logits.ndim < 1:
        raise ValueError(f"logits must have at least 1 dim, got {tuple(logits.shape)}")

    # log_probs = logits - logsumexp(logits)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)   # (B, T, 1)
    log_probs = logits - log_z                              # (B, T, V)
    probs = torch.exp(log_probs)                            # (B, T, V)

    # H(p) = - sum_v p(v) * log p(v)
    entropy = -(probs * log_probs).sum(dim=-1)              # (B, T)
    return entropy



def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-token conditional log-probabilities for a causal LM.

    Args:
        model: HF causal LM, already on correct device.
        input_ids: (B, T)
        labels:    (B, T) token ids
        return_token_entropy: if True, also return per-token entropy (B, T) computed from logits.

    Returns:
        dict with:
          - "log_probs": (B, T)
          - "token_entropy": (B, T) if requested
    """
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            f"input_ids and labels must be (B, T). Got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have same shape. Got {tuple(input_ids.shape)} vs {tuple(labels.shape)}"
        )

    # forward
    logits = model(input_ids=input_ids).logits  # (B, T, V)

    # stable log-probs over vocab
    log_probs_vocab = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # gather log p(label_t | prefix) for each position
    # labels: (B, T) -> (B, T, 1) for gather
    gathered = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1))  # (B, T, 1)
    token_log_probs = gathered.squeeze(-1)  # (B, T)

    out: Dict[str, torch.Tensor] = {"log_probs": token_log_probs}

    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)  # (B, T)

    return out