from typing import Dict, List, Tuple, Any, Callable

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import torch.nn.functional as F


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Tensor]:
    """
    Tokenize prompt and output separately, concatenate them, then build:
      - input_ids: concat[:-1]
      - labels:    concat[1:]
      - response_mask: 1 on label positions corresponding to output tokens, else 0
    Shapes: (batch_size, max_len - 1)
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt_strs and output_strs must have same length, got {len(prompt_strs)} vs {len(output_strs)}"
        )
    
    batch_size = len(prompt_strs)
    if batch_size == 0:
        empty = torch.empty((0, 0), dtype=torch.long)
        return {"input_ids": empty, "labels": empty, "response_mask": empty}
    
    # pad id fallback
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # tokenize separately then concat (no special tokens)
    prompt_ids_list: List[List[int]] = []
    output_ids_list: List[List[int]] = []
    concat_ids_list: List[List[int]] = []
    prompt_lens: List[int] = []
    output_lens: List[int] = []

    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        o_ids = tokenizer(o, add_special_tokens=False).input_ids
        prompt_ids_list.append(list(p_ids))
        output_ids_list.append(list(o_ids))
        prompt_lens.append(len(p_ids))
        output_lens.append(len(o_ids))
        concat_ids_list.append(list(p_ids) + list(o_ids))

    max_len = max(len(x) for x in concat_ids_list)  # length of full concat (prompt+output)

    # full concatenated and padded (batch, max_len)
    full = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    # response_mask aligns with labels (= full[:, 1:]) => (batch, max_len - 1)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.long)

    for i, ids in enumerate(concat_ids_list):
        L = len(ids)
        full[i, :L] = torch.tensor(ids, dtype=torch.long)

        P = prompt_lens[i]
        O = output_lens[i]

        # output tokens are full indices: [P, P+O-1]
        # labels are full shifted left => label indices: [P-1, P+O-2] (length O)
        if O > 0:
            start = max(P - 1, 0)
            end = min(start + O, max_len - 1)  # cap to mask length
            response_mask[i, start:end] = 1

    input_ids = full[:, :-1].contiguous()
    labels = full[:, 1:].contiguous()

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

