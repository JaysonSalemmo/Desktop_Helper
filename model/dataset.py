import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from model import chat_format
from model.tokenizer import DesktopHelperTokenizer


def result_span_mask(ids: list[int], result_start_id: int, result_end_id: int) -> list[bool]:
    """Target-position mask (True = exclude from loss) covering every
    [RESULT]...[/RESULT] block, delimiters included.

    At inference the dispatcher injects the real result — the model never
    generates those tokens, so training loss on them only teaches it to
    *invent* results (the OPT-era unfaithfulness bug). Loss stays ON for the
    [CALL: tool] token (routing) and the reply after [/RESULT].

    Targets are shifted: y[k] predicts ids[k+1], so token ids[j] is masked at
    index j-1.
    """
    mask = [False] * (len(ids) - 1)
    inside = False
    for j, tok in enumerate(ids):
        if tok == result_start_id:
            inside = True
        if inside and j >= 1:
            mask[j - 1] = True
        if tok == result_end_id:
            inside = False
    return mask


class InstructDataset(Dataset):
    def __init__(
        self,
        tokenizer: DesktopHelperTokenizer,
        tool_calls_path: str | Path,
        context_len: int = 1024,
        seed: int = 42,
    ):
        self._pad_id = tokenizer.pad_id
        self.context_len = context_len

        # load Dolly and normalise into (prompt, response) pairs.
        # keeping general instruction data in the mix matters MORE now than in
        # the OPT era: the base model already converses well, and a diet of
        # pure synthetic tool-calls would erode that (catastrophic forgetting).
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        examples = []
        for ex in dolly:
            prompt = ex["instruction"].strip()
            if ex.get("context", "").strip():
                prompt += "\n\n" + ex["context"].strip()
            examples.append({"prompt": prompt, "response": ex["response"].strip()})

        # append synthetic tool-call (and no-tool chat) examples
        with open(tool_calls_path) as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        random.Random(seed).shuffle(examples)

        # tokenize everything upfront in ChatML framing (chat_format is the
        # single source of truth shared with inference priming). each sequence
        # is padded/truncated to context_len + 1 for custom-fn-free collation.
        #
        # NOTE on padding: SmolLM2 sets pad_id == eos_id (both <|im_end|>), so
        # padding can NOT be masked by comparing token ids — that would mask
        # every legitimate end-of-turn <|im_end|> target and the model would
        # never learn to stop. we track each example's true length and mask by
        # POSITION instead.
        self.sequences: list[torch.Tensor] = []
        self.prompt_lens: list[int] = []
        self.seq_lens: list[int] = []
        self.result_masks: list[torch.Tensor] = []
        target_len = context_len + 1
        skipped = 0
        for ex in examples:
            ids, prompt_len = chat_format.example_ids(
                tokenizer, ex["prompt"], ex["response"]
            )
            seq_len = min(len(ids), target_len)
            # an example whose prompt fills (or overflows) the window has no
            # trainable response targets left after masking — cross_entropy
            # over zero targets is NaN, and at batch_size=1 a single such
            # example poisons every weight in the model (found the hard way:
            # Colab run 2026-07-19, loss went NaN mid-epoch). skip them.
            if prompt_len >= seq_len - 1:
                skipped += 1
                continue
            if len(ids) >= target_len:
                ids = ids[:target_len]
            else:
                ids = ids + [self._pad_id] * (target_len - len(ids))
            self.sequences.append(torch.tensor(ids, dtype=torch.long))
            self.prompt_lens.append(min(prompt_len, target_len))
            self.seq_lens.append(seq_len)
            # mask [RESULT]...[/RESULT] out of the loss — the dispatcher
            # injects real results at inference; training on fabricated ones
            # teaches exactly the unfaithful behaviour we intercept
            self.result_masks.append(torch.tensor(
                result_span_mask(ids, tokenizer.result_start_id, tokenizer.result_end_id),
                dtype=torch.bool,
            ))
        if skipped:
            print(f"InstructDataset: skipped {skipped} examples with no trainable "
                  f"targets (prompt fills the {context_len}-token window)")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:].clone()
        # mask prompt tokens: y is shifted by one, so the first (prompt_len - 1)
        # targets are prompt tokens. the position at prompt_len - 1 is the last
        # prompt token predicting the first response token — kept.
        n_mask = max(0, self.prompt_lens[idx] - 1)
        y[:n_mask] = -100
        y[self.result_masks[idx]] = -100  # never learn to generate result blocks
        # padding masked BY POSITION (see __init__ note): the final real token
        # sits at seq_len-1, its target at y[seq_len-2] — everything after is
        # padding. the true <|im_end|> target at y[seq_len-2] stays in the loss.
        y[self.seq_lens[idx] - 1:] = -100
        return x, y
