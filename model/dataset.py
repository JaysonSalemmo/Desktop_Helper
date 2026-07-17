import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from model.tokenizer import DesktopHelperTokenizer


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

        # load Dolly and normalise into (prompt, response) pairs
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        examples = []
        for ex in dolly:
            prompt = ex["instruction"].strip()
            if ex.get("context", "").strip():
                prompt += "\n\n" + ex["context"].strip()
            examples.append({"prompt": prompt, "response": ex["response"].strip()})

        # append synthetic tool-call examples
        with open(tool_calls_path) as f:
            for line in f:
                examples.append(json.loads(line.strip()))

        random.Random(seed).shuffle(examples)

        # tokenize everything upfront — each sequence is padded/truncated to
        # exactly context_len + 1 so the DataLoader can collate without a custom fn.
        # we build <bos>prompt\n and response<eos> separately so we can record where
        # the prompt ends — the loss is masked over the prompt tokens (see __getitem__)
        # so the model only learns to generate the response, not to predict the user's
        # input. standard instruction-tuning practice; matters most for a small model
        # whose limited capacity shouldn't be spent modelling prompt phrasings.
        self.sequences: list[torch.Tensor] = []
        self.prompt_lens: list[int] = []
        target_len = context_len + 1
        for ex in examples:
            prompt_ids = [tokenizer.bos_id] + tokenizer.encode(f"{ex['prompt']}\n")
            response_ids = tokenizer.encode(ex["response"]) + [tokenizer.eos_id]
            ids = prompt_ids + response_ids
            if len(ids) >= target_len:
                ids = ids[:target_len]
            else:
                ids = ids + [self._pad_id] * (target_len - len(ids))
            self.sequences.append(torch.tensor(ids, dtype=torch.long))
            self.prompt_lens.append(min(len(prompt_ids), target_len))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:].clone()
        # mask prompt tokens: y is shifted by one, so the first (prompt_len - 1)
        # targets are prompt tokens. the position at prompt_len - 1 is the last
        # prompt token predicting the first response token — kept, so the model
        # still learns to start the response from the prompt.
        n_mask = max(0, self.prompt_lens[idx] - 1)
        y[:n_mask] = -100
        y[y == self._pad_id] = -100  # cross_entropy ignores -100 by default
        return x, y
