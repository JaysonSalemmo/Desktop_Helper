"""
Tokenizer for the SmolLM2 era: the base model's own tokenizer plus our tool
tokens, wrapped in the same interface the rest of the codebase already uses.

The OPT-era mistake this fixes: a from-scratch 32K tokenizer meant NO
pretrained embedding could transfer — all 32,000 rows started random, which
caused the entire faithfulness saga (see docs/TIMELINE.md). Now we adopt
SmolLM2's 49,152-token vocabulary as-is and append 11 special tokens, so only
11 embedding rows start fresh.

Vocabulary layout:
    0..49151   — SmolLM2's tokens (ChatML specials included: <|im_start|>=1,
                 <|im_end|>=2; note pad and eos are BOTH id 2 — dataset code
                 must mask padding by position, never by id equality)
    49152..    — our 9 [CALL: tool] tokens + [RESULT] + [/RESULT]

The tokenizer is saved to model/hf_tokenizer/ (committed) so Colab and the
Mac load the identical vocabulary without touching the network.
"""
from pathlib import Path

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LOCAL_TOKENIZER_DIR = Path(__file__).parent / "hf_tokenizer"

# our tool-calling protocol tokens, appended after SmolLM2's vocabulary.
# the model learns to emit [CALL: ...]; the app injects [RESULT]...[/RESULT].
TOOL_TOKENS: dict[str, str] = {
    "calendar":  "[CALL: calendar]",
    "screen":    "[CALL: screen]",
    "reminders": "[CALL: reminders]",
    "notes":     "[CALL: notes]",
    "spotify":   "[CALL: spotify]",
    "launcher":  "[CALL: launcher]",
    "weather":   "[CALL: weather]",
    "news":      "[CALL: news]",
    "stocks":    "[CALL: stocks]",
    "files":     "[CALL: files]",
}

# Order is load-bearing: the fine-tuned checkpoint learned these exact token
# ids, so the original tokens MUST keep their positions. New tools are appended
# AFTER the result markers so they don't shift the old ids — the checkpoint
# resize is then a clean one-row append at the end.
_LEGACY_TOOLS = ["calendar", "screen", "reminders", "notes", "spotify",
                 "launcher", "weather", "news", "stocks"]
SPECIAL_TOKENS: list[str] = (
    [TOOL_TOKENS[t] for t in _LEGACY_TOOLS]
    + ["[RESULT]", "[/RESULT]"]
    + [TOOL_TOKENS[t] for t in TOOL_TOKENS if t not in _LEGACY_TOOLS]
)


class DesktopHelperTokenizer:
    # wraps a HuggingFace tokenizer, preserving the exact interface the
    # dispatcher/dataset/tests were built against in the OPT era.

    def __init__(self, tokenizer):
        self._tok = tokenizer
        # cache tool ids once — is_tool_call runs per generated token
        self._tool_ids = {
            self._tok.convert_tokens_to_ids(tok): name
            for name, tok in TOOL_TOKENS.items()
        }

    # --- construction ---

    @classmethod
    def build(cls) -> "DesktopHelperTokenizer":
        # download SmolLM2's tokenizer and append our special tokens.
        # run once (model/load_base.py does this); afterwards use load().
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path = LOCAL_TOKENIZER_DIR) -> "DesktopHelperTokenizer":
        from transformers import AutoTokenizer
        return cls(AutoTokenizer.from_pretrained(str(path)))

    def save(self, path: str | Path = LOCAL_TOKENIZER_DIR) -> None:
        self._tok.save_pretrained(str(path))

    # --- encoding / decoding ---

    def encode(self, text: str) -> list[int]:
        # add_special_tokens=False is load-bearing: HF tokenizers otherwise
        # auto-prepend specials, and every call site in this codebase adds
        # framing tokens manually — a silent double-BOS would shift the input
        # distribution away from what the pretrained weights expect.
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    # --- token id helpers (interface preserved from the OPT era) ---

    @property
    def vocab_size(self) -> int:
        # len() includes added tokens (49163); .vocab_size on the raw HF
        # tokenizer does NOT (49152) — using it would silently undersize the
        # embedding table and send tool-token ids out of range.
        return len(self._tok)

    @property
    def pad_id(self) -> int:
        # NOTE: SmolLM2 sets pad == eos (both <|im_end|>, id 2). Anything
        # masking padding must do it by position, not by comparing ids.
        return self._tok.pad_token_id

    @property
    def bos_id(self) -> int:
        return self._tok.bos_token_id  # <|im_start|>

    @property
    def eos_id(self) -> int:
        return self._tok.eos_token_id  # <|im_end|>

    @property
    def result_start_id(self) -> int:
        return self._tok.convert_tokens_to_ids("[RESULT]")

    @property
    def result_end_id(self) -> int:
        return self._tok.convert_tokens_to_ids("[/RESULT]")

    def tool_token_id(self, tool: str) -> int:
        token = TOOL_TOKENS.get(tool)
        if token is None:
            raise ValueError(f"unknown tool '{tool}'. valid tools: {list(TOOL_TOKENS)}")
        tok_id = self._tok.convert_tokens_to_ids(token)
        if tok_id is None or tok_id == self._tok.unk_token_id:
            raise RuntimeError(f"token '{token}' not in vocabulary — rebuild the tokenizer")
        return tok_id

    def is_tool_call(self, token_id: int) -> str | None:
        # O(1) via the cached id→name map (runs on every generated token)
        return self._tool_ids.get(token_id)
