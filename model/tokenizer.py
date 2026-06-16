from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# these tokens are always treated as single atomic units regardless of what bpe
# would normally do with the characters inside them. the model learns to emit
# the [CALL: ...] tokens to request data, and the app injects [RESULT]...[/RESULT]
# back into the context before the model continues generating.
SPECIAL_TOKENS: list[str] = [
    "<pad>",             # 0 — padding added to short sequences in a batch
    "<bos>",             # 1 — beginning of a conversation turn
    "<eos>",             # 2 — end of a conversation turn
    "<unk>",             # 3 — unknown (rarely fires with byte-level bpe)
    "[CALL: calendar]",  # model emits this when it needs calendar events
    "[CALL: screen]",    # model emits this to request a screen description
    "[CALL: reminders]", # model emits this to read/write reminders
    "[CALL: notes]",     # model emits this to read/write notes
    "[CALL: spotify]",   # model emits this for spotify control
    "[CALL: launcher]",  # model emits this to open an app
    "[CALL: weather]",   # model emits this to get the current weather
    "[CALL: news]",      # model emits this to fetch headlines
    "[CALL: stocks]",    # model emits this to get stock prices
    "[RESULT]",          # marks the start of tool output injected by the app
    "[/RESULT]",         # marks the end of tool output
]

# maps each tool name to its special token string.
# used by the dispatcher in src/assistant/ to look up the right token.
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
}


class DesktopHelperTokenizer:
    # wraps a byte-level bpe tokenizer with our special tokens baked in.
    #
    # byte-level bpe (same approach as gpt-2) encodes every byte as a unicode
    # character before running bpe merges. this means any utf-8 string can be
    # encoded without unknowns — no character is ever out-of-vocabulary.

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    # --- construction ---

    @classmethod
    def train(cls, files: list[str | Path], vocab_size: int = 32000) -> "DesktopHelperTokenizer":
        # train a brand new tokenizer from scratch on the given text files.
        # each file should be plain utf-8 text, one document per line or free-form.
        # special tokens are added first so they always get the lowest ids (0-14).
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            # ignore tokens that appear fewer than twice — reduces noise from typos
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train([str(f) for f in files], trainer)
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str | Path) -> "DesktopHelperTokenizer":
        # load a tokenizer that was previously trained and saved.
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        # save the tokenizer to a json file — this is the only file needed to reload it.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    # --- encoding / decoding ---

    def encode(self, text: str) -> list[int]:
        # convert a string to a list of token ids.
        return self._tok.encode(text).ids

    def encode_turn(self, text: str) -> list[int]:
        # wrap text in bos/eos tokens and encode — use this for full conversation turns.
        return [self.bos_id] + self.encode(text) + [self.eos_id]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        # convert token ids back to a string.
        # skip_special=True strips <bos>, <eos>, <pad> etc from the output.
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    # --- token id helpers ---

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id("<pad>")

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id("<eos>")

    @property
    def result_start_id(self) -> int:
        # id of [RESULT] — the app injects this before tool output
        return self._tok.token_to_id("[RESULT]")

    @property
    def result_end_id(self) -> int:
        # id of [/RESULT] — the app injects this after tool output
        return self._tok.token_to_id("[/RESULT]")

    def tool_token_id(self, tool: str) -> int:
        # look up the token id for a tool by name, e.g. tool_token_id("calendar").
        # used by the dispatcher to detect which tool the model is calling.
        token = TOOL_TOKENS.get(tool)
        if token is None:
            raise ValueError(f"unknown tool '{tool}'. valid tools: {list(TOOL_TOKENS)}")
        tok_id = self._tok.token_to_id(token)
        if tok_id is None:
            raise RuntimeError(f"token '{token}' not found in vocabulary — was the tokenizer trained?")
        return tok_id

    def is_tool_call(self, token_id: int) -> str | None:
        # returns the tool name if the token id is a [CALL: ...] token, else None.
        # useful in the generation loop to detect when the model wants a tool.
        token = self._tok.id_to_token(token_id)
        for tool, call_token in TOOL_TOKENS.items():
            if token == call_token:
                return tool
        return None
