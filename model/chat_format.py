"""
ChatML framing — the single source of truth for how turns are laid out.

SmolLM2-Instruct's entire instruction-following behavior is conditioned on
this exact format (its chat template):

    <|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>

Every producer and consumer of model input goes through THIS module —
dataset.py (training examples), dispatcher.py (inference priming),
generate.py and eval_faithfulness.py — so the framing can never drift
between training and inference. The dispatcher's first generated token (the
confidence-gated routing decision) lands exactly after "<|im_start|>assistant\n";
one wrong character there is a silent distribution shift.

The skeleton is hardcoded rather than calling tokenizer.apply_chat_template()
in the hot loop; tests/test_chat_format.py asserts the two stay identical, so
upstream template changes are caught cheaply instead of silently.
"""

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# the model was instruction-tuned expecting a system turn to open every
# conversation (its template injects one automatically). we define our own —
# short (every request pays its token cost) and IDENTICAL between training
# and inference, which is all that actually matters.
# ⚠️ Deliberately still says "Desktop Helper" after the 2026-08-02 rename to
# Buddy. The shipped checkpoint was fine-tuned with this exact string, and this
# must stay IDENTICAL between training and inference — renaming it here would
# put every inference out of distribution from training and quietly degrade
# routing. It changes only as part of a retrain, where the generated data
# carries the new name too.
SYSTEM = ("You are Desktop Helper, a personal assistant on the user's Mac. "
          "You can call tools to get live data.")


def user_turn(message: str) -> str:
    """System + user turns, closed and ready for the assistant."""
    return (f"{IM_START}system\n{SYSTEM}{IM_END}\n"
            f"{IM_START}user\n{message}{IM_END}\n{IM_START}assistant\n")


def assistant_turn(content: str) -> str:
    """The assistant's half: content closed with <|im_end|> (the eos token)."""
    return f"{content}{IM_END}"


def prime_ids(tokenizer, message: str) -> list[int]:
    """Token ids priming the model to respond to `message` — used by the
    dispatcher/generate/eval. Generation starts right after 'assistant\\n'."""
    return tokenizer.encode(user_turn(message))


def example_ids(tokenizer, prompt: str, response: str) -> tuple[list[int], int]:
    """(full token ids, prompt_len) for one training example.

    prompt_len covers everything through '<|im_start|>assistant\\n' — the
    dataset masks those positions from the loss so the model only learns to
    produce responses, exactly mirroring what prime_ids feeds at inference.
    """
    prompt_ids = tokenizer.encode(user_turn(prompt))
    response_ids = tokenizer.encode(assistant_turn(response))
    return prompt_ids + response_ids, len(prompt_ids)
