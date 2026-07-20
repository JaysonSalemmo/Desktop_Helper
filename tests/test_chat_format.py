import pytest

from model import chat_format
from model.tokenizer import DesktopHelperTokenizer


@pytest.fixture(scope="module")
def tok() -> DesktopHelperTokenizer:
    return DesktopHelperTokenizer.load()


def test_skeleton_matches_official_chat_template(tok):
    # our hardcoded ChatML skeleton must stay byte-identical to what the
    # model's own chat template produces — this test is the tripwire for
    # upstream template changes (we deliberately don't call apply_chat_template
    # in the inference hot loop)
    ours = chat_format.user_turn("What's the weather like today?")
    official = tok._tok.apply_chat_template(
        [{"role": "system", "content": chat_format.SYSTEM},
         {"role": "user", "content": "What's the weather like today?"}],
        tokenize=False, add_generation_prompt=True,
    )
    assert ours == official


def test_prime_ids_end_with_assistant_opening(tok):
    # the first generated token (the routing decision) must land right after
    # "<|im_start|>assistant\n" — decode the tail and check
    ids = chat_format.prime_ids(tok, "Hi")
    tail = tok.decode(ids[-3:], skip_special=False)
    assert tail.endswith("assistant\n")


def test_example_ids_prompt_response_split(tok):
    ids, prompt_len = chat_format.example_ids(tok, "What's 2+2?", "4")
    # the prompt span IS the priming — training and inference see the same prefix
    assert ids[:prompt_len] == chat_format.prime_ids(tok, "What's 2+2?")
    # response ends with the eos (<|im_end|>) the model must learn to emit
    assert ids[-1] == tok.eos_id
    response_text = tok.decode(ids[prompt_len:])
    assert response_text == "4"
