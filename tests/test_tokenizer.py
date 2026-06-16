import os
import tempfile

import pytest

from model.tokenizer import DesktopHelperTokenizer, SPECIAL_TOKENS, TOOL_TOKENS


@pytest.fixture(scope="module")
def tok() -> DesktopHelperTokenizer:
    # train once and reuse across tests in this file — bpe training is the
    # slow part, and none of these tests mutate the tokenizer.
    corpus = (
        "the user asked what was on their calendar today. "
        "open spotify and play some music please. "
        "what is the weather like in new york today. "
        "add a reminder to call the doctor tomorrow. "
        "take a screenshot and describe what is on screen. "
    ) * 50
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(corpus)
        path = f.name
    try:
        return DesktopHelperTokenizer.train([path], vocab_size=300)
    finally:
        os.unlink(path)


def test_special_tokens_occupy_first_ids(tok: DesktopHelperTokenizer):
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        assert tok._tok.token_to_id(token) == expected_id


def test_encode_decode_roundtrip(tok: DesktopHelperTokenizer):
    text = "what is on my calendar today?"
    assert tok.decode(tok.encode(text)) == text


def test_encode_turn_wraps_bos_eos(tok: DesktopHelperTokenizer):
    ids = tok.encode_turn("open spotify")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_every_tool_token_resolves_and_round_trips(tok: DesktopHelperTokenizer):
    for tool in TOOL_TOKENS:
        tool_id = tok.tool_token_id(tool)
        assert tok.is_tool_call(tool_id) == tool


def test_is_tool_call_returns_none_for_non_tool_token(tok: DesktopHelperTokenizer):
    assert tok.is_tool_call(tok.pad_id) is None


def test_unknown_tool_name_raises(tok: DesktopHelperTokenizer):
    with pytest.raises(ValueError):
        tok.tool_token_id("not_a_real_tool")


def test_save_and_load_round_trip(tok: DesktopHelperTokenizer, tmp_path):
    save_path = tmp_path / "tokenizer.json"
    tok.save(save_path)
    loaded = DesktopHelperTokenizer.load(save_path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("what is the weather") == tok.encode("what is the weather")
