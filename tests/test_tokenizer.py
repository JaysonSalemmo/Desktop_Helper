import pytest

from model.tokenizer import DesktopHelperTokenizer, TOOL_TOKENS


@pytest.fixture(scope="module")
def tok() -> DesktopHelperTokenizer:
    # loads the committed SmolLM2-derived tokenizer (model/hf_tokenizer/)
    return DesktopHelperTokenizer.load()


def test_vocab_layout(tok: DesktopHelperTokenizer):
    # 49152 SmolLM2 tokens + 11 appended tool/protocol tokens
    assert tok.vocab_size == 49163
    assert tok.bos_id == 1   # <|im_start|>
    assert tok.eos_id == 2   # <|im_end|>
    # SmolLM2 quirk: pad IS eos — anything masking padding must go by
    # position, never by id (see dataset.py)
    assert tok.pad_id == tok.eos_id


def test_tool_tokens_appended_after_base_vocab(tok: DesktopHelperTokenizer):
    for tool in TOOL_TOKENS:
        assert tok.tool_token_id(tool) >= 49152


def test_encode_decode_roundtrip(tok: DesktopHelperTokenizer):
    text = "What is on my calendar today?"
    assert tok.decode(tok.encode(text)) == text


def test_encode_adds_no_specials(tok: DesktopHelperTokenizer):
    # double-BOS was flagged as a silent distribution-shift risk — encode()
    # must be raw; call sites add framing via chat_format
    ids = tok.encode("hello")
    assert tok.bos_id not in ids


def test_special_tokens_are_atomic(tok: DesktopHelperTokenizer):
    ids = tok.encode("[CALL: weather][RESULT]72°F, sunny[/RESULT]")
    assert ids[0] == tok.tool_token_id("weather")
    assert ids[1] == tok.result_start_id
    assert ids[-1] == tok.result_end_id


def test_every_tool_token_resolves_and_round_trips(tok: DesktopHelperTokenizer):
    for tool in TOOL_TOKENS:
        assert tok.is_tool_call(tok.tool_token_id(tool)) == tool


def test_is_tool_call_returns_none_for_non_tool_token(tok: DesktopHelperTokenizer):
    assert tok.is_tool_call(tok.eos_id) is None
    assert tok.is_tool_call(0) is None


def test_unknown_tool_name_raises(tok: DesktopHelperTokenizer):
    with pytest.raises(ValueError):
        tok.tool_token_id("not_a_real_tool")


def test_save_and_load_round_trip(tok: DesktopHelperTokenizer, tmp_path):
    tok.save(tmp_path / "tok")
    loaded = DesktopHelperTokenizer.load(tmp_path / "tok")
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("what is the weather") == tok.encode("what is the weather")
    assert loaded.tool_token_id("spotify") == tok.tool_token_id("spotify")
