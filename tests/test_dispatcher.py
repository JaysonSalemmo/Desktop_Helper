from types import SimpleNamespace

import torch

from model.tokenizer import DesktopHelperTokenizer
from src.assistant.dispatcher import ToolDispatcher


class ScriptedModel:
    """Emits a fixed sequence of token ids, one per forward call, by putting all
    logit mass on the scripted token. Lets us test the dispatch loop deterministically
    without a real checkpoint."""

    def __init__(self, script: list[int], vocab_size: int, context_len: int = 1024):
        self.script = script
        self.i = 0
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(context_len=context_len)

    def eval(self):
        return self

    def __call__(self, idx):
        logits = torch.full((1, idx.shape[1], self.vocab_size), -10.0)
        tok = self.script[min(self.i, len(self.script) - 1)]
        logits[0, -1, tok] = 10.0
        self.i += 1
        return logits, None


def _tokenizer():
    return DesktopHelperTokenizer.load("model/tokenizer.json")


def _dispatcher(model, tok, handlers):
    # top_k=1 → greedy, so the scripted token always wins deterministically
    return ToolDispatcher(model, tok, handlers, device=torch.device("cpu"), top_k=1)


def test_tool_call_intercepted_and_real_result_injected():
    tok = _tokenizer()
    word = tok.encode("okay")[0]
    script = [tok.tool_token_id("spotify"), word, tok.eos_id]
    model = ScriptedModel(script, tok.vocab_size)

    calls = {}

    def fake_spotify(message: str) -> str:
        calls["msg"] = message
        return "Bohemian Rhapsody by Queen"

    result = _dispatcher(model, tok, {"spotify": fake_spotify}).respond("what's playing?")

    assert result.tool == "spotify"
    assert result.tool_result == "Bohemian Rhapsody by Queen"
    assert calls["msg"] == "what's playing?"          # handler got the user message
    assert "Bohemian Rhapsody" not in result.response  # injected result != final reply


def test_no_tool_call_returns_plain_text():
    tok = _tokenizer()
    word = tok.encode("hello")[0]
    model = ScriptedModel([word, tok.eos_id], tok.vocab_size)

    result = _dispatcher(model, tok, {}).respond("say hi")

    assert result.tool is None
    assert result.tool_result is None
    assert len(result.response) > 0


def test_unregistered_tool_falls_back_gracefully():
    tok = _tokenizer()
    script = [tok.tool_token_id("weather"), tok.eos_id]
    model = ScriptedModel(script, tok.vocab_size)

    result = _dispatcher(model, tok, {}).respond("what's the weather?")

    assert result.tool == "weather"
    assert "not available" in result.tool_result
