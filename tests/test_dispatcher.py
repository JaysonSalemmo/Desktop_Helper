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


class AfterCallModel:
    """Emits the given call token first, then fixed logits forever after."""

    def __init__(self, call_id: int, vocab_size: int, later: dict[int, float],
                 context_len: int = 1024):
        self.call_id = call_id
        self.vocab_size = vocab_size
        self.later = later  # token id → logit; everything else 0
        self.first = True
        self.config = SimpleNamespace(context_len=context_len)

    def eval(self):
        return self

    def __call__(self, idx):
        logits = torch.zeros((1, idx.shape[1], self.vocab_size))
        if self.first:
            logits[0, -1, :] = -10.0
            logits[0, -1, self.call_id] = 10.0
            self.first = False
        else:
            for tok_id, value in self.later.items():
                logits[0, -1, tok_id] = value
        return logits, None


def test_reply_after_tool_is_deterministic():
    # near-tied logits would flip under temperature sampling; post-tool decoding
    # must be greedy, so every run picks the same (higher) token
    tok = _tokenizer()
    a, b = tok.encode("alpha")[0], tok.encode("beta")[0]

    responses = set()
    for _ in range(5):
        model = AfterCallModel(tok.tool_token_id("weather"), tok.vocab_size,
                               later={a: 0.6, b: 0.5})
        d = ToolDispatcher(model, tok, {"weather": lambda m: "72°F, sunny"},
                           device=torch.device("cpu"), max_new_tokens=5, copy_boost=0.0)
        responses.add(d.respond("weather?").response)
    assert len(responses) == 1


def test_copy_bias_pulls_reply_tokens_from_result():
    # flat logits after the call: without the copy boost any token could win;
    # with it, generation can only emit tokens from the injected result
    tok = _tokenizer()
    result_text = "Bohemian Rhapsody by Queen"
    model = AfterCallModel(tok.tool_token_id("spotify"), tok.vocab_size, later={})
    d = ToolDispatcher(model, tok, {"spotify": lambda m: result_text},
                       device=torch.device("cpu"), max_new_tokens=6)

    response = d.respond("what's playing?").response
    # argmax over flat+boost picks the lowest boosted id each step; the
    # repetition penalty then knocks it below the rest, so generation walks
    # the boosted ids in ascending order
    expected_ids = sorted(set(tok.encode(result_text)))[:5]
    assert response == tok.decode(expected_ids, skip_special=True).strip()


def test_verbatim_tool_short_circuits_generation():
    tok = _tokenizer()
    # script only contains the call token — if the dispatcher tried to keep
    # generating, it would emit the call token forever and never terminate cleanly
    model = ScriptedModel([tok.tool_token_id("calendar")], tok.vocab_size)
    d = ToolDispatcher(
        model, tok, {"calendar": lambda m: "FIFA World Cup 3rd Place Match at 9am"},
        device=torch.device("cpu"), top_k=1,
        verbatim={"calendar": lambda r: f"Today: {r}."},
    )

    result = d.respond("What's on my calendar?")

    assert result.response == "Today: FIFA World Cup 3rd Place Match at 9am."
    assert result.tool == "calendar"
    assert model.i == 1  # exactly one forward pass — the routing token


def test_non_verbatim_tool_still_generates():
    tok = _tokenizer()
    word = tok.encode("okay")[0]
    model = ScriptedModel([tok.tool_token_id("weather"), word, tok.eos_id], tok.vocab_size)
    d = ToolDispatcher(model, tok, {"weather": lambda m: "72°F, sunny"},
                       device=torch.device("cpu"), top_k=1,
                       verbatim={"calendar": lambda r: r})

    result = d.respond("weather?")
    assert result.tool == "weather"
    assert model.i > 1  # generation continued past the tool call


def test_fallback_router_rescues_unrouted_messages():
    tok = _tokenizer()
    word = tok.encode("hello")[0]
    model = ScriptedModel([word, tok.eos_id], tok.vocab_size)  # no tool call
    d = ToolDispatcher(
        model, tok, {"spotify": lambda m: "Now playing: Passionfruit by Drake"},
        device=torch.device("cpu"), top_k=1,
        verbatim={"spotify": lambda r: r},
        fallback_router=lambda m: "spotify" if "play" in m.lower() else None,
    )

    result = d.respond("Play Passionfruit by Drake on Spotify")
    assert result.tool == "spotify"
    assert result.response == "Now playing: Passionfruit by Drake"

    # non-play chat stays plain text
    model2 = ScriptedModel([word, tok.eos_id], tok.vocab_size)
    d.model = model2.eval()
    result = d.respond("hello there")
    assert result.tool is None


def test_unregistered_tool_falls_back_gracefully():
    tok = _tokenizer()
    script = [tok.tool_token_id("weather"), tok.eos_id]
    model = ScriptedModel(script, tok.vocab_size)

    result = _dispatcher(model, tok, {}).respond("what's the weather?")

    assert result.tool == "weather"
    assert "not available" in result.tool_result
