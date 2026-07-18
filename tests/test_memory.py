import pytest

from src.memory.memory import ChromaMemory


class TrivialEmbedder:
    """Deterministic char-histogram embedding — keeps tests offline (Chroma's
    default embedder downloads a model on first use)."""

    def __call__(self, input):
        return [self._embed(text) for text in input]

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * 128
        for word in text.lower().split():
            vec[hash(word) % 128] += 1.0
        return vec

    def name(self) -> str:
        return "trivial"

    # chroma 1.5 calls these for query/add paths (input is a list of texts)
    def embed_query(self, input):
        return [self._embed(t) for t in input]

    def embed_documents(self, input):
        return [self._embed(t) for t in input]


@pytest.fixture
def memory(tmp_path):
    return ChromaMemory(str(tmp_path / "mem"), embedding_function=TrivialEmbedder())


def test_record_and_search(memory):
    memory.record("Play a song by Drake", "Now playing: Passionfruit by Drake", "spotify")
    memory.record("What's the weather like?", "It's 73°F and rainy", "weather")

    hits = memory.search("Play a song by Drake")
    assert hits[0]["message"] == "Play a song by Drake"
    assert hits[0]["tool"] == "spotify"


def test_search_empty_store(memory):
    assert memory.search("anything") == []


def test_recent_orders_newest_first(memory):
    memory.record("first", "reply one")
    memory.record("second", "reply two")
    recent = memory.recent(2)
    assert recent[0]["message"] == "second"
    assert recent[1]["message"] == "first"


def test_memory_handler_formats_hits(tmp_path):
    from src.assistant.tools import build_handlers

    mem = ChromaMemory(str(tmp_path / "mem"), embedding_function=TrivialEmbedder())
    mem.record("Play a song by Drake", "Now playing: Passionfruit by Drake", "spotify")
    config = {"allowed_apps": [], "stocks": {"watchlist": []},
              "news": {"rss_feeds": []}, "weather": {"location": "X"}}
    handlers = build_handlers(config, memory=mem)

    reply = handlers["memory"]("What did you play earlier?")
    assert "Play a song by Drake" in reply
    assert "Passionfruit" in reply

    # no memory wired → graceful
    handlers = build_handlers(config, memory=None)
    assert handlers["memory"]("what did we talk about") == "Memory isn't set up"


def test_fallback_router_memory_keywords():
    from src.assistant.tools import build_fallback_router

    fallback = build_fallback_router()
    assert fallback("What did you play earlier?") == "memory"
    assert fallback("Do you remember what I asked?") == "memory"
    # play requests without memory words still go to spotify
    assert fallback("Play Passionfruit by Drake") == "spotify"
