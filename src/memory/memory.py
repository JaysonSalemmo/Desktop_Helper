"""
Persistent conversation memory — ChromaDB, fully local.

Every completed exchange is recorded; questions about the past ("what song
did you play earlier?") are answered by semantic search over the store.

Deliberately retrieval-only in v1: the model was trained on single-turn
exchanges, so injecting history into its prompt would push every message
out-of-distribution and destabilise routing. Memory answers come straight
from the store via templates instead.

Embeddings run locally (Chroma's default MiniLM, ~80MB one-time download).
The store lives in data/memory/ (gitignored).
"""
import time
import uuid


class ChromaMemory:
    def __init__(self, path: str, embedding_function=None):
        import chromadb  # slow import — deferred so app startup stays fast
        self._client = chromadb.PersistentClient(path=path)
        kwargs = {}
        if embedding_function is not None:  # tests inject a trivial one
            kwargs["embedding_function"] = embedding_function
        self._col = self._client.get_or_create_collection("exchanges", **kwargs)

    def record(self, message: str, response: str, tool: str | None = None) -> None:
        self._col.add(
            ids=[str(uuid.uuid4())],
            documents=[f"You: {message}\nAssistant: {response}"],
            metadatas=[{"ts": time.time(), "message": message,
                        "response": response, "tool": tool or ""}],
        )

    def search(self, query: str, n: int = 3) -> list[dict]:
        """Most relevant past exchanges, in relevance order."""
        if self._col.count() == 0:
            return []
        result = self._col.query(query_texts=[query],
                                 n_results=min(n, self._col.count()))
        return result["metadatas"][0]

    def recent(self, n: int = 5) -> list[dict]:
        """The n most recent exchanges."""
        got = self._col.get(include=["metadatas"])
        metas = sorted(got["metadatas"], key=lambda m: m["ts"], reverse=True)
        return metas[:n]
