"""
Shared model/dispatcher loader — the one place the assistant "engine" is
assembled. Both frontends (Textual TUI, menu bar app) call `load_engine` on a
worker thread and get back a ready ToolDispatcher.

Paths go through src.paths so the same code resolves correctly from source and
inside a frozen bundle: the tokenizer is a read-only bundled resource, the
checkpoint and memory store are writable per-user data (Application Support
when frozen, the project dir from source).
"""
from pathlib import Path

import torch

from model.device import get_device
from model.generate import load_model
from model.tokenizer import DesktopHelperTokenizer
from src.assistant.dispatcher import ToolDispatcher
from src.assistant.tools import (build_fallback_router, build_handlers,
                                 build_pre_router, build_reprompts,
                                 build_verbatim)
from src.paths import resource_path, user_data_dir, user_data_path

TOKENIZER_PATH = resource_path("model", "hf_tokenizer")


class CheckpointMissing(FileNotFoundError):
    pass


def load_engine(config: dict) -> tuple[ToolDispatcher, torch.device]:
    """Load tokenizer + checkpoint + handlers. Slow (~4GB read) — call off
    the UI thread. Raises CheckpointMissing with a human-readable message."""
    checkpoint = Path(config["model"]["checkpoint"])
    if not checkpoint.is_absolute():
        # relative → under the user-data dir (project root from source, App
        # Support when frozen), so the checkpoint lives outside the read-only .app
        checkpoint = user_data_dir() / checkpoint
    if not checkpoint.exists():
        raise CheckpointMissing(
            f"Model checkpoint not found: {checkpoint}\n"
            f"Place the weights at that path (frozen app: under {user_data_dir()}), "
            "then restart."
        )

    device = get_device(require_cuda=False)
    tokenizer = DesktopHelperTokenizer.load(str(TOKENIZER_PATH))
    model = load_model(str(checkpoint), device)

    memory = None
    if config.get("features", {}).get("memory", True):
        try:
            from src.memory.memory import ChromaMemory
            memory = ChromaMemory(str(user_data_path("data", "memory")))
        except Exception:
            pass  # memory is a convenience — the assistant works without it

    dispatcher = ToolDispatcher(model, tokenizer, build_handlers(config, memory), device,
                                verbatim=build_verbatim(),
                                reprompt=build_reprompts(),
                                pre_router=build_pre_router(config),
                                fallback_router=build_fallback_router(config),
                                memory=memory)
    return dispatcher, device
