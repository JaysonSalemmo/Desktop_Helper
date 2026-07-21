"""
Shared model/dispatcher loader — the one place the assistant "engine" is
assembled. Both frontends (Textual TUI, menu bar app) call `load_engine` on a
worker thread and get back a ready ToolDispatcher.

Paths are resolved relative to the project root (not the cwd) so a future
.app bundle can launch from anywhere.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = PROJECT_ROOT / "model" / "hf_tokenizer"


class CheckpointMissing(FileNotFoundError):
    pass


def load_engine(config: dict) -> tuple[ToolDispatcher, torch.device]:
    """Load tokenizer + checkpoint + handlers. Slow (~4GB read) — call off
    the UI thread. Raises CheckpointMissing with a human-readable message."""
    checkpoint = Path(config["model"]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise CheckpointMissing(
            f"Model checkpoint not found: {checkpoint}\n"
            "Download it from Google Drive into model/checkpoints/ "
            "(weights are gitignored), then restart."
        )

    device = get_device(require_cuda=False)
    tokenizer = DesktopHelperTokenizer.load(str(TOKENIZER_PATH))
    model = load_model(str(checkpoint), device)

    memory = None
    if config.get("features", {}).get("memory", True):
        try:
            from src.memory.memory import ChromaMemory
            memory = ChromaMemory(str(PROJECT_ROOT / "data" / "memory"))
        except Exception:
            pass  # memory is a convenience — the assistant works without it

    dispatcher = ToolDispatcher(model, tokenizer, build_handlers(config, memory), device,
                                verbatim=build_verbatim(),
                                reprompt=build_reprompts(),
                                pre_router=build_pre_router(config),
                                fallback_router=build_fallback_router(config),
                                memory=memory)
    return dispatcher, device
