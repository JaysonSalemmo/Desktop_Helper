"""File logging for the app — the whole point is the frozen bundle.

A frozen menu bar app has no console, so an unhandled error or a caught
model-load failure would otherwise vanish. This writes a rotating-ish single
log under the user-data dir (``~/Library/Application Support/Desktop
Helper/logs/`` when frozen, the project dir from source) so there's always a
place to read what happened.
"""
import logging

from src.paths import user_data_path

_LOGGER_NAME = "desktop_helper"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:  # already configured this process
        return logger
    log_dir = user_data_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "desktop-helper.log")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
