import json
import shutil
from pathlib import Path

from src.paths import resource_path, user_data_path

# config.json is writable per-user state (App Support when frozen, project dir
# from source); the template ships read-only with the code.
CONFIG_PATH = user_data_path("config.json")
CONFIG_EXAMPLE_PATH = resource_path("config.example.json")


def load() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        # frozen first run: seed a starter config from the bundled template so
        # the app has something to edit instead of dying on a missing file
        if CONFIG_EXAMPLE_PATH.exists() and CONFIG_PATH != CONFIG_EXAMPLE_PATH:
            shutil.copy(CONFIG_EXAMPLE_PATH, CONFIG_PATH)
            raise SystemExit(
                f"Created a starter config at {CONFIG_PATH}. "
                "Fill in your details (name, checkpoint, API keys), then relaunch."
            )
        raise SystemExit(
            f"Missing {CONFIG_PATH.name}. Copy the template and fill in your details:\n"
            f"  cp {CONFIG_EXAMPLE_PATH.name} {CONFIG_PATH.name}"
        )


def save(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
