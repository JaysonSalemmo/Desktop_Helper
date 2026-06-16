import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"
CONFIG_EXAMPLE_PATH = CONFIG_PATH.parent / "config.example.json"


def load() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"Missing {CONFIG_PATH.name}. Copy the template and fill in your details:\n"
            f"  cp {CONFIG_EXAMPLE_PATH.name} {CONFIG_PATH.name}"
        )


def save(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
