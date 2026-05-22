import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"


def load() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
