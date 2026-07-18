"""
Build a plain-text corpus for training the BPE tokenizer.

Pulls the same data the model is fine-tuned on — Dolly + synthetic tool-call
examples — and writes it as .txt so the tokenizer learns merges over exactly
the vocabulary it will see at training and inference time. Run once, before
train_tokenizer.py.

usage:
    uv run python -m model.data.build_corpus --tool-calls data/tool_calls.jsonl --output data/corpus/
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="build tokenizer training corpus")
    parser.add_argument("--tool-calls", default="data/tool_calls.jsonl")
    parser.add_argument("--output", default="data/corpus/")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Dolly...")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    with open(out_dir / "dolly.txt", "w") as f:
        for ex in dolly:
            prompt = ex["instruction"].strip()
            if ex.get("context", "").strip():
                prompt += "\n\n" + ex["context"].strip()
            f.write(f"{prompt}\n{ex['response'].strip()}\n")
    print(f"  wrote {len(dolly)} Dolly examples")

    print("Loading tool-call examples...")
    n = 0
    with open(out_dir / "tool_calls.txt", "w") as f:
        with open(args.tool_calls) as src:
            for line in src:
                ex = json.loads(line)
                f.write(f"{ex['prompt']}\n{ex['response']}\n")
                n += 1
    print(f"  wrote {n} tool-call examples")

    print(f"Corpus ready in {out_dir}")


if __name__ == "__main__":
    main()
