"""OPT-ERA ARTIFACT — the SmolLM2 era adopts the base model's
tokenizer (model/tokenizer.py builds it); this from-scratch trainer is kept
only for history. See docs/TIMELINE.md.
"""
"""
train the bpe tokenizer on a text corpus and save it to disk.
run this once before training the model — the saved tokenizer.json
is what the training loop and the app will both load.

usage:
    uv run python -m model.train_tokenizer --data data/corpus/ --output model/tokenizer.json

the data directory should contain .txt files (one or many).
in phase 2 we'll download and place the actual training corpus there.
"""
import argparse
from pathlib import Path

from model.tokenizer import DesktopHelperTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="train the desktop helper bpe tokenizer")
    parser.add_argument(
        "--data",
        required=True,
        help="directory containing .txt corpus files",
    )
    parser.add_argument(
        "--output",
        default="model/tokenizer.json",
        help="where to save the trained tokenizer (default: model/tokenizer.json)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="number of bpe tokens to learn (default: 32000)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    files = sorted(data_dir.glob("**/*.txt"))
    if not files:
        raise ValueError(f"no .txt files found in {data_dir}")

    print(f"found {len(files)} file(s) — starting tokenizer training...")
    tokenizer = DesktopHelperTokenizer.train(files, vocab_size=args.vocab_size)

    tokenizer.save(args.output)
    print(f"done. vocab size: {tokenizer.vocab_size}")
    print(f"saved to: {args.output}")
    print(f"pad={tokenizer.pad_id}  bos={tokenizer.bos_id}  eos={tokenizer.eos_id}")


if __name__ == "__main__":
    main()
