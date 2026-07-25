"""
Grow a fine-tuned checkpoint's embedding table to match a larger tokenizer,
preserving every existing weight and initializing only the new row(s).

Used when a new tool token is appended (e.g. [CALL: files], id 49163): the
token was added AFTER the existing ids, so all learned rows keep their meaning
and this just appends. The new row is seeded from the mean of the existing
tool-call rows — so it starts in "tool space" and the embeddings-only warm start
converges fast instead of learning a routing token from noise.

usage:
  uv run python -m model.resize_embeddings \
    --in  model/checkpoints/smol_run1_warm_epoch_02.pt \
    --out model/checkpoints/smol_run1_warm_files_init.pt
"""
import argparse

import torch

from model.tokenizer import DesktopHelperTokenizer

_LEGACY_TOOLS = ("calendar", "screen", "reminders", "notes", "spotify",
                 "launcher", "weather", "news", "stocks")


def resize(in_path: str, out_path: str) -> None:
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    sd = ckpt["model_state_dict"]

    old_emb = sd["token_emb.weight"]
    old_vocab, d_model = old_emb.shape

    tok = DesktopHelperTokenizer.load()
    new_vocab = tok.vocab_size
    if new_vocab <= old_vocab:
        raise SystemExit(
            f"tokenizer vocab {new_vocab} is not larger than the checkpoint "
            f"({old_vocab}) — nothing to add. Did you add the token to TOOL_TOKENS "
            "and regenerate model/hf_tokenizer?")

    new_emb = torch.empty(new_vocab, d_model, dtype=old_emb.dtype)
    new_emb[:old_vocab] = old_emb  # every existing row, unchanged

    # seed the new row(s) from the mean of the existing tool-call embeddings
    tool_ids = [tok.tool_token_id(t) for t in _LEGACY_TOOLS]
    new_emb[old_vocab:] = old_emb[tool_ids].mean(dim=0, keepdim=True)

    sd["token_emb.weight"] = new_emb
    if "lm_head.weight" in sd:
        sd["lm_head.weight"] = new_emb  # weight-tied; keep them the same tensor
    config.vocab_size = new_vocab

    torch.save({"config": config, "model_state_dict": sd}, out_path)
    print(f"Resized {old_vocab} → {new_vocab} (new rows seeded from tool-mean). "
          f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grow a checkpoint's vocab by appending rows")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()
    resize(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
