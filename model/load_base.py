"""
Transplant SmolLM2-1.7B-Instruct into DesktopHelperLM.

Successor to load_opt.py (kept for history). Differences that matter:
- The base model's tokenizer is adopted (+11 tool tokens), so 49,152 of the
  49,163 embedding rows transplant directly — only the 11 tool-token rows
  start fresh. In the OPT era ALL rows started random, which caused the
  faithfulness saga (docs/TIMELINE.md).
- The 11 new rows are initialized to the mean of the pretrained embeddings
  plus small noise — the standard recipe for added special tokens; a
  mean-embedding starts "neutral" instead of nowhere.
- No positional embeddings to transplant at all (RoPE is weight-free) — one
  fewer place to hide an OPT-style off-by-two.

After transplanting, run the logit-equivalence test (tests/test_transplant.py)
BEFORE spending Colab hours — it compares our model against HF's reference
implementation and catches RoPE-convention/eps/mapping bugs in seconds. The
OPT era learned this the expensive way.

usage:
    uv run python -m model.load_base            # → model/checkpoints/smol_transplant.pt
"""
import argparse
from pathlib import Path

import torch

from model.config import ModelConfig
from model.tokenizer import BASE_MODEL_ID, DesktopHelperTokenizer
from model.transformer import DesktopHelperLM


def transplant(output_path: str) -> DesktopHelperLM:
    from transformers import AutoModelForCausalLM

    print("Building tokenizer (base + 11 tool tokens)...")
    tokenizer = DesktopHelperTokenizer.build()
    tokenizer.save()  # → model/hf_tokenizer/, committed to git

    print(f"Downloading {BASE_MODEL_ID} (~3.4GB)...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32)
    src = base.state_dict()

    print("Building DesktopHelperLM...")
    config = ModelConfig.from_tokenizer(tokenizer)
    model = DesktopHelperLM(config)
    dst = model.state_dict()

    print("Transplanting weights...")

    # embeddings: pretrained rows copy over; the 11 tool-token rows get
    # mean-of-vocab + noise (std well below the embedding's own spread)
    pretrained = src["model.embed_tokens.weight"]  # (49152, 2048)
    n_base = pretrained.shape[0]
    emb = dst["token_emb.weight"]
    emb[:n_base] = pretrained.clone()
    mean_emb = pretrained.mean(dim=0)
    for row in range(n_base, config.vocab_size):
        emb[row] = mean_emb + torch.randn_like(mean_emb) * 0.02
    # lm_head is tied to token_emb — nothing separate to copy

    for i in range(config.n_layers):
        s = f"model.layers.{i}"
        d = f"blocks.{i}"

        dst[f"{d}.ln1.weight"] = src[f"{s}.input_layernorm.weight"].clone()
        dst[f"{d}.ln2.weight"] = src[f"{s}.post_attention_layernorm.weight"].clone()

        # SmolLM2 stores q/k/v separately (all bias-free, plain MHA) — fuse
        # into our single qkv weight, same pattern load_opt.py used
        q = src[f"{s}.self_attn.q_proj.weight"]
        k = src[f"{s}.self_attn.k_proj.weight"]
        v = src[f"{s}.self_attn.v_proj.weight"]
        dst[f"{d}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0).clone()
        dst[f"{d}.attn.out_proj.weight"] = src[f"{s}.self_attn.o_proj.weight"].clone()

        dst[f"{d}.mlp.gate_proj.weight"] = src[f"{s}.mlp.gate_proj.weight"].clone()
        dst[f"{d}.mlp.up_proj.weight"] = src[f"{s}.mlp.up_proj.weight"].clone()
        dst[f"{d}.mlp.down_proj.weight"] = src[f"{s}.mlp.down_proj.weight"].clone()

    dst["ln_f.weight"] = src["model.norm.weight"].clone()

    model.load_state_dict(dst)
    model.lm_head.weight = model.token_emb.weight  # re-tie after load

    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.zeros((1, 8), dtype=torch.long))
    assert logits.shape == (1, 8, config.vocab_size), f"shape mismatch: {logits.shape}"
    print(f"Forward pass {logits.shape} ✓")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out)
    print(f"Saved → {out}  ({model.num_params() / 1e9:.2f}B params + "
          f"{model.token_emb.weight.numel() / 1e6:.0f}M embeddings)")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transplant SmolLM2-1.7B-Instruct into DesktopHelperLM"
    )
    parser.add_argument(
        "--output", default="model/checkpoints/smol_transplant.pt",
        help="output checkpoint path (default: model/checkpoints/smol_transplant.pt)"
    )
    args = parser.parse_args()
    transplant(args.output)


if __name__ == "__main__":
    main()
