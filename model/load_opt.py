import argparse
from pathlib import Path

import torch
from transformers import OPTForCausalLM

from model.config import ModelConfig
from model.transformer import DesktopHelperLM


OPT_MODEL_ID = "facebook/opt-350m"
# OPT positional embeddings use a +2 index offset (Fairseq legacy convention):
# position 0 lives at embed_positions[2], position 1 at embed_positions[3], etc.
_OPT_POS_OFFSET = 2


def transplant(output_path: str) -> None:
    print(f"Downloading {OPT_MODEL_ID}...")
    opt = OPTForCausalLM.from_pretrained(OPT_MODEL_ID)
    src = opt.state_dict()

    print("Building DesktopHelperLM...")
    config = ModelConfig()
    model = DesktopHelperLM(config)
    dst = model.state_dict()

    print("Transplanting weights...")

    # positional embeddings — slice first context_len positions, correct for OPT offset
    dst["pos_emb.weight"] = src["model.decoder.embed_positions.weight"][
        _OPT_POS_OFFSET : _OPT_POS_OFFSET + config.context_len
    ].clone()

    # OPT-350M has no decoder-level final_layer_norm (unlike larger OPT variants).
    # ln_f stays as fresh init and is learned during fine-tuning.

    for i in range(config.n_layers):
        o = f"model.decoder.layers.{i}"
        d = f"blocks.{i}"

        dst[f"{d}.ln1.weight"] = src[f"{o}.self_attn_layer_norm.weight"].clone()
        dst[f"{d}.ln1.bias"]   = src[f"{o}.self_attn_layer_norm.bias"].clone()

        # OPT uses separate Q/K/V projections; fuse into our single qkv weight.
        # OPT has biases on these projections — our qkv uses bias=False, so drop them.
        q = src[f"{o}.self_attn.q_proj.weight"]
        k = src[f"{o}.self_attn.k_proj.weight"]
        v = src[f"{o}.self_attn.v_proj.weight"]
        dst[f"{d}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0).clone()

        # out_proj: drop OPT's bias (our out_proj uses bias=False)
        dst[f"{d}.attn.out_proj.weight"] = src[f"{o}.self_attn.out_proj.weight"].clone()

        dst[f"{d}.ln2.weight"] = src[f"{o}.final_layer_norm.weight"].clone()
        dst[f"{d}.ln2.bias"]   = src[f"{o}.final_layer_norm.bias"].clone()

        # fc1/fc2: drop OPT's biases (our MLP uses bias=False throughout)
        dst[f"{d}.mlp.fc1.weight"] = src[f"{o}.fc1.weight"].clone()
        dst[f"{d}.mlp.fc2.weight"] = src[f"{o}.fc2.weight"].clone()

    # token_emb and lm_head are left as fresh random init — OPT's vocab (50272 tokens)
    # cannot map to our 32000-token custom tokenizer with baked-in tool-call tokens.
    # these weights are learned entirely during fine-tuning.
    model.load_state_dict(dst)
    print("token_emb + lm_head: fresh init (learned during fine-tuning)")

    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.zeros((1, 8), dtype=torch.long))
    assert logits.shape == (1, 8, config.vocab_size), f"shape mismatch: {logits.shape}"
    print(f"Forward pass {logits.shape} ✓")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, out)
    print(f"Saved → {out}  ({model.num_params() / 1e6:.1f}M params)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transplant OPT-350M weights into DesktopHelperLM"
    )
    parser.add_argument(
        "--output", default="model/checkpoints/opt_transplant.pt",
        help="output checkpoint path (default: model/checkpoints/opt_transplant.pt)"
    )
    args = parser.parse_args()
    transplant(args.output)


if __name__ == "__main__":
    main()
