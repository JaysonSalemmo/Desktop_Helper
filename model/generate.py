"""
Load a fine-tuned checkpoint and generate a response to a prompt.

Runs on Mac (MPS/CPU) — inference doesn't need CUDA. This is the quickest way
to check whether fine-tuning actually taught the model anything.

usage:
    uv run python -m model.generate \
        --checkpoint model/checkpoints/epoch_01.pt \
        --prompt "What's on my calendar today?"
"""
import argparse

import torch

from model.device import get_device
from model.tokenizer import DesktopHelperTokenizer
from model.transformer import DesktopHelperLM


def load_model(checkpoint: str, device: torch.device) -> DesktopHelperLM:
    # weights_only=False: checkpoint stores a ModelConfig object (our own file).
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = DesktopHelperLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def generate(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k):
    # match the training format: <bos> prompt \n  ...then let the model continue
    # until it emits <eos>. training turns were <bos>prompt\nresponse<eos>.
    ids = [tokenizer.bos_id] + tokenizer.encode(f"{prompt}\n")
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )

    # decode only the newly generated tokens, not the prompt we primed with
    generated = out[0, len(ids):].tolist()
    return tokenizer.decode(generated, skip_special=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="generate from a fine-tuned checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="model/tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = get_device(require_cuda=False)
    print(f"Device: {device}")

    tokenizer = DesktopHelperTokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, device)

    response = generate(
        model, tokenizer, args.prompt, device,
        args.max_new_tokens, args.temperature, args.top_k,
    )

    print(f"\nPrompt:   {args.prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
