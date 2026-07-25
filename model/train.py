"""
Fine-tune the SmolLM2 transplant on Dolly + tool-call data. Requires CUDA.

Recipe changes from the OPT era (see docs/TIMELINE.md):
- Pure bf16, no GradScaler. fp16 AMP keeps fp32 master weights+grads
  (~13.6GB at 1.7B — alone blowing a T4's budget); bf16 params throughout
  need no loss scaling and halve that. This is what makes 1.7B fit.
- bitsandbytes 8-bit AdamW: optimizer states drop from ~13.6GB (fp32 Adam)
  to ~3.4GB. Installed on Colab (not in pyproject — no CUDA locally anyway).
- Gradient checkpointing: recompute activations in backward, ~30% slower,
  large memory win. Total budget ≈ 3.4 (weights) + 3.4 (grads) + 3.4 (optim)
  + activations ⇒ workable on a 15GB T4/L4 at batch 1-2.
- Single LR (2e-5), split-LR is gone: its whole reason was 32k random-init
  embedding rows; now only 11 tool-token rows start fresh.
- Few epochs (default 3): the instruct base needs light adaptation — the
  danger is no longer undertraining but catastrophically forgetting the
  conversational ability we transplanted. A held-out Dolly slice is
  monitored as the forgetting signal.
"""
import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model.dataset import InstructDataset
from model.device import get_device
from model.tokenizer import DesktopHelperTokenizer
from model.transformer import DesktopHelperLM


def configure_embeddings_only(model, first_new_row: int) -> None:
    """Warm-start mode: freeze everything except the token embedding, and
    gradient-mask the embedding so ONLY rows >= first_new_row (the appended
    tool tokens) receive updates. The 49,152 pretrained rows and every block
    stay bit-identical — this is the surgical version of the OPT era's
    split-LR, scoped to the 11 rows that actually start from scratch.

    Because lm_head is tied to token_emb, this trains the tool tokens' OUTPUT
    logit rows too — which is precisely the circuit that failed to fire after
    the gentle main fine-tune (probed 2026-07-20: correct tool often argmax
    but at p≈0.02-0.04, far under any usable confidence).
    """
    for param in model.parameters():
        param.requires_grad = False
    model.token_emb.weight.requires_grad = True

    def mask_pretrained_rows(grad):
        grad = grad.clone()
        grad[:first_new_row] = 0
        return grad

    model.token_emb.weight.register_hook(mask_pretrained_rows)


def _lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    # linear warmup, then cosine decay to min_ratio × peak. Era-2 lesson:
    # every "plateau" in runs 3-5 was actually this schedule hitting its
    # floor — judge convergence against the schedule, not the loss alone.
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return fn


@torch.no_grad()
def _held_out_loss(model, loader, device) -> float:
    # the catastrophic-forgetting alarm: loss on general instruction data the
    # model never trains on. if this climbs while train loss falls, the
    # narrow tool-call data is eating the base model's general ability.
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


def train(args) -> None:
    device = get_device(require_cuda=True)
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = DesktopHelperTokenizer.load(args.tokenizer)

    print("Loading model from checkpoint...")
    # weights_only=False: the checkpoint stores a ModelConfig object (our own
    # file, produced by model/load_base.py — not untrusted input)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = DesktopHelperLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.lm_head.weight = model.token_emb.weight  # re-tie after load
    model = model.to(device=device, dtype=torch.bfloat16)  # bf16 throughout
    model.gradient_checkpointing = True

    if args.embeddings_only:
        # default: the first special-token id (calendar, 49152) — trains every
        # from-scratch tool row. Overridable so a post-resize run can warm-start
        # ONLY the newly-appended row(s) and leave the already-trained rows frozen.
        first_new_row = args.first_new_row
        if first_new_row is None:
            first_new_row = tokenizer.tool_token_id("calendar")
        n_rows = tokenizer.vocab_size - first_new_row
        configure_embeddings_only(model, first_new_row)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"WARM-START MODE: training embedding rows {first_new_row}+ only "
              f"({trainable / 1e6:.0f}M-param tensor, {n_rows} rows unmasked)")

    print("Building dataset...")
    dataset = InstructDataset(
        tokenizer=tokenizer,
        tool_calls_path=args.tool_calls,
        context_len=config.context_len,
        include_dolly=not args.no_dolly,
    )
    held_out_size = min(200, len(dataset) // 20)
    train_set, held_out = random_split(
        dataset, [len(dataset) - held_out_size, held_out_size],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)
    held_loader = DataLoader(held_out, batch_size=args.batch_size)
    print(f"{len(train_set)} train / {held_out_size} held-out, "
          f"{len(loader)} batches per epoch")

    # 8-bit AdamW (bitsandbytes): quantized optimizer states, ~75% memory
    # saving vs fp32 Adam — the difference between fitting a T4 and not.
    # imported lazily: only installed on Colab (train requires CUDA anyway).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # weight decay must be 0 in embeddings-only mode: AdamW's decoupled decay
    # moves EVERY param in the group regardless of gradient, which would
    # shrink the gradient-masked pretrained rows (caught by unit test)
    weight_decay = 0.0 if args.embeddings_only else 0.1
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.lr,
                                        weight_decay=weight_decay)
        print("optimizer: AdamW8bit (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                      weight_decay=weight_decay)
        print("optimizer: torch AdamW (bitsandbytes not installed — "
              "needs more VRAM; fine on A100)")

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(args.warmup_steps, total_steps)
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logdir = args.logdir or str(out_dir / "runs")
    writer = SummaryWriter(log_dir=logdir)
    print(f"TensorBoard logs → {logdir}")

    global_step = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        running_loss = 0.0

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # bf16 autocast, and NO GradScaler: bf16 has fp32's exponent
            # range, so fp16-style loss scaling is unnecessary
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            # belt & braces vs the dataset-level filter: a non-finite loss
            # must never reach backward() — one NaN gradient poisons every
            # weight permanently and the rest of the run trains garbage
            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss at step {step}, skipping micro-batch")
                continue

            loss.backward()
            running_loss += loss.item()

            if (step + 1) % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if global_step % args.log_every == 0:
                avg = running_loss / args.log_every
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("loss/train", avg, global_step)
                writer.add_scalar("lr", lr, global_step)
                print(f"epoch {epoch}  step {global_step:>6}  loss {avg:.4f}  lr {lr:.2e}")
                running_loss = 0.0

            global_step += 1

        held = _held_out_loss(model, held_loader, device)
        writer.add_scalar("loss/held_out", held, global_step)
        print(f"epoch {epoch}  held-out loss {held:.4f}  (forgetting alarm — "
              "should not climb while train loss falls)")

        ckpt_path = out_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "config": config,
        }, ckpt_path)
        print(f"Saved → {ckpt_path}")

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the SmolLM2 transplant on Dolly + tool-call data. Requires CUDA."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="path to smol_transplant.pt")
    parser.add_argument("--tokenizer", default="model/hf_tokenizer",
                        help="path to the saved tokenizer dir")
    parser.add_argument("--tool-calls", default="data/tool_calls.jsonl")
    parser.add_argument("--output", default="model/checkpoints/finetune",
                        help="directory to save per-epoch checkpoints")
    parser.add_argument("--logdir", default=None,
                        help="TensorBoard log directory (default: <output>/runs)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="light-touch adaptation — more risks forgetting (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="per-device batch size (default: 1 — 1.7B on a 15GB card is tight)")
    parser.add_argument("--grad-accum", type=int, default=32,
                        help="effective batch = batch_size × grad_accum (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="single LR for the whole model (default: 2e-5)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--embeddings-only", action="store_true",
                        help="warm-start mode: train ONLY the appended "
                             "tool-token embedding rows (use --lr 1e-3)")
    parser.add_argument("--first-new-row", type=int, default=None,
                        help="embeddings-only: first embedding row to train "
                             "(rows below it are gradient-masked). Default: the "
                             "first special-token id, i.e. every tool row. Pass "
                             "the files-token id to warm-start only that row "
                             "after a resize.")
    parser.add_argument("--no-dolly", action="store_true",
                        help="tool-call data only (for the warm start — "
                             "concentrated routing signal, forgetting impossible)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
