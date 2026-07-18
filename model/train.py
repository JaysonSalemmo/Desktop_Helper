import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.dataset import InstructDataset
from model.device import get_device
from model.tokenizer import DesktopHelperTokenizer
from model.transformer import DesktopHelperLM


def _lr_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
    # linear warmup, then cosine decay from the peak LR down to min_ratio × peak.
    # decaying the LR (rather than holding it flat) reliably reaches a lower final
    # loss and generalises better. applied as a multiplier to both param groups,
    # so the embedding and block LRs decay proportionally together.
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return fn


def train(args) -> None:
    device = get_device(require_cuda=True)
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = DesktopHelperTokenizer.load(args.tokenizer)

    print("Loading model from checkpoint...")
    # weights_only=False: the checkpoint stores a ModelConfig object, which the
    # PyTorch 2.6+ default (weights_only=True) refuses to unpickle. Safe here —
    # this is our own checkpoint produced by model/load_opt.py, not untrusted input.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = DesktopHelperLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Building dataset...")
    dataset = InstructDataset(
        tokenizer=tokenizer,
        tool_calls_path=args.tool_calls,
        context_len=config.context_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    print(f"{len(dataset)} examples, {len(loader)} batches per epoch")

    # Split-LR: token_emb/lm_head started from random init (OPT's vocab couldn't
    # transfer to our custom tokenizer) and must be learned from scratch, so they
    # get a much higher LR. The transplanted transformer blocks + positional
    # embeddings are already pre-trained and only need gentle fine-tuning — a high
    # LR there would wipe out OPT's language knowledge. lm_head.weight is tied to
    # token_emb.weight (same tensor), so model.parameters() yields it once and the
    # id() partition below cleanly separates the two groups.
    embed_params = list(model.token_emb.parameters())
    embed_ids = {id(p) for p in embed_params}
    block_params = [p for p in model.parameters() if id(p) not in embed_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embed_params, "lr": args.embed_lr},
            {"params": block_params, "lr": args.lr},
        ],
        weight_decay=0.1,
    )
    print(f"embedding LR {args.embed_lr:.1e} (from scratch)  |  block LR {args.lr:.1e} (fine-tune)")
    # total optimizer steps drive the cosine decay horizon
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(args.warmup_steps, total_steps)
    )
    scaler = torch.cuda.amp.GradScaler()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard: writes live loss/LR curves to logdir (default <output>/runs).
    # Point Colab's %tensorboard at this dir to watch training in real time.
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

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item()

            if (step + 1) % args.grad_accum != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if global_step % args.log_every == 0:
                avg = running_loss / args.log_every
                embed_lr, block_lr = scheduler.get_last_lr()
                writer.add_scalar("loss/train", avg, global_step)
                writer.add_scalar("lr/embed", embed_lr, global_step)
                writer.add_scalar("lr/block", block_lr, global_step)
                print(f"epoch {epoch}  step {global_step:>6}  loss {avg:.4f}  emb_lr {embed_lr:.2e}  blk_lr {block_lr:.2e}")
                running_loss = 0.0

            global_step += 1

        ckpt_path = out_dir / f"epoch_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, ckpt_path)
        print(f"Saved → {ckpt_path}")

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DesktopHelperLM on Dolly + tool-call data. Requires CUDA."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="path to opt_transplant.pt")
    parser.add_argument("--tokenizer", required=True,
                        help="path to trained tokenizer.json")
    parser.add_argument("--tool-calls", default="data/tool_calls.jsonl",
                        help="path to tool-call JSONL (default: data/tool_calls.jsonl)")
    parser.add_argument("--output", default="model/checkpoints/finetune",
                        help="directory to save per-epoch checkpoints")
    parser.add_argument("--logdir", default=None,
                        help="TensorBoard log directory (default: <output>/runs)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="per-device batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="gradient accumulation steps — effective batch = batch_size × grad_accum (default: 8 → effective 32)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="LR for the pre-trained transformer blocks (default: 5e-5)")
    parser.add_argument("--embed-lr", type=float, default=3e-4,
                        help="LR for token_emb/lm_head, learned from scratch (default: 3e-4)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10,
                        help="print loss every N optimizer steps (default: 10)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
