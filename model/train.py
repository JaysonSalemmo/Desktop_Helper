import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.dataset import InstructDataset
from model.device import get_device
from model.tokenizer import DesktopHelperTokenizer
from model.transformer import DesktopHelperLM


def _warmup_lambda(warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return fn


def train(args) -> None:
    device = get_device(require_cuda=True)
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = DesktopHelperTokenizer.load(args.tokenizer)

    print("Loading model from checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _warmup_lambda(args.warmup_steps)
    )
    scaler = torch.cuda.amp.GradScaler()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                print(f"epoch {epoch}  step {global_step:>6}  loss {avg:.4f}  lr {scheduler.get_last_lr()[0]:.2e}")
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="per-device batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="gradient accumulation steps — effective batch = batch_size × grad_accum (default: 8 → effective 32)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10,
                        help="print loss every N optimizer steps (default: 10)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
