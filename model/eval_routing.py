"""
Routing eval: does the MODEL send each prompt to the right tool?

The counterpart eval_faithfulness never was. Run 6's lesson: 100% faithfulness
coexisted with the files token over-firing live on spotify/reminders phrasings
("Play a song by Bruno Mars" → files), because nothing measured routing on
OTHER tools' prompts or on adversarial near-misses. This eval runs the bare
model — NO pre-router, NO fallback, no dispatcher band-aids — so the number
reflects the weights alone. The gate mirrors the dispatcher exactly: argmax
must be a tool token AND clear `route_confidence` softmax probability.

Scores three failure kinds separately:
  over-fire  — routed a prompt that should be chat, or stole another tool's
  under-fire — emitted no call where a tool was expected
  (correct routes are just counted)

usage:
    uv run python -m model.eval_routing --checkpoint model/checkpoints/<ckpt>.pt
"""
import argparse

import torch
import torch.nn.functional as F

from model import chat_format
from model.device import get_device
from model.generate import load_model
from model.tokenizer import DesktopHelperTokenizer

# (prompt, expected tool or None for chat). Grouped so failures read clearly.
CASES = [
    # -- run-6 live over-fires: the reason this eval exists ------------------
    ("Play a song by Bruno Mars.", "spotify"),
    ("When are my reminders set for?", "reminders"),
    ("Add milk to my shopping list", "reminders"),
    # -- more hard negatives in the same families ----------------------------
    ("Play something by The Weeknd", "spotify"),
    ("Put on some jazz", "spotify"),
    ("Put paper towels on my groceries list", "reminders"),
    ("Add a dentist appointment to my to-do list", "reminders"),
    ("Remind me to call the dentist tomorrow", "reminders"),
    ("What's due today?", "reminders"),
    # -- files positives: the run-6 win must survive the rebalance -----------
    ("Find Kai's resume", "files"),
    ("Where's my budget spreadsheet?", "files"),
    ("Find the most recent version of my resume", "files"),
    ("What did I work on last week?", "files"),
    ("Locate invoice.pdf", "files"),
    ("Search for my tax documents", "files"),
    ("Find my grocery list", "files"),  # "list" but a file query — must not flip
    # -- every tool's canonical phrasing: regression floor -------------------
    ("What's the weather like today?", "weather"),
    ("What's on my calendar today?", "calendar"),
    ("Check my reminders.", "reminders"),
    ("What did I note today?", "notes"),
    ("What's playing?", "spotify"),
    ("Pause the music.", "spotify"),
    ("Give me the headlines.", "news"),
    ("How's my watchlist looking?", "stocks"),
    ("What's on my screen?", "screen"),
    ("Open Blender.", "launcher"),
    # -- chat: must NOT route ------------------------------------------------
    ("Tell me a joke", None),
    ("What's the capital of France?", None),
    ("How are you today?", None),
    ("Tell me a fun fact about space.", None),
]


def route(model, tokenizer, prompt: str, device, confidence: float):
    """(routed tool or None, argmax prob) for one prompt — dispatcher's gate."""
    ids = chat_format.prime_ids(tokenizer, prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(x)
    logits = logits[0, -1, :].float()
    top = int(torch.argmax(logits))
    prob = F.softmax(logits, dim=-1)[top].item()
    tool = tokenizer.is_tool_call(top)
    if tool is not None and prob < confidence:
        tool = None  # not confident it's a tool request — dispatcher treats as chat
    return tool, prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Score bare-model tool routing")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="model/hf_tokenizer")
    parser.add_argument("--route-confidence", type=float, default=0.6)
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU — lets the eval run while the menu-bar "
                             "app holds MPS (one model process per GPU)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device(require_cuda=False)
    tokenizer = DesktopHelperTokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, device)

    correct, over, under = 0, [], []
    for prompt, expected in CASES:
        got, prob = route(model, tokenizer, prompt, device, args.route_confidence)
        ok = got == expected
        correct += ok
        if not ok:
            if got is None:
                under.append((prompt, expected))
            else:
                over.append((prompt, expected, got))
        mark = "✓" if ok else "✗"
        want = expected or "chat"
        have = got or "chat"
        print(f"{mark} [{want:9} → {have:9} p={prob:.2f}] {prompt}")

    n = len(CASES)
    print(f"\nRouting: {correct}/{n} ({correct / n:.0%})  "
          f"over-fires: {len(over)}  under-fires: {len(under)}")
    for prompt, expected, got in over:
        print(f"  OVER  {prompt!r}: {got} stole from {expected or 'chat'}")
    for prompt, expected in under:
        print(f"  UNDER {prompt!r}: no call, wanted {expected}")


if __name__ == "__main__":
    main()
