"""
Faithfulness eval: does the model's reply copy the facts from an injected
[RESULT] block, or invent its own?

Runs the real dispatcher path (interception + injection + resume) against fake
handlers returning fixed held-out results — entities chosen to NOT appear in
the training generator's pools, so the only way to score is genuine copying
from context. Each case lists key facts (numbers, names, conditions); score is
the fraction of facts present verbatim (case-insensitive) in the reply.

This is the counterpart of the routing test: routing measured "does it pick the
right tool", this measures "does it echo the real data". Run it on every new
checkpoint. Baseline epoch_08 (pre-faithfulness-retrain): expect near zero.

usage:
    uv run python -m model.eval_faithfulness --checkpoint model/checkpoints/epoch_08.pt
"""
import argparse

import torch

from model.device import get_device
from model.generate import load_model
from model.tokenizer import DesktopHelperTokenizer
from src.assistant.dispatcher import ToolDispatcher

# (tool, user message, injected result, key facts that must appear in the reply)
CASES = [
    ("weather", "What's the weather like today?",
     "43°F, freezing rain", ["43", "freezing rain"]),
    ("weather", "How hot is it today?",
     "91°F, partly cloudy", ["91", "partly cloudy"]),
    ("stocks", "How's my watchlist looking?",
     "RBLX: $67.21 (+2.8%)", ["RBLX", "67.21", "2.8"]),
    ("stocks", "Check ADBE for me.",
     "ADBE: $512.09 (-1.1%)", ["ADBE", "512.09", "1.1"]),
    ("calendar", "What's on my calendar today?",
     "Pottery workshop at 4:35pm, Dinner with Wojciech at 8pm",
     ["Pottery", "4:35", "Wojciech", "8pm"]),
    ("calendar", "Any meetings today?",
     "Thesis defense at 11:05am", ["Thesis", "11:05"]),
    ("reminders", "Check my reminders.",
     "Return library books, Rotate the compost, Text Priyanka",
     ["library books", "compost", "Priyanka"]),
    ("reminders", "What's on my to-do list?",
     "Defrost the freezer, Upload receipts to Concur",
     ["freezer", "receipts", "Concur"]),
    ("notes", "What did I note today?",
     "Wifi password for studio: grapefruit-42", ["studio", "grapefruit-42"]),
    ("spotify", "What's playing?",
     "Paranoid Android by Radiohead, volume 37%",
     ["Paranoid Android", "Radiohead", "37"]),
    ("news", "Give me the headlines.",
     "Ferry workers strike enters third day; Comet visible over Tasmania tonight",
     ["Ferry", "strike", "Comet", "Tasmania"]),
    ("screen", "What's on my screen?",
     "Blender in front, also open: Anki, Transmit", ["Blender", "Anki", "Transmit"]),
    ("launcher", "Open Blender.",
     "Blender launched", ["Blender"]),
    ("calendar", "What do I have scheduled?",
     "Calendar access not granted", ["access", "granted"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score RESULT-copying faithfulness of a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="model/tokenizer.json")
    parser.add_argument("--copy-boost", type=float, default=2.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    args = parser.parse_args()

    device = get_device(require_cuda=False)
    tokenizer = DesktopHelperTokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, device)

    # one handler per tool, each returning the active case's fixed result
    current = {}
    handlers = {tool: (lambda m, t=tool: current[t]) for tool, _, _, _ in CASES}
    dispatcher = ToolDispatcher(model, tokenizer, handlers, device,
                                copy_boost=args.copy_boost,
                                repetition_penalty=args.repetition_penalty)

    lines = []
    scores = []
    for tool, message, injected, facts in CASES:
        current[tool] = injected
        torch.manual_seed(0)
        result = dispatcher.respond(message)
        reply = result.response.lower()
        hit = [f for f in facts if f.lower() in reply]
        scores.append(len(hit) / len(facts))
        routed = "✓" if result.tool == tool else f"✗ routed to {result.tool}"
        lines.append(f"[{tool:9}] {scores[-1]:4.0%}  route {routed}\n"
                     f"    inject: {injected}\n"
                     f"    reply:  {result.response}")

    print("\n".join(lines))
    print(f"\nOverall faithfulness: {sum(scores) / len(scores):.0%}  ({args.checkpoint})")


if __name__ == "__main__":
    main()
