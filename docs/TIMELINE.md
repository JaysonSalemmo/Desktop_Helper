# Desktop Helper — Project Timeline

A chronological record of the project's eras, decisions, and measured results.
Kept for research and note-taking; the raw day-by-day log lives in the
(untracked) DEVLOG.md.

---

## Era 0 — The from-scratch plan (June 2026)

**Premise:** build a conversational desktop assistant where *everything* is
from scratch: a custom GPT-style transformer (PyTorch), a custom 32K BPE
tokenizer with tool-call tokens baked in, pre-trained on a general corpus
(SmolLM-Corpus was the candidate), then fine-tuned on instruction + synthetic
tool-call data. Tool calls via special tokens: the model emits
`[CALL: weather]`, the app runs the real tool and injects
`[RESULT]...[/RESULT]`, the model continues.

Phase 1 shipped: `model/transformer.py` (decoder-only, pre-norm, learned
positional embeddings, ~303M params), tokenizer, tests. Estimated pre-training
cost: 1–2 weeks of GPU time — which prompted the first pivot.

## Era 1 — The OPT pivot (2026-06-19, branch `pretrained-model`)

**Decision:** skip pre-training entirely. Meta's OPT-350M (May 2022, open
weights) matched the custom architecture *exactly* — 24 layers, d_model 1024,
16 heads, pre-norm, learned positions — so its transformer blocks were
transplanted directly into the custom code (`model/load_opt.py`).

**The fateful constraint:** the custom 32K tokenizer couldn't map to OPT's
50,272-token vocabulary, so the token embeddings and LM head started as
**100% random init**. Nearly every training problem for the next month traces
back to this single decision.

## Era 2 — The training campaign (runs 1–5, July 2026)

All training on Google Colab (free tier), fp16, checkpoints to Drive.
Data: databricks-dolly-15k + synthetic tool-call examples.

| Run | Recipe | Outcome |
|---|---|---|
| 1 | 1 epoch, uniform LR | **Garbage.** Diagnosed: random-init embeddings undertrained at a fine-tuning LR. Fix: split-LR (3e-4 embeddings / 5e-5 blocks). |
| 2 | 3 epochs, split-LR | Plateau diagnosed; three tuning changes queued. |
| 3 | 8 epochs + prompt-loss masking + cosine decay + prompt diversity | **Routing achieved: 8/8 tools**, clean `[CALL][RESULT]` structure, real generalization ("Am I free this afternoon?" → calendar). Loss ~4.4 (masked regime). |
| 4 | 12 epochs + RESULT-masked loss + high-entropy data | Faithfulness 4% → 29% → **36%** (epoch 12). Blending shrank from word-level to letter-level ("Wojciech" → "Wofciech"). |
| 5 | 16 epochs, 8000 examples | **88% faithfulness, routing 14/14.** "The plateau was the cosine schedule, not the model" — every apparent convergence was the LR hitting its floor. |

**The faithfulness saga** (the era's defining fight): with real
out-of-distribution data the model *paraphrased toward its training
distribution* instead of copying injected results — real reminders became
"call mom", 86°F became "72°F". Fixes, in order: mask `[RESULT]` spans from
the loss (stop teaching result-invention); regenerate training data with
high-entropy compositional content (memorization stops working, copying
becomes the only low-loss strategy); decoding-side copy-bias + repetition
penalty (helped only after the copy circuit began forming — a boost sweep on
run 3 proved decoding can't summon a circuit training didn't build).

Memorable artifacts: the first faithfulness eval scored **4%**; "CSFA World
Cup 3rd" (for "FIFA World Cup 3rd Place Match"); the copy-boost Goodhart
regime where the metric climbed while replies collapsed into "volume volume
volume".

## Era 3 — App-ification (2026-07-17 → 07-18)

The model became an actual macOS product in roughly 48 hours:

- **TUI wired to the model** (async Textual workers), then **all 9 tools to
  real backends**: EventKit calendar/reminders, Open-Meteo→NWS weather, RSS
  news, yfinance stocks, AppleScript+Web-API Spotify (search, random artist
  picks), NSWorkspace screen, launcher, notes.
- **Menu bar app** (rumps) → **launcher .app bundle** (own Info.plist — ended
  the calendar-permission saga: macOS silently refuses prompts for apps
  without usage-description keys) → app icon → /Applications.
- **Voice**: local whisper input (push-to-talk), spoken replies (`say`,
  system voice), wake word built ("hey Jarvis", openWakeWord — parked pending
  a custom "hey helper" model). A pynput/macOS-15 Caps-Lock crash forced a
  from-scratch Quartz event-tap hotkey listener (active tap, consuming,
  multi-keybind).
- **ChromaDB memory** (retrieval-only by design — single-turn model, history
  injection would break routing), **chat-bubble reply panel** with follow-ups,
  morning briefing.
- **The verbatim design** — the era's core architectural lesson, learned when
  Kai's first real calendar query returned "CSFA World Cup 3rd": *the model
  routes, the tools know, the templates speak.* Fact-heavy tools (calendar,
  reminders, notes, spotify, stocks, launcher) bypass model paraphrasing
  entirely; the model's 88% copying was never going to be the 100% calendars
  deserve. Free-form chat was retired to canned replies after producing
  **"Ojomala Adar is a JavaScript that combines the family"** in response to
  "Hello".
- Routing hardening, each layer forced by a live failure: greedy routing
  (deterministic) → confidence gate at 0.9 (measured: real prompts route at
  p≈1.000, chat maxes at 0.68 — "Hello." had been fetching the current track
  at p=0.13) → fallback keyword router ("Play X on Spotify" was OOD and routed
  nowhere). Plus the Weird Al ambush: "Another song by Bruno Mars" literally
  matched *"Another Tattoo (Parody of… by B.o.B feat. Bruno Mars)"* — vague
  artist requests got their own grammar + random pick.

## Era 4 — The ceiling, and the SmolLM2 pivot (2026-07-18 → 07-19)

**What OPT-350M could do:** route 14/14 with real semantic understanding,
copy short results at 88%, wrap simple replies fluently.
**What it could never do:** hold a conversation, summarize, describe — 350M
parameters of May-2022 pretraining with ~19k fine-tuning examples has a hard
generative ceiling; every workaround (verbatim templates, canned chat) was a
way of routing around it.

**Decision (2026-07-19, branch `smollm2-1.7b`):** upgrade all three legs at
once —
1. **Architecture** → modern Llama-style: RoPE (rotate_half convention),
   RMSNorm, SwiGLU — still implemented from scratch in our own code.
2. **Base model** → SmolLM2-1.7B-**Instruct** (2024, Apache 2.0): 5× the
   parameters, 2+ years newer pretraining, and *instruction-tuned* — the
   conversational ability the project lacked cannot be trained at home, but it
   can be transplanted and preserved.
3. **Tokenizer strategy** → adopt the base model's tokenizer + append 11 tool
   tokens, fixing Era 1's original sin: only 11 embedding rows start random
   this time, not 49,152.

Process lesson institutionalized from the Era 2 saga: a **logit-equivalence
test** (our transplant vs. HuggingFace's reference implementation, asserted
in seconds, locally) so no transplant bug ever again costs an 8-hour training
run to discover.

The `opt-350m` branch preserves Era 1–3 exactly as they ended;
`run5_epoch16.pt` remains the era's final artifact.

## Era 5 — The warm start, and the cutover (2026-07-19 → 07-20)

**Run 1** (3 epochs, uniform 2e-5, Dolly + tool calls): the era's core goal —
kill the word salad — achieved immediately. Chat became genuinely
conversational ("Hey there — what do you need?"), and held-out Dolly loss
*fell* every epoch (1.336→1.268): zero catastrophic forgetting, the
instruction-tuned base preserved by design. But faithfulness eval read **0%**
— the model never emitted a `[CALL]` token. A first-token probe showed the
routing circuit *sprouting* (correct tool often argmax, at ~2–4%
probability) but nowhere near usable confidence. Diagnosis: the 11
from-scratch tool-token rows — the only random-init params — undertrained at
the deliberately-gentle LR. Era 1's original sin, shrunk 3000× but not gone.

**The fix — embeddings-only warm start** (`train.py --embeddings-only`):
freeze everything, gradient-mask all embedding rows except the 11 tool rows,
train only those at 1e-3 on tool-call data. The tied lm_head means the output
logit rows train too — exactly the circuit that must fire. Forgetting is
impossible *by construction*, and a unit test asserts pretrained rows and
blocks stay bit-identical. (Gotcha the test caught: AdamW's decoupled weight
decay moves every param in the group regardless of gradient — masked rows
were shrinking; this mode runs weight_decay=0.)

**Result of a ~40-minute L4 run:** 93% faithfulness, 13/14 routing, and
**byte-perfect proper nouns** — "Pottery workshop at 4:35pm, and Dinner with
Wojciech at 8pm", exact. The OPT era's white whale (letter-level blending,
"Wofciech") is dead. Chat untouched, frozen by construction.

**The cutover (07-20):** gate re-measured — every routing prompt fires at
p=0.73–0.999, chat never argmaxes a tool; the old 0.9 gate (calibrated for
OPT) was itself the missing 7%, so `route_confidence` → 0.6 → **eval 100%**.
Decoding knobs re-swept: 98% with *zero* assistance (the model copies
natively now); defaults settled at copy_boost 2.0 / repetition_penalty 1.1.
The verbatim templates stay — 100% is still better than 99.9% for calendars —
but they're now belt-and-suspenders, not load-bearing.

**Chat policy inverted:** Era 3 retired free chat to canned replies
("Ojomala Adar…"); Era 5 retires the canned replies. Unrouted chat now goes
to the model — with one carve-out, found in live testing: asked "What can you
do?", the model fluently *invents* its own feature list (Wikipedia, Apple
Watch). Capability questions stay canned; everything else is the model's.

Checkpoint: `smol_run1_warm_epoch_02.pt` — main fine-tune + surgical warm
start. Speed on MPS: ~2.3 tok/s (1.7B bf16, no KV cache) — the KV cache is
the next era's first fight.
