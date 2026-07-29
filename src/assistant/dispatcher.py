"""
Tool dispatcher — the loop that turns the model into an actual assistant.

The fine-tuned model emits a `[CALL: tool]` token when it wants live data, then
(in training) fabricated its own `[RESULT]...[/RESULT]`. At inference we intercept
the moment a `[CALL: tool]` token appears: we run the *real* tool, inject the real
`[RESULT]...[/RESULT]` into the context, and let the model resume — so the final
response is written from real data, never the model's hallucinated result.

The model chooses the *tool*; the handler parses the user message to choose the
specific *action* within that tool (e.g. pause vs. skip vs. current track). This
keeps the model's job simple (routing only) and the action deterministic.

Three decoding measures nudge the reply toward the injected result (the small
model otherwise paraphrases toward its training distribution — wrong
temperatures, invented reminders):
- after a result is injected, decoding turns greedy — sampling at temperature
  actively randomises number/name copying;
- tokens that appear in the injected result get a logit boost (`copy_boost`);
- tokens already generated in the reply are penalised (`repetition_penalty`) —
  greedy decoding on this model loops without it.

Measured honestly (2026-07-20, smol_run1_warm): the SmolLM2-era model copies
natively — 98% faithfulness with both knobs off, 100% with the light defaults.
The knobs are retained as cheap insurance, not as a crutch.

Generation runs on a per-turn KV cache (model.transformer.KVCache): the prompt
is prefilled once, each later step feeds only the newest token, and a result
injection prefills its chunk in one forward pass.
"""
import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.transformer import KVCache


@dataclass
class DispatchResult:
    response: str            # the model's final natural-language reply
    tool: str | None         # which tool was called, if any
    tool_result: str | None  # the real data the tool returned


# a handler takes the user's message and returns a result string to inject
Handler = Callable[[str], str]

# smart typography → ASCII, applied to every incoming message. macOS quote
# substitution turns the typed ' into ’ (U+2019), which no extraction regex
# matches (seen live: "Find kai’s resume" searched for the possessive verbatim).
# ASCII also matches the training data, so the model sees its own distribution.
_TYPOGRAPHY = str.maketrans({
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en/em dash
})


def normalize_typography(message: str) -> str:
    return message.translate(_TYPOGRAPHY)


class ToolDispatcher:
    def __init__(
        self,
        model,
        tokenizer,
        handlers: dict[str, Handler],
        device: torch.device,
        max_new_tokens: int = 120,
        temperature: float = 0.7,
        top_k: int | None = 40,
        copy_boost: float = 2.0,  # SmolLM2-era re-sweep (2026-07-20): this model copies
                                  # NATIVELY — 98% faithfulness at boost 0, 100% at boost 2.
                                  # The OPT era needed boost 8 to reach 88%; that aggression
                                  # now just risks Goodhart glue. Light touch is enough.
        repetition_penalty: float = 1.1,
        verbatim: "dict[str, Callable[[str], str]] | None" = None,
        reprompt: "dict[str, Callable[[str], str]] | None" = None,
        pre_router: "Callable[[str], str | None] | None" = None,
        fallback_router: "Callable[[str], str | None] | None" = None,
        route_confidence: float = 0.6,
        memory=None,  # optional ChromaMemory — records every completed exchange
    ):
        self.model = model.eval()
        self.tok = tokenizer
        self.handlers = handlers
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.copy_boost = copy_boost  # 0 disables the copy bias
        self.repetition_penalty = repetition_penalty  # 1.0 disables
        # tools whose reply is templated from the real result instead of
        # model-generated — fact-heavy tools (calendar, reminders, notes) where
        # a 350M paraphrase garbles proper nouns. The model still does the
        # routing; it just doesn't get to rewrite the facts.
        self.verbatim = verbatim or {}
        # the third reply mode (after verbatim templates and inject-and-resume):
        # tool → function turning the real result into a NEW chat prompt, which
        # the model answers with its preserved instruct ability. For tools whose
        # result should be UNDERSTOOD rather than recited — screen OCR text
        # wants "you're looking at a checkout page", not a read-back.
        self.reprompt = reprompt or {}
        # consulted BEFORE the model — forces a tool for queries the model
        # can't route but reliably mis-routes (file search has no tool token
        # yet argmaxes reminders/launcher). Returns a tool name or None.
        self.pre_router = pre_router
        # consulted only when the model emits NO tool call — safety net for
        # phrasings outside the training distribution ("Play X on Spotify"
        # produced gibberish chat instead of routing)
        self.fallback_router = fallback_router
        # a tool call is only accepted if the model is at least this sure.
        # Re-measured on smol_run1_warm (2026-07-20): genuine tool prompts
        # route at p=0.73-0.999 (min: "What's playing on Spotify?" 0.734,
        # "Check my reminders." 0.786 — the old 0.9 gate wrongly blocked
        # both); chat prompts never have a tool as top token at all. 0.6
        # clears every real route with margin while still blocking the
        # flat-mush regime (p≈0.02-0.04) seen in undertrained checkpoints.
        self.route_confidence = route_confidence
        self.memory = memory
        # built from the tool-token registry, NOT by scanning an id range —
        # in the SmolLM2 tokenizer the tool ids live at 49152+, and the old
        # range(100) scan would silently find none (disabling the gate's
        # tool-token masking without any error)
        from model.tokenizer import TOOL_TOKENS
        self._tool_ids = {tokenizer.tool_token_id(name) for name in TOOL_TOKENS}

    def _next_token(self, seq: list[int], cache: KVCache | None = None,
                    greedy: bool = False,
                    boost_ids: set[int] | None = None,
                    penalize_ids: set[int] | None = None,
                    gate_tools: bool = False,
                    mask_tools: bool = False) -> int:
        if cache is not None and len(seq) <= self.model.config.context_len:
            # feed only what the cache hasn't seen — one token on a normal
            # step, the whole injected chunk right after a tool result
            idx = torch.tensor([seq[cache.seq_len:]], dtype=torch.long,
                               device=self.device)
            logits, _ = self.model(idx, cache=cache)
        else:
            # overflow (or no cache): the pre-cache windowed path. the trimmed
            # window restarts positions at 0, so the cache can't serve it —
            # and seq only grows, so once here, every later step is here too.
            context = seq[-self.model.config.context_len:]
            idx = torch.tensor([context], dtype=torch.long, device=self.device)
            logits, _ = self.model(idx)
        logits = logits[0, -1, :]
        if mask_tools:
            # generation that must stay pure text (the reprompt pass — a tool
            # call emitted THERE would recurse the dispatch loop)
            logits[list(self._tool_ids)] = float("-inf")
        if gate_tools:
            top = int(torch.argmax(logits))
            if self.tok.is_tool_call(top) is not None:
                prob = F.softmax(logits.float(), dim=-1)[top]
                if prob < self.route_confidence:
                    # not confident it's a tool request — treat as chat
                    logits[list(self._tool_ids)] = float("-inf")
        if boost_ids and self.copy_boost:
            # only boost result tokens not yet copied into the reply — a boosted
            # token that keeps its boost after emission out-muscles the repetition
            # penalty and loops ("granted granted granted...")
            remaining = boost_ids - penalize_ids if penalize_ids else boost_ids
            if remaining:
                logits[list(remaining)] += self.copy_boost
        if penalize_ids and self.repetition_penalty != 1.0:
            for tok_id in penalize_ids:
                if logits[tok_id] > 0:
                    logits[tok_id] = logits[tok_id] / self.repetition_penalty
                else:
                    logits[tok_id] = logits[tok_id] * self.repetition_penalty
        if greedy:
            return int(torch.argmax(logits))
        logits = logits / self.temperature
        if self.top_k is not None:
            v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))

    def _describe(self, prompt: str) -> str:
        """One plain chat generation over `prompt` — the reprompt pass. Same
        decoding policy as no-tool chat (greedy first token, sampled middle),
        but with tool tokens hard-masked every step."""
        from model import chat_format
        seq = chat_format.prime_ids(self.tok, prompt)
        start = len(seq)
        cache = KVCache(self.model.config.n_layers)
        for step in range(self.max_new_tokens):
            next_id = self._next_token(seq, cache=cache, greedy=step == 0,
                                       mask_tools=True)
            if next_id == self.tok.eos_id:
                break
            seq.append(next_id)
        return self.tok.decode(seq[start:], skip_special=True).strip()

    def _reply_via_reprompt(self, tool: str, tool_result: str) -> "DispatchResult":
        prompt = self.reprompt[tool](tool_result)
        return DispatchResult(response=self._describe(prompt),
                              tool=tool, tool_result=tool_result)

    def _finish_via_router(self, tool: str, message: str) -> "DispatchResult":
        """A router (pre- or fallback) picked the tool without the model. Run
        it and reply in the tool's mode: reprompt tools describe, verbatim
        tools template, the rest return the raw result — the model didn't route
        it, so it doesn't get to wrap it."""
        tool_result = self._run_tool(tool, message)
        if tool in self.reprompt:
            return self._reply_via_reprompt(tool, tool_result)
        template = self.verbatim.get(tool)
        response = template(tool_result) if template else tool_result
        return DispatchResult(response=response, tool=tool,
                              tool_result=tool_result)

    def _run_tool(self, tool: str, message: str) -> str:
        handler = self.handlers.get(tool)
        if handler is None:
            return f"{tool} is not available yet"
        try:
            return handler(message)
        except Exception as exc:  # a failing tool shouldn't crash the whole turn
            return f"{tool} error: {exc}"

    @torch.no_grad()
    def respond(self, message: str) -> DispatchResult:
        message = normalize_typography(message)
        result = self._respond(message)
        if self.memory is not None:
            try:
                self.memory.record(message, result.response, result.tool)
            except Exception:
                pass  # memory is a convenience — never fail the turn over it
        return result

    def _respond(self, message: str) -> DispatchResult:
        # degenerate input first: a message with no letters or digits (". . .",
        # "???") gives the model nothing to condition on, and it free-associates
        # training-shaped junk (seen live: dots in → a Python fruit list out).
        # The voice path already guards this ("Didn't catch that"); this is the
        # typed equivalent. No generation, no tool.
        if not re.search(r"[^\W_]", message):
            return DispatchResult(
                response="Didn't catch that — what do you need?",
                tool=None, tool_result=None)

        # pre-router first: for queries the model can't route but reliably
        # mis-routes (file search), force the tool with no generation at all
        if self.pre_router is not None:
            forced = self.pre_router(message)
            if forced is not None:
                return self._finish_via_router(forced, message)

        # ChatML priming via the shared format module — the first generated
        # token (the routing decision) lands exactly after "assistant\n",
        # identically to how every training example was framed
        from model import chat_format
        seq = chat_format.prime_ids(self.tok, message)
        response_start = len(seq)  # where the final reply begins (updated after a tool call)
        tool_used: str | None = None
        tool_result: str | None = None
        result_ids: set[int] = set()  # injected result tokens → copy-bias targets
        reply_ids: set[int] = set()   # reply tokens so far → repetition-penalty targets
        cache = KVCache(self.model.config.n_layers)  # per-turn — never shared

        for step in range(self.max_new_tokens):
            # the first token is the routing decision — greedy, so the same
            # question always reaches the same tool (sampling made routing
            # flaky on marginal phrasings). Post-injection stays greedy too;
            # sampling only shapes the middle of no-tool chat replies.
            next_id = self._next_token(seq, cache=cache,
                                       greedy=step == 0 or tool_used is not None,
                                       boost_ids=result_ids, penalize_ids=reply_ids,
                                       gate_tools=step == 0)
            tool = self.tok.is_tool_call(next_id)

            if step == 0 and tool is None and self.fallback_router is not None:
                # the model declined to route on its routing token. if the
                # keyword net catches the message, run the tool NOW — before
                # the cache work, a full chat reply was generated here only to
                # be discarded. (a model that would have emitted a mid-reply
                # call loses that chance, but the fallback patterns are narrow
                # and training always puts the call first.)
                routed = self.fallback_router(message)
                if routed is not None:
                    return self._finish_via_router(routed, message)

            # (A "files-veto" stopgap lived here: the files row was warm-started
            # alone and over-fired on other tools' phrasings, so a files route
            # was treated as provisional. Run 7 retrained routing with hard
            # negatives and `model/eval_routing.py` now measures ZERO files
            # over-fires across 30 cases, including every prompt that motivated
            # the veto — so it was removed 2026-07-29. If a future checkpoint
            # regresses, that eval is what will catch it.)

            if tool is not None and tool_used is None:
                # intercept: keep the [CALL: tool] token, then inject the *real* result
                # instead of letting the model generate its fabricated one
                seq.append(next_id)
                tool_used = tool
                tool_result = self._run_tool(tool, message)
                if tool in self.reprompt:
                    # reprompt tool: the result becomes a fresh chat prompt the
                    # model answers with its instruct ability — understand, not
                    # recite (inject-and-resume + copy bias would read it back)
                    return self._reply_via_reprompt(tool, tool_result)
                template = self.verbatim.get(tool)
                if template is not None:
                    # verbatim tool: the reply is the templated result, word-perfect
                    return DispatchResult(response=template(tool_result),
                                          tool=tool_used, tool_result=tool_result)
                encoded_result = self.tok.encode(tool_result)
                result_ids = set(encoded_result)
                seq.append(self.tok.result_start_id)
                seq.extend(encoded_result)
                seq.append(self.tok.result_end_id)
                response_start = len(seq)
                continue

            if next_id == self.tok.eos_id:
                break
            seq.append(next_id)
            if tool_used is not None:
                reply_ids.add(next_id)

        response = self.tok.decode(seq[response_start:], skip_special=True).strip()
        return DispatchResult(response=response, tool=tool_used, tool_result=tool_result)
