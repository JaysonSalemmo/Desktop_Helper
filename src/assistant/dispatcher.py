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

Measured honestly (2026-07-17, epoch_08): these make replies deterministic and
slightly more likely to borrow real digits, but they cannot make this
checkpoint faithful — it never learned to copy from RESULT context. The real
fix is training-side (RESULT-masked loss + high-entropy result content).
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DispatchResult:
    response: str            # the model's final natural-language reply
    tool: str | None         # which tool was called, if any
    tool_result: str | None  # the real data the tool returned


# a handler takes the user's message and returns a result string to inject
Handler = Callable[[str], str]


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
        copy_boost: float = 8.0,  # re-sweep per checkpoint — the optimum rises as the
                                  # copy circuit strengthens (run-3 epoch_08 → 4,
                                  # run-4 epoch_12 → 8 under the decaying scheme).
                                  # Higher scores the eval but degrades glue (Goodhart).
        repetition_penalty: float = 1.3,
        verbatim: "dict[str, Callable[[str], str]] | None" = None,
        fallback_router: "Callable[[str], str | None] | None" = None,
        route_confidence: float = 0.9,
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
        # consulted only when the model emits NO tool call — safety net for
        # phrasings outside the training distribution ("Play X on Spotify"
        # produced gibberish chat instead of routing)
        self.fallback_router = fallback_router
        # a tool call is only accepted if the model is at least this sure.
        # Measured on epoch_16: genuine tool prompts route at p≈1.000, chat
        # tops out at p≈0.68 ("Hello." wanted spotify at p=0.13) — 0.9 splits
        # them cleanly. Below the gate, tool tokens are masked and the turn
        # becomes plain chat.
        self.route_confidence = route_confidence
        self.memory = memory
        self._tool_ids = {i for i in range(100)
                          if tokenizer.is_tool_call(i) is not None}

    def _next_token(self, seq: list[int], greedy: bool = False,
                    boost_ids: set[int] | None = None,
                    penalize_ids: set[int] | None = None,
                    gate_tools: bool = False) -> int:
        context = seq[-self.model.config.context_len:]
        idx = torch.tensor([context], dtype=torch.long, device=self.device)
        logits, _ = self.model(idx)
        logits = logits[0, -1, :]
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
        result = self._respond(message)
        if self.memory is not None:
            try:
                self.memory.record(message, result.response, result.tool)
            except Exception:
                pass  # memory is a convenience — never fail the turn over it
        return result

    def _respond(self, message: str) -> DispatchResult:
        # prime with the training format: <bos> prompt \n
        seq = [self.tok.bos_id] + self.tok.encode(f"{message}\n")
        response_start = len(seq)  # where the final reply begins (updated after a tool call)
        tool_used: str | None = None
        tool_result: str | None = None
        result_ids: set[int] = set()  # injected result tokens → copy-bias targets
        reply_ids: set[int] = set()   # reply tokens so far → repetition-penalty targets

        for step in range(self.max_new_tokens):
            # the first token is the routing decision — greedy, so the same
            # question always reaches the same tool (sampling made routing
            # flaky on marginal phrasings). Post-injection stays greedy too;
            # sampling only shapes the middle of no-tool chat replies.
            next_id = self._next_token(seq, greedy=step == 0 or tool_used is not None,
                                       boost_ids=result_ids, penalize_ids=reply_ids,
                                       gate_tools=step == 0)
            tool = self.tok.is_tool_call(next_id)

            if tool is not None and tool_used is None:
                # intercept: keep the [CALL: tool] token, then inject the *real* result
                # instead of letting the model generate its fabricated one
                seq.append(next_id)
                tool_used = tool
                tool_result = self._run_tool(tool, message)
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

        if tool_used is None and self.fallback_router is not None:
            tool = self.fallback_router(message)
            if tool is not None:
                tool_result = self._run_tool(tool, message)
                template = self.verbatim.get(tool)
                # no template → return the raw result; the model already
                # declined to route, so it doesn't get to wrap this one
                response = template(tool_result) if template else tool_result
                return DispatchResult(response=response, tool=tool,
                                      tool_result=tool_result)

        response = self.tok.decode(seq[response_start:], skip_special=True).strip()
        return DispatchResult(response=response, tool=tool_used, tool_result=tool_result)
