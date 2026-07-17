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
        copy_boost: float = 4.0,  # swept 2026-07-17 on run-3 epoch_08: metric keeps
                                  # rising past 8 but replies degrade into token salad
                                  # (Goodhart) — 4 is the fluency/faithfulness balance
        repetition_penalty: float = 1.3,
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

    def _next_token(self, seq: list[int], greedy: bool = False,
                    boost_ids: set[int] | None = None,
                    penalize_ids: set[int] | None = None) -> int:
        context = seq[-self.model.config.context_len:]
        idx = torch.tensor([context], dtype=torch.long, device=self.device)
        logits, _ = self.model(idx)
        logits = logits[0, -1, :]
        if boost_ids and self.copy_boost:
            logits[list(boost_ids)] += self.copy_boost
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
        # prime with the training format: <bos> prompt \n
        seq = [self.tok.bos_id] + self.tok.encode(f"{message}\n")
        response_start = len(seq)  # where the final reply begins (updated after a tool call)
        tool_used: str | None = None
        tool_result: str | None = None
        result_ids: set[int] = set()  # injected result tokens → copy-bias targets
        reply_ids: set[int] = set()   # reply tokens so far → repetition-penalty targets

        for _ in range(self.max_new_tokens):
            # post-injection the reply must echo real data, so decode greedily
            # with the copy bias; sampling is only for the routing/no-tool path
            next_id = self._next_token(seq, greedy=tool_used is not None,
                                       boost_ids=result_ids, penalize_ids=reply_ids)
            tool = self.tok.is_tool_call(next_id)

            if tool is not None and tool_used is None:
                # intercept: keep the [CALL: tool] token, then inject the *real* result
                # instead of letting the model generate its fabricated one
                seq.append(next_id)
                tool_used = tool
                tool_result = self._run_tool(tool, message)
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
