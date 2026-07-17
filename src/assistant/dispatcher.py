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
    ):
        self.model = model.eval()
        self.tok = tokenizer
        self.handlers = handlers
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

    def _next_token(self, seq: list[int]) -> int:
        context = seq[-self.model.config.context_len:]
        idx = torch.tensor([context], dtype=torch.long, device=self.device)
        logits, _ = self.model(idx)
        logits = logits[0, -1, :] / self.temperature
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

        for _ in range(self.max_new_tokens):
            next_id = self._next_token(seq)
            tool = self.tok.is_tool_call(next_id)

            if tool is not None and tool_used is None:
                # intercept: keep the [CALL: tool] token, then inject the *real* result
                # instead of letting the model generate its fabricated one
                seq.append(next_id)
                tool_used = tool
                tool_result = self._run_tool(tool, message)
                seq.append(self.tok.result_start_id)
                seq.extend(self.tok.encode(tool_result))
                seq.append(self.tok.result_end_id)
                response_start = len(seq)
                continue

            if next_id == self.tok.eos_id:
                break
            seq.append(next_id)

        response = self.tok.decode(seq[response_start:], skip_special=True).strip()
        return DispatchResult(response=response, tool=tool_used, tool_result=tool_result)
