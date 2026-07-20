import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.config import ModelConfig


class RMSNorm(nn.Module):
    # modern replacement for LayerNorm (used by Llama, SmolLM2, Qwen...).
    # LayerNorm subtracts the mean and divides by the standard deviation;
    # RMSNorm skips the mean subtraction and just divides by the root mean
    # square, then scales by a learned weight. Simpler, slightly faster, and
    # empirically just as good — and it has no bias term at all.

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in float32 for numerical stability (matters in bf16 training),
        # then cast back to the input dtype — same as HF's LlamaRMSNorm.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # RoPE helper. CRITICAL: this is the HF/Llama "split-half" convention —
    # the rotation pairs dimension i with dimension i + head_dim/2.
    # The original RoPE paper pairs (2i, 2i+1) instead ("interleaved").
    # The pretrained Q/K weights were trained under THIS convention; using the
    # interleaved one produces plausible-but-wrong attention — silent
    # corruption that only a logit-equivalence test catches.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    # rotary position embeddings (RoPE): instead of ADDING a learned position
    # vector to the token embedding (what GPT-2/OPT did), RoPE ROTATES each
    # query/key vector by an angle proportional to its position. relative
    # offsets between tokens then fall out of the dot product naturally.
    # there are no learned weights — position information is pure geometry.

    def __init__(self, head_dim: int, rope_theta: float):
        super().__init__()
        # each pair of dims rotates at its own frequency: low dims spin fast
        # (fine-grained local order), high dims spin slowly (long-range order)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (T,) → cos/sin: (T, head_dim)
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # duplicated to cover both halves
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, n_heads, T, head_dim); cos/sin: (T, head_dim) → broadcast
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class KVCache:
    # generation bottleneck fix: without a cache, emitting token N re-runs
    # attention over all N-1 previous tokens in every layer, so a T-token reply
    # costs O(T²) forward work. The keys/values of past tokens never change
    # (causal attention — nothing behind you can see you), so we store each
    # layer's rotated k/v once and each new step only computes its own q/k/v.
    # Not an nn.Module: no weights, just per-turn scratch state.

    def __init__(self, n_layers: int):
        self.layers: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * n_layers

    @property
    def seq_len(self) -> int:
        entry = self.layers[0]
        return 0 if entry is None else entry[0].shape[2]

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # append this step's k/v to the layer's cache, return the full history
        entry = self.layers[layer]
        if entry is not None:
            k = torch.cat((entry[0], k), dim=2)
            v = torch.cat((entry[1], v), dim=2)
        self.layers[layer] = (k, v)
        return k, v


class CausalSelfAttention(nn.Module):
    # attention lets every token "look at" every other token and decide how
    # much to borrow from each one. "causal" = a token can only look backwards.
    # position information enters HERE via RoPE (rotating q and k) rather than
    # being baked into the embeddings like the old architecture did.

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout

        # one projection producing queries, keys, and values all at once.
        # SmolLM2 stores q/k/v as separate matrices; the transplant script
        # concatenates them into this fused weight (valid because all three
        # are the same size under plain multi-head attention).
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryEmbedding(self.d_head, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
        past: int = 0,
    ) -> torch.Tensor:
        # `past` (cache length BEFORE this chunk) comes from the caller — it
        # can't be read off the cache here, because earlier layers have
        # already appended this chunk to their entries by the time later
        # layers run.
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # rotate q and k by their ABSOLUTE positions — with a cache, this
        # chunk starts at position past, not 0. Keys are cached post-rotation,
        # so each token is rotated exactly once, at its true position.
        cos, sin = self.rope(torch.arange(past, past + T, device=x.device))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer, k, v)

        dropout_p = self.dropout if self.training else 0.0
        if past == 0:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p,
                                               is_causal=T > 1)
        elif T == 1:
            # one new query attending to the whole cache — no mask needed
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # multi-token chunk on top of an existing cache (the dispatcher's
            # result injection). SDPA's is_causal aligns the diagonal top-left
            # (query i sees keys ≤ i), but here query i sits at absolute
            # position past+i — build the bottom-right-aligned mask explicitly.
            S = past + T
            mask = (torch.arange(S, device=x.device)[None, :]
                    <= (past + torch.arange(T, device=x.device))[:, None])
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    # the modern feed-forward sublayer (Llama-family). instead of one linear
    # + GELU, two parallel projections are computed from the same input:
    #   gate — passed through SiLU (the "switch")
    #   up   — the actual content
    # multiplied elementwise, then projected back down. the gate learns which
    # features to let through — a smarter nonlinearity for the same FLOPs.

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    # one full transformer layer: attention then feed-forward, pre-norm style
    # (normalize before each sublayer), residual connections throughout.
    # same skeleton as the old architecture — only the parts changed:
    # LayerNorm → RMSNorm, GELU MLP → SwiGLU, positions → RoPE (in attention).

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, config.rms_norm_eps)   # pre-attention norm
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model, config.rms_norm_eps)   # pre-mlp norm
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
        past: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache=cache, layer=layer, past=past)
        x = x + self.mlp(self.ln2(x))
        return x


class DesktopHelperLM(nn.Module):
    # the full language model. takes token ids, returns next-token logits.
    #
    #   token ids → embeddings (no positional embeddings — RoPE handles
    #   position inside attention) → n_layers blocks → final RMSNorm →
    #   lm head (tied to the embedding matrix)

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying: the "meaning" of a token is the same whether encoding
        # it as input or predicting it as output. SmolLM2 ties too
        # (tie_word_embeddings=true), so the transplant stays consistent.
        self.lm_head.weight = self.token_emb.weight

        # when True, recompute each block's activations during backward
        # instead of storing them — trades ~30% speed for a large activation
        # memory saving. enabled by train.py; irrelevant at inference.
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # only matters for the 11 tool-token embedding rows and any layer the
        # transplant doesn't cover — everything else is overwritten by
        # pretrained weights in load_base.py.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        past = cache.seq_len if cache is not None else 0
        assert past + T <= self.config.context_len, (
            f"sequence length {past + T} exceeds context length {self.config.context_len}"
        )

        x = self.token_emb(idx)  # no positional add — RoPE lives in attention

        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, cache=cache, layer=i, past=past)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def num_params(self) -> int:
        # subtract the tied embedding weights so they're not counted twice
        total = sum(p.numel() for p in self.parameters())
        return total - self.token_emb.weight.numel()

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        # autoregressively sample one token at a time. the KV cache prefills
        # the prompt once, then each step feeds only the newest token. if the
        # sequence outgrows context_len we drop the cache and fall back to
        # windowed re-processing — the trimmed window restarts RoPE positions
        # at 0, which a cache built at absolute positions can't serve.
        cache: KVCache | None = KVCache(self.config.n_layers)
        new_tokens = idx
        for _ in range(max_new_tokens):
            if cache is not None and \
                    cache.seq_len + new_tokens.shape[1] > self.config.context_len:
                cache = None
            if cache is not None:
                logits, _ = self(new_tokens, cache=cache)
            else:
                logits, _ = self(idx[:, -self.config.context_len:])

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            new_tokens = idx_next

            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx
