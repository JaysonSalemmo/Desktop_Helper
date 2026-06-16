import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    # attention lets every token "look at" every other token and decide
    # how much to borrow from each one. "causal" means a token can only
    # look backwards in time — it cannot see tokens that come after it.
    # this is enforced by a mask so the model can't cheat during training.
    #
    # "multi-head" means we do this in parallel n_heads times, each head
    # looking for different kinds of relationships (syntax, semantics, etc).

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads  # dimensions per head
        self.d_model = config.d_model
        self.dropout = config.dropout

        # one projection that produces queries, keys, and values all at once.
        # splitting into 3 separate linears would give the same result but this is faster.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)

        # mixes the outputs of all heads back into a single d_model vector
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, d_model

        # project to queries, keys, values and split along the last dim
        q, k, v = self.qkv(x).split(C, dim=2)

        # reshape so each head gets its own slice: (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product attention with flash attention under the hood.
        # is_causal=True applies the look-ahead mask so token i can't see token i+1.
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)

        # merge all heads back into (B, T, d_model) and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    # the feed-forward sublayer. applied independently to each token position
    # after attention. this is where most of the model's "knowledge" is stored —
    # attention moves information around, mlp processes it.
    #
    # structure: d_model → d_ff → d_model (expand then contract)

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)   # expand
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)   # contract
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gelu is a smooth approximation of relu — slightly better in practice
        # for language models. dropout applied after the second linear.
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    # one full transformer layer: attention then feed-forward.
    # we use "pre-norm" style — layernorm is applied before each sublayer,
    # not after. this trains more stably than the original post-norm design.
    #
    # residual connections (x = x + sublayer(x)) let gradients flow directly
    # back through the network during training, preventing vanishing gradients.

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)   # norm before attention
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)   # norm before mlp
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attend, then add back to residual stream
        x = x + self.mlp(self.ln2(x))    # process, then add back to residual stream
        return x


class DesktopHelperLM(nn.Module):
    # the full language model. takes a sequence of token ids, returns logits
    # (unnormalized scores) over the vocabulary for the next token at each position.
    #
    # structure:
    #   token ids → embeddings + positional embeddings
    #   → n_layers transformer blocks
    #   → final layernorm
    #   → linear projection to vocab_size (the "lm head")

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # turns each token id into a learned d_model-dimensional vector
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # adds position information — the model has no built-in sense of order,
        # so we learn a separate embedding for each position 0..context_len-1
        self.pos_emb = nn.Embedding(config.context_len, config.d_model)

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)  # final norm before the lm head

        # projects from d_model back to vocab_size to get next-token scores
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying: share the token embedding matrix with the lm head.
        # the intuition is that the "meaning" of a token should be the same
        # whether we're encoding it as input or predicting it as output.
        # also saves ~32m parameters.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # small normal init is standard for transformers.
        # std=0.02 keeps activations from exploding at the start of training.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.config.context_len, (
            f"sequence length {T} exceeds context length {self.config.context_len}"
        )

        # build position indices [0, 1, 2, ..., T-1] on the same device as the input
        pos = torch.arange(T, device=idx.device)

        # combine token meaning and position information
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # shape: (B, T, vocab_size)

        # cross-entropy loss is only computed when training targets are provided
        loss = None
        if targets is not None:
            # flatten to (B*T, vocab_size) vs (B*T,) for the loss function
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
        # autoregressively sample one token at a time.
        # temperature > 1 = more random, temperature < 1 = more focused.
        # top_k limits sampling to the k most likely tokens at each step.
        # eos_id, if given, stops generation once every sequence in the batch
        # has produced it — for batch size 1 (the normal chat case) this just
        # means "stop as soon as the model says it's done".
        #
        # note: this does not stop on [CALL: tool] tokens. the phase 4 dispatcher
        # will run its own token-by-token loop so it can pause, run the tool, and
        # feed the result back in before continuing — this method is for plain
        # text completion and quick testing.
        for _ in range(max_new_tokens):
            # trim the context window if the sequence has grown too long
            idx_cond = idx[:, -self.config.context_len:]
            logits, _ = self(idx_cond)

            # take only the last position's logits — that's the next token prediction
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # mask out everything below the kth highest score
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx
