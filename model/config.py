from dataclasses import dataclass


# all the numbers that define the model's size and shape.
# the defaults match SmolLM2-1.7B exactly — they must, because we transplant
# its pretrained weights into this architecture (see model/load_base.py).
# change them only together with a different base model.
@dataclass
class ModelConfig:
    # how many unique tokens the model knows about.
    # SmolLM2's tokenizer has 49152 base tokens; we append 11 tool-call
    # special tokens, so the real value comes from the tokenizer at build time
    # via from_tokenizer() — never trust this default for a real model.
    vocab_size: int = 49163

    # the longest sequence the model can process at once.
    # SmolLM2 supports 8192, but we train/infer at 1024 to keep VRAM (Colab)
    # and latency (MPS) manageable — a desktop-assistant turn fits easily.
    context_len: int = 1024

    # the "width" of the model. every token is represented as a vector of this size.
    d_model: int = 2048

    # how many attention heads to split each layer into.
    # 2048 / 32 = 64 dims per head — SmolLM2 uses standard multi-head
    # attention (every head has its own K/V — no grouped-query trickery).
    n_heads: int = 32

    # how many transformer blocks to stack.
    n_layers: int = 24

    # width of the SwiGLU feed-forward sublayer inside each block.
    d_ff: int = 8192

    # dropout is 0: SmolLM2 was pretrained without it, and our fine-tuning is
    # a light-touch adaptation — injecting noise the base model never saw
    # works against preserving its abilities.
    dropout: float = 0.0

    # RMSNorm epsilon. SmolLM2 uses 1e-5 — some implementations default to
    # 1e-6, which is exactly the kind of silent mismatch the logit-equivalence
    # test in tests/test_transplant.py exists to catch.
    rms_norm_eps: float = 1e-5

    # base frequency for rotary position embeddings (RoPE).
    # SmolLM2 uses 130000 (Llama-3-style long-context scaling), not the
    # original RoPE paper's 10000.
    rope_theta: float = 130000.0

    @classmethod
    def from_tokenizer(cls, tokenizer, **overrides) -> "ModelConfig":
        # build a config whose vocab_size always matches the tokenizer,
        # INCLUDING appended special tokens — an undersized embedding table
        # would make tool-token ids index out of range.
        return cls(vocab_size=tokenizer.vocab_size, **overrides)
