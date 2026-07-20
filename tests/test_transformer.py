import math

import torch

from model import ModelConfig, DesktopHelperLM
from model.transformer import KVCache


def _tiny_config() -> ModelConfig:
    # small enough to run instantly on cpu, large enough to exercise every shape
    return ModelConfig(vocab_size=64, context_len=16, d_model=32, n_heads=4, n_layers=2, d_ff=64)


def test_forward_pass_shapes():
    config = _tiny_config()
    model = DesktopHelperLM(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is None


def test_loss_near_random_baseline_at_init():
    config = _tiny_config()
    model = DesktopHelperLM(config)
    idx = torch.randint(0, config.vocab_size, (4, 16))
    targets = torch.randint(0, config.vocab_size, (4, 16))
    _, loss = model(idx, targets)
    expected = math.log(config.vocab_size)
    assert abs(loss.item() - expected) < 1.0


def test_weight_tying():
    model = DesktopHelperLM(_tiny_config())
    assert model.lm_head.weight is model.token_emb.weight


def test_num_params_does_not_double_count_tied_weights():
    model = DesktopHelperLM(_tiny_config())
    raw_total = sum(p.numel() for p in model.parameters())
    assert model.num_params() == raw_total - model.token_emb.weight.numel()


def test_context_length_is_enforced():
    config = _tiny_config()
    model = DesktopHelperLM(config)
    idx = torch.randint(0, config.vocab_size, (1, config.context_len + 1))
    try:
        model(idx)
        assert False, "expected an assertion error for exceeding context length"
    except AssertionError:
        pass


def test_generate_produces_requested_length():
    config = _tiny_config()
    model = DesktopHelperLM(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_generate_stops_early_at_eos():
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=2, context_len=16, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = DesktopHelperLM(config)
    idx = torch.randint(0, config.vocab_size, (1, 4))
    # high temperature flattens the distribution so eos (token 1) is reached
    # well before the 50-step cap, proving the loop actually breaks early
    out = model.generate(idx, max_new_tokens=50, temperature=1000.0, eos_id=1)
    assert out.shape[1] < 4 + 50


def test_kv_cache_matches_full_forward():
    # cached decoding must be a pure speedup: identical logits to feeding the
    # whole sequence at once. exercises all three attention paths — prefill
    # (is_causal), single-token step (no mask), and a multi-token chunk on an
    # existing cache (the dispatcher's result injection → explicit mask, the
    # branch where SDPA's top-left-aligned is_causal would silently be wrong)
    torch.manual_seed(0)
    config = _tiny_config()
    model = DesktopHelperLM(config).eval()
    ids = torch.randint(0, config.vocab_size, (1, 10))

    with torch.no_grad():
        full, _ = model(ids)

        cache = KVCache(config.n_layers)
        pieces = [model(ids[:, :4], cache=cache)[0]]        # prefill
        for t in range(4, 7):                               # one-token steps
            pieces.append(model(ids[:, t:t + 1], cache=cache)[0])
        pieces.append(model(ids[:, 7:], cache=cache)[0])    # chunk on cache

    assert cache.seq_len == 10
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-5)
