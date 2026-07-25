"""
The anti-silent-corruption suite — the institutional lesson of the OPT era,
where a transplant subtlety could only be discovered by burning an 8-hour
Colab run and getting garbage out.

The logit-equivalence test feeds identical token ids through (a) HuggingFace's
own reference implementation and (b) our from-scratch architecture with the
transplanted weights, and asserts the logits match. A wrong RoPE convention,
RMSNorm epsilon, fused-qkv ordering, or weight-mapping bug fails HERE, in
seconds, locally — never on Colab.

These tests are skipped unless the transplant checkpoint exists (it's 6GB+ and
gitignored) — run `uv run python -m model.load_base` first. CI/fresh clones
skip cleanly; the Colab notebook runs them before training.
"""
from pathlib import Path

import pytest
import torch

CHECKPOINT = Path("model/checkpoints/smol_transplant.pt")

needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="transplant checkpoint missing — run `uv run python -m model.load_base`",
)


@pytest.fixture(scope="module")
def tokenizer():
    from model.tokenizer import DesktopHelperTokenizer
    return DesktopHelperTokenizer.load()


@pytest.fixture(scope="module")
def our_model():
    from model.transformer import DesktopHelperLM
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = DesktopHelperLM(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.lm_head.weight = model.token_emb.weight
    model.eval()
    # checkpoint is bf16 (Colab RAM constraint); compute in fp32 for comparison
    return model.float()


def test_embedding_table_matches_tokenizer():
    # the bug the logit test CANNOT see: an undersized embedding table only
    # explodes when a tool-token id (49152+) is actually used
    from model.tokenizer import DesktopHelperTokenizer
    tok = DesktopHelperTokenizer.load()
    assert tok.vocab_size == 49164, f"expected 49152+12, got {tok.vocab_size}"
    assert tok.tool_token_id("calendar") >= 49152


def test_no_double_bos():
    from model.tokenizer import DesktopHelperTokenizer
    tok = DesktopHelperTokenizer.load()
    ids = tok.encode("hello")
    assert ids[0] != tok.bos_id, "encode() must not auto-prepend specials"


@needs_checkpoint
def test_logit_equivalence_with_reference(tokenizer, our_model):
    from transformers import AutoModelForCausalLM
    from model.tokenizer import BASE_MODEL_ID

    reference = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    reference.eval()

    # a real ChatML-framed prompt — exercises specials, text, and positions
    from model import chat_format
    ids = chat_format.prime_ids(tokenizer, "What's the weather like today?")
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        ref_logits = reference(x).logits            # (1, T, 49152)
        our_logits, _ = our_model(x)                # (1, T, 49163)

    # compare over the shared vocabulary — our extra 11 rows are fresh.
    # tolerance accounts for the checkpoint's bf16 rounding (~0.1-0.3 logit
    # noise vs the fp32 reference); a wrong RoPE convention or eps produces
    # diffs of 5+ with different argmaxes, so the separation stays clean
    ours = our_logits[..., : ref_logits.shape[-1]]
    max_diff = (ours - ref_logits).abs().max().item()
    assert max_diff < 1.0, (
        f"logits diverge from reference (max abs diff {max_diff:.2e}) — "
        "suspect RoPE convention, rms_norm_eps, or weight mapping"
    )
    # every position must agree on the most likely next token
    agree = (ours.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    assert agree == 1.0, f"argmax agreement only {agree:.1%}"


@needs_checkpoint
def test_transplanted_model_generates_language(tokenizer, our_model):
    # beyond numerics: the untouched instruct model should produce coherent
    # text for a trivial prompt (pre-fine-tuning sanity, not a quality bar)
    from model import chat_format
    ids = chat_format.prime_ids(tokenizer, "Say hello in one short sentence.")
    x = torch.tensor([ids], dtype=torch.long)
    out = our_model.generate(x, max_new_tokens=20, temperature=0.7, top_k=40,
                             eos_id=tokenizer.eos_id)
    text = tokenizer.decode(out[0, len(ids):].tolist())
    assert len(text.strip()) > 0
    print(f"\ntransplant says: {text.strip()!r}")


def test_embeddings_only_mode_moves_only_new_rows():
    # warm-start safety: after a training step, the pretrained rows and all
    # blocks must be bit-identical; only the appended rows may change
    import torch.nn.functional as F
    from model.config import ModelConfig
    from model.train import configure_embeddings_only
    from model.transformer import DesktopHelperLM

    cfg = ModelConfig(vocab_size=100, context_len=32, d_model=64, n_heads=4,
                      n_layers=2, d_ff=128)
    model = DesktopHelperLM(cfg)
    configure_embeddings_only(model, first_new_row=90)

    before_emb = model.token_emb.weight.detach().clone()
    before_block = model.blocks[0].attn.qkv.weight.detach().clone()

    # weight_decay=0 mirrors train.py's embeddings-only mode — AdamW decay
    # would move gradient-masked rows otherwise
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-2, weight_decay=0.0)
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(90, 100, (2, 16))  # targets in the new-row range
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    after_emb = model.token_emb.weight.detach()
    assert torch.equal(after_emb[:90], before_emb[:90]), "pretrained rows moved!"
    assert not torch.equal(after_emb[90:], before_emb[90:]), "new rows did not move"
    assert torch.equal(model.blocks[0].attn.qkv.weight.detach(), before_block), "blocks moved!"
