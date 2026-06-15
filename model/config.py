from dataclasses import dataclass


# all the numbers that define the model's size and shape.
# changing these is the main lever for trading off quality vs. speed vs. memory.
# the defaults give us roughly 300m parameters — a reasonable starting point.
@dataclass
class ModelConfig:
    # how many unique tokens the model knows about.
    # 32k is standard for a bpe tokenizer — large enough to cover english well
    # without making the embedding table too big.
    vocab_size: int = 32000

    # the longest sequence the model can process at once.
    # tokens beyond this get cut off — the model has no memory of them.
    # 1024 tokens is ~750 words, enough for most desktop assistant interactions.
    context_len: int = 1024

    # the "width" of the model. every token is represented as a vector of this size.
    # larger = more expressive, but more memory and slower to train.
    d_model: int = 1024

    # how many attention heads to split each layer into.
    # each head learns to attend to different kinds of relationships.
    # must divide evenly into d_model (1024 / 16 = 64 dims per head).
    n_heads: int = 16

    # how many transformer blocks to stack.
    # more layers = more reasoning steps per forward pass, but more memory.
    n_layers: int = 24

    # width of the feed-forward sublayer inside each block.
    # convention is 4x d_model — this is where most of the model's "storage" lives.
    d_ff: int = 4096

    # fraction of activations randomly zeroed during training.
    # forces the model to not rely on any single neuron — reduces overfitting.
    # set to 0.0 at inference time automatically.
    dropout: float = 0.1
