import torch


def get_device(require_cuda: bool = False) -> torch.device:
    # picks the best available device.
    # for quick sanity checks (forward pass, shape tests) any device works.
    # for actual training, set require_cuda=True — this will refuse to run
    # on a mac or cpu so you don't accidentally kick off a multi-hour training
    # run on your laptop and brick it.
    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_cuda:
        raise RuntimeError(
            "cuda is not available on this machine. "
            "training must run on the desktop gpu — not on the mac. "
            "if you're just testing a forward pass, call get_device(require_cuda=False)."
        )

    # mps = apple silicon gpu. fine for development and inference, not for training.
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
