import os
import sys

# Make sure we can import nanoGPT.model from our project root
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
NANOGPT_DIR = os.path.join(PROJECT_ROOT, "nanoGPT")

if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)

from nanoGPT.model import GPT, GPTConfig


def create_nanogpt_model(
    vocab_size: int,
    block_size: int = 256,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
):
    """
    Small helper to create a nanoGPT model with given vocabulary size.
    You can tweak the default depth/width later.
    """
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = GPT(config)
    return model, config
