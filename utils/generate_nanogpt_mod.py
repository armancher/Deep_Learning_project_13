import os
import sys
import argparse
import torch

# Ensure project root visible
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.nanogpt_wrapper import create_nanogpt_model
from tokenizers import get_tokenizer


# ---------------------------------------
# Sampling function
# ---------------------------------------
def sample(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond, idx_cond)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    tokenizer_name = ckpt["tokenizer_name"]
    config = ckpt["config"]
    vocab_size = config["vocab_size"]
    vocab_size_arg = ckpt.get("vocab_size_arg", vocab_size)

    print(f"Tokenizer used: {tokenizer_name}")

    # Load tokenizer
    if tokenizer_name == "byt5":
        tok = get_tokenizer("byt5")
    else:
        # For BPE/char we rebuild vocab from prompt (tiny)
        # but this assumes merges/id maps were stored —
        # simple project version → train on prompt's chars
        tok = get_tokenizer(tokenizer_name, text=args.prompt, vocab_size=vocab_size_arg)

    # Create model
    model, _ = create_nanogpt_model(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    # Encode prompt
    encoded = tok.encode(args.prompt)
    idx = torch.tensor(encoded, dtype=torch.long, device=device)[None, :]

    print("\nGenerating...\n")

    out = sample(
        model,
        idx,
        max_new_tokens=args.max_new_tokens,
        block_size=config["block_size"],
        temperature=args.temperature,
        top_k=args.top_k,
    )

    tokens = out[0].tolist()
    text = tok.decode(tokens)
    print(text)


if __name__ == "__main__":
    main()
