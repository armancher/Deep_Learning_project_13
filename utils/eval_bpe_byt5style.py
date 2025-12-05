import os
import sys
import argparse
import json
import math

import torch
from torch.utils.data import Dataset, DataLoader

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.nanogpt_wrapper import create_nanogpt_model
from tokenizers import get_tokenizer


class IDDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
        return x, y


def load_text(path: str) -> str:
    if os.path.isdir(path):
        txts = []
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".txt"):
                with open(os.path.join(path, fn), "r", encoding="utf-8") as f:
                    txts.append(f.read())
        return "\n\n".join(txts)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--metrics_out",
        default="results/bpe_byt5_experiment/metrics_bpe_eval.json",
        help="Path to output metrics JSON"
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Loading checkpoint from {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    config = ckpt["config"]
    arch = ckpt.get("arch", "unknown")
    tokenizer_name = ckpt.get("tokenizer_name", "bpe")

    vocab_size = config["vocab_size"]
    block_size = config["block_size"]

    # Load evaluation text first (needed for BPE training)
    full_text = load_text(args.dataset)
    chars = len(full_text)
    print(f"Loaded dataset with {chars} raw characters.")

    # Load tokenizer depending on checkpoint
    if tokenizer_name == "bpe":
        # Rebuild BPE tokenizer with same vocab size on the eval text
        tok = get_tokenizer("bpe", text=full_text, vocab_size=vocab_size)
    elif tokenizer_name == "byt5":
        tok = get_tokenizer("byt5")
    elif tokenizer_name == "char":
        tok = get_tokenizer("char", text=full_text)
    else:
        raise ValueError(f"Unknown tokenizer_name in checkpoint: {tokenizer_name}")

    assert tok.vocab_size == vocab_size, "Vocab size mismatch."

    # Create model and load weights
    model, _ = create_nanogpt_model(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Encode eval text
    ids = tok.encode(full_text)
    eval_tokens = len(ids)
    print(f"Total tokens for eval = {eval_tokens}")

    val_ds = IDDataset(ids, args.block_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    avg_loss = evaluate_loss(model, val_loader, device)
    # avg_loss is mean negative log-likelihood in nats per token
    bits_per_token = avg_loss / math.log(2)

    # adjust to bits-per-character (approx): tokens/char ratio
    if chars > 0:
        ratio = eval_tokens / chars
        bits_per_char = bits_per_token * ratio
    else:
        bits_per_char = float("nan")

    print(f"Eval loss (nats/token): {avg_loss:.4f}")
    print(f"Bits per token: {bits_per_token:.4f}")
    print(f"Bits per character (approx): {bits_per_char:.4f}")

    num_params = ckpt.get("num_params", sum(p.numel() for p in model.parameters()))
    train_steps = ckpt.get("train_steps", None)
    train_time_sec = ckpt.get("train_time_sec", None)
    train_steps_per_sec = ckpt.get("train_steps_per_sec", None)

    # Ensure output directory exists
    metrics_dir = os.path.dirname(args.metrics_out)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    metrics = {
        "checkpoint": args.checkpoint,
        "tokenizer": f"{tokenizer_name}-{arch}",
        "vocab_size": vocab_size,
        "loss": avg_loss,
        "bits_per_token": bits_per_token,
        "bits_per_char": bits_per_char,
        "eval_tokens": eval_tokens,
        "eval_chars": chars,
        "num_params": num_params,
        "train_steps": train_steps,
        "train_time_sec": train_time_sec,
        "train_steps_per_sec": train_steps_per_sec,
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics → {args.metrics_out}")


if __name__ == "__main__":
    main()
