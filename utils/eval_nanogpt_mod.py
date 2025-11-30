import os
import sys
import argparse
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root visible
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
        # number of non-overlapping blocks
        return (len(self.ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = torch.tensor(self.ids[start:start + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[start + 1:start + 1 + self.block_size], dtype=torch.long)
        return x, y


def load_text(path):
    if os.path.isdir(path):
        buf = []
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".txt"):
                with open(os.path.join(path, fn), "r", encoding="utf-8") as f:
                    buf.append(f.read())
        return "\n\n".join(buf)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()
            count += 1
    return total_loss / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics_out", type=str, default=None,
                        help="Optional path to save metrics as JSON")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    tokenizer_name = ckpt["tokenizer_name"]
    vocab_size = ckpt["config"]["vocab_size"]
    vocab_size_arg = ckpt.get("vocab_size_arg", vocab_size)
    num_params = ckpt.get("num_params", None)
    train_steps = ckpt.get("train_steps", None)
    train_time_sec = ckpt.get("train_time_sec", None)
    train_steps_per_sec = ckpt.get("train_steps_per_sec", None)
    train_dataset_path = ckpt.get("dataset_path", None)

    print(f"Tokenizer used in model: {tokenizer_name}")
    print(f"Vocabulary size from checkpoint: {vocab_size}")

    text = load_text(args.dataset)
    print(f"Loaded {len(text)} raw chars for evaluation.")

    if tokenizer_name == "byt5":
        tok = get_tokenizer("byt5")
    else:
        tok = get_tokenizer(tokenizer_name, text=text, vocab_size=vocab_size_arg)

    eval_ids = tok.encode(text)
    print(f"Eval tokens: {len(eval_ids)}")

    cfg = ckpt["config"]
    model, _ = create_nanogpt_model(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    eval_dataset = IDDataset(eval_ids, args.block_size)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    loss = evaluate(model, eval_loader, device)
    perplexity = float(torch.exp(torch.tensor(loss)))

    bits_per_token = loss / math.log(2.0)
    tokens_per_char = len(eval_ids) / max(len(text), 1)
    bits_per_char = bits_per_token * tokens_per_char

    print("====================================")
    print(" Evaluation Results")
    print("====================================")
    print(f"Loss:           {loss:.4f}")
    print(f"Perplexity:     {perplexity:.4f}")
    print(f"Bits/token:     {bits_per_token:.4f}")
    print(f"Bits/character: {bits_per_char:.4f}")
    if train_steps_per_sec is not None:
        print(f"Train steps/sec: {train_steps_per_sec:.4f}")
    if num_params is not None:
        print(f"Model parameters: {num_params}")
    print("====================================")

    if args.metrics_out is not None:
        metrics = {
            "checkpoint": args.checkpoint,
            "dataset_eval": args.dataset,
            "dataset_train": train_dataset_path,
            "tokenizer": tokenizer_name,
            "vocab_size": vocab_size,
            "loss": loss,
            "perplexity": perplexity,
            "bits_per_token": bits_per_token,
            "bits_per_char": bits_per_char,
            "eval_tokens": len(eval_ids),
            "eval_chars": len(text),
            "num_params": num_params,
            "train_steps": train_steps,
            "train_time_sec": train_time_sec,
            "train_steps_per_sec": train_steps_per_sec,
        }

        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics JSON → {args.metrics_out}")


if __name__ == "__main__":
    main()
