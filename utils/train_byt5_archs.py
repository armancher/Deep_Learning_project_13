import os
import sys
import argparse
import time
import math

import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is visible (same pattern as train_nanogpt_mod.py)
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.nanogpt_wrapper import create_nanogpt_model
from tokenizers import get_tokenizer


# =============== Dataset over ID sequences ===============

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


def load_text(path: str) -> str:
    """Load a single file or all .txt files from a folder."""
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


def get_dataloaders_byt5(text, block_size, batch_size, train_frac=0.9):
    # byt5 tokenizer: bytes 0–255
    tok = get_tokenizer("byt5")
    print(f"Tokenizer: byt5, vocab_size={tok.vocab_size}")

    ids = tok.encode(text)
    print(f"Total byte tokens = {len(ids)}")

    split = int(train_frac * len(ids))
    train_ids = ids[:split]
    val_ids = ids[split:]

    print(f"Train tokens = {len(train_ids)}, Val tokens = {len(val_ids)}")

    train_ds = IDDataset(train_ids, block_size)
    val_ds = IDDataset(val_ids, block_size)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Dataset too small for this block_size.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return tok, train_loader, val_loader


def evaluate(model, loader, device):
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

    parser.add_argument("--dataset", required=True, help="Path to WikiText txt or folder")
    parser.add_argument("--save_path", default="checkpoints_byt5")

    # Only byt5 tokenizer
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")

    # Two architectures: baseline and byt5-style
    parser.add_argument(
        "--arch",
        choices=["baseline", "byt5style"],
        default="baseline",
        help="baseline = 6-layer small model, byt5style = deeper/wider ByT5-inspired model"
    )

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load raw text
    full_text = load_text(args.dataset)
    print(f"Loaded dataset with {len(full_text)} raw characters.")

    # Tokenizer + dataloaders (byt5 only)
    tok, train_loader, val_loader = get_dataloaders_byt5(
        text=full_text,
        block_size=args.block_size,
        batch_size=args.batch_size,
    )

    # Choose architecture
    if args.arch == "baseline":
        n_layer = 6
        n_head = 6
        n_embd = 384
    else:  # byt5style
        # ByT5-style: more layers + wider embeddings, reallocating capacity from vocab
        n_layer = 12
        n_head = 8
        n_embd = 512

    print(f"Architecture = {args.arch} | n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

    # Create model
    model, config = create_nanogpt_model(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_path, exist_ok=True)

    train_start_time = time.time()
    last_log_time = train_start_time

    train_iter = iter(train_loader)

    for step in range(1, args.max_steps + 1):
        # cycle over train_loader
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if step % 50 == 0 or step == 1:
            now = time.time()
            dt = now - last_log_time
            print(f"step {step:5d} | loss {loss.item():.4f} | dt {dt:.1f}s")
            last_log_time = now

        # Evaluation + checkpoint
        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.time() - train_start_time
            steps_per_sec = step / max(elapsed, 1e-8)

            print(f"--- Step {step} | Val loss = {val_loss:.4f}")
            print(f"(avg steps/sec so far: {steps_per_sec:.3f})")

            ckpt_path = os.path.join(
                args.save_path,
                f"nanogpt_byt5_{args.arch}_step{step}.pt"
            )

            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_name": "byt5",
                "arch": args.arch,
                "config": {
                    "vocab_size": tok.vocab_size,
                    "block_size": args.block_size,
                    "n_layer": n_layer,
                    "n_head": n_head,
                    "n_embd": n_embd,
                },
                "num_params": num_params,
                "train_steps": step,
                "train_time_sec": elapsed,
                "train_steps_per_sec": steps_per_sec,
                "dataset_path": args.dataset,
            }, ckpt_path)

            print(f"Saved checkpoint → {ckpt_path}", flush=True)

    print("Training complete.")


if __name__ == "__main__":
    main()
