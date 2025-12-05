import os
import sys
import argparse
import time
import math
import json

import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is visible
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.nanogpt_wrapper import create_nanogpt_model
from tokenizers import get_tokenizer


# =============== Dataset over ID sequences (non-overlapping blocks) ===============

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


def get_dataloaders_bpe(text, block_size, batch_size, vocab_size, train_frac=0.9):
    """
    Build BPE tokenizer on the full text, then create train/val loaders.
    """
    tok = get_tokenizer("bpe", text=text, vocab_size=vocab_size)
    print(f"Tokenizer: bpe, vocab_size={tok.vocab_size}")

    ids = tok.encode(text)
    print(f"Total tokens = {len(ids)}")

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

    return tok, ids, train_loader, val_loader


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
    parser.add_argument("--save_path", default="checkpoints/bpe_byt5_experiment")


    # BPE-specific
    parser.add_argument("--vocab_size", type=int, default=32000)

    # Model / training hyperparameters
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")

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

    # Tokenizer + dataloaders (BPE only)
    tok, all_ids, train_loader, val_loader = get_dataloaders_bpe(
        text=full_text,
        block_size=args.block_size,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
    )

    # ByT5-style architecture: deeper + wider
    n_layer = 12
    n_head = 8
    n_embd = 512
    arch = "byt5style"

    print(f"Architecture = {arch} | n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

    # Create model
    model, _ = create_nanogpt_model(
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

    # -------------------------
    # Training loop
    # -------------------------
    for step in range(1, args.max_steps + 1):
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

        # Periodic val loss + checkpoint
        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            elapsed = time.time() - train_start_time
            steps_per_sec = step / max(elapsed, 1e-8)

            print(f"--- Step {step} | Val loss = {val_loss:.4f}")
            print(f"(avg steps/sec so far: {steps_per_sec:.3f})")

            ckpt_path = os.path.join(
                args.save_path,
                f"nanogpt_bpe_{arch}_step{step}.pt"
            )

            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_name": "bpe",
                "arch": arch,
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

    # -------------------------
    # Final checkpoint + metrics JSON
    # -------------------------
    elapsed = time.time() - train_start_time
    steps_per_sec = args.max_steps / max(elapsed, 1e-8)

    final_ckpt_path = os.path.join(
        args.save_path,
        f"nanogpt_bpe_{arch}_step{args.max_steps}.pt"
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "tokenizer_name": "bpe",
        "arch": arch,
        "config": {
            "vocab_size": tok.vocab_size,
            "block_size": args.block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
        },
        "num_params": num_params,
        "train_steps": args.max_steps,
        "train_time_sec": elapsed,
        "train_steps_per_sec": steps_per_sec,
        "dataset_path": args.dataset,
    }, final_ckpt_path)
    print(f"Saved FINAL checkpoint → {final_ckpt_path}", flush=True)

    # Evaluate on full dataset for metrics
    eval_tokens = len(all_ids)
    eval_chars = len(full_text)

    eval_ds = IDDataset(all_ids, args.block_size)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    avg_loss = evaluate(model, eval_loader, device)
    bits_per_token = avg_loss / math.log(2)

    if eval_chars > 0:
        ratio = eval_tokens / eval_chars
        bits_per_char = bits_per_token * ratio
    else:
        bits_per_char = float("nan")

    print(f"Eval loss (nats/token): {avg_loss:.4f}")
    print(f"Bits per token: {bits_per_token:.4f}")
    print(f"Bits per character (approx): {bits_per_char:.4f}")

    # Metrics directory for enhanced architecture
    metrics_dir = os.path.join("results", "bpe_byt5_experiment")

    os.makedirs(metrics_dir, exist_ok=True)

    metrics_out = os.path.join(
        metrics_dir,
        f"metrics_bpe_{arch}_step{args.max_steps}.json"
    )

    metrics = {
        "checkpoint": final_ckpt_path,
        "tokenizer": "bpe",
        "arch": arch,
        "vocab_size": tok.vocab_size,
        "loss": avg_loss,
        "bits_per_token": bits_per_token,
        "bits_per_char": bits_per_char,
        "eval_tokens": eval_tokens,
        "eval_chars": eval_chars,
        "num_params": num_params,
        "train_steps": args.max_steps,
        "train_time_sec": elapsed,
        "train_steps_per_sec": steps_per_sec,
    }

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics → {metrics_out}")
    print("Training + evaluation complete.")


if __name__ == "__main__":
    main()
