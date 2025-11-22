import os, sys
import argparse
import math
import time

import torch
from torch.utils.data import Dataset, DataLoader

# Make sure we can import project modules from root
THIS_DIR = os.path.dirname(__file__)
os.chdir(THIS_DIR)  # ensure CWD = project root

# Ensure project root is in PYTHONPATH
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.nanogpt_wrapper import create_nanogpt_model
from tokenizers import get_tokenizer


# ==========================
# Dataset over token IDs
# ==========================
class IDDataset(Dataset):
    def __init__(self, ids, block_size: int):
        """
        ids: a long list of token ids (train or val split)
        block_size: context length
        """
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        # number of possible blocks
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1 : idx + 1 + self.block_size], dtype=torch.long)
        return x, y


# ==========================
# Helpers
# ==========================
def load_text(dataset_path: str) -> str:
    """
    If dataset_path is a file: load that file.
    If it's a directory: concatenate all .txt files.
    """
    if os.path.isdir(dataset_path):
        parts = []
        for fn in sorted(os.listdir(dataset_path)):
            if fn.lower().endswith(".txt"):
                full = os.path.join(dataset_path, fn)
                with open(full, "r", encoding="utf-8") as f:
                    parts.append(f.read())
        text = "\n\n".join(parts)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text:
        raise ValueError(f"Dataset at {dataset_path} is empty or unreadable.")
    return text


def get_dataloaders(tokenizer_name: str,
                    dataset_path: str,
                    block_size: int,
                    batch_size: int,
                    vocab_size: int | None = None,
                    train_frac: float = 0.9):

    print(f"Loading raw text from: {dataset_path}")
    full_text = load_text(dataset_path)
    print(f"Loaded corpus with {len(full_text)} characters")

    # Build tokenizer
    print(f"Initializing tokenizer: {tokenizer_name}")
    if tokenizer_name in ["char", "bpe", "byte_bpe"]:
        tok = get_tokenizer(tokenizer_name, text=full_text, vocab_size=vocab_size or 32000)
    elif tokenizer_name == "byte":
        tok = get_tokenizer("byte")
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer_name}")

    print(f"Tokenizer vocab_size = {tok.vocab_size}")

    # Encode whole corpus once
    print("Encoding full corpus...")
    ids = tok.encode(full_text)
    print(f"Total tokens: {len(ids)}")

    # train/val split by tokens
    split_idx = int(train_frac * len(ids))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

    train_dataset = IDDataset(train_ids, block_size)
    val_dataset = IDDataset(val_ids, block_size)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(
            f"Dataset too small for block_size={block_size}. "
            f"Train len={len(train_dataset)}, Val len={len(val_dataset)}"
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return tok, train_loader, val_loader


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)


# ==========================
# Training
# ==========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to .txt file or folder with .txt files.")
    parser.add_argument("--tokenizer", type=str, default="byte",
                        choices=["byte", "char", "bpe", "byte_bpe"],
                        help="Which tokenizer to use.")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Target vocab size for BPE / byte-BPE.")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--save_path", type=str, default="checkpoints",
                        help="Folder to save model checkpoints.")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Data + tokenizer + loaders
    tok, train_loader, val_loader = get_dataloaders(
        tokenizer_name=args.tokenizer,
        dataset_path=args.dataset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
    )

    # Model
    model, config = create_nanogpt_model(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )
    model.to(device)

    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create checkpoint folder
    os.makedirs(args.save_path, exist_ok=True)

    # Training loop
    step = 0
    train_iter = iter(train_loader)
    t0 = time.time()

    while step < args.max_steps:
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

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step:5d} | train loss {loss.item():.4f} | time {dt:.1f}s")
            t0 = time.time()

        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"***** step {step} | val loss {val_loss:.4f} *****")

            # Save checkpoint
            ckpt_path = os.path.join(args.save_path,
                                     f"nanogpt_{args.tokenizer}_step{step}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": tok.vocab_size,
                        "block_size": args.block_size,
                        "n_layer": 6,
                        "n_head": 6,
                        "n_embd": 384,
                    },
                    "tokenizer_name": args.tokenizer,
                    "vocab_size_arg": args.vocab_size,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

        step += 1

    print("Training finished.")


if __name__ == "__main__":
    main()
