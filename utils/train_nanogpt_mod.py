import os
import sys
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is visible
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.nanogpt_wrapper import create_nanogpt_model
from tokenizers import get_tokenizer


# ==========================
# Dataset over ID sequences
# ==========================
class IDDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y


def load_text(path: str) -> str:
    """Loads single file or all .txt files in folder."""
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


def get_dataloaders(tokenizer_name, text, block_size, batch_size, vocab_size, train_frac=0.9):

    # Build tokenizer
    if tokenizer_name == "byt5":
        tok = get_tokenizer("byt5")  # no training needed
    else:
        tok = get_tokenizer(tokenizer_name, text=text, vocab_size=vocab_size)

    print(f"Tokenizer: {tokenizer_name}, vocab_size={tok.vocab_size}")

    # Encode whole dataset
    ids = tok.encode(text)
    print(f"Total tokens = {len(ids)}")

    # Split tokens
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
    total_loss = 0
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

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", default="byt5",
                        choices=["byt5", "char", "bpe"])
    parser.add_argument("--vocab_size", type=int, default=32000)

    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save_path", default="checkpoints")

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

    # Build tokenizer & dataloaders
    tok, train_loader, val_loader = get_dataloaders(
        tokenizer_name=args.tokenizer,
        text=full_text,
        block_size=args.block_size,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size
    )

    # Create GPT model
    model, config = create_nanogpt_model(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.save_path, exist_ok=True)

    step = 0
    train_iter = iter(train_loader)
    train_start_time = time.time()
    last_log_time = train_start_time

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
            now = time.time()
            dt = now - last_log_time
            print(f"step {step:5d} | loss {loss.item():.4f} | time {dt:.1f}s")
            last_log_time = now

        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"--- Step {step} | Val loss = {val_loss:.4f}")

            elapsed = time.time() - train_start_time
            steps_per_sec = step / max(elapsed, 1e-8)

            ckpt_path = os.path.join(
                args.save_path,
                f"nanogpt_{args.tokenizer}_step{step}.pt"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_name": args.tokenizer,
                "vocab_size_arg": args.vocab_size,
                "config": {
                    "vocab_size": tok.vocab_size,
                    "block_size": args.block_size,
                    "n_layer": 6,
                    "n_head": 6,
                    "n_embd": 384,
                },
                "num_params": num_params,
                "train_steps": step,
                "train_time_sec": elapsed,
                "train_steps_per_sec": steps_per_sec,
                "dataset_path": args.dataset,
            }, ckpt_path)

            print(f"Saved checkpoint → {ckpt_path}")
            print(f"(avg steps/sec so far: {steps_per_sec:.3f})")

        step += 1

    print("Training complete.")


if __name__ == "__main__":
    main()