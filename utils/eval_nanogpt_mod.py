import os
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root visible
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.nanogpt_wrapper import create_nanogpt_model
from tokenizers import get_tokenizer


# ==================================
# Dataset over token IDs
# ==================================
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

    args = parser.parse_args()

    # Device
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

    print(f"Tokenizer used in model: {tokenizer_name}")
    print(f"Vocabulary size from checkpoint: {vocab_size}")

    # Load evaluation text
    text = load_text(args.dataset)
    print(f"Loaded {len(text)} raw chars for evaluation.")

    # Load tokenizer (must match training)
    if tokenizer_name == "byt5":
        tok = get_tokenizer("byt5")
    else:
        tok = get_tokenizer(tokenizer_name, text=text, vocab_size=vocab_size_arg)

    # Encode dataset
    eval_ids = tok.encode(text)
    print(f"Eval tokens: {len(eval_ids)}")

    # Recreate model
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

    # Compute loss & perplexity
    loss = evaluate(model, eval_loader, device)
    perplexity = float(torch.exp(torch.tensor(loss)))

    print("====================================")
    print(" Evaluation Results")
    print("====================================")
    print(f"Loss:       {loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("====================================")


if __name__ == "__main__":
    main()