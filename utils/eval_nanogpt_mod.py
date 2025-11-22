import os
import sys
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on PYTHONPATH
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
    """Loads either a file or folder of .txt files."""
    if os.path.isdir(path):
        parts = []
        for fn in sorted(os.listdir(path)):
            if fn.lower().endswith(".txt"):
                with open(os.path.join(path, fn), "r", encoding="utf-8") as f:
                    parts.append(f.read())
        return "\n\n".join(parts)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


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
    return total_loss / max(1, count)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved .pt checkpoint")
    parser.add_argument("--dataset", required=True,
                        help="Path to evaluation .txt or folder of .txt")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Extract tokenizer name
    tokenizer_name = ckpt["tokenizer_name"]
    vocab_size = ckpt["config"]["vocab_size"]
    print(f"Tokenizer used for model: {tokenizer_name} (vocab={vocab_size})")

    # Load evaluation text
    raw_text = load_text(args.dataset)
    print(f"Loaded evaluation text ({len(raw_text)} raw chars)")

    # Create tokenizer with correct settings
    if tokenizer_name in ["char", "bpe", "byte_bpe"]:
        tok = get_tokenizer(tokenizer_name, text=raw_text,
                            vocab_size=ckpt["vocab_size_arg"])
    else:
        tok = get_tokenizer("byte")

    # Encode evaluation data
    eval_ids = tok.encode(raw_text)
    print(f"Eval tokens: {len(eval_ids)}")

    # Create model
    print("Reconstructing model...")
    model, _ = create_nanogpt_model(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=ckpt["config"]["n_layer"],
        n_head=ckpt["config"]["n_head"],
        n_embd=ckpt["config"]["n_embd"],
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Build eval dataset/dataloader
    eval_dataset = IDDataset(eval_ids, args.block_size)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Run evaluation
    loss = evaluate(model, eval_loader, device)
    perplexity = torch.exp(torch.tensor(loss))

    print("=====================================")
    print(" Evaluation Results")
    print("=====================================")
    print(f"Loss:       {loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("=====================================")


if __name__ == "__main__":
    main()