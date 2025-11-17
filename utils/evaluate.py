# evaluate.py
import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from model import TransformerLM
from config import get_experiment_config
from dataloader import create_dataloaders

from tokenizers.basic import BasicTokenizer
from tokenizers.byte_tokenizer import ByteTokenizer
from tokenizers.char_tokenizer import CharTokenizer

# ---------------------------------------------------------
# 1. Compute Perplexity
# ---------------------------------------------------------
def compute_perplexity(model, loader, device, vocab_size):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")  # sum for full-sequence loss
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens  # average cross-entropy
    ppl = np.exp(avg_loss)

    return ppl, avg_loss


# ---------------------------------------------------------
# 2. Compute Bits Per Character (BPC)
# ---------------------------------------------------------
def compute_bpc(avg_loss_nats):
    """
    BPC = bits per character
    Convert from natural log (nats) to bits:
    bits = nats / ln(2)
    """
    return avg_loss_nats / np.log(2)


# ---------------------------------------------------------
# 3. Load model from checkpoint
# ---------------------------------------------------------
def load_model(cfg, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


# ---------------------------------------------------------
# 4. Text generation
# ---------------------------------------------------------
def generate_text(model, device, start_ids, max_new_tokens=100):
    """
    start_ids: list[int] initial context
    """
    model.eval()
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(x)
        next_token_logits = logits[0, -1]  # last token prediction

        # sampling — use temperature = 1.0 for now
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id.unsqueeze(0)], dim=1)

        if x.shape[1] >= model.pos_emb.num_embeddings:
            break  # reached max seq len

    return x.squeeze(0).tolist()

# ---------------------------------------------------------
# 5. Helper for real text
# ---------------------------------------------------------
def load_tokenizer(cfg):
    kind = cfg.tokenizer_type

    if kind == "byte_bpe":
        tok = BasicTokenizer()
        # IMPORTANT: load trained merges
        tok.load("../data/byte_bpe.model")   # adapt filename if different
        return tok

    if kind == "byte":
        return ByteTokenizer()

    if kind == "char":
        tok = CharTokenizer()
        tok.load("../data/char_vocab.json")  # or re-train on full text if needed
        return tok

    raise ValueError("Unknown tokenizer:", kind)

# ---------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    # ADD THIS HERE
    parser.add_argument("--prompt", type=str, default="Hello")

    args = parser.parse_args()

    cfg = get_experiment_config(args.experiment)

    # Load data
    _, val_loader = create_dataloaders(cfg)

    # Load model
    model, device = load_model(cfg, args.checkpoint)

    print("Evaluating model...")

    # ---- Perplexity + BPC ----
    ppl, avg_loss_nats = compute_perplexity(model, val_loader, device, cfg.vocab_size)
    bpc = compute_bpc(avg_loss_nats)

    print(f"Perplexity: {ppl:.4f}")
    print(f"Average loss (nats/token): {avg_loss_nats:.4f}")
    print(f"BPC (bits per character): {bpc:.4f}")

    # ---- Text generation ----
    tok = load_tokenizer(cfg)

    prompt = args.prompt if args.prompt is not None else "Hello"
    start_ids = tok.encode(prompt)

    gen_ids = generate_text(model, device, start_ids, max_new_tokens=100)
    decoded = tok.decode(gen_ids)

    print("\nGenerated text:")
    print(decoded)
