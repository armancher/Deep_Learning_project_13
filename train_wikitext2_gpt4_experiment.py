from pathlib import Path
import sys
import time
import math
import json
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F

# --- Load nanoGPT's model.py explicitly from the nanoGPT folder ---
import importlib.util

ROOT = Path(__file__).resolve().parent          # .../Deep_Learning_project_13
NANOGPT_DIR = ROOT / "nanoGPT"                  # .../Deep_Learning_project_13/nanoGPT
MODEL_PY = NANOGPT_DIR / "model.py"             # .../Deep_Learning_project_13/nanoGPT/model.py

if not MODEL_PY.exists():
    raise FileNotFoundError(f"Could not find nanoGPT model.py at {MODEL_PY}")

spec = importlib.util.spec_from_file_location("nanogpt_model", MODEL_PY)
nanogpt_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nanogpt_model)

GPT = nanogpt_model.GPT
GPTConfig = nanogpt_model.GPTConfig


# =========================================================
# 1. High-level experiment configuration
# =========================================================

# Choose model size: "tiny" or "small"
MODEL_SIZE = "small"

# Data paths (relative to project root)
TEXT_PATH = Path("data/train.txt")
TRAIN_BIN_PATH = Path("data/gpt4_tokens/train.bin")
VAL_BIN_PATH = Path("data/gpt4_tokens/val.bin")

# Where to save training metrics / checkpoints
RESULTS_DIR = Path("results/experiments")
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints_wikitext2_gpt4"


@dataclass
class ModelSizeConfig:
    n_layer: int
    n_head: int
    n_embd: int


MODEL_SIZES: Dict[str, ModelSizeConfig] = {
    "tiny": ModelSizeConfig(n_layer=4, n_head=4, n_embd=256),
    "small": ModelSizeConfig(n_layer=8, n_head=8, n_embd=512),
}


# Training hyperparameters (shared across tokenizers)
block_size = 256
batch_size = 64
max_iters = 20_000
learning_rate = 3e-4
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
warmup_iters = 200
eval_interval = 500      # evaluate every N steps
eval_iters = 50          # average over this many batches
grad_clip = 1.0          # gradient clipping (max norm)


# =========================================================
# 2. Dataset utilities
# =========================================================

class TokenDataset(torch.utils.data.Dataset):
    """Simple dataset that holds a flat array of token IDs and samples random blocks."""
    def __init__(self, ids: np.ndarray, block_size: int):
        assert ids.ndim == 1
        self.ids = torch.from_numpy(ids.astype(np.int64))  # long tensor
        self.block_size = block_size

    def __len__(self):
        # Number of possible starting positions for a block
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        # idx is ignored; we sample randomly each time to mimic nanoGPT behavior
        # (we don't want deterministic ordering here)
        i = torch.randint(low=0, high=len(self.ids) - self.block_size - 1, size=(1,)).item()
        x = self.ids[i : i + self.block_size]
        y = self.ids[i + 1 : i + 1 + self.block_size]
        return x, y


def get_batch(dataset: TokenDataset, device: torch.device, batch_size: int):
    """Sample a batch of (input, target) from a TokenDataset."""
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = dataset[0]  # index is ignored; sampling is random internally
        xs.append(x.unsqueeze(0))
        ys.append(y.unsqueeze(0))
    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return x, y


# =========================================================
# 3. Helper: compute char/byte statistics
# =========================================================

def compute_text_stats() -> Dict[str, Any]:
    """Compute #chars, #bytes and ratios from the raw train.txt."""
    text = TEXT_PATH.read_text(encoding="utf-8")
    raw_bytes = text.encode("utf-8")
    num_chars = len(text)
    num_bytes = len(raw_bytes)
    return {
        "num_chars": num_chars,
        "num_bytes": num_bytes,
    }


# =========================================================
# 4. Training loop with metrics logging
# =========================================================

def main():
    assert MODEL_SIZE in MODEL_SIZES, f"Unknown MODEL_SIZE={MODEL_SIZE!r}"
    size_conf = MODEL_SIZES[MODEL_SIZE]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- load tokenized data -----
    print("Loading tokenized data...")
    train_ids = np.fromfile(TRAIN_BIN_PATH, dtype=np.uint32)
    val_ids = np.fromfile(VAL_BIN_PATH, dtype=np.uint32)

    # vocab_size = max token id + 1 to be safe
    vocab_size = int(max(train_ids.max(), val_ids.max())) + 1
    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens  : {len(val_ids):,}")
    print(f"vocab_size  : {vocab_size:,}")

    train_dataset = TokenDataset(train_ids, block_size=block_size)
    val_dataset = TokenDataset(val_ids, block_size=block_size)

    # char / byte stats for bits/char, bits/byte
    text_stats = compute_text_stats()
    num_chars = text_stats["num_chars"]
    num_bytes = text_stats["num_bytes"]
    num_tokens_total = len(train_ids) + len(val_ids)
    tokens_per_char = num_tokens_total / num_chars
    tokens_per_byte = num_tokens_total / num_bytes
    print(f"chars: {num_chars:,}, bytes: {num_bytes:,}, tokens_total: {num_tokens_total:,}")
    print(f"tokens per char:  {tokens_per_char:.6f}")
    print(f"tokens per byte:  {tokens_per_byte:.6f}")

    # ----- model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gpt_conf = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=size_conf.n_layer,
        n_head=size_conf.n_head,
        n_embd=size_conf.n_embd,
        bias=True,
    )
    model = GPT(gpt_conf).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ----- optimizer -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # ----- training loop -----
    def evaluate_split(split: str) -> float:
        """Return average loss (in nats per token) on given split."""
        model.eval()
        dataset = train_dataset if split == "train" else val_dataset
        losses = []
        with torch.no_grad():
            for _ in range(eval_iters):
                x, y = get_batch(dataset, device, batch_size)
                logits, loss = model(x, y)
                losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    metrics_history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    start_time = time.time()

    print("Starting training...")
    for step in range(1, max_iters + 1):
        # warmup LR
        if step <= warmup_iters:
            lr = learning_rate * step / warmup_iters
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        x, y = get_batch(train_dataset, device, batch_size)
        logits, loss = model(x, y)  # loss is in nats per token (PyTorch CE)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % 10 == 0:
            # simple progress logging
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"step {step:6d}/{max_iters} | train loss (nats/token): {loss.item():.4f} | lr {current_lr:.2e}")

        # periodic evaluation
        if step % eval_interval == 0 or step == max_iters:
            train_loss = evaluate_split("train")
            val_loss = evaluate_split("val")  # nats per token

            # convert to bits/token
            train_loss_bits = train_loss / math.log(2.0)
            val_loss_bits = val_loss / math.log(2.0)

            # perplexity per token
            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(val_loss)

            # bits per char / byte from model
            bits_per_char = val_loss_bits * tokens_per_char
            bits_per_byte = val_loss_bits * tokens_per_byte

            elapsed = time.time() - start_time
            tokens_seen = step * batch_size * block_size
            tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else float("nan")

            print("\n--- EVAL ---")
            print(f"step {step}/{max_iters}")
            print(f"  train loss (nats/token): {train_loss:.4f}")
            print(f"  val   loss (nats/token): {val_loss:.4f}")
            print(f"  train loss (bits/token): {train_loss_bits:.4f}")
            print(f"  val   loss (bits/token): {val_loss_bits:.4f}")
            print(f"  train perplexity        : {train_ppl:.3f}")
            print(f"  val   perplexity        : {val_ppl:.3f}")
            print(f"  val bits per char       : {bits_per_char:.4f}")
            print(f"  val bits per byte       : {bits_per_byte:.4f}")
            print(f"  elapsed (s)             : {elapsed:.1f}")
            print(f"  tokens seen             : {tokens_seen:,}")
            print(f"  tokens/sec              : {tokens_per_sec:.1f}")
            print("-----------\n")

            # save metrics snapshot
            metrics_history.append({
                "step": step,
                "train_loss_nats_per_token": train_loss,
                "val_loss_nats_per_token": val_loss,
                "train_loss_bits_per_token": train_loss_bits,
                "val_loss_bits_per_token": val_loss_bits,
                "train_perplexity": train_ppl,
                "val_perplexity": val_ppl,
                "val_bits_per_char": bits_per_char,
                "val_bits_per_byte": bits_per_byte,
                "elapsed_seconds": elapsed,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_per_sec,
            })

            # checkpointing: save best val loss so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = CHECKPOINTS_DIR / f"wikitext2_gpt4_{MODEL_SIZE}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "val_loss_nats_per_token": val_loss,
                        "config": {
                            "MODEL_SIZE": MODEL_SIZE,
                            "block_size": block_size,
                            "batch_size": batch_size,
                            "max_iters": max_iters,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "beta1": beta1,
                            "beta2": beta2,
                        },
                        "text_stats": text_stats,
                    },
                    ckpt_path,
                )
                print(f"Saved new best checkpoint to {ckpt_path}")

            # also save metrics history so far
            metrics_path = RESULTS_DIR / f"wikitext2_gpt4_{MODEL_SIZE}_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_size": MODEL_SIZE,
                        "vocab_size": vocab_size,
                        "block_size": block_size,
                        "batch_size": batch_size,
                        "max_iters": max_iters,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "beta1": beta1,
                        "beta2": beta2,
                        "grad_clip": grad_clip,
                        "num_chars": num_chars,
                        "num_bytes": num_bytes,
                        "num_tokens_total": int(num_tokens_total),
                        "tokens_per_char": tokens_per_char,
                        "tokens_per_byte": tokens_per_byte,
                        "metrics_history": metrics_history,
                    },
                    f,
                    indent=2,
                )
                print(f"Saved metrics history to {metrics_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()
