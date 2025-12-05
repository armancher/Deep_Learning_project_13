import os
import sys
import argparse
import json
import math
import time
import torch
from datetime import datetime
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


def calculate_comprehensive_metrics(loss, tokenizer, eval_ids, text, model_params, train_steps_per_sec):
    """Calculate all enhanced metrics including bits per byte and tokenization efficiency"""
    
    # Basic metrics from original
    perplexity = float(torch.exp(torch.tensor(loss)))
    bits_per_token = loss / math.log(2.0)
    tokens_per_char = len(eval_ids) / max(len(text), 1)
    bits_per_char = bits_per_token * tokens_per_char
    
    # NEW: Bits per byte (requested by TA)
    bytes_count = len(text.encode('utf-8'))
    bits_per_byte = bits_per_token * (len(eval_ids) / bytes_count)
    
    # NEW: Tokenization efficiency analysis
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_tokens = tokenizer.encode(sample_text)
    sample_chars = len(sample_text)
    sample_bytes = len(sample_text.encode('utf-8'))
    
    # NEW: Compression ratios
    compression_ratio_chars = len(text) / len(eval_ids)
    compression_ratio_bytes = bytes_count / len(eval_ids)
    
    # NEW: Tokens per parameter (TPP - requested by TA)
    tokens_per_parameter = len(eval_ids) / model_params if model_params else None
    
    return {
        # Original metrics
        "loss": loss,
        "perplexity": perplexity,
        "bits_per_token": bits_per_token,
        "bits_per_character": bits_per_char,
        
        # NEW: Enhanced metrics
        "bits_per_byte": bits_per_byte,
        "compression_ratio_chars": compression_ratio_chars,
        "compression_ratio_bytes": compression_ratio_bytes,
        "tokens_per_parameter": tokens_per_parameter,
        "total_bytes_processed": bytes_count,
        "total_tokens_processed": len(eval_ids),
        "total_chars_processed": len(text),
        
        # NEW: Tokenization efficiency
        "tokenization_efficiency": {
            "tokens_per_char": len(eval_ids) / len(text),
            "chars_per_token": len(text) / len(eval_ids),
            "bytes_per_token": bytes_count / len(eval_ids),
            "sample_tokens_per_char": len(sample_tokens) / sample_chars,
            "sample_compression_ratio": sample_chars / len(sample_tokens),
            "sample_bits_per_byte": (bits_per_token * len(sample_tokens)) / sample_bytes
        },
        
        # NEW: Performance metrics
        "training_efficiency": {
            "train_steps_per_sec": train_steps_per_sec,
            "estimated_tokens_per_second": train_steps_per_sec * 32 * 256 if train_steps_per_sec else None,  # Assuming batch_size=32, block_size=256
        },
        
        "timestamp": datetime.now().isoformat()
    }


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
    parser = argparse.ArgumentParser(description="Enhanced nanoGPT evaluation with comprehensive metrics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics_out", type=str, default=None,
                    help="Optional path to save enhanced metrics as JSON")


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

    # Auto-generate metrics filename if user didn't supply one
    if args.metrics_out is None:
        step_str = f"step{train_steps}" if train_steps is not None else "stepUnknown"
        tokenizer_str = tokenizer_name.replace("/", "_")  # sanitize
    
        auto_name = f"enhanced_metrics_{tokenizer_str}_{step_str}.json"

        # Always save inside results/ directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        args.metrics_out = os.path.join(results_dir, auto_name)

        print(f"Auto-generated metrics filename → {args.metrics_out}")

    # Load and tokenize dataset
    text = load_text(args.dataset)
    print(f"Loaded {len(text)} raw chars for evaluation.")

    if tokenizer_name == "byt5":
        tok = get_tokenizer("byt5")
    else:
        tok = get_tokenizer(tokenizer_name, text=text, vocab_size=vocab_size_arg)

    eval_ids = tok.encode(text)
    print(f"Eval tokens: {len(eval_ids)}")

    # Load model
    cfg = ckpt["config"]
    model, _ = create_nanogpt_model(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    # Evaluation
    eval_dataset = IDDataset(eval_ids, args.block_size)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    loss = evaluate(model, eval_loader, device)
    
    # NEW: Calculate comprehensive metrics
    enhanced_metrics = calculate_comprehensive_metrics(
        loss, tok, eval_ids, text, num_params, train_steps_per_sec
    )

    # Enhanced output
    print("\n" + "="*50)
    print(" ENHANCED EVALUATION RESULTS")
    print("="*50)
    print(f"Loss:                 {enhanced_metrics['loss']:.4f}")
    print(f"Perplexity:           {enhanced_metrics['perplexity']:.4f}")
    print(f"Bits/token:           {enhanced_metrics['bits_per_token']:.4f}")
    print(f"Bits/character:       {enhanced_metrics['bits_per_character']:.4f}")
    print(f"Bits/byte:            {enhanced_metrics['bits_per_byte']:.4f}")  # NEW
    print(f"Compression ratio:    {enhanced_metrics['compression_ratio_chars']:.2f} chars/token")  # NEW
    print(f"Tokens/parameter:     {enhanced_metrics['tokens_per_parameter']:.2f}" if enhanced_metrics['tokens_per_parameter'] else "Tokens/parameter:     N/A")  # NEW
    
    if train_steps_per_sec is not None:
        print(f"Train steps/sec:      {train_steps_per_sec:.4f}")
    if num_params is not None:
        print(f"Model parameters:     {num_params}")
    
    # NEW: Tokenization efficiency
    print(f"Tokens/char:          {enhanced_metrics['tokenization_efficiency']['tokens_per_char']:.3f}")
    print(f"Chars/token:          {enhanced_metrics['tokenization_efficiency']['chars_per_token']:.3f}")
    print("="*50)

    # Save enhanced metrics
    if args.metrics_out is not None:
        # Add basic info to metrics
        enhanced_metrics.update({
            "checkpoint": args.checkpoint,
            "dataset_eval": args.dataset,
            "dataset_train": train_dataset_path,
            "tokenizer": tokenizer_name,
            "vocab_size": vocab_size,
            "num_params": num_params,
            "train_steps": train_steps,
            "train_time_sec": train_time_sec,
            "train_steps_per_sec": train_steps_per_sec,
        })

        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(enhanced_metrics, f, indent=2)
        print(f"Saved ENHANCED metrics JSON → {args.metrics_out}")


if __name__ == "__main__":
    main()
