# tokenize_gpt4.py
from pathlib import Path
import numpy as np
from datetime import datetime
import json


from tokenizers.gpt4 import GPT4Tokenizer  # your copied file

def main():
    # 1) load raw text (WikiText-2 or whatever you saved as train.txt)
    data_path = Path("data/train.txt")
    text = data_path.read_text(encoding="utf-8")

    # 2) instantiate Karpathy's GPT-4 tokenizer
    tokenizer = GPT4Tokenizer()

    # 3) tokenize
    token_ids = tokenizer.encode(text)  # list[int]

    # 4) some quick stats (useful later for evaluation)
    print(f"Number of characters : {len(text):,}")
    print(f"Number of tokens     : {len(token_ids):,}")
    print(f"Chars per token      : {len(text) / len(token_ids):.2f}")

    # 5) save as binary for nanoGPT-style training
    ids = np.array(token_ids, dtype=np.uint32)

    # simple 90/10 split
    n = int(0.9 * len(ids))
    train_ids = ids[:n]
    val_ids = ids[n:]

    out_dir = Path("data/gpt4_tokens")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    print("Saved tokenized data to data/gpt4_tokens/train.bin and val.bin")

        # ---- save stats for later analysis ----
    stats = {
        "num_characters": len(text),
        "num_tokens": len(token_ids),
        "chars_per_token": len(text) / len(token_ids),
        "source_file": str(data_path),
        "created_at": datetime.now().isoformat(),
    }

    results_dir = Path("results/tokenization")
    results_dir.mkdir(parents=True, exist_ok=True)

    stats_path = results_dir / "gpt4_wikitext2_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
