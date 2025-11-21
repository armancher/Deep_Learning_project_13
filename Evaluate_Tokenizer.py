# evaluate_gpt4_tokenizer.py
from pathlib import Path
from collections import Counter
import math
import json

import numpy as np
from tokenizers.gpt4 import GPT4Tokenizer


def percentile(arr, q: float):
    """
    Simple percentile for a sorted list.
    q in [0,1], e.g. 0.5 for median.
    """
    if not arr:
        return None
    idx = int(q * (len(arr) - 1))
    return arr[idx]


def main():
    # ---- paths ----
    text_path = Path("data/train.txt")
    tokens_path = Path("data/gpt4_tokens/train.bin")
    results_dir = Path("results/tokenization")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    text = text_path.read_text(encoding="utf-8")
    ids = np.fromfile(tokens_path, dtype=np.uint32)
    tokenizer = GPT4Tokenizer()

    # ---- basic counts: chars, bytes, tokens ----
    raw_bytes = text.encode("utf-8")

    num_chars = len(text)
    num_bytes = len(raw_bytes)
    num_tokens = len(ids)

    print(f"Characters : {num_chars:,}")
    print(f"Bytes      : {num_bytes:,}")
    print(f"Tokens     : {num_tokens:,}")
    print(f"Chars/token: {num_chars / num_tokens:.3f}")
    print(f"Bytes/token: {num_bytes / num_tokens:.3f}")

    # ---- word-level stats ----
    words = text.split()
    num_words = len(words)
    tokens_per_word_global = num_tokens / num_words

    print(f"Words      : {num_words:,}")
    print(f"Tokens/word (global): {tokens_per_word_global:.3f}")

    # ---- token frequency & entropy (unigram over tokens) ----
    counter = Counter(ids.tolist())
    vocab_size = len(counter)
    print(f"Vocab size (used in corpus): {vocab_size:,}")

    total_tokens = num_tokens
    token_entropy_bits = 0.0  # H(T) in bits/token
    for c in counter.values():
        p = c / total_tokens
        token_entropy_bits -= p * math.log2(p)

    # character- and byte-normalized versions of token entropy
    bits_per_char = token_entropy_bits * (total_tokens / num_chars)
    tokens_per_byte = total_tokens / num_bytes
    bits_per_byte_from_tokens = token_entropy_bits * tokens_per_byte

    print(f"Token entropy (bits/token)   : {token_entropy_bits:.3f}")
    print(f"Bits per character (from T)  : {bits_per_char:.3f}")
    print(f"Tokens per byte              : {tokens_per_byte:.6f}")
    print(f"Bits per byte (from tokens)  : {bits_per_byte_from_tokens:.3f}")

    # ---- pure byte-level entropy of the corpus ----
    byte_counter = Counter(raw_bytes)
    byte_entropy_bits = 0.0  # H(B) in bits/byte
    for c in byte_counter.values():
        p = c / num_bytes
        byte_entropy_bits -= p * math.log2(p)

    print(f"Byte entropy (bits/byte)     : {byte_entropy_bits:.3f}")

    # ---- fragmentation: tokens per word distribution (sample) ----
    N = min(100_000, num_words)
    sample_words = words[:N]
    lens = [len(tokenizer.encode(w)) for w in sample_words]
    lens_sorted = sorted(lens)

    avg_tokens_per_word = sum(lens) / len(lens)
    median_tokens_per_word = percentile(lens_sorted, 0.5)
    p90_tokens_per_word = percentile(lens_sorted, 0.9)

    print(f"\nFragmentation on first {N:,} words:")
    print(f"  mean tokens/word  : {avg_tokens_per_word:.3f}")
    print(f"  median tokens/word: {median_tokens_per_word}")
    print(f"  90th pct tokens/w : {p90_tokens_per_word}")

    # ---- top tokens ----
    print("\nTop 20 most frequent tokens:")
    for token_id, count in counter.most_common(20):
        try:
            decoded = tokenizer.decode([int(token_id)])
        except Exception:
            decoded = "<decode error>"
        print(f"  id={token_id:>6} freq={count:>8}  repr={decoded!r}")

    # ---- aggregate metrics into JSON ----
    metrics = {
        # corpus size
        "num_characters": num_chars,
        "num_bytes": num_bytes,
        "num_tokens": num_tokens,
        "num_words": num_words,
        # size ratios
        "chars_per_token": num_chars / num_tokens,
        "bytes_per_token": num_bytes / num_tokens,
        "tokens_per_word_global": tokens_per_word_global,
        # vocabulary / entropy
        "vocab_size_used": vocab_size,
        "token_entropy_bits_per_token": token_entropy_bits,
        "bits_per_character_from_tokens": bits_per_char,
        "tokens_per_byte": tokens_per_byte,
        "bits_per_byte_from_tokens": bits_per_byte_from_tokens,
        "byte_entropy_bits_per_byte": byte_entropy_bits,
        # fragmentation (per-word token counts)
        "fragmentation_sample_words": N,
        "mean_tokens_per_word": avg_tokens_per_word,
        "median_tokens_per_word": median_tokens_per_word,
        "p90_tokens_per_word": p90_tokens_per_word,
    }

    out_path = results_dir / "gpt4_wikitext2_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
