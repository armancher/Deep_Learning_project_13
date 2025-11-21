from pathlib import Path
import numpy as np

from tokenizers.gpt4 import GPT4Tokenizer


def main():
    data_dir = Path("data/gpt4_tokens")
    train_ids = np.fromfile(data_dir / "train.bin", dtype=np.uint32)

    tokenizer = GPT4Tokenizer()

    # decode a small slice back to text
    sample_ids = train_ids[:200].tolist()
    text = tokenizer.decode(sample_ids)

    print("First 200 token IDs:", sample_ids[:20], "...")
    print("\nDecoded text snippet:\n")
    print(text)


if __name__ == "__main__":
    main()
