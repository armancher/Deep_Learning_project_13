# prepare_data.py
import argparse
import numpy as np
import os
import sys

sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

# import your tokenizers
from tokenizers.basic import BasicTokenizer        # byte-level BPE (Karpathy style)
from tokenizers.byte_tokenizer import ByteTokenizer
from tokenizers.char_tokenizer import CharTokenizer
# for BPE with SentencePiece or other, import your own
from tokenizers.sp_bpe_tokenizer import SentencePieceBPE

def load_raw_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(data, train_ratio=0.9, val_ratio=0.05):
    n = len(data)
    train = data[: int(n * train_ratio)]
    val   = data[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    test  = data[int(n * (train_ratio + val_ratio)) :]
    return train, val, test

def get_tokenizer(kind, vocab_size):
    if kind == "byte_bpe":
        tok = BasicTokenizer()
        return tok, vocab_size
    elif kind == "byte":
        tok = ByteTokenizer()
        return tok, 256
    elif kind == "char":
        tok = CharTokenizer()
        return tok, None  # inferred from training text
    elif kind == "bpe":
        # your sentencepiece-based tokenizer
        tok = SentencePieceBPE("bpe.model")
        return tok, tok.vocab_size
    else:
        raise ValueError("Unknown tokenizer: " + kind)

def prepare(kind, vocab_size, raw_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # load text
    raw = load_raw_text(raw_path)

    # split
    train_txt, val_txt, test_txt = split_text(raw)

    # load tokenizer
    tokenizer, _ = get_tokenizer(kind, vocab_size)

    # train tokenizer if needed
    if kind == "byte_bpe":
        tokenizer.train(train_txt, vocab_size)
        tokenizer.save(os.path.join(out_dir, f"{kind}"))

    elif kind == "char":
        tokenizer.train(train_txt)
        tokenizer.save(os.path.join(out_dir, f"{kind}"))

    # encode all splits
    train_ids = np.array(tokenizer.encode(train_txt), dtype=np.int64)
    val_ids   = np.array(tokenizer.encode(val_txt), dtype=np.int64)
    test_ids  = np.array(tokenizer.encode(test_txt), dtype=np.int64)

    # save
    np.save(os.path.join(out_dir, f"{kind}_train.npy"), train_ids)
    np.save(os.path.join(out_dir, f"{kind}_val.npy"), val_ids)
    np.save(os.path.join(out_dir, f"{kind}_test.npy"), test_ids)

    print("Saved:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, help="one of: byte_bpe, byte, char, bpe")
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, default="data")
    args = parser.parse_args()

    prepare(args.kind, args.vocab_size, args.raw, args.out)
