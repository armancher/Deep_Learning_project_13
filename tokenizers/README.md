tokenizers/
│
├── __init__.py                # get_tokenizer registry
├── base_tokenizer.py          # tokenizer interface
│
├── byt5_tokenizer.py          # pure byte level (0–255)
├── char_tokenizer.py          # character tokenizer
├── basic_bpe.py               # BPE model implementation (minBPE-style)
└── basic_bpe_tokenizer.py     # tokenizer wrapper for BPE

