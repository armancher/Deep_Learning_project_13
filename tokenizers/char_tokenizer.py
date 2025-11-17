from .base import Tokenizer

class CharTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.char2idx = {}
        self.idx2char = {}

    def train(self, text, vocab_size=None, verbose=False):
        chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self.char2idx[c] for c in text]

    def decode(self, ids):
        return "".join(self.idx2char[i] for i in ids)
