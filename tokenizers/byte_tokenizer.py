from .base_tokenizer import BaseTokenizer

class ByteTokenizer(BaseTokenizer):
    def __init__(self):
        self._vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return self._vocab_size