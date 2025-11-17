from .base import Tokenizer


class ByteTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text=None, vocab_size=256, verbose=False):
        # nothing to train, just keep vocab = bytes 0..255
        pass

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")