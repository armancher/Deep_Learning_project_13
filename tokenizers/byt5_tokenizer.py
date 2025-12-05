from .base_tokenizer import BaseTokenizer

class ByT5Tokenizer(BaseTokenizer):
    """
    Pure byte-level tokenizer like ByT5.
    Token IDs = raw bytes 0–255.
    """

    def __init__(self):
        self._vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self):
        return self._vocab_size