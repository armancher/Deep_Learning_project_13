from .base_tokenizer import BaseTokenizer
from .basic_bpe import BasicBPE

class BasicBPETokenizer(BaseTokenizer):
    """
    Wrapper around BasicBPE to match our tokenizer interface.
    """

    def __init__(self, text: str, vocab_size: int = 32000):
        self.bpe = BasicBPE(vocab_size=vocab_size)
        self.bpe.symbol_to_id = {}
        self.bpe.id_to_symbol = {}
        self.bpe.train(text)
        self._vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return self.bpe.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.bpe.decode(ids)

    @property
    def vocab_size(self):
        return self._vocab_size