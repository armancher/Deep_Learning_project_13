from .base_tokenizer import BaseTokenizer

class CharTokenizer(BaseTokenizer):
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self._vocab_size = len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self):
        return self._vocab_size