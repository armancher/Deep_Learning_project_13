from .base_tokenizer import BaseTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BaselineBPETokenizer(BaseTokenizer):
    def __init__(self, text: str, vocab_size=32000):
        self.vocab_size_target = vocab_size

        # Initialize empty tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

        # Train
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
        )
        self.tokenizer.train_from_iterator([text], trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()