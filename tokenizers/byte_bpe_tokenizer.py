from .base_tokenizer import BaseTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

class ByteLevelBPETokenizer(BaseTokenizer):
    def __init__(self, text: str, vocab_size=32000):
        self.vocab_size_target = vocab_size

        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
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