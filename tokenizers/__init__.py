from .byt5_tokenizer import ByT5Tokenizer
from .basic_bpe_tokenizer import BasicBPETokenizer
from .char_tokenizer import CharTokenizer


def get_tokenizer(name: str, text: str = None, vocab_size: int = 32000):

    name = name.lower()

    if name in ["byte", "byt5"]:
        return ByT5Tokenizer()

    if name == "char":
        if text is None:
            raise ValueError("Char tokenizer requires raw text.")
        return CharTokenizer(text)

    if name == "bpe":
        if text is None:
            raise ValueError("BPE tokenizer requires raw text.")
        return BasicBPETokenizer(text=text, vocab_size=vocab_size)

    raise ValueError(f"Unknown tokenizer: {name}")