from .byte_tokenizer import ByteTokenizer
from .char_tokenizer import CharTokenizer
from .bpe_tokenizer import BaselineBPETokenizer
from .byte_bpe_tokenizer import ByteLevelBPETokenizer


def get_tokenizer(name: str, text: str = None, vocab_size: int = 32000):
    name = name.lower()

    if name == "byte":
        return ByteTokenizer()

    if name == "char":
        if text is None:
            raise ValueError("Char tokenizer requires the raw text to build vocab.")
        return CharTokenizer(text)

    if name == "bpe":
        if text is None:
            raise ValueError("BPE tokenizer requires text for training.")
        return BaselineBPETokenizer(text, vocab_size=vocab_size)

    if name == "byte_bpe":
        if text is None:
            raise ValueError("Byte-level BPE tokenizer requires text for training.")
        return ByteLevelBPETokenizer(text, vocab_size=vocab_size)

    raise ValueError(f"Unknown tokenizer '{name}'")