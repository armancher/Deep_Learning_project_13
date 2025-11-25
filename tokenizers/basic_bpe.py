import re
from collections import Counter, defaultdict

class BasicBPE:
    """
    A minimal Byte Pair Encoding (BPE) implementation.
    Inspired by minBPE: simple, readable, and correct.
    """

    def __init__(self, vocab_size=32000):
        self.vocab_size_target = vocab_size
        self.merges = {}
        self.reverse_merges = {}
        self.vocab = set()

    # ------------------------------------
    # Helpers
    # ------------------------------------
    def _word_to_symbols(self, word):
        # Represent each character as individual symbol
        return list(word)

    def _get_stats(self, corpus):
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair, corpus):
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")

        merged_corpus = {}
        merged_symbol = "".join(pair)

        for word, freq in corpus.items():
            w = " ".join(word)
            w_merged = pattern.sub(merged_symbol, w)
            merged_corpus[tuple(w_merged.split(" "))] = freq

        return merged_corpus

    # ------------------------------------
    # Training
    # ------------------------------------
    def train(self, text):
        # Split text into words
        words = text.split()
        corpus = Counter()

        for w in words:
            corpus[tuple(self._word_to_symbols(w))] += 1

        while len(self.merges) < (self.vocab_size_target - 256):

            stats = self._get_stats(corpus)
            if not stats:
                break

            # Most frequent pair
            best_pair = max(stats, key=stats.get)

            self.merges[best_pair] = len(self.merges)
            self.reverse_merges["".join(best_pair)] = best_pair

            corpus = self._merge_pair(best_pair, corpus)

        # Save vocabulary
        self.vocab = set()
        for word in corpus:
            for symbol in word:
                self.vocab.add(symbol)

    # ------------------------------------
    # Encoding
    # ------------------------------------
    def encode_word(self, word):
        symbols = self._word_to_symbols(word)

        # Apply merges in merge-rank order
        merges_sorted = sorted(self.merges.items(), key=lambda x: x[1])
        for (a, b), _ in merges_sorted:
            merged = "".join((a, b))

            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text):
        ids = []
        for word in text.split():
            symbols = self.encode_word(word)
            for s in symbols:
                idx = self._get_id(s)
                ids.append(idx)
            ids.append(self._get_id(" "))  # space as token
        return ids

    # ------------------------------------
    # Decoding
    # ------------------------------------
    def decode(self, ids):
        tokens = [self._get_symbol(i) for i in ids]
        text = "".join(tokens)
        return text

    # ------------------------------------
    # Vocab mapping
    # ------------------------------------
    def _get_id(self, symbol):
        # map each symbol to unique integer ID
        if symbol not in self.symbol_to_id:
            self.symbol_to_id[symbol] = len(self.symbol_to_id)
            self.id_to_symbol[len(self.id_to_symbol)] = symbol
        return self.symbol_to_id[symbol]

    def _get_symbol(self, idx):
        return self.id_to_symbol[idx]

    @property
    def vocab_size(self):
        return len(self.vocab)