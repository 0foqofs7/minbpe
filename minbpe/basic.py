"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        vocab = self.vocab
        text_bytes = b"".join([vocab[idx] for idx in ids])
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        merges = self.merges
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            pair = None
            idx = None
            for i in range(len(ids) - 1):
                p = (ids[i], ids[i+1])
                merge_idx = merges.get(p)
                if merge_idx is not None and (idx is None or merge_idx < idx):
                    pair = p
                    idx = merge_idx
            # if there are no more merges available, break
            if pair is None:
                break # nothing else can be merged anymore
            newids = []
            append = newids.append
            i = 0
            n = len(ids)
            p0, p1 = pair
            while i < n:
                if ids[i] == p0 and i < n - 1 and ids[i+1] == p1:
                    append(idx)
                    i += 2
                else:
                    append(ids[i])
                    i += 1
            ids = newids
        return ids
