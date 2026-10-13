"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from collections import Counter

from .base import Tokenizer, get_stats, merge, build_merge_map


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
            stats = Counter(zip(ids, ids[1:]))
            # find the pair with the highest count
            pair = stats.most_common(1)[0][0]
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            newids = []
            append = newids.append
            p0, p1 = pair
            j = 0
            n = len(ids)
            while j < n:
                if ids[j] == p0 and j < n - 1 and ids[j+1] == p1:
                    append(idx)
                    j += 2
                else:
                    append(ids[j])
                    j += 1
            ids = newids
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
        if len(vocab) and max(vocab) == len(vocab) - 1 and isinstance(ids, (list, tuple)):
            vocab_list = [vocab[i] for i in range(len(vocab))]
            try:
                text_bytes = b"".join([vocab_list[idx] if idx >= 0 else vocab[idx] for idx in ids])
            except (IndexError, KeyError, TypeError):
                text_bytes = b"".join([vocab[idx] for idx in ids])
        else:
            text_bytes = b"".join([vocab[idx] for idx in ids])
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        merge_map = build_merge_map(self.merges)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            pair = None
            idx = None
            for i in range(len(ids) - 1):
                nexts = merge_map.get(ids[i])
                if nexts is None:
                    continue
                merge_idx = nexts.get(ids[i+1])
                if merge_idx is not None and (idx is None or merge_idx < idx):
                    pair = (ids[i], ids[i+1])
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
