"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from collections import Counter
from itertools import islice

from .base import Tokenizer, get_stats, merge, build_merge_lookup


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        # the ids are kept as a str with one code point per token id, so that
        # pair counting and merging run in C (str.replace replaces
        # non-overlapping occurrences left to right, exactly like merge())
        ids = text_bytes.decode("latin-1")

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        # count up the number of times every consecutive pair appears; the
        # counts are then updated around the occurrences each merge replaces
        stats = dict(Counter(zip(ids, ids[1:])))
        stats_get = stats.get
        for i in range(num_merges):
            # find the pair with the highest count; ties go to the pair that
            # occurs first in ids, as with a freshly counted stats dict
            vals = list(stats.values())
            best = max(vals)
            if vals.count(best) == 1:
                cpair = next(islice(stats, vals.index(best), None))
            else:
                tied = [p for p, c in stats.items() if c == best]
                cpair = min(tied, key=lambda p: ids.find(p[0] + p[1]))
            pair = (ord(cpair[0]), ord(cpair[1]))
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            # (str.split, like str.replace, cuts non-overlapping occurrences
            # left to right; only the pairs next to an occurrence change)
            a, b = cpair
            new = chr(idx)
            parts = ids.split(a + b)
            removed = []
            added = []
            prev = parts[0]
            for k in range(1, len(parts)):
                part = parts[k]
                if prev:
                    c = prev[-1]
                    removed.append((c, a))
                    added.append((c, new))
                elif k > 1:
                    removed.append((b, a))
                    added.append((new, new))
                if part:
                    c = part[0]
                    removed.append((b, c))
                    added.append((new, c))
                prev = part
            for p in removed:
                c = stats[p] - 1
                if c:
                    stats[p] = c
                else:
                    del stats[p]
            # no occurrence of the merged pair is left
            del stats[cpair]
            for p in added:
                stats[p] = stats_get(p, 0) + 1
            ids = new.join(parts)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {best} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        # when every merge produces an id larger than both of its inputs (and
        # ids are distinct), the lowest-index pair present only ever increases,
        # so applying each merge once in index order gives the same result.
        # tokens are held one per character so each merge is a str.replace,
        # which also replaces left to right without overlaps.
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        if all(type(p0) is int and type(p1) is int and type(idx) is int
               and 0 <= p0 < idx and 0 <= p1 < idx and idx < 0x110000
               for (p0, p1), idx in ordered) and \
                len({idx for _, idx in ordered}) == len(ordered):
            s = text_bytes.decode("latin-1")
            for (p0, p1), idx in ordered:
                if len(s) < 2:
                    break
                s = s.replace(chr(p0) + chr(p1), chr(idx))
            return list(map(ord, s))
        merge_lookup = self._get_merge_lookup()
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            best_idx = None
            it = iter(ids)
            prev = next(it)
            for curr in it:
                nexts = merge_lookup.get(prev)
                if nexts is not None:
                    idx = nexts.get(curr)
                    if idx is not None and (best_idx is None or idx < best_idx):
                        best_p0 = prev
                        best_p1 = curr
                        best_idx = idx
                prev = curr
            if best_idx is None:
                break # nothing else can be merged anymore
            newids = []
            append = newids.append
            i = 0
            n = len(ids)
            last = n - 1
            while i < last:
                # if the pair matches, replace it
                if ids[i] == best_p0 and ids[i+1] == best_p1:
                    append(best_idx)
                    i += 2
                else:
                    append(ids[i])
                    i += 1
            if i == last:
                append(ids[i])
            ids = newids
        return ids
