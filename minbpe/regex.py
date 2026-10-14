"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

from collections import Counter
from heapq import heapify, heappop, heappush, heapreplace
from operator import add
from unicodedata import category
from .base import Tokenizer


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# the GPT-4 pattern for the standard library re module; it splits the same way
# on text whose characters all have one of the categories below (letters are
# then [^\W\d_], numbers \d, and \s agrees with the regex module), plus ASCII
# other than \x1c-\x1f (the possessive quantifiers never need to backtrack)
_GPT4_SPLIT_PATTERN_STD = r"""'(?i:[sdmt]|ll|ve|re)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d{1,3}| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
_STD_SAFE_CATEGORIES = frozenset(("Lu", "Ll", "Lt", "Lm", "Lo", "Nd",
                                  "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
                                  "Sm", "Sc", "Sk", "So"))


def _std_splits_alike(text):
    for c in set(text):
        if c < "\x80":
            if "\x1c" <= c <= "\x1f":
                return False
        elif c == "\u017f" or category(c) not in _STD_SAFE_CATEGORIES:
            return False
    return True


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        # the default pattern is compiled (with the regex module) on first use
        self._split_pattern = self.pattern
        self._compiled_pattern = None
        if self.pattern != GPT4_SPLIT_PATTERN:
            self.compiled_pattern
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    @property
    def compiled_pattern(self):
        if self._compiled_pattern is None:
            import regex as re
            self._compiled_pattern = re.compile(self._split_pattern)
        return self._compiled_pattern

    @compiled_pattern.setter
    def compiled_pattern(self, value):
        self._compiled_pattern = value

    def _split_chunks(self, text):
        if (self._compiled_pattern is None and
                self._split_pattern == GPT4_SPLIT_PATTERN and
                _std_splits_alike(text)):
            import re
            return re.findall(_GPT4_SPLIT_PATTERN_STD, text)
        return self.compiled_pattern.findall(text)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks and encode them
        # identical chunks are kept once (in first-seen order) with a count
        chunk_counts = Counter(self._split_chunks(text))
        ids = []
        counts = []
        for chunk, count in chunk_counts.items():
            # one code point per token id, so merges can use str.replace
            # (non-overlapping, left to right, exactly like merge())
            chunk_ids = chunk.encode("utf-8").decode("latin-1")
            if len(chunk_ids) >= 2:
                ids.append(chunk_ids)
                counts.append(count)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        # count the number of times every consecutive pair appears; the counts
        # are then kept up to date by re-counting only the chunks a merge changes
        # pair -> indices of the chunks that contain it (once per occurrence;
        # it may also list chunks that no longer hold the pair); a pair is
        # kept as the str of its two tokens, which caches its hash
        where = {}
        where_get = where.get
        for ci, chunk_ids in enumerate(ids):
            for pair in map(add, chunk_ids, chunk_ids[1:]):
                w = where_get(pair)
                if w is None:
                    where[pair] = [ci]
                else:
                    w.append(ci)
        counts_at = counts.__getitem__
        stats = {p: sum(map(counts_at, w)) for p, w in where.items()}
        stats_get = stats.get
        # lazy max-heap of (-count, pair); every pair in stats has an entry
        # at or above its count (entries are pushed only when a count rises)
        heap = [(-c, p) for p, c in stats.items()]
        heapify(heap)
        for i in range(num_merges):
            # find the pair with the highest count; ties go to the pair that
            # occurs first in the chunks, as with a freshly built stats dict
            if not stats:
                max(stats.values()) # nothing left to merge: raise as before
            while True:
                negc, p = heap[0]
                c = stats_get(p)
                if c == -negc:
                    break
                if c is None:
                    heappop(heap)
                else:
                    heapreplace(heap, (-c, p))
            best = -negc
            # take every current entry tied at the top count off the heap
            tied = []
            tied_seen = set()
            while heap and heap[0][0] == negc:
                p = heappop(heap)[1]
                c = stats_get(p)
                if c == best:
                    if p not in tied_seen:
                        tied.append(p)
                        tied_seen.add(p)
                elif c is not None:
                    heappush(heap, (-c, p))
            if len(tied) == 1:
                cpair = tied[0]
            else:
                # only the earliest chunk holding a tied pair needs scanning
                # (where may also list chunks that no longer hold the pair)
                cands = set()
                for p in tied:
                    cands.update(where[p])
                tied_set = set(tied)
                cpair = None
                while cpair is None:
                    ci = min(cands)
                    cands.discard(ci)
                    chunk_ids = ids[ci]
                    cpair = next((p for p in map(add, chunk_ids, chunk_ids[1:]) if p in tied_set), None)
                for p in tied:
                    if p != cpair:
                        heappush(heap, (negc, p))
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            # (str.split cuts non-overlapping occurrences left to right, like
            # merge(); only the pairs next to an occurrence change)
            a, b = cpair
            old = cpair
            new = chr(idx)
            # the chunks holding an occurrence, by the token before it (""
            # when that is the end of an adjacent occurrence) and by the token
            # after it; a chunk is listed once per occurrence
            left = {}
            left_get = left.get
            right = {}
            right_get = right.get
            for ci in where.pop(cpair):
                parts = ids[ci].split(old)
                if len(parts) == 1:
                    continue
                ids[ci] = new.join(parts)
                if len(parts) == 2:
                    # a single occurrence: at most one pair on each side
                    prev, part = parts
                    if prev:
                        c = prev[-1]
                        w = left_get(c)
                        if w is None:
                            left[c] = [ci]
                        else:
                            w.append(ci)
                    if part:
                        c = part[0]
                        w = right_get(c)
                        if w is None:
                            right[c] = [ci]
                        else:
                            w.append(ci)
                    continue
                # None marks the start of the chunk ("" marks two adjacent
                # occurrences)
                prev = parts[0] or None
                for part in parts[1:]:
                    if prev is not None:
                        c = prev[-1] if prev else ""
                        w = left_get(c)
                        if w is None:
                            left[c] = [ci]
                        else:
                            w.append(ci)
                    if part:
                        c = part[0]
                        w = right_get(c)
                        if w is None:
                            right[c] = [ci]
                        else:
                            w.append(ci)
                    prev = part
            # update the pair counts (the order of stats does not matter: ties
            # are settled above by chunk order); a lowered count keeps its
            # heap entry, and pairs with the new token did not exist before
            for c, w in left.items():
                n = sum(map(counts_at, w))
                if c:
                    p = c + a
                    q = c + new
                else:
                    p = b + a
                    q = new + new
                if p != cpair:
                    n0 = stats[p] - n
                    if n0:
                        stats[p] = n0
                    else:
                        del stats[p]
                stats[q] = n
                heappush(heap, (-n, q))
                where[q] = w
            for c, w in right.items():
                n = sum(map(counts_at, w))
                p = b + c
                if p != cpair:
                    n0 = stats[p] - n
                    if n0:
                        stats[p] = n0
                    else:
                        del stats[p]
                q = new + c
                stats[q] = n
                heappush(heap, (-n, q))
                where[q] = w
            # no occurrence of the merged pair is left
            del stats[cpair]
            # save the merge
            pair = (ord(cpair[0]), ord(cpair[1]))
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {best} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _get_merge_table(self):
        # the merge lookup as a list indexed by the first token id (None where
        # no merge starts with that id), so the encode scan can subscript it
        lookup = self._get_merge_lookup()
        if getattr(self, "_merge_table_src", None) is not lookup:
            size = 256
            for p0, nexts in lookup.items():
                size = max(size, p0 + 1, max(nexts.values()) + 1)
            table = [None] * size
            for p0, nexts in lookup.items():
                table[p0] = nexts
            self._merge_table = table
            self._merge_table_src = lookup
        return self._merge_table

    def _encode_chunk(self, text_bytes, merge_lookup=None):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        if type(merge_lookup) is not list:
            merge_lookup = self._get_merge_table()
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            best_idx = None
            it = iter(ids)
            prev = next(it)
            for curr in it:
                nexts = merge_lookup[prev]
                if nexts is not None:
                    idx = nexts.get(curr)
                    if idx is not None:
                        if best_idx is None or idx < best_idx:
                            best_p0 = prev
                            best_p1 = curr
                            best_idx = idx
                            repeated = False
                        elif prev == best_p0 and curr == best_p1:
                            repeated = True
                prev = curr
            if best_idx is None:
                break # nothing else can be merged anymore
            if not repeated:
                # the pair occurs exactly once: merge it in place
                i = ids.index(best_p0)
                while ids[i+1] != best_p1:
                    i = ids.index(best_p0, i + 1)
                ids[i:i+2] = [best_idx]
                continue
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

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = self.compiled_pattern.findall(text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        merge_lookup = self._get_merge_table()
        # identical chunks encode identically, so reuse results while the
        # merges are unchanged (the cache is reset whenever the lookup is)
        cache = self._chunk_cache
        if len(cache) > 100000:
            cache.clear()
        cache_get = cache.get
        for chunk in text_chunks:
            chunk_ids = cache_get(chunk)
            if chunk_ids is None:
                chunk_bytes = chunk.encode("utf-8") # raw bytes
                chunk_ids = self._encode_chunk(chunk_bytes, merge_lookup)
                cache[chunk] = chunk_ids
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        if all(type(k) is str and k for k in special):
            # split at the leftmost occurrence of any special token (the first
            # listed one on a tie), which is what the alternation below matches
            ids = []
            start = 0
            while True:
                pos = -1
                for k in special:
                    j = text.find(k, start)
                    if j != -1 and (pos == -1 or j < pos):
                        pos = j
                        tok = k
                if pos == -1:
                    break
                ids.extend(self.encode_ordinary(text[start:pos]))
                ids.append(special[tok])
                start = pos + len(tok)
            ids.extend(self.encode_ordinary(text[start:]))
            return ids
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        import regex as re
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
