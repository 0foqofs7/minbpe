"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer, get_stats, merge, build_merge_lookup


def _merge_if_present(ids, pair, idx):
    newids = None
    i = 0
    n = len(ids)
    last = n - 1
    p0, p1 = pair
    while i < last:
        if ids[i] == p0 and ids[i+1] == p1:
            if newids is None:
                newids = ids[:i]
                append = newids.append
            append(idx)
            i += 2
        else:
            if newids is not None:
                append(ids[i])
            i += 1
    if i == last and newids is not None:
        append(ids[i])
    return ids if newids is None else newids


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks and encode them
        # identical chunks are kept once (in first-seen order) with a count
        chunk_counts = {}
        for chunk in self.compiled_pattern.findall(text):
            chunk_counts[chunk] = chunk_counts.get(chunk, 0) + 1
        ids = []
        counts = []
        for chunk, count in chunk_counts.items():
            chunk_ids = list(chunk.encode("utf-8"))
            if len(chunk_ids) >= 2:
                ids.append(chunk_ids)
                counts.append(count)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            stats_get = stats.get
            for chunk_ids, count in zip(ids, counts):
                it = iter(chunk_ids)
                prev = next(it)
                for curr in it:
                    pair = (prev, curr)
                    stats[pair] = stats_get(pair, 0) + count
                    prev = curr
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            p0, p1 = pair
            merged_ids = []
            for chunk_ids in ids:
                newids = None
                j = 0
                n = len(chunk_ids)
                last = n - 1
                while j < last:
                    if chunk_ids[j] == p0 and chunk_ids[j+1] == p1:
                        if newids is None:
                            newids = chunk_ids[:j]
                            append = newids.append
                        append(idx)
                        j += 2
                    else:
                        if newids is not None:
                            append(chunk_ids[j])
                        j += 1
                if j == last and newids is not None:
                    append(chunk_ids[j])
                merged_ids.append(chunk_ids if newids is None else newids)
            ids = merged_ids
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

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

    def _encode_chunk(self, text_bytes, merge_lookup=None):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        if merge_lookup is None:
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

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        merge_lookup = self._get_merge_lookup()
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
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
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
