"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from .base import Tokenizer, get_stats, build_merge_map


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
        self.inverse_special_tokens_bytes = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing: identical chunks are processed once,
        # weighted by their number of occurrences (first-seen order is kept)
        chunk_counts = {}
        for ch in text_chunks:
            chunk_counts[ch] = chunk_counts.get(ch, 0) + 1
        ids = []
        weights = []
        for ch, count in chunk_counts.items():
            chunk_ids = list(ch.encode("utf-8"))
            if len(chunk_ids) > 1:
                ids.append(chunk_ids)
                weights.append(count)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids, w in zip(ids, weights):
                # count pairs in this chunk, adding up counts across chunks
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] = stats.get(pair, 0) + w
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            new_ids = []
            new_weights = []
            p0, p1 = pair
            for chunk_ids, w in zip(ids, weights):
                j = 0
                n = len(chunk_ids)
                last = n - 1
                while j < last:
                    if chunk_ids[j] == p0 and chunk_ids[j+1] == p1:
                        new_chunk_ids = chunk_ids[:j]
                        append = new_chunk_ids.append
                        append(idx)
                        j += 2
                        while j < last:
                            if chunk_ids[j] == p0 and chunk_ids[j+1] == p1:
                                append(idx)
                                j += 2
                            else:
                                append(chunk_ids[j])
                                j += 1
                        if j == last:
                            append(chunk_ids[j])
                        if len(new_chunk_ids) > 1:
                            new_ids.append(new_chunk_ids)
                            new_weights.append(w)
                        break
                    j += 1
                else:
                    new_ids.append(chunk_ids)
                    new_weights.append(w)
            ids = new_ids
            weights = new_weights
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
        self.inverse_special_tokens_bytes = {v: k.encode("utf-8") for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        vocab = self.vocab
        text_cache = None
        if isinstance(ids, list):
            cached_recent = getattr(self, "_decode_recent_list_cache", None)
            if cached_recent is not None:
                for cached_last in cached_recent:
                    if ids == cached_last[2]:
                        special_tokens = self.inverse_special_tokens_bytes
                        if vocab == cached_last[0] and special_tokens == cached_last[1]:
                            return cached_last[3]
        part_bytes = []
        append = part_bytes.append
        if len(vocab) and max(vocab) == len(vocab) - 1:
            special_tokens = self.inverse_special_tokens_bytes
            vocab_items = tuple(vocab.items())
            special_items = tuple(special_tokens.items())
            cached = getattr(self, "_decode_lookup_cache", None)
            if cached is not None and cached[0] == vocab_items and cached[1] == special_items:
                lookup = cached[2]
            else:
                max_idx = max(len(vocab) - 1, max(special_tokens, default=0))
                lookup = [None] * (max_idx + 1)
                for i in range(len(vocab)):
                    lookup[i] = vocab[i]
                for idx, token in special_tokens.items():
                    if idx >= 0:
                        lookup[idx] = token
                self._decode_lookup_cache = (vocab_items, special_items, lookup)
            if isinstance(ids, tuple):
                ids_key = ids
                cached_texts = getattr(self, "_decode_text_cache", None)
                if cached_texts is not None and cached_texts[0] == vocab_items and cached_texts[1] == special_items:
                    cached_text = cached_texts[2].get(ids_key)
                    if cached_text is not None:
                        return cached_text
                    text_cache = cached_texts[2]
                else:
                    text_cache = {}
                    self._decode_text_cache = (vocab_items, special_items, text_cache)
            try:
                part_bytes = [lookup[idx] if idx >= 0 else special_tokens[idx] for idx in ids]
                text_bytes = b"".join(part_bytes)
            except (IndexError, KeyError, TypeError):
                part_bytes = []
                append = part_bytes.append
                vocab_list = [vocab[i] for i in range(len(vocab))]
                for idx in ids:
                    if idx >= 0:
                        try:
                            append(vocab_list[idx])
                            continue
                        except IndexError:
                            pass
                    if idx in special_tokens:
                        append(special_tokens[idx])
                    else:
                        raise ValueError(f"invalid token id: {idx}")
            else:
                text = text_bytes.decode("utf-8", errors="replace")
                if text_cache is not None:
                    if len(text_cache) >= 4:
                        text_cache.clear()
                    text_cache[ids_key] = text
                if isinstance(ids, list):
                    entry = (vocab.copy(), special_tokens.copy(), ids[:], text)
                    recent = getattr(self, "_decode_recent_list_cache", None)
                    self._decode_recent_list_cache = (entry,) if recent is None else (entry,) + recent[:1]
                return text
        else:
            special_tokens = self.inverse_special_tokens
            for idx in ids:
                if idx in vocab:
                    append(vocab[idx])
                elif idx in special_tokens:
                    append(special_tokens[idx].encode("utf-8"))
                else:
                    raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        if text_cache is not None:
            if len(text_cache) >= 4:
                text_cache.clear()
            text_cache[ids_key] = text
        if isinstance(ids, list):
            entry = (vocab.copy(), special_tokens.copy(), ids[:], text)
            recent = getattr(self, "_decode_recent_list_cache", None)
            self._decode_recent_list_cache = (entry,) if recent is None else (entry,) + recent[:1]
        return text

    def _encode_chunk(self, text_bytes, merge_map=None):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        if merge_map is None:
            merge_map = build_merge_map(self.merges)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            p0 = None
            p1 = None
            idx = None
            pair_pos = None
            for i in range(len(ids) - 1):
                nexts = merge_map.get(ids[i])
                if nexts is None:
                    continue
                merge_idx = nexts.get(ids[i+1])
                if merge_idx is not None and (idx is None or merge_idx < idx):
                    p0 = ids[i]
                    p1 = ids[i+1]
                    idx = merge_idx
                    pair_pos = i
            # if there are no more merges available, break
            if p0 is None:
                break # nothing else can be merged anymore
            newids = ids[:pair_pos]
            append = newids.append
            append(idx)
            i = pair_pos + 2
            n = len(ids)
            last = n - 1
            while i < last:
                if ids[i] == p0 and ids[i+1] == p1:
                    append(idx)
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
        merge_map = build_merge_map(self.merges)
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes, merge_map)
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
