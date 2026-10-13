"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

from .base import Tokenizer, get_stats, build_merge_map, ordered_merge_steps


class _LazyRegexModule:
    # imports the regex module on first use (save/load never need it)
    def __getattr__(self, name):
        import regex
        globals()["re"] = regex
        return getattr(regex, name)

re = _LazyRegexModule()


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
        # the split pattern is compiled on first use (save/load never need it)
        self._compiled_source = self.pattern
        self._compiled_pattern = None
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.inverse_special_tokens_bytes = {}

    @property
    def compiled_pattern(self):
        compiled = self._compiled_pattern
        if compiled is None:
            compiled = self._compiled_pattern = re.compile(self._compiled_source)
        return compiled

    @compiled_pattern.setter
    def compiled_pattern(self, value):
        self._compiled_pattern = value

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
        # each chunk is kept as a str with one character per token id
        ids = []
        weights = []
        for ch, count in chunk_counts.items():
            chunk_ids = ch.encode("utf-8").decode("latin-1")
            if len(chunk_ids) > 1:
                ids.append(chunk_ids)
                weights.append(count)

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        # count the number of times every consecutive pair appears, once;
        # where maps each pair to the indices of the chunks containing it
        stats = {}
        where = {}
        for j, (chunk_ids, w) in enumerate(zip(ids, weights)):
            for cpair in zip(chunk_ids, chunk_ids[1:]):
                stats[cpair] = stats.get(cpair, 0) + w
                if cpair in where:
                    where[cpair].add(j)
                else:
                    where[cpair] = {j}
        import heapq
        # lazy max-heap of (-count, pair); an entry is current only while
        # stats[pair] still equals its count, stale entries are skipped
        heap = [(-c, p) for p, c in stats.items()]
        heapq.heapify(heap)
        heappush = heapq.heappush
        heappop = heapq.heappop
        for i in range(num_merges):
            if not stats:
                raise ValueError("max() arg is an empty sequence")
            # find the pair with the highest count; ties go to the pair
            # that occurs first in chunk order
            while stats.get(heap[0][1]) != -heap[0][0]:
                heappop(heap)
            top = -heap[0][0]
            tied = []
            seen = set()
            while heap and heap[0][0] == -top:
                p = heappop(heap)[1]
                if p not in seen and stats.get(p) == top:
                    seen.add(p)
                    tied.append(p)
            for p in tied:
                heappush(heap, (-top, p))
            touched = set()
            if len(tied) == 1:
                cpair = tied[0]
            else:
                first = min(min(where[p]) for p in tied)
                tied = set(tied)
                chunk_ids = ids[first]
                cpair = next(p for p in zip(chunk_ids, chunk_ids[1:]) if p in tied)
            pair = (ord(cpair[0]), ord(cpair[1]))
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in the affected chunks with idx
            # (str.replace is left to right and non-overlapping),
            # updating the pair counts of those chunks only
            old = cpair[0] + cpair[1]
            new = chr(idx)
            for j in sorted(where[cpair]):
                chunk_ids = ids[j]
                w = weights[j]
                new_ids = chunk_ids.replace(old, new)
                ids[j] = new_ids
                for p in zip(chunk_ids, chunk_ids[1:]):
                    stats[p] -= w
                for p in zip(new_ids, new_ids[1:]):
                    if p in stats:
                        stats[p] += w
                    else:
                        stats[p] = w
                old_pairs = set(zip(chunk_ids, chunk_ids[1:]))
                new_pairs = set(zip(new_ids, new_ids[1:]))
                for p in old_pairs - new_pairs:
                    where[p].discard(j)
                    if not stats[p]:
                        del stats[p]
                        del where[p]
                for p in new_pairs - old_pairs:
                    if p in where:
                        where[p].add(j)
                    else:
                        where[p] = {j}
                touched |= old_pairs
                touched |= new_pairs
            for p in touched:
                c = stats.get(p)
                if c:
                    heappush(heap, (-c, p))
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")

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
        if type(self)._encode_chunk is RegexTokenizer._encode_chunk:
            # when every merge creates a token id larger than both of its children
            # (and ids are distinct), the merges can be applied in index order with
            # str.replace; the unique chunks are joined by a separator character
            # above every token id, so no pair can span two chunks
            steps = ordered_merge_steps(self.merges)
            if steps is not None and (not steps or (steps[0][2] > 255 and steps[-1][2] < 0x10ffff)):
                sep = chr(steps[-1][2] + 1 if steps else 256)
                unique = list(dict.fromkeys(text_chunks))
                s = sep.join([ch.encode("utf-8").decode("latin-1") for ch in unique])
                for pair, new, _ in steps:
                    if pair in s:
                        s = s.replace(pair, new)
                piece_of = dict(zip(unique, s.split(sep)))
                return list(map(ord, "".join(map(piece_of.__getitem__, text_chunks))))
        # all chunks of text are encoded separately, then results are joined
        ids = []
        merge_map = build_merge_map(self.merges)
        # identical chunks encode to identical ids, so each is encoded once
        cache = {}
        extend = ids.extend
        for chunk in text_chunks:
            chunk_ids = cache.get(chunk)
            if chunk_ids is None:
                chunk_bytes = chunk.encode("utf-8") # raw bytes
                chunk_ids = self._encode_chunk(chunk_bytes, merge_map)
                cache[chunk] = chunk_ids
            extend(chunk_ids)
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
