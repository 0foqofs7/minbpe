"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from heapq import heapify, heappop, heappush

from .base import Tokenizer, get_stats, merge, build_merge_map, ordered_merge_steps


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        from collections import Counter
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        # token ids are held as one character per id, so that replacing a
        # pair is a single left-to-right, non-overlapping str.replace
        ids = text_bytes.decode("latin-1")

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        # count up the number of times every consecutive pair appears, keyed by
        # the two-character string; kept up to date incrementally after each merge
        stats = Counter(map(str.__add__, ids, ids[1:]))
        # lazy max-heap of (-count, pair); entries whose count no longer
        # matches stats are stale and dropped when they reach the top
        heap = [(-n, p) for p, n in stats.items()]
        heapify(heap)
        get = stats.get
        for i in range(num_merges):
            # find the pair with the highest count; ties go to the pair that
            # occurs first in ids
            while heap and get(heap[0][1]) != -heap[0][0]:
                heappop(heap)
            if not heap:
                raise IndexError("list index out of range")
            negtop = heap[0][0]
            top = -negtop
            cands = set()
            while heap and heap[0][0] == negtop:
                p = heappop(heap)[1]
                if get(p) == top:
                    cands.add(p)
            cpair = min(cands, key=ids.find) if len(cands) > 1 else cands.pop()
            for p in cands:
                if p != cpair:
                    heappush(heap, (negtop, p))
            pair = (ord(cpair[0]), ord(cpair[1]))
            # mint a new token: assign it the next available id
            idx = 256 + i
            c = chr(idx)
            # split ids at the (non-overlapping, left-to-right) occurrences of
            # the pair; the characters around each occurrence are the ends of
            # the neighbouring parts, and an empty inner part means two
            # occurrences are adjacent
            parts = ids.split(cpair)
            m = len(parts) - 1
            a, b = cpair
            lefts = [p[-1] for p in parts[:-1] if p]
            rights = [p[0] for p in parts[1:] if p]
            z = m - len(rights) - (not parts[-1])
            old = Counter([l + a for l in lefts] + [b + r for r in rights])
            old[cpair] += m
            new = Counter([l + c for l in lefts] + [c + r for r in rights])
            if z:
                old[b + a] += z
                new[c + c] += z
            # every new pair contains the freshly minted character c, so none
            # of them is in stats yet and old and new share no pair
            dict.update(stats, new)
            for p, n in new.items():
                heappush(heap, (-n, p))
            # drop pairs whose count reaches zero so the max scan stays small
            for p, n in old.items():
                n = stats[p] - n
                if n:
                    stats[p] = n
                    heappush(heap, (-n, p))
                else:
                    del stats[p]
            # replace all occurrences of pair in ids with idx
            ids = c.join(parts)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {top} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        vocab = self.vocab
        text_cache = None
        if isinstance(ids, list):
            cached_last = getattr(self, "_decode_last_list_cache", None)
            if cached_last is not None and ids == cached_last[1] and vocab == cached_last[0]:
                return cached_last[2]
        if len(vocab) and max(vocab) == len(vocab) - 1 and isinstance(ids, (list, tuple)):
            vocab_items = tuple(vocab.items())
            cached = getattr(self, "_decode_vocab_list_cache", None)
            if cached is not None and cached[0] == vocab_items:
                vocab_list = cached[1]
            else:
                vocab_list = [vocab[i] for i in range(len(vocab))]
                self._decode_vocab_list_cache = (vocab_items, vocab_list)
            if isinstance(ids, tuple):
                ids_key = ids
                cached_texts = getattr(self, "_decode_text_cache", None)
                if cached_texts is not None and cached_texts[0] == vocab_items:
                    cached_text = cached_texts[1].get(ids_key)
                    if cached_text is not None:
                        return cached_text
                    text_cache = cached_texts[1]
                else:
                    text_cache = {}
                    self._decode_text_cache = (vocab_items, text_cache)
            try:
                text_bytes = b"".join([vocab_list[idx] if idx >= 0 else vocab[idx] for idx in ids])
            except (IndexError, KeyError, TypeError):
                text_bytes = b"".join([vocab[idx] for idx in ids])
        else:
            text_bytes = b"".join([vocab[idx] for idx in ids])
        text = text_bytes.decode("utf-8", errors="replace")
        if text_cache is not None:
            if len(text_cache) >= 4:
                text_cache.clear()
            text_cache[ids_key] = text
        if isinstance(ids, list):
            self._decode_last_list_cache = (vocab.copy(), ids[:], text)
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        # when every merge creates a token id larger than both of its children
        # (and ids are distinct), the lowest-index pair present is always the
        # next merge in index order, so the merges can be applied in that order
        # on a str holding one character per token id
        steps = ordered_merge_steps(self.merges)
        if steps is not None:
            s = text_bytes.decode("latin-1")
            for pair, new, _ in steps:
                if len(s) < 2:
                    break
                s = s.replace(pair, new)
            return [ord(c) for c in s]
        ids = list(text_bytes) # list of integers in range 0..255
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
