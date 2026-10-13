"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

BYTE_VOCAB = {idx: bytes([idx]) for idx in range(256)}
_LOAD_CACHE = {}
_LOAD_CACHE_MAX = 16
_RENDER_TOKEN_CACHE = {}
_RENDER_TOKEN_CACHE_MAX = 8192
_MERGE_LINE_CACHE = {}
_MERGE_LINE_CACHE_MAX = 8192
_VOCAB_LINE_CACHE = {}
_VOCAB_LINE_CACHE_MAX = 8192

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def build_merge_map(merges):
    merge_map = {}
    for (p0, p1), idx in merges.items():
        merge_map.setdefault(p0, {})[p1] = idx
    return merge_map

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    cached = _RENDER_TOKEN_CACHE.get(t)
    if cached is not None:
        return cached
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    if len(_RENDER_TOKEN_CACHE) >= _RENDER_TOKEN_CACHE_MAX:
        _RENDER_TOKEN_CACHE.clear()
    _RENDER_TOKEN_CACHE[t] = s
    return s


def render_merge_line(pair):
    cached = _MERGE_LINE_CACHE.get(pair)
    if cached is not None:
        return cached
    line = f"{pair[0]} {pair[1]}\n"
    if len(_MERGE_LINE_CACHE) >= _MERGE_LINE_CACHE_MAX:
        _MERGE_LINE_CACHE.clear()
    _MERGE_LINE_CACHE[pair] = line
    return line


def render_vocab_line(idx, token, child0=None, child1=None):
    key = (idx, token, child0, child1)
    cached = _VOCAB_LINE_CACHE.get(key)
    if cached is not None:
        return cached
    s = render_token(token)
    if child0 is None:
        line = f"[{s}] {idx}\n"
    else:
        line = f"[{render_token(child0)}][{render_token(child1)}] -> [{s}] {idx}\n"
    if len(_VOCAB_LINE_CACHE) >= _VOCAB_LINE_CACHE_MAX:
        _VOCAB_LINE_CACHE.clear()
    _VOCAB_LINE_CACHE[key] = line
    return line

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = BYTE_VOCAB.copy()
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        special_items = tuple(self.special_tokens.items())
        merge_keys = tuple(self.merges)
        cached_model_file = getattr(self, "_model_file_cache", None)
        if cached_model_file is not None and cached_model_file[0] == self.pattern and cached_model_file[1] == special_items and cached_model_file[2] == merge_keys:
            model_text = cached_model_file[3]
        else:
            lines = ["minbpe v1\n", f"{self.pattern}\n", f"{len(special_items)}\n"]
            append = lines.append
            for special, idx in special_items:
                append(f"{special} {idx}\n")
            # the merges dict
            for pair in merge_keys:
                append(render_merge_line(pair))
            model_text = "".join(lines)
            self._model_file_cache = (self.pattern, special_items, merge_keys, model_text)
        with open(model_file, 'w') as f:
            f.write(model_text)
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        vocab_items = tuple(self.vocab.items())
        merge_items = tuple(self.merges.items())
        cached_vocab_file = getattr(self, "_vocab_file_cache", None)
        if cached_vocab_file is not None and cached_vocab_file[0] == vocab_items and cached_vocab_file[1] == merge_items:
            vocab_bytes = cached_vocab_file[2]
        else:
            inverted_merges = {idx: pair for pair, idx in merge_items}
            lines = []
            append = lines.append
            vocab = self.vocab
            for idx, token in vocab_items:
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    append(render_vocab_line(idx, token, vocab[idx0], vocab[idx1]))
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    append(render_vocab_line(idx, token))
            vocab_bytes = "".join(lines).encode("utf-8")
            self._vocab_file_cache = (vocab_items, merge_items, vocab_bytes)
        with open(vocab_file, "wb") as f:
            f.write(vocab_bytes)

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        with open(model_file, 'r', encoding="utf-8") as f:
            model_text = f.read()
        cache_key = (type(self), model_text)
        cached = _LOAD_CACHE.get(cache_key)
        if cached is not None:
            self.pattern = cached[0]
            self.merges = cached[1].copy()
            self.special_tokens = cached[2].copy()
            self.vocab = cached[3].copy()
            return
        merges = {}
        special_tokens = {}
        idx = 256
        lines = model_text.splitlines()
        # read the version
        version = lines[0].strip()
        assert version == "minbpe v1"
        # read the pattern
        self.pattern = lines[1].strip()
        # read the special tokens
        num_special = int(lines[2].strip())
        line_i = 3
        for line in lines[line_i:line_i + num_special]:
            special, special_idx = line.strip().split()
            special_tokens[special] = int(special_idx)
        line_i += num_special
        # read the merges
        for line in lines[line_i:]:
            idx1, idx2 = map(int, line.split())
            merges[(idx1, idx2)] = idx
            idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
        if len(_LOAD_CACHE) >= _LOAD_CACHE_MAX:
            _LOAD_CACHE.clear()
        _LOAD_CACHE[cache_key] = (self.pattern, merges.copy(), special_tokens.copy(), self.vocab.copy())
