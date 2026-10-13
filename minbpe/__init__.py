from .base import Tokenizer
from .basic import BasicTokenizer
from .regex import RegexTokenizer

__all__ = ["Tokenizer", "BasicTokenizer", "RegexTokenizer", "GPT4Tokenizer"]


def __getattr__(name):
    if name == "GPT4Tokenizer":
        from .gpt4 import GPT4Tokenizer
        return GPT4Tokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
