import math


def num_tokens_from_string(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len]
