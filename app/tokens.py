"""Token counting with the tiktoken cl100k_base encoding (TASK.pdf section 4.3)."""

from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text, disallowed_special=()))
