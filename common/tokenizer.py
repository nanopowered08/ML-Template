from __future__ import annotations

from typing import Iterable

import tiktoken
from tiktoken import Encoding


class Tokenizer:
    """
    Wrapper around tiktoken.

    All Fl-x models should use this interface instead of
    calling tiktoken directly.

    Responsibilities
    ----------------
    - Encode text -> token IDs
    - Decode token IDs -> text
    - Manage special ChatML tokens
    - Expose vocabulary size
    """

    DEFAULT_SPECIAL_TOKENS = [
        "<|im_start|>",
        "<|im_end|>",
        "<|pad|>",
        "<|eos|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ]

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        special_tokens: Iterable[str] | None = None,
    ) -> None:

        base_encoding = tiktoken.get_encoding(encoding_name)

        self.special_tokens = list(
            special_tokens or self.DEFAULT_SPECIAL_TOKENS
        )

        # Assign IDs after the existing vocabulary
        special_token_ids = {
            token: base_encoding.n_vocab + i
            for i, token in enumerate(self.special_tokens)
        }

        self.encoding = Encoding(
            name=f"{encoding_name}_flx",
            pat_str=base_encoding._pat_str,
            mergeable_ranks=base_encoding._mergeable_ranks,
            special_tokens={
                **base_encoding._special_tokens,
                **special_token_ids,
            },
        )
    
    def has_special_tokens(self) -> bool:
        """
        Check that all Fl-x special tokens exist
        as single tokens.
        """

        for token in self.special_tokens:
            ids = self.encode(token)

            if len(ids) != 1:
                return False
        return True

    @property
    def pad_token_id(self) -> int:
        return self.encoding.encode(
            "<|pad|>",
            allowed_special="all"
        )[0]


    @property
    def eos_token_id(self) -> int:
       return self.encoding.encode(
           "<|eos|>",
           allowed_special="all"
       )[0]

    @property
    def vocab_size(self) -> int:
        return self.encoding.max_token_value + 1

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return self.encoding.encode(
            text,
            allowed_special="all",
        )

    def decode(self, tokens: Iterable[int]) -> str:
        """
        Decode token IDs back into text.
        """

        return self.encoding.decode(list(tokens))

    def token_count(self, text: str) -> int:
        return len(self.encode(text))

    def __len__(self) -> int:
        return self.vocab_size