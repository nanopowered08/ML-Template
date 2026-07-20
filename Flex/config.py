from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlexConfig:
    # Tokenizer
    vocab_size: int = 100285

    # Context
    max_seq_len: int = 512

    # Transformer
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024

    # Regularization
    dropout: float = 0.1

    # Attention
    bias: bool = False

    # Training
    pad_token_id: int = 100279

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads."
            )

        return self.d_model // self.n_heads
