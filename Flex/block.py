from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        norm = x.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x = x * torch.rsqrt(
            norm + self.eps
        )

        return x * self.weight



class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.

    (SiLU(W1x) * W2x)W3
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()

        self.gate = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.up = nn.Linear(
            d_model,
            d_ff,
            bias=False,
        )

        self.down = nn.Linear(
            d_ff,
            d_model,
            bias=False,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.down(
            torch.nn.functional.silu(
                self.gate(x)
            )
            *
            self.up(x)
        )



class TransformerBlock(nn.Module):
    """
    Single Flex decoder block.

    RMSNorm
    Attention
    Residual
    RMSNorm
    SwiGLU
    Residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()


        self.norm1 = RMSNorm(
            d_model
        )

        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )


        self.norm2 = RMSNorm(
            d_model
        )


        self.mlp = SwiGLU(
            d_model,
            d_ff,
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x + self.dropout(
            self.attention(
                self.norm1(x)
            )
        )


        x = x + self.dropout(
            self.mlp(
                self.norm2(x)
            )
        )

        return x
