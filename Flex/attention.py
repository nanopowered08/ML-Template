from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import (
    RotaryEmbedding,
    apply_rope,
)


class MultiHeadAttention(nn.Module):
    """
    Decoder-only multi-head self attention
    with RoPE and causal masking.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads


        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=bias,
        )


        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len,
        )

        self.dropout = nn.Dropout(
            dropout
        )


        mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len,
            )
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False,
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch, seq_len, _ = x.shape


        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        # [B,T,D] -> [B,H,T,HD]

        q = q.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)


        cos, sin = self.rope(q)


        q = apply_rope(
            q,
            cos,
            sin,
        )

        k = apply_rope(
            k,
            cos,
            sin,
        )


        scores = (
            q @ k.transpose(-2, -1)
        )

        scores = scores / (
            self.head_dim ** 0.5
        )


        mask = self.causal_mask[
            :seq_len,
            :seq_len,
        ]

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )


        attention = F.softmax(
            scores,
            dim=-1,
        )

        attention = self.dropout(
            attention
        )


        output = attention @ v


        output = output.transpose(
            1,
            2,
        ).contiguous()


        output = output.view(
            batch,
            seq_len,
            self.d_model,
        )


        return self.out_proj(output)
