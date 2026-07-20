from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)

    Applies rotary position encoding to query and key tensors.

    Expected input:
        [batch, heads, seq_len, head_dim]

    head_dim must be even.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                "RoPE dimension must be even."
            )

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta


        inv_freq = 1.0 / (
            theta ** (
                torch.arange(
                    0,
                    dim,
                    2,
                ).float()
                /
                dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )


    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Generate cosine and sine values.

        x:
            [batch, heads, seq, dim]
        """

        seq_len = x.shape[-2]

        positions = torch.arange(
            seq_len,
            device=x.device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(
            positions,
            self.inv_freq,
        )

        emb = torch.cat(
            [
                freqs,
                freqs,
            ],
            dim=-1,
        )

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin



def rotate_half(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate pairs:

    [x1, x2]
    becomes
    [-x2, x1]
    """

    x1 = x[..., : x.shape[-1] // 2]

    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat(
        [
            -x2,
            x1,
        ],
        dim=-1,
    )


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embedding.

    x:
        [batch, heads, seq, dim]

    cos/sin:
        [seq, dim]
    """

    cos = cos[
        None,
        None,
        :,
        :
    ]

    sin = sin[
        None,
        None,
        :,
        :
    ]

    return (
        x * cos
        +
        rotate_half(x) * sin
    )
