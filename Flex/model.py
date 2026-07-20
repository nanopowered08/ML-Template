from __future__ import annotations

import torch
import torch.nn as nn

from .config import FlexConfig
from .block import TransformerBlock, RMSNorm


class FlexModel(nn.Module):
    """
    Flex decoder-only Transformer.
    """

    def __init__(
        self,
        config: FlexConfig,
    ):
        super().__init__()

        self.config = config


        # Token embeddings
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )


        # Transformer stack
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.n_layers)
            ]
        )


        # Final normalization
        self.norm = RMSNorm(
            config.d_model
        )


        # Language modeling head
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )


        # Weight tying
        self.lm_head.weight = (
            self.embedding.weight
        )


    def forward(
        self,
        input_ids: torch.Tensor,
    ):

        x = self.embedding(
            input_ids
        )


        for block in self.blocks:
            x = block(x)


        x = self.norm(x)


        logits = self.lm_head(
            x
        )


        return logits
