from __future__ import annotations

import torch
import torch.nn as nn

from .config import FlexConfig
from .block import TransformerBlock, RMSNorm


class FlexModel(nn.Module):
    """
    Flex decoder-only Transformer model.
    """

    def __init__(
        self,
        config: FlexConfig,
    ):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

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

        self.norm = RMSNorm(
            config.d_model
        )

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        # Tie input and output embeddings
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)


    def _init_weights(
        self,
        module,
    ):
        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )


        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )


    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        x = self.embedding(
            input_ids
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(
            x
        )

        logits = self.lm_head(
            x
        )

        return logits
