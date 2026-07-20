import torch
import torch.nn as nn

from .config import FlexConfig


class FlexModel(nn.Module):

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

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:

        return self.embedding(input_ids)
