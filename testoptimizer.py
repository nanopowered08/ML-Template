import torch
from torch import nn

from common.optimizer import create_optimizer


class TinyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            10,
            10
        )

        self.norm = nn.LayerNorm(
            10
        )


model = TinyModel()

optimizer = create_optimizer(
    model
)

print(type(optimizer))

print(
    len(optimizer.param_groups)
)

for group in optimizer.param_groups:
    print(
        group["weight_decay"]
    )
