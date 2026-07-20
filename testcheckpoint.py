import torch
from torch import nn

from common.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


model = nn.Linear(
    10,
    10,
)

optimizer = torch.optim.AdamW(
    model.parameters()
)


save_checkpoint(
    "checkpoints/test.pt",
    model,
    optimizer,
    epoch=5,
    step=100,
    config={
        "model": "tiny"
    },
)


model2 = nn.Linear(
    10,
    10,
)

optimizer2 = torch.optim.AdamW(
    model2.parameters()
)


info = load_checkpoint(
    "checkpoints/test.pt",
    model2,
    optimizer2,
)


print(info["epoch"])
print(info["step"])
print(info["config"])
