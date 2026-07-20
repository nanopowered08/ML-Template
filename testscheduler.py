import torch
from torch import nn

from common.optimizer import create_optimizer
from common.scheduler import create_scheduler


model = nn.Linear(
    10,
    10,
)

optimizer = create_optimizer(
    model,
    learning_rate=1e-3,
)

scheduler = create_scheduler(
    optimizer,
    warmup_steps=5,
    total_steps=20,
)

print("Learning Rates:")

for step in range(20):

    lr = optimizer.param_groups[0]["lr"]

    print(
        f"{step:02d}: {lr:.6f}"
    )

    optimizer.step()
    scheduler.step()
