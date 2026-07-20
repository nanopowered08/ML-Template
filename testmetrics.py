import torch

from common.metrics import (
    calculate_perplexity,
    calculate_accuracy,
    AverageMeter,
)


loss = torch.tensor(1.0)

print(
    calculate_perplexity(loss)
)


logits = torch.tensor(
    [
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.4, 0.6],
        ]
    ]
)

labels = torch.tensor(
    [
        [1, 0, -100]
    ]
)

print(
    calculate_accuracy(
        logits,
        labels,
    )
)


meter = AverageMeter()

meter.update(2)
meter.update(4)

print(
    meter.average
)
