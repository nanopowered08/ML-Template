from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a learning-rate scheduler with:

    - Linear warmup
    - Cosine decay

    Parameters
    ----------
    optimizer:
        Optimizer instance.

    warmup_steps:
        Number of warmup steps.

    total_steps:
        Total number of training steps.

    min_lr_ratio:
        Final LR as a fraction of the initial LR.
        Example:
            0.1 = decay to 10% of initial LR.
    """

    if total_steps <= 0:
        raise ValueError(
            "total_steps must be greater than zero."
        )

    if warmup_steps >= total_steps:
        raise ValueError(
            "warmup_steps must be smaller than total_steps."
        )

    def lr_lambda(step: int) -> float:

        # Warmup
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        # Cosine decay
        progress = (
            (step - warmup_steps)
            /
            (total_steps - warmup_steps)
        )

        cosine = 0.5 * (
            1.0
            + math.cos(math.pi * progress)
        )

        return (
            min_lr_ratio
            + (1.0 - min_lr_ratio) * cosine
        )

    return LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )
