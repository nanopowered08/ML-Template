from __future__ import annotations

import torch
from torch import nn


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Creates AdamW optimizer with proper parameter groups.

    Parameters:
        model:
            PyTorch model.

        learning_rate:
            Initial learning rate.

        weight_decay:
            Decay applied only to suitable parameters.

        betas:
            Adam momentum values.

        eps:
            Numerical stability term.
    """

    decay = []
    no_decay = []


    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue


        if (
            name.endswith("bias")
            or "norm" in name.lower()
            or "ln" in name.lower()
        ):
            no_decay.append(parameter)

        else:
            decay.append(parameter)


    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer
