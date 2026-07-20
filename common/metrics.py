from __future__ import annotations

import math

import torch


def calculate_perplexity(
    loss: torch.Tensor | float,
) -> float:
    """
    Calculate perplexity from language model loss.

    Formula:
        PPL = exp(loss)

    Lower is better.
    """

    if isinstance(loss, torch.Tensor):
        loss_value = loss.item()
    else:
        loss_value = float(loss)

    if loss_value < 0:
        raise ValueError(
            "Loss cannot be negative"
        )

    return math.exp(loss_value)


def calculate_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Calculate token-level accuracy.

    Parameters
    ----------
    logits:
        Model predictions.

        Shape:
            [batch, sequence, vocab]

    labels:
        Target tokens.

        Shape:
            [batch, sequence]

    ignore_index:
        Tokens ignored during evaluation.
        Default matches PyTorch CrossEntropyLoss.
    """

    predictions = torch.argmax(
        logits,
        dim=-1,
    )

    mask = labels != ignore_index

    correct = (
        predictions[mask]
        ==
        labels[mask]
    )

    total = mask.sum().item()

    if total == 0:
        return 0.0

    return (
        correct.sum().item()
        /
        total
    )


class AverageMeter:
    """
    Tracks running averages.

    Example:
        meter.update(loss)
        print(meter.average)
    """

    def __init__(self) -> None:
        self.reset()


    def reset(self) -> None:
        """
        Clear stored values.
        """

        self.total = 0.0
        self.count = 0


    def update(
        self,
        value: float,
        n: int = 1,
    ) -> None:
        """
        Add a new value.

        Parameters
        ----------
        value:
            Metric value.

        n:
            Number of samples represented.
        """

        self.total += value * n
        self.count += n


    @property
    def average(self) -> float:
        """
        Current average.
        """

        if self.count == 0:
            return 0.0

        return self.total / self.count
