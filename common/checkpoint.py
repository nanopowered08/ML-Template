from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    step: int = 0,
    config: dict | None = None,
) -> None:
    """
    Save training state.
    """

    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }


    if optimizer is not None:
        checkpoint["optimizer"] = (
            optimizer.state_dict()
        )


    if scheduler is not None:
        checkpoint["scheduler"] = (
            scheduler.state_dict()
        )


    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str = "cpu",
) -> dict:
    """
    Load training state.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
    )


    model.load_state_dict(
        checkpoint["model"]
    )


    if (
        optimizer is not None
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )


    if (
        scheduler is not None
        and "scheduler" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )


    return checkpoint
