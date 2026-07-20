from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from common.tokenizer import Tokenizer
from chattemplate.chatml import ChatMLFormatter


class ChatMLDataset(Dataset):
    """
    JSONL dataset loader for Fl-x models.

    Expected format:

    {
        "messages": [
            {
                "role": "user",
                "content": "Hello"
            },
            {
                "role": "assistant",
                "content": "Hi!"
            }
        ]
    }

    The dataset is loaded lazily to avoid storing
    the entire file in memory.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        max_length: int = 4096,
    ) -> None:

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.offsets = self._build_index()


    def _build_index(self) -> list[int]:
        """
        Creates a list of byte offsets.

        Allows random access without loading
        the entire dataset.
        """

        offsets = []

        with self.path.open(
            "rb"
        ) as file:

            while True:
                position = file.tell()

                line = file.readline()

                if not line:
                    break

                offsets.append(position)

        if len(offsets) == 0:
            raise ValueError(
                "Dataset is empty"
            )

        return offsets


    def _read_line(
        self,
        index: int,
    ) -> dict[str, Any]:

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            file.seek(
                self.offsets[index]
            )

            line = file.readline()

        return json.loads(line)


    def __len__(self) -> int:
        return len(self.offsets)


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:

        sample = self._read_line(index)

        if "messages" not in sample:
            raise ValueError(
                "Dataset entry missing 'messages'"
            )

        text = ChatMLFormatter.from_dicts(
            sample["messages"]
        )

        tokens = self.tokenizer.encode(text)

        # Truncate
        tokens = tokens[: self.max_length]

        input_ids = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

class ChatMLCollator:
    """
    Collates ChatMLDataset samples into padded batches.

    input_ids:
        Padding token used.

    labels:
        Padding uses -100 because PyTorch loss functions
        ignore this value.
    """

    def __init__(
        self,
        pad_token_id: int,
    ) -> None:

        self.pad_token_id = pad_token_id


    def __call__(
        self,
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:

        input_ids = [
            item["input_ids"]
            for item in batch
        ]

        labels = [
            item["labels"]
            for item in batch
        ]

        max_length = max(
            tensor.size(0)
            for tensor in input_ids
        )


        padded_input_ids = []
        padded_labels = []


        for ids, target in zip(
            input_ids,
            labels,
        ):

            padding_length = (
                max_length - ids.size(0)
            )


            padded_input_ids.append(
                torch.cat(
                    [
                        ids,
                        torch.full(
                            (
                                padding_length,
                            ),
                            self.pad_token_id,
                            dtype=torch.long,
                        ),
                    ]
                )
            )


            padded_labels.append(
                torch.cat(
                    [
                        target,
                        torch.full(
                            (
                                padding_length,
                            ),
                            -100,
                            dtype=torch.long,
                        ),
                    ]
                )
            )


        return {
            "input_ids": torch.stack(
                padded_input_ids
            ),
            "labels": torch.stack(
                padded_labels
            ),
        }