from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass(slots=True)
class ChatMessage:
    """
    Represents a single ChatML message.
    """

    role: str
    content: str


class ChatMLFormatter:
    """
    Formats conversations into ChatML.

    Every Fl-x model should use this class before tokenization.
    """

    VALID_ROLES = {
        "system",
        "user",
        "assistant",
        "tool",
    }

    @classmethod
    def format(
        cls,
        messages: Iterable[ChatMessage],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Convert a list of ChatMessage objects into ChatML.

        Parameters
        ----------
        messages:
            Iterable of ChatMessage objects.

        add_generation_prompt:
            If True, append an assistant header so the model
            knows to generate the assistant response.

        Returns
        -------
        str
            ChatML formatted conversation.
        """

        output: list[str] = []

        for message in messages:

            if message.role not in cls.VALID_ROLES:
                raise ValueError(
                    f"Unknown role '{message.role}'"
                )

            output.append(
                f"{IM_START}{message.role}\n"
                f"{message.content.strip()}\n"
                f"{IM_END}"
            )

        if add_generation_prompt:
            output.append(f"{IM_START}assistant\n")

        return "\n\n".join(output)

    @classmethod
    def from_dicts(
        cls,
        messages: list[dict],
        add_generation_prompt: bool = False,
    ) -> str:
        """
        Convenience wrapper for OpenAI-style dictionaries.

        Example
        -------
        [
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
        """

        parsed = [
            ChatMessage(
                role=msg["role"],
                content=msg["content"],
            )
            for msg in messages
        ]

        return cls.format(
            parsed,
            add_generation_prompt=add_generation_prompt,
        )