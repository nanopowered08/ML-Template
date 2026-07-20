import torch

from common.tokenizer import Tokenizer

from Flex.config import FlexConfig
from Flex.model import FlexModel


tokenizer = Tokenizer()

config = FlexConfig(
    vocab_size=tokenizer.vocab_size
)


model = FlexModel(
    config
)


input_ids = torch.tensor(
    [
        [
            100277,
            882,
            198,
            13347,
            198,
            100278,
        ]
    ]
)


logits = model(
    input_ids
)


print("Input:")
print(input_ids.shape)

print()

print("Logits:")
print(logits.shape)

print()

print("Finite:")
print(torch.isfinite(logits).all())


params = sum(
    p.numel()
    for p in model.parameters()
)

print()

print("Parameters:")
print(params)
