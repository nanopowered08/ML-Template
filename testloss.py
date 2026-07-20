import torch
import torch.nn as nn

from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset
from common.dataset import ChatMLCollator

from Flex.config import FlexConfig
from Flex.model import FlexModel

from common.optimizer import create_optimizer


# Device
device = "cpu"


# Tokenizer
tokenizer = Tokenizer()


# Config
config = FlexConfig(
    vocab_size=tokenizer.vocab_size
)


# Model
model = FlexModel(
    config
).to(device)


# Dataset
dataset = ChatMLDataset(
    "data/data.jsonl",
    tokenizer,
)


# Take one sample
sample = dataset[0]


input_ids = sample["input_ids"].unsqueeze(0).to(device)


# For causal LM:
# predict next token
labels = input_ids.clone()

labels[:, :-1] = input_ids[:, 1:]
labels[:, -1] = -100

print("Input:")
print(input_ids.shape)


# Forward
logits = model(
    input_ids
)


print()

print("Logits:")
print(logits.shape)

print(
    "Logit range:",
    logits.min().item(),
    logits.max().item()
)

# Loss
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(
    logits.view(
        -1,
        config.vocab_size,
    ),
    labels.view(-1),
)


print()

print("Loss:")
print(loss.item())


print()

print("Finite loss:")
print(torch.isfinite(loss))


# Backprop
loss.backward()


has_grad = True

for name, param in model.named_parameters():

    if param.grad is None:
        print(
            "Missing gradient:",
            name,
        )
        has_grad = False
        break


print()

print("Gradients:")
print(has_grad)


# Optimizer step
optimizer = create_optimizer(
    model,
    learning_rate=1e-4,
)


optimizer.step()

print()

print("Optimizer step:")
print("Success")
