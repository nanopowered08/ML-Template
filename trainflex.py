import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset
from common.optimizer import create_optimizer

from Flex.config import FlexConfig
from Flex.model import FlexModel


device = "cpu"


# Settings
epochs = 35
learning_rate = 1e-2


# Tokenizer
tokenizer = Tokenizer()


# Dataset
dataset = ChatMLDataset(
    "data/data.jsonl",
    tokenizer,
)


loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)


# Model
config = FlexConfig(
    vocab_size=tokenizer.vocab_size
)

model = FlexModel(
    config
).to(device)


optimizer = create_optimizer(
    model,
    learning_rate,
)


loss_fn = nn.CrossEntropyLoss()


print("Starting training")
print("Samples:", len(dataset))


for epoch in range(epochs):

    total_loss = 0


    for step, batch in enumerate(loader):

        input_ids = batch["input_ids"].to(device)


        # causal shift
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]


        logits = model(
            inputs
        )


        loss = loss_fn(
            logits.reshape(
                -1,
                config.vocab_size,
            ),
            labels.reshape(-1),
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Step {step+1}/{len(loader)} "
            f"Loss: {loss.item():.4f}"
        )


    avg = total_loss / len(loader)

    print(
        f"\nEpoch {epoch+1} average loss: {avg:.4f}\n"
    )


print("Training complete")
