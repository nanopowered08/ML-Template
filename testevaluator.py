import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from Flex.config import FlexConfig
from Flex.model import FlexModel

from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset, ChatMLCollator
from common.evaluator import Evaluator


device = torch.device("cpu")

# -------------------------
# Config
# -------------------------

config = FlexConfig()

# -------------------------
# Model
# -------------------------

model = FlexModel(config).to(device)

# -------------------------
# Tokenizer
# -------------------------

# Replace the encoding string below with whatever your tokenizer expects
# if it is different.
tokenizer = Tokenizer()

# -------------------------
# Dataset
# -------------------------

dataset = ChatMLDataset(
    path="data/data.jsonl",
    tokenizer=tokenizer,
    max_length=config.max_seq_len,
)

collator = ChatMLCollator(
    pad_token_id=config.pad_token_id,
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collator,
)

# -------------------------
# Loss
# -------------------------

criterion = nn.CrossEntropyLoss(
    ignore_index=-100,
)

# -------------------------
# Evaluator
# -------------------------

evaluator = Evaluator(
    model=model,
    criterion=criterion,
    device=device,
)

results = evaluator.evaluate(
    dataloader
)

print("=" * 50)
print("Evaluation Results")
print("=" * 50)

for key, value in results.items():
    print(f"{key}: {value}")
