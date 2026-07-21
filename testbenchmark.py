import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from Flex.config import FlexConfig
from Flex.model import FlexModel

from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset, ChatMLCollator

from common.inference import Generator
from common.evaluator import Evaluator
from common.benchmark import Benchmark


device = torch.device("cpu")


# -------------------------
# Model
# -------------------------

config = FlexConfig()

model = FlexModel(
    config
).to(device)


# -------------------------
# Tokenizer
# -------------------------

tokenizer = Tokenizer()


# -------------------------
# Dataset
# -------------------------

dataset = ChatMLDataset(
    "data/data.jsonl",
    tokenizer,
    max_length=config.max_seq_len,
)


collator = ChatMLCollator(
    config.pad_token_id
)


dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collator,
)


# -------------------------
# Generator
# -------------------------

generator = Generator(
    model=model,
    tokenizer=tokenizer,
    device=device,
)


# -------------------------
# Evaluator
# -------------------------

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)


evaluator = Evaluator(
    model=model,
    criterion=criterion,
    device=device,
)


# -------------------------
# Benchmark
# -------------------------

benchmark = Benchmark(
    generator,
    evaluator,
)


results = benchmark.run(
    dataloader=dataloader,
    prompts=[
        "Hello!",
        "Explain recursion.",
        "Write Python code for a loop."
    ],
    human_eval=True,
)


print()
print("=" * 60)
print("Final Results")
print("=" * 60)

for key, value in results.items():
    print(
        f"{key}: {value}"
    )
