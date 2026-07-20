from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset

from Flex.config import FlexConfig
from Flex.TestBed1 import FlexModel


tokenizer = Tokenizer()

config = FlexConfig(
    vocab_size=tokenizer.vocab_size
)

model = FlexModel(config)

dataset = ChatMLDataset(
    "data/data.jsonl",
    tokenizer,
)

sample = dataset[0]

embeddings = model(
    sample["input_ids"].unsqueeze(0)
)

print("Input shape:")
print(sample["input_ids"].unsqueeze(0).shape)

print()

print("Embedding shape:")
print(embeddings.shape)
