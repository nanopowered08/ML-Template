from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset
from Flex.config import FlexConfig


tokenizer = Tokenizer()

config = FlexConfig(
    vocab_size=tokenizer.vocab_size
)

dataset = ChatMLDataset(
    "data/data.jsonl",
    tokenizer,
)

sample = dataset[0]

print("=== Flex Configuration ===")
print(config)

print()

print("=== Dataset Sample ===")
print(sample["input_ids"])

print()

print("Sequence length:")
print(sample["input_ids"].shape[0])

print()

print("Vocabulary size:")
print(config.vocab_size)

print()

print("Fits context window:")
print(
    sample["input_ids"].shape[0]
    <=
    config.max_seq_len
)
