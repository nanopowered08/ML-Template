from common.dataset import ChatMLDataset
from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset, ChatMLCollator

dataset = ChatMLDataset(
    "data/data.jsonl",
    Tokenizer()
)

print(len(dataset))

item = dataset[0]

print(item["input_ids"])
print(item["labels"])
collator = ChatMLCollator(
    Tokenizer.pad_token_id
)
