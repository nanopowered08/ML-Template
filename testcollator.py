from common.dataset import ChatMLDataset, ChatMLCollator
from common.tokenizer import Tokenizer


tokenizer = Tokenizer()

dataset = ChatMLDataset(
    "data/data.jsonl",
    tokenizer
)


collator = ChatMLCollator(
    tokenizer.pad_token_id
)


batch = collator(
    [
        dataset[0],
        dataset[1],
    ]
)


print(batch["input_ids"])
print(batch["labels"])
print(batch["input_ids"].shape)
