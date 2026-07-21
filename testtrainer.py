from common.tokenizer import Tokenizer
from common.dataset import ChatMLDataset

from common.trainer import Trainer

from Flex.config import FlexConfig
from Flex.model import FlexModel

device = "cpu"

tokenizer = Tokenizer()

config = FlexConfig()

dataset = ChatMLDataset(
    "data/data.jsonl",
    tokenizer,
)

model = FlexModel(config)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    config=config,
    device=device,
)

trainer.train()