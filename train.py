tokenizer = Tokenizer()

config = FlexConfig()

dataset = ChatMLDataset(
    "data/evol/train.jsonl",
    tokenizer,
)

model = FlexModel(config)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    config=config,
)

trainer.train()