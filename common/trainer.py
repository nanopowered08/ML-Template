import os

import torch
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from common.dataset import ChatMLCollator
from common.optimizer import create_optimizer
from common.scheduler import create_scheduler
from common.checkpoint import save_checkpoint


class Trainer:

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        config,
        device="cpu",
    ):

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset = train_dataset
        self.config = config
        self.device = device

        self.loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=ChatMLCollator(
                tokenizer.pad_token_id
            ),
        )

        self.optimizer = create_optimizer(
            self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=config.epochs * len(self.loader),
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100
        )

        os.makedirs(
            config.checkpoint_dir,
            exist_ok=True,
        )


    def train(self):

        print("=" * 60)
        print("Starting Training")
        print("=" * 60)

        print(f"Device : {self.device}")
        print(f"Samples: {len(self.dataset)}")
        print(f"Batches: {len(self.loader)}")
        print()

        global_step = 0

        for epoch in range(self.config.epochs):

            self.model.train()

            epoch_loss = 0.0

            print(
                f"Epoch {epoch + 1}/{self.config.epochs}"
            )

            for step, batch in enumerate(self.loader):

                input_ids = batch["input_ids"].to(
                    self.device
                )

                labels = batch["labels"].to(
                    self.device
                )

                logits = self.model(
                    input_ids
                )

                loss = self.loss_fn(
                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),
                    labels.reshape(-1),
                )

                self.optimizer.zero_grad()

                loss.backward()

                clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

                self.optimizer.step()

                self.scheduler.step()

                epoch_loss += loss.item()

                global_step += 1

                if (
                    global_step
                    % self.config.log_every
                    == 0
                ):

                    lr = (
                        self.optimizer.param_groups[0]["lr"]
                    )

                    print(
                        f"Step {global_step:6d} | "
                        f"Loss {loss.item():7.4f} | "
                        f"LR {lr:.6e}"
                    )

            avg_loss = epoch_loss / len(self.loader)

            print()
            print(
                f"Epoch {epoch+1} Average Loss: "
                f"{avg_loss:.4f}"
            )

            save_checkpoint(
                path=os.path.join(
                    self.config.checkpoint_dir,
                    f"epoch_{epoch+1}.pt",
                ),
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                step=global_step,
                config=self.config,
            )

            print("Checkpoint saved.")
            print()

        print("=" * 60)
        print("Training Complete")
        print("=" * 60)