import torch
import math


class Evaluator:

    def __init__(
        self,
        model,
        criterion,
        device="cpu",
        generator=None,
    ):

        self.model = model
        self.criterion = criterion
        self.device = device
        self.generator = generator


    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
    ):

        self.model.eval()

        total_loss = 0.0
        total_tokens = 0


        for batch in dataloader:

            input_ids = batch["input_ids"].to(
                self.device
            )

            labels = batch["labels"].to(
                self.device
            )


            logits = self.model(
                input_ids
            )


            loss = self.criterion(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                labels.view(-1)
            )


            tokens = labels.numel()


            total_loss += loss.item() * tokens
            total_tokens += tokens



        avg_loss = (
            total_loss /
            total_tokens
        )


        perplexity = math.exp(
            avg_loss
        )


        return {
            "loss": avg_loss,
            "perplexity": perplexity
        }



    @torch.no_grad()
    def human_test(
        self,
        prompts,
        max_tokens=50,
    ):

        self.model.eval()


        results = []


        for prompt in prompts:

            if self.generator:

                output = self.generator.generate(
                    self.generator.encode(prompt),
                    max_new_tokens=max_tokens
                )


                text = self.generator.decode(
                    output[0].tolist()
                )

            else:
                text = "[Generator missing]"



            results.append(
                {
                    "prompt": prompt,
                    "response": text,
                    "rating": None
                }
            )


        return results
