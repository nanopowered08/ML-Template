import torch
from common.sampling import sample


class Generator:

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
    ):

        self.model = model.to(device)
        self.model.eval()

        self.tokenizer = tokenizer
        self.device = device


    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        do_sample=True,
    ):

        input_ids = input_ids.to(
            self.device
        )


        for _ in range(max_new_tokens):

            logits = self.model(
                input_ids
            )


            # last token only
            logits = logits[:, -1, :]


            if temperature > 0:

                logits = logits / temperature


            if do_sample:

                probs = torch.softmax(
                    logits,
                    dim=-1,
                )

                next_token = sample(
                     logits,
                    temperature_value=temperature,
                    top_k_value=top_k,
                    top_p_value=top_p,
                )

            else:

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )


            input_ids = torch.cat(
                [
                    input_ids,
                    next_token,
                ],
                dim=1,
            )


        return input_ids


    def encode(
        self,
        text,
    ):

        tokens = self.tokenizer.encode(
            text
        )

        return torch.tensor(
            [tokens],
            dtype=torch.long,
        )


    def decode(
        self,
        tokens,
    ):

        return self.tokenizer.decode(
            tokens
        )