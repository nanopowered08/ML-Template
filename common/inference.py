import torch

from common.sampling import sample


class Generator:

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
    ):

        self.device = device

        self.model = model.to(device)
        self.model.eval()

        self.tokenizer = tokenizer


    def load_checkpoint(
        self,
        path,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )


        if "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint


        self.model.load_state_dict(
            state
        )

        self.model.eval()


    def encode(
        self,
        text,
    ):

        ids = self.tokenizer.encode(
            text
        )

        return torch.tensor(
            [ids],
            dtype=torch.long,
            device=self.device,
        )


    def decode(
        self,
        ids,
    ):

        return self.tokenizer.decode(
            ids
        )


    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
    ):


        for _ in range(max_new_tokens):

            logits = self.model(
                input_ids
            )


            # last position
            logits = logits[:, -1, :]


            token = sample(
                logits,
                temperature_value=temperature,
                top_k_value=top_k,
                top_p_value=top_p,
            )


            input_ids = torch.cat(
                [
                    input_ids,
                    token,
                ],
                dim=1,
            )


        return input_ids