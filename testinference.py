import torch

from Flex.config import FlexConfig
from Flex.model import FlexModel

from common.tokenizer import Tokenizer
from common.inference import Generator


device = "cpu"


tokenizer = Tokenizer()

config = FlexConfig()

model = FlexModel(
    config
)

generator = Generator(
    model,
    tokenizer,
    device="cpu"
)


generator.load_checkpoint(
    "checkpoints/epoch_10.pt"
)


prompt = (
    "<|im_start|>user\n"
    "Hello"
    "<|im_end|>\n"
    "<|im_start|>assistant"
)


tokens = generator.encode(
    prompt
)


output = generator.generate(
    generator.encode(prompt),
    max_new_tokens=40,
)


print(
    generator.decode(
        output[0].tolist()
    )
)