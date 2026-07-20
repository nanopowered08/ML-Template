import torch

from Flex.block import TransformerBlock


block = TransformerBlock(
    d_model=256,
    n_heads=4,
    d_ff=1024,
)


x = torch.randn(
    1,
    13,
    256,
)


out = block(x)


print("Input:")
print(x.shape)

print()

print("Output:")
print(out.shape)

print()

print("Finite:")
print(torch.isfinite(out).all())
