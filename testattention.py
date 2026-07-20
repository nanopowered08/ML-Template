import torch

from Flex.attention import MultiHeadAttention


attention = MultiHeadAttention(
    d_model=256,
    n_heads=4,
    max_seq_len=512,
)


x = torch.randn(
    1,
    13,
    256,
)


out = attention(x)


print("Input:")
print(x.shape)

print()

print("Output:")
print(out.shape)

print()

print("Finite:")
print(torch.isfinite(out).all())
