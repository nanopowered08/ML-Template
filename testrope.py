import torch

from Flex.rope import (
    RotaryEmbedding,
    apply_rope,
)


batch = 2
heads = 4
seq = 16
head_dim = 64


x = torch.randn(
    batch,
    heads,
    seq,
    head_dim,
)


rope = RotaryEmbedding(
    head_dim,
    max_seq_len=512,
)


cos, sin = rope(x)

out = apply_rope(
    x,
    cos,
    sin,
)


print("Input:")
print(x.shape)

print()

print("Output:")
print(out.shape)

print()

print(
    "Changed:",
    not torch.equal(x, out)
)
