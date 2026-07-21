import torch

from common.sampling import (
    greedy,
    temperature,
    top_k,
    top_p,
    sample,
)


# Fake logits:
# batch = 1
# vocab = 100284

logits = torch.randn(
    1,
    100284
)


print("Input:")
print(logits.shape)


# Greedy

token = greedy(logits)

print("\nGreedy:")
print(token)
print(token.shape)


# Temperature

temp_logits = temperature(
    logits,
    0.8,
)

print("\nTemperature:")
print(temp_logits.shape)
print("Finite:")
print(torch.isfinite(temp_logits).all())


# Top K

topk_logits = top_k(
    logits,
    50,
)

print("\nTop K:")
print(topk_logits.shape)
print("Finite:")
print(torch.isfinite(topk_logits).any())


# Top P

topp_logits = top_p(
    logits,
    0.95,
)

print("\nTop P:")
print(topp_logits.shape)
print("Finite:")
print(torch.isfinite(topp_logits).any())


# Full sampler

token = sample(
    logits,
    temperature_value=0.8,
    top_k_value=50,
    top_p_value=0.95,
)

print("\nSample:")
print(token)
print(token.shape)


print("\nAll tests passed.")