import torch


def greedy(logits):
    """
    Always pick highest probability token.
    """

    return torch.argmax(
        logits,
        dim=-1,
        keepdim=True,
    )


def temperature(
    logits,
    value=1.0,
):
    """
    Adjust randomness.

    <1.0 = more deterministic
    >1.0 = more random
    """

    return logits / value


def top_k(
    logits,
    k=50,
):
    """
    Keep only top k tokens.
    """

    values, indices = torch.topk(
        logits,
        k,
        dim=-1,
    )

    filtered = torch.full_like(
        logits,
        float("-inf"),
    )

    filtered.scatter_(
        -1,
        indices,
        values,
    )

    return filtered


def top_p(
    logits,
    p=0.95,
):
    """
    Nucleus sampling.
    """

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )


    probs = torch.softmax(
        sorted_logits,
        dim=-1,
    )


    cumulative = torch.cumsum(
        probs,
        dim=-1,
    )


    mask = cumulative > p

    # keep at least one token
    mask[..., 0] = False


    sorted_logits[mask] = float("-inf")


    output = torch.full_like(
        logits,
        float("-inf"),
    )


    output.scatter_(
        -1,
        sorted_indices,
        sorted_logits,
    )


    return output


def sample(
    logits,
    temperature_value=1.0,
    top_k_value=None,
    top_p_value=None,
):

    logits = temperature(
        logits,
        temperature_value,
    )


    if top_k_value:
        logits = top_k(
            logits,
            top_k_value,
        )


    if top_p_value:
        logits = top_p(
            logits,
            top_p_value,
        )


    probs = torch.softmax(
        logits,
        dim=-1,
    )


    return torch.multinomial(
        probs,
        num_samples=1,
    )