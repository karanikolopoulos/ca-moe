import torch

_EPSILON = 1e-7


def max_norm(w: torch.Tensor, max_value=2.0, axis=0) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(torch.square(w), dim=axis, keepdim=True))
    desired = torch.clamp(norm, 0.0, max_value)
    return w * (desired / (_EPSILON + norm))


def non_neg(w: torch.Tensor) -> torch.Tensor:
    return w * (w >= 0.0).to(w.dtype)


def unit_norm(w: torch.Tensor, axis=0) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(torch.square(w), dim=axis, keepdim=True))
    return w / (_EPSILON + norm)


def min_max_norm(
    w: torch.Tensor, min_value=0.0, max_value=1.0, rate=1.0, axis=0
) -> torch.Tensor:
    norm = torch.sqrt(torch.sum(torch.square(w), dim=axis, keepdim=True))
    clipped = torch.clamp(input=norm, min=min_value, max=max_value)
    desired = rate * clipped + (1.0 - rate) * norm
    return w * (desired / (_EPSILON + norm))


def bypass(w: torch.Tensor) -> torch.Tensor:
    return w
