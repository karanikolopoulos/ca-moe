import torch
import math
from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_, normal_, uniform_

TRUN_NORMAL_STDDEV_CORRECTION = 0.87962566103423978

def validate_inputs(scale: float, mode: str, distribution: str):
    if scale <= 0.0:
        raise ValueError(
                "Argument `scale` must be positive float. "
                f"Received: scale={scale}"
            )
    
    allowed_modes = {"fan_in", "fan_out", "fan_avg"}
    if mode not in allowed_modes:
        raise ValueError(
            f"Invalid `mode` argument: {mode}. "
            f"Please use one of {allowed_modes}"
        )

    allowed_distributions = {
            "uniform",
            "truncated_normal",
            "untruncated_normal",
        }
    if distribution not in allowed_distributions:
        raise ValueError(
            f"Invalid `distribution` argument: {distribution}."
            f"Please use one of {allowed_distributions}"
        )

def variance_scaling(
    tensor: torch.Tensor,
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "truncated_normal"
) -> torch.Tensor:
    
    validate_inputs(scale=scale, mode=mode, distribution=distribution)

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    denom = {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "fan_avg": (fan_in + fan_out) / 2,
    }[mode]
    denom = max(1.0, denom)
    variance = scale / denom

    if distribution == "truncated_normal":
        stddev = math.sqrt(variance) / TRUN_NORMAL_STDDEV_CORRECTION
        return trunc_normal_(tensor, mean=0.0, std=stddev, a=-2*stddev, b=2*stddev)

    elif distribution == "untruncated_normal":
        stddev = math.sqrt(variance)
        return normal_(tensor, mean=0.0, std=stddev)

    elif distribution == "uniform":
        limit = math.sqrt(3.0 * variance)
        return uniform_(tensor, a=-limit, b=limit)

def lecun_uniform(tensor: torch.Tensor) -> torch.Tensor:
    return variance_scaling(tensor, scale=1.0, mode="fan_in", distribution="uniform")

def lecun_normal(tensor: torch.Tensor) -> torch.Tensor:
    return variance_scaling(tensor, scale=1.0, mode="fan_in", distribution="truncated_normal")

def bypass(tensor: torch.Tensor) -> torch.Tensor:
    return tensor