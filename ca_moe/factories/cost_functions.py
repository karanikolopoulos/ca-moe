import math
from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from lightning.pytorch.utilities import measure_flops


class CostFunction(ABC):
    """Base class for expert cost functions.

    Each implementation receives a model and returns a scalar float
    on a scale comparable to cross-entropy loss (~1–10).
    """

    @abstractmethod
    def __call__(self, model: nn.Module) -> float:
        """Transformed cost for use in loss (log scale, comparable to CE)."""
        pass

    def raw(self, model: nn.Module) -> float:
        """Untransformed cost for cost gain computation.

        Override when __call__ applies a transformation (e.g. log10) that makes
        ratio-based metrics (cost gain) meaningless in the transformed space.
        Defaults to __call__ for already-linear cost functions.
        """
        return self.__call__(model)


_REGISTRY: dict[str, type[CostFunction]] = {}


def register_cost(name: str) -> Callable:
    def decorator(cls: type[CostFunction]) -> type[CostFunction]:
        _REGISTRY[name] = cls
        return cls
    return decorator



@register_cost("logflops")
class LogFlopsCost(CostFunction):
    """log10(FLOPs) — requires input_shape to run a forward pass."""

    def __init__(self, input_shape: tuple[int, ...], **kwargs):
        self.input_shape = tuple(input_shape)

    def _measure(self, model: nn.Module) -> int:
        def forward():
            return model(torch.randn(*self.input_shape))

        return int(measure_flops(model, forward))

    def __call__(self, model: nn.Module) -> float:
        flops = self._measure(model)
        return math.log10(flops) if flops > 0 else 0.0

    def raw(self, model: nn.Module) -> float:
        """Linear FLOPs — used for cost gain (ratio in original space)."""
        return float(self._measure(model))


@register_cost("logparams")
class LogParamsCost(CostFunction):
    """log10(trainable parameters)"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, model: nn.Module) -> float:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return math.log10(params) if params > 0 else 0.0


@register_cost("numlayers")
class NumLayersCost(CostFunction):
    """Count of Linear + Conv2d layers — simple depth proxy."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, model: nn.Module) -> float:
        return float(
            sum(1 for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d)))
        )


def get_cost_fn(name: str, **kwargs) -> CostFunction:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown cost function '{name}'. Choose from: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)
