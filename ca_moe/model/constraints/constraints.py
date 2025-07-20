import torch

_EPSILON = 1e-7


class Constraint: # fallback - null object pattern
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return w

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MaxNorm(Constraint):
    def __init__(self, max_value: float = 2.0, axis: int | list[int] = 0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(torch.square(w), dim=self.axis, keepdim=True))
        desired = torch.clamp(norm, 0.0, self.max_value)
        return w * (desired / (_EPSILON + norm))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_value={self.max_value}, axis={self.axis})"
        )


class NonNeg(Constraint):
    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return w * (w >= 0.0).to(w.dtype)


class UnitNorm(Constraint):
    def __init__(self, axis: int | list[int] = 0):
        self.axis = axis

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        return w / (
            _EPSILON
            + torch.sqrt(torch.sum(torch.square(w), dim=self.axis, keepdim=True))
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis})"


class MinMaxNorm(Constraint):
    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        rate: float = 1.0,
        axis: int | list[int] = 0,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.axis = axis

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.sum(torch.square(w), dim=self.axis, keepdim=True))
        clipped = torch.clamp(norm, self.min_value, self.max_value)
        desired = self.rate * clipped + (1.0 - self.rate) * norm
        return w * (desired / (_EPSILON + norm))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_value={self.min_value}, "
            f"max_value={self.max_value}, rate={self.rate}, axis={self.axis})"
        )