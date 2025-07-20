import math
import torch
import torch.nn.functional as F

class Regularizer: # fallback - null object pattern
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0)

class L1(Regularizer):
    """Adds L1 regularization penalty to tensor."""
    def __init__(self, l1=0.01) -> None:
        l1 = 0.01 if l1 is None else l1
        validate_float_arg(value=l1, name="l1")
        
        self.l1 = l1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.l1 * tensor.abs().sum()

class L2(Regularizer):
    """Adds L2 regularization penalty to tensor."""
    def __init__(self, l2=0.01) -> None:
        l2 = 0.01 if l2 is None else l2
        validate_float_arg(value=l2, name="l2")
        
        self.l2 = l2

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.l2 * (tensor ** 2).sum()

class L1L2(Regularizer):
    """Adds L1 and L2 regularization penalties to tensor."""
    def __init__(self, l1=0.0, l2=0.0) -> None:
        l1 = 0.0 if l1 is None else l1
        l2 = 0.0 if l2 is None else l2
        validate_float_arg(value=l1, name="l1")
        validate_float_arg(value=l2, name="l2")
        
        self.l1 = l1
        self.l2 = l2

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:      
        regularization = 0
        
        if self.l1:
            regularization += self.l1 * tensor.abs().sum()
        if self.l2:
            regularization += self.l2 * (tensor ** 2).sum()
        
        return regularization

class Orthogonal(Regularizer):
    "Adds orthogonal regularization penalty to tensor."
    def __init__(self, factor=0.01, mode="rows"):
        factor = 0.01 if factor is None else factor
        
        validate_float_arg(value=factor, name="factor")        
        assert mode in {"rows", "columns"}, f"Invalid `mode`: {mode}. Must be 'rows' or 'columns'."
        
        self.factor = factor
        self.mode = mode

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim == 2, (
            "Inputs to OrthogonalRegularizer must have rank 2.",
            f"Received: tensor.shape={tensor.shape}"
            )
        
        if self.mode == "rows":
            tensor = F.normalize(tensor, dim=1, eps=1e-12) # normalize rows
            product = tensor @ tensor.T
            size, _ = tensor.shape
            # size = tensor.size(0)
        else:
            tensor = F.normalize(tensor, dim=0, eps=1e-12) # normalize columns
            product = tensor.T @ tensor
            _, size = tensor.shape  
            # size = tensor.size(1)

        # remove diagonal elements (self-products should be 1), mask to zero out diagonal elements
        diagonal_mask = 1.0 - torch.eye(size, device=tensor.device, dtype=tensor.dtype)
        product_no_diagonal = product * diagonal_mask
        # calculate number of unique pairs (combinations)
        num_pairs = size * (size - 1.0) / 2.0
        # penalty: 0.5 * sum of absolute values of off-diagonal elements / num_pairs
        penalty = 0.5 * product_no_diagonal.abs().sum() / num_pairs
        
        return self.factor * penalty
    

def validate_float_arg(value: float | int, name: str) -> float:
    """Check penalty number availability, raise ValueError if failed."""
    if (
        not isinstance(value, (float, int))
        or (math.isinf(value) or math.isnan(value))
        or value < 0
    ):
        raise ValueError(
            f"Invalid value for argument {name}: expected a non-negative float."
            f"Received: {name}={value}"
        )
    return float(value)