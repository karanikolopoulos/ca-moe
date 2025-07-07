import torch


def bypass(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _lecun(tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
    fan_in = torch.nn.init._calculate_correct_fan(tensor, mode="fan_in")
    std = torch.sqrt(torch.tensor(1.0 / fan_in, dtype=tensor.dtype, device=tensor.device))
    return fan_in, std


def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    _, std = _lecun(tensor)
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)


def lecun_uniform_(tensor: torch.Tensor) -> torch.Tensor:
    _, std = _lecun(tensor)
    bound = torch.sqrt(torch.tensor(3.0, dtype=tensor.dtype, device=tensor.device)) * std
    return torch.nn.init.uniform_(tensor, a=-bound.item(), b=bound.item())


def variance_scaling(scale=1.0, mode="fan_in", distribution="truncated_normal"):
    raise NotImplementedError
