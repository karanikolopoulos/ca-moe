from logging import getLogger

import torch
import torch.nn as nn

from .initializers import bypass, lecun_normal, lecun_uniform, variance_scaling

logger = getLogger(__name__)

init_snake: dict[str, callable] = {
    # constant fillers
    "zeros": nn.init.zeros_,  # fill with 0
    "ones": nn.init.ones_,  # fill with 1
    "constant": nn.init.constant_,  # fill with a custom scalar
    # identity & sparse (structured)
    "identity": nn.init.eye_,  # identity matrix (only for 2D)
    "dirac": nn.init.dirac_,  # dirac delta (preserve features in conv layers)
    "sparse": nn.init.sparse_,  # initialize sparsely (for large fc layers)
    # distributions
    "random_uniform": nn.init.uniform_,  # uniform U(a, b)
    "random_normal": nn.init.normal_,  # normal N(μ, σ)
    "truncated_normal": nn.init.trunc_normal_,  # limited range
    # Glorot (Xavier) - based on fan_avg (good for sigmoid, tanh)
    "glorot_uniform": nn.init.xavier_uniform_,
    "glorot_normal": nn.init.xavier_normal_,
    # He (Kaiming) - based on fan_in (good for ReLU)
    "he_uniform": nn.init.kaiming_uniform_,
    "he_normal": nn.init.kaiming_normal_,
    # Orthogonal - maintains orthogonality in linear ops
    "orthogonal": nn.init.orthogonal_,
    # Lecun - based on fan_in (good for SELU or scaled tanh)
    "lecun_uniform": lecun_uniform,
    "lecun_normal": lecun_normal,
    # scaled - generalized, supports all modes + distributions
    "variance_scaling": variance_scaling,
}


init_pascal: dict[str, callable] = {
    "Zeros": nn.init.zeros_,
    "Ones": nn.init.ones_,
    "Constant": nn.init.constant_,
    "Identity": nn.init.eye_,
    "Dirac": nn.init.dirac_,
    "Sparse": nn.init.sparse_,
    "RandomUniform": nn.init.uniform_,
    "RandomNormal": nn.init.normal_,
    "TruncatedNormal": nn.init.trunc_normal_,
    "GlorotUniform": nn.init.xavier_uniform_,
    "GlorotNormal": nn.init.xavier_normal_,
    "HeUniform": nn.init.kaiming_uniform_,
    "HeNormal": nn.init.kaiming_normal_,
    "Orthogonal": nn.init.orthogonal_,
    "LecunUniform": lecun_uniform,
    "LecunNormal": lecun_normal,
    "VarianceScaling": variance_scaling,
}

INITIALIZER_REGISTRY = init_snake | init_pascal | {"bypass": bypass}


def get_initializer(name: str | None, **kwargs) -> callable:
    if name is None:
        logger.debug("No initializer name provided.")
        initializer_fn = bypass

    try:
        initializer_fn = INITIALIZER_REGISTRY[name]
        logger.debug(f"Using initializer: {name}")
    except KeyError as e:
        options = ", ".join(INITIALIZER_REGISTRY.keys())

        logger.debug(repr(e))
        logger.debug(f"Unknown initializer: {name}. Available options: {options}")
        logger.debug("Fallback to no initializer.")
        initializer_fn = bypass

    def wrap(tensor: torch.Tensor) -> torch.Tensor:
        initializer_fn(tensor=tensor, **kwargs)

    return wrap
