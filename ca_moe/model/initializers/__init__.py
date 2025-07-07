import torch.nn as nn
from logging import getLogger

from .initializers import bypass, lecun_uniform_, lecun_normal_, variance_scaling

logger = getLogger(__name__)

INITIALIZER_REGISTRY: dict[str, callable] = {
    # constant Fillers
    "zeros": nn.init.zeros_, # fill with 0
    "ones": nn.init.ones_, # fill with 1
    "constant": nn.init.constant_, # fill with a custom scalar
    # identity & sparse (structured)
    "identity": nn.init.eye_, # identity matrix (only for 2D)
    "dirac": nn.init.dirac_, # dirac delta (preserve features in conv layers)
    "sparse": nn.init.sparse_, # initialize sparsely (for large fc layers)
    # distributions
    "random_uniform": nn.init.uniform_, # uniform U(a, b)
    "random_normal": nn.init.normal_, # normal N(μ, σ)
    "truncated_normal": nn.init.trunc_normal_, # limited range
    # Glorot (Xavier) - based on fan_avg (good for sigmoid, tanh)
    "glorot_uniform": nn.init.xavier_uniform_,
    "glorot_normal": nn.init.xavier_normal_,
    # He (Kaiming) - based on fan_in (good for ReLU)
    "he_uniform": nn.init.kaiming_uniform_,
    "he_normal": nn.init.kaiming_normal_,
    # Lecun - based on fan_in (good for SELU or scaled tanh) 
    "lecun_uniform": lecun_uniform_,
    "lecun_normal": lecun_normal_,
    # Orthogonal - maintains orthogonality in linear ops
    "orthogonal": nn.init.orthogonal_,
    # scaled - generalized, supports all modes + distributions 
    "variance_scaling": variance_scaling,
    # debug - passthrough fn
    "bypass": bypass 
}

def get_initializer(name: str | None, **kwargs) -> callable:
    if name is None:
        logger.info("No initializer name provided.")
        return bypass
    
    try:
        initializer_fn = INITIALIZER_REGISTRY[name]
        logger.info(f"Using regulizer: {name}")
        return initializer_fn(**kwargs)
    except KeyError as e:
        options = ", ".join(INITIALIZER_REGISTRY.keys())
        
        logger.info(repr(e))
        logger.info(f"Unknown initializer: {name}. Available options: {options}")
        logger.info("Fallback to no initializer.")
        return bypass



