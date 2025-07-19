from logging import getLogger

from .constraints import Constraint, MaxNorm, MinMaxNorm, NonNeg, UnitNorm

logger = getLogger(__name__)


CONSTRAINT_REGISTRY: dict[str, Constraint] = {
    "MaxNorm": MaxNorm,
    "max_norm": MaxNorm,
    "NonNeg": NonNeg,
    "non_neg": NonNeg,
    "UnitNorm": UnitNorm,
    "unit_norm": UnitNorm,
    "MixMaxNorm": MinMaxNorm,
    "mix_max_norm": MinMaxNorm,
}


def get_constraint(name: str | None, **kwargs) -> Constraint:
    if name is None:
        logger.debug("No constraint name provided.")
        return Constraint()

    try:
        constraint_fn = CONSTRAINT_REGISTRY[name]
        logger.debug(f"Using constraint: {name}")
        return constraint_fn(**kwargs)
    except KeyError:
        options = ", ".join(CONSTRAINT_REGISTRY.keys())

        logger.debug(f"Unknown constraint: {name}. Available options: {options}")
        logger.debug("Fallback to no constraint.")
        return Constraint()
