from logging import getLogger
from .regularizer import Regularizer, L1, L2, L1L2, Orthogonal

logger = getLogger(__name__)

REGULARIZER_REGISTRY: dict[str, Regularizer] = {
    "L1": L1,
    "l1": L1,
    "L2": L2,
    "l2": L2,
    "L1L2": L1L2,
    "l1_l2": L1L2,
    "orthogonal_regularizer": Orthogonal,
    "OrthogonalRegularizer": Orthogonal,
}


def get_regularizer(name: str | None, **kwargs) -> Regularizer:
    if name is None:
        logger.info("No regularizer name provided.")
        return Regularizer()

    try:
        regularizer_cls = REGULARIZER_REGISTRY[name]
        logger.info(f"Using regularizer: {name}")
        return regularizer_cls(**kwargs)
    except KeyError as e:
        logger.info(repr(e))
        logger.info(
            f"Unknown regularizer: {name}. Available options: {', '.join(REGULARIZER_REGISTRY.keys())}"
        )
        logger.info("Fallback to no regularizer.")
        return Regularizer()
