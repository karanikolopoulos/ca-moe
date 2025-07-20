import logging

import torch
import torch.nn as nn

from ca_moe.model.layers.keras_mixins import KerasMixin
from ca_moe.model.activations import activations


logger = logging.getLogger(__name__)


class TorchDense(nn.Module, KerasMixin):
    """Torch equivalent of tf.keras.Dense with Keras-style functionality"""

    def __init__(
        self, in_features: int, out_features: int, activation: str, bias: bool = True, **kwargs
    ):  
        logger.debug("TorchDense.__init__ START")
        nn.Module.__init__(self)

        self.core = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.activation = activations[activation]
        KerasMixin.__init__(self, **kwargs)
        logger.debug("TorchDense.__init__ END")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.core(x)
        return self.activation(output)

    def reset_parameters(self):
        self.core.reset_parameters()
        self.apply_initializers()