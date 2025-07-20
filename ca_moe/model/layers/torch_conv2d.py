import logging

import torch
import torch.nn as nn

from ca_moe.model.layers.keras_mixins import KerasMixin
from ca_moe.model.activations import activations


logger = logging.getLogger(__name__)


class TorchConv2D(nn.Module, KerasMixin):
    """Torch equivalent of tf.keras.Conv2D with Keras-style functionality"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int | str,
        dilation: int,
        groups: int,
        bias: bool,
        activation: str,
        **kwargs,
    ):
        logger.debug("TorchConv2D.__init__ START")
        nn.Module.__init__(self)

        self.core = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.activation = activations[activation]
        KerasMixin.__init__(self, **kwargs)
        logger.debug("TorchConv2D.__init__ END")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.core(x)
        return self.activation(output)

    def reset_parameters(self):
        self.core.reset_parameters()
        self.apply_initializers()
