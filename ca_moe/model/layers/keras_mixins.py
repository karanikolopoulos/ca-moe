import torch

import logging

from ca_moe.model.constraints import get_constraint
from ca_moe.model.initializers import get_initializer
from ca_moe.model.regularizers import get_regularizer

logger = logging.getLogger(__name__)


class InitMixin:
    """Mixin for weights and biases initialization"""

    def __init__(self, kernel_initializer: str, bias_initializer: str, **kwargs):
        logger.debug("InitMixin.__init__ START")
        super().__init__(**kwargs)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        logger.debug("InitMixin.__init__ END")

    def apply_initializers(self) -> None:
        with torch.no_grad():
            self.kernel_initializer(self.core.weight)
            if self.core.bias is not None:
                self.bias_initializer(self.core.bias)


class RegMixin:
    """Mixin for regularization"""

    def __init__(
        self,
        kernel_regularizer: str | None,
        bias_regularizer: str | None,
        activity_regularizer: str | None,
        **kwargs,
    ):
        logger.debug("RegMixin.__init__ START")
        super().__init__(**kwargs)
        self.kernel_regularizer = get_regularizer(kernel_regularizer)
        self.bias_regularizer = get_regularizer(bias_regularizer)
        self.activity_regularizer = get_regularizer(activity_regularizer)
        logger.debug("RegMixin.__init__ END")

    def get_kernel_regularization_loss(self) -> torch.Tensor:
        return self.kernel_regularizer(self.core.weight)

    def get_bias_regularization_loss(self) -> torch.Tensor:
        return self.bias_regularizer(self.core.bias)

    def get_activity_regularization_loss(self, output: torch.Tensor) -> torch.Tensor:
        return self.activity_regularizer(output)

    def get_total_regularization_loss(self, output: torch.Tensor) -> torch.Tensor:
        kernel_reg = self.get_kernel_regularization_loss()
        bias_reg = self.get_bias_regularization_loss()
        activity_reg = self.get_activity_regularization_loss(output)

        logger.debug(f"Kernel regularization loss: {kernel_reg}")
        logger.debug(f"Bias regularization loss: {bias_reg}")
        logger.debug(f"Activity regularization loss: {activity_reg}")

        return kernel_reg + bias_reg + activity_reg


class ConstraintMixin:
    """Mixin for weights and biases constraints"""

    def __init__(
        self, kernel_constraint: str | None, bias_constraint: str | None, **kwargs
    ):
        # end of MRO - no super call here
        logger.debug("ConstraintMixin.__init__ START")
        self.kernel_constraint = get_constraint(kernel_constraint)
        self.bias_constraint = get_constraint(bias_constraint)
        logger.debug("ConstraintMixin.__init__ END")

    def apply_constraints(self) -> None:
        with torch.no_grad():
            self.core.weight.copy_(self.kernel_constraint(self.core.weight))
            self.core.bias.copy_(self.bias_constraint(self.core.bias))


class KerasMixin(InitMixin, RegMixin, ConstraintMixin):
    """
    Combined mixin for Keras functionality

    MRO: (
            caller class (should define `core` nn.Module),
            ca_moe.model.layers.KerasMixin,
            ca_moe.model.layers.InitMixin,
            ca_moe.model.layers.RegMixin,
            ca_moe.model.layers.ConstraintMixin,
            object
        )
    """

    def __init__(self, **kwargs):
        logger.debug("KerasMixin.__init__ START")
        super().__init__(**kwargs)
        assert hasattr(self, "core"), "Parent class should define `core` nn.Module"

        self.apply_initializers()
        logger.debug("KerasMixin.__init__ END")
