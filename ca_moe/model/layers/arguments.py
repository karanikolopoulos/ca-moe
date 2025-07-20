from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass()
class LayerArgs(ABC):
    def __post_init__(self):
        if not self.use_bias:  # are you drunk
            self.bias_initializer = None
            self.bias_regularizer = None
            self.bias_constraint = None

    def to_dict(self):
        return asdict(self)

    @abstractmethod
    def map_to_torch_args(self):
        pass


@dataclass
class Dense(LayerArgs):
    input_shape: int
    units: int
    use_bias: bool = True
    activation: str = "relu"
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"
    kernel_regularizer: str | None = None
    bias_regularizer: str | None = None
    activity_regularizer: str | None = None
    kernel_constraint: str | None = None
    bias_constraint: str | None = None

    def __post_init__(self):
        return super().__post_init__()

    def map_to_torch_args(self):
        return {
            "in_features": self.input_shape,
            "out_features": self.units,
            "bias": self.use_bias,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
        }


@dataclass
class Conv2D(LayerArgs):
    input_shape: int | None
    filters: int
    kernel_size: int
    strides: int = 1
    padding: Literal["valid", "same"] = "valid"
    data_format: Literal["channels_first"] = "channels_first"
    dilation_rate: int = 1
    groups: int = 1
    activation: str = None
    use_bias: bool = True
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"
    kernel_regularizer: str | None = None
    bias_regularizer: str | None = None
    activity_regularizer: str | None = None
    kernel_constraint: str | None = None
    bias_constraint: str | None = None

    def __post_init__(self):
        return super().__post_init__()

    def map_to_torch_args(self):
        *_, in_channels = self.input_shape

        return {
            "in_channels": in_channels,
            "out_channels": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.strides,
            "padding": self.padding,
            "dilation": self.dilation_rate,
            "groups": self.groups,
            "activation": self.activation,
            "bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
        }


@dataclass
class AvgMaxPool(LayerArgs):
    input_shape: list | None
    pool_size: int
    strides: int | None = None
    padding: Literal["valid", "same"] = "valid"
    data_format: Literal["channels_first"] = "channels_first"

    def __post_init__(self):
        if self.strides is None:
            self.strides = self.pool_size

    def map_to_torch_args(self):
        if self.padding == "same":
            raise NotImplementedError
        else:
            padding = 0

        return {
            "kernel_size": self.pool_size,
            "stride": self.strides,
            "padding": padding,
        }
