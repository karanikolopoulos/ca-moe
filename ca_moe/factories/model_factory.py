from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Literal

from omegaconf import DictConfig
import torch
import torch.nn as nn
from hydra.utils import call, get_class
from tensorflow import keras


@dataclass
class Dense:
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


@dataclass
class Conv2D:
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


@dataclass
class AvgMaxPool:
    input_shape: list | None
    pool_size: int
    strides: int | None = None
    padding: Literal["valid", "same"] = "valid"

    def __post_init__(self):
        if self.strides is None:
            self.strides = self.pool_size


class Builder(ABC):
    @abstractmethod
    def build(self, layers):
        pass

    @abstractmethod
    def get_dense_args(self, args: Dense) -> dict:
        pass

    @abstractmethod
    def get_conv2d_args(self, args: Conv2D) -> dict:
        pass

    @abstractmethod
    def get_avg_max_pool_args(self, args: AvgMaxPool) -> dict:
        pass

    def get_args(self, args: DictConfig) -> dict:
        if not args:  # e.g. flatten layers with no arguments
            return {}

        args = call(args)  # instantiate the dataclass

        # call the appropriate methods, implemented by the concrete builders
        if isinstance(args, Dense):
            return self.get_dense_args(args=args)

        if isinstance(args, Conv2D):
            return self.get_conv2d_args(args=args)

        if isinstance(args, AvgMaxPool):
            return self.get_avg_max_pool_args(args=args)

        raise NotImplementedError


class TorchBuilder(Builder):
    def __init__(self):
        pass

    def build(self, layers) -> nn.Module:
        model = torch.nn.Sequential()

        for name, layer in layers.items():
            layer_cls = get_class(layer.cls)
            args = self.get_args(layer.args)
            module = layer_cls(**args)
            model.add_module(name=name, module=module)

        return model

    def get_dense_args(self, args: Dense) -> dict:
        return {
            "in_features": args.input_shape,
            "out_features": args.units,
            "bias": args.use_bias,
            "activation": args.activation,
            "kernel_initializer": args.kernel_initializer,
            "bias_initializer": args.bias_initializer,
            "kernel_regularizer": args.kernel_regularizer,
            "bias_regularizer": args.bias_regularizer,
            "activity_regularizer": args.activity_regularizer,
            "kernel_constraint": args.kernel_constraint,
            "bias_constraint": args.bias_constraint,
        }

    def get_conv2d_args(self, args: Conv2D):
        *_, in_channels = args.input_shape

        return {
            "in_channels": in_channels,
            "out_channels": args.filters,
            "kernel_size": args.kernel_size,
            "stride": args.strides,
            "padding": args.padding,
            "dilation": args.dilation_rate,
            "groups": args.groups,
            "activation": args.activation,
            "bias": args.use_bias,
            "kernel_initializer": args.kernel_initializer,
            "bias_initializer": args.bias_initializer,
            "kernel_regularizer": args.kernel_regularizer,
            "bias_regularizer": args.bias_regularizer,
            "activity_regularizer": args.activity_regularizer,
            "kernel_constraint": args.kernel_constraint,
            "bias_constraint": args.bias_constraint,
        }

    def get_avg_max_pool_args(self, args: AvgMaxPool) -> dict:
        if args.padding == "same":
            raise NotImplementedError
        else:
            padding = 0

        return {
            "kernel_size": args.pool_size,
            "stride": args.strides,
            "padding": padding,
        }


class KerasBuilder(Builder):
    def build(self, layers):
        model = keras.Sequential()

        for _, layer in layers.items():
            layer_cls = get_class(layer.cls)
            args = self.get_args(layer.args)
            module = layer_cls(**args)
            model.add(module)

        return model

    def get_dense_args(self, args: Dense) -> dict:
        return asdict(args)

    def get_conv2d_args(self, args: Conv2D):
        args = asdict(args)
        args.pop("data_format", None)
        return args

    def get_avg_max_pool_args(self, args: AvgMaxPool) -> dict:
        return asdict(args)
