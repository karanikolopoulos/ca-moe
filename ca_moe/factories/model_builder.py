from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from hydra.utils import call, get_class
from omegaconf import DictConfig
from tensorflow import keras

from ca_moe.model.layers import LayerArgs


class Builder(ABC):
    """Conversion point of arguments for different backends."""

    def __init__(self):
        self.reset()

    @abstractmethod
    def build(self, layers):
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def get_args(self, args: DictConfig) -> dict:
        if not args:  # e.g. flatten layers with no arguments
            return {}

        args = call(args)  # instantiate the dataclass
        assert isinstance(args, LayerArgs)

        return self._process_args(args)
    
    @abstractmethod
    def _process_args(self, args: LayerArgs) -> dict:
        pass


class TorchBuilder(Builder):
    def build(self, layers) -> nn.Module:
        self.reset()

        for name, layer in layers.items():
            layer_cls = get_class(layer.cls)
            args = self.get_args(layer.args)
            module = layer_cls(**args)
            self.model.add_module(name=name, module=module)

        return self.model

    def reset(self):
        self.model = torch.nn.Sequential()

    def _process_args(self, args: LayerArgs) -> dict:
        return args.map_to_torch_args()


class KerasBuilder(Builder):
    def build(self, layers) -> keras.Model:
        self.reset()

        for _, layer in layers.items():
            layer_cls = get_class(layer.cls)
            args = self.get_args(layer.args)
            module = layer_cls(**args)
            self.model.add(module)

        return self.model

    def reset(self):
        self.model = keras.Sequential()

    def _process_args(self, args: LayerArgs) -> dict:
        return args.to_dict()
