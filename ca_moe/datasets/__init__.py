"""Dataset exports - point Hydra's _target_ to instantiate"""

from ca_moe.datasets.vision import (
    MNISTDataModule,
    FashionMNISTDataModule,
    CIFAR100DataModule,
)

__all__ = [
    "MNISTDataModule",
    "FashionMNISTDataModule",
    "CIFAR100DataModule",
]
