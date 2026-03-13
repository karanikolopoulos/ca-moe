from torchvision import datasets

from ca_moe.datasets.base import VisionDataModule


class MNISTDataModule(VisionDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=datasets.MNIST, **kwargs)


class FashionMNISTDataModule(VisionDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=datasets.FashionMNIST, **kwargs)


class CIFAR100DataModule(VisionDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_class=datasets.CIFAR100, **kwargs)
