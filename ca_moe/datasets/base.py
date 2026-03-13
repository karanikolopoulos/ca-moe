import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from ca_moe.utils.constants import DATA_DIR


class VisionDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_class,
        batch_size: int = 32,
        num_workers: int = 0,
        val_size: int = 10000,
        seed: int = 2021,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset_class"])
        self.dataset_class = dataset_class
        self.data_dir = str(DATA_DIR)

        self.transform = transforms.ToTensor()

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            train_ds = self.dataset_class(
                self.data_dir,
                train=True,
                transform=self.transform,
                download=False,
            )

            train_size = len(train_ds) - self.hparams.val_size
            generator = torch.Generator().manual_seed(self.hparams.seed)
            self.train_ds, self.val_ds = random_split(
                dataset=train_ds,
                lengths=[train_size, self.hparams.val_size],
                generator=generator,
            )

        if stage == "test":
            self.test_ds = self.dataset_class(
                self.data_dir,
                train=False,
                transform=self.transform,
                download=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


