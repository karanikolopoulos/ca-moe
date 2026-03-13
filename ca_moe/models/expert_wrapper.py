"""
Lightning wrapper for training individual expert models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from ca_moe.utils.tools import build_optimizer


class ExpertWrapper(L.LightningModule):
    def __init__(
        self,
        expert: nn.Module,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["expert"])

        self.expert = expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch

        # Forward pass
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("test/loss", loss)
        self.log("test/acc", acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return build_optimizer(
            self.parameters(), self.hparams.optimizer, self.hparams.learning_rate
        )
