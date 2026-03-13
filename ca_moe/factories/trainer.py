import logging
from pathlib import Path

import lightning as L

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig

from ca_moe.utils.wandb_config import WandbConfig

log = logging.getLogger(__name__)


def setup_callbacks(cfg: DictConfig) -> list:
    callbacks = []

    callbacks.append(TQDMProgressBar(leave=True))

    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_last=True,
        save_top_k=1,
        filename="best",
        enable_version_counter=False,
    )
    callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.patience,
        mode="min",
        verbose=False,
    )
    callbacks.append(early_stop_callback)

    return callbacks


def setup_loggers(cfg: DictConfig, log_dir: Path) -> list:
    csv_logger = CSVLogger(
        save_dir=str(log_dir),
        name="",
        version="",
    )

    loggers = [csv_logger]

    if raw_wandb := cfg.get("wandb"):
        wandb_cfg = WandbConfig.from_omegaconf(raw_wandb, log_dir)
        loggers.append(WandbLogger(**wandb_cfg.to_logger_kwargs()))

    return loggers


def setup_trainer(cfg: DictConfig, log_dir: Path) -> L.Trainer:

    callbacks = setup_callbacks(cfg)
    loggers = setup_loggers(cfg, log_dir)

    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="auto",
        devices="auto" if cfg.device == "cuda" else 1,
        callbacks=callbacks,
        logger=loggers,
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    return trainer
