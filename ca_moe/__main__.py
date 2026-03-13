from logging import getLogger
from pathlib import Path

import hydra
import lightning as L
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from ca_moe.factories import setup_trainer
from ca_moe.utils import CONFIG_DIR
from ca_moe.utils.constants import ROOT
from ca_moe.utils.tools import get_last_checkpoint, setup_hydra

logger = getLogger(__name__)

load_dotenv()
setup_hydra()


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    log_dir = Path(HydraConfig.get().runtime.output_dir).relative_to(ROOT)
    L.seed_everything(cfg.seed, workers=True)

    logger.info("Instantiating model")
    model: L.LightningModule = instantiate(cfg.model)
    logger.info("Instantiating data")
    datamodule: L.LightningDataModule = instantiate(cfg.dataset.module)

    ckpt_path = get_last_checkpoint(log_dir, cfg.training.resume)
    trainer = setup_trainer(cfg, log_dir)

    logger.info(f"Training {cfg.model_alias} on {cfg.dataset.name}...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    logger.info("Testing best model...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    logger.info(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
