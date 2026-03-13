import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing_extensions import Self

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    project: str
    group: str
    name: str
    entity: str | None
    tags: list[str]
    mode: str
    notes: str
    job_type: str
    log_dir: Path

    def __post_init__(self) -> None:
        if not self._has_api_key:
            log.info("WANDB_API_KEY not set - WandB logger disabled")
        elif not self.entity:
            log.debug("WANDB_ENTITY not set - using account default entity")

    @property
    def _has_api_key(self) -> bool:
        return bool(os.environ.get("WANDB_API_KEY"))

    def to_logger_kwargs(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "entity": self.entity or None,
            "group": self.group,
            "name": self.name,
            "tags": self.tags or None,
            "mode": self.mode,
            "notes": self.notes or None,
            "job_type": self.job_type or None,
            "save_dir": str(self.log_dir),
        }

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig, log_dir: Path) -> Self:
        d = OmegaConf.to_container(cfg, resolve=True)
        return cls(**d, log_dir=log_dir)
