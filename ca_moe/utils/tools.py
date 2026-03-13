import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        module = record.module
        if "ca_moe" in record.pathname:
            module = "CA-MoE"

        return "[{} - {}] [{} {}:{}] {}".format(
            module,
            record.levelname,
            self.formatTime(record, datefmt="%m-%d %H:%M:%S"),
            record.filename,
            record.lineno,
            record.getMessage(),
        )


@contextmanager
def eval_mode(module: torch.nn.Module):
    training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(training)


def setup_hydra() -> None:
    OmegaConf.register_new_resolver("get_run_name", get_run_name)
    OmegaConf.register_new_resolver("decide_device", decide_device)


def decide_device(device: Literal["cuda", "cpu"]) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return device

    return "cpu"


def get_run_name(model_name: str, training_dict: DictConfig, seed: int) -> str:
    model, _ = model_name.split("/")

    run_name = f"{model_name}_{training_dict.batch_size}_{training_dict.optimizer}_{training_dict.learning_rate}"
    if model == "moe":
        run_name += (
            f"_{training_dict.gamma}_{training_dict.lambda_}_{training_dict.cost_fn}"
        )

    return run_name + f"_{seed}"


def get_last_checkpoint(log_dir: Path, resume: bool) -> str | None:
    last_ckpt = log_dir / "checkpoints" / "last.ckpt"

    if not (resume and last_ckpt.exists()):
        return None

    logger.info(f"Resuming from {last_ckpt}")

    return str(last_ckpt)


_OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
}


def build_optimizer(
    params: Iterable[nn.Parameter],
    name: str,
    lr: float,
) -> torch.optim.Optimizer:
    key = name.lower()
    if key not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: {sorted(_OPTIMIZERS)}"
        )
    return _OPTIMIZERS[key](params, lr=lr)
