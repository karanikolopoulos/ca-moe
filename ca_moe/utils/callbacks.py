import logging

from typing import Any

from omegaconf import DictConfig
from hydra.types import TaskFunction
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback



class MyCallback(Callback):

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        pass

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        pass