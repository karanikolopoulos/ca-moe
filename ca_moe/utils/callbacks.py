import logging

from typing import Any

from omegaconf import DictConfig
from hydra.types import TaskFunction
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback

from pathlib import Path
import pandas as pd

class MyCallback(Callback):

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_start(self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any) -> None:
        pass

    def on_job_end(self, config: DictConfig, job_return: JobReturn, **kwargs: Any) -> None:
        output_dir = Path(config.hydra.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None

        history = job_return.return_value  # can raise
        history = pd.DataFrame(history.history)
        history.index = range(1, len(history)+1)
        
        history.to_json(output_dir/"history.json", indent=4)
        