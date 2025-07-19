import logging

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import DictConfig, OmegaConf
from hydra.utils import call

from ca_moe.definitions.constants import CONFIG_PATH
from ca_moe.utils.tools import calc_channels, flatten

import numpy as np

logger = logging.getLogger(__name__)

_ = load_dotenv()

OmegaConf.register_new_resolver(name="flatten", resolver=flatten)
OmegaConf.register_new_resolver(name="calc_channels", resolver=calc_channels)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(config: DictConfig):
    logging.info("Resolving Hydra Config")
    OmegaConf.resolve(cfg=config)

    builder = call(config.backend.builder)
    model = builder.build(config.model.layers)

if __name__ == "__main__":
    main()
