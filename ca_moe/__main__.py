import logging

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import DictConfig, OmegaConf
from hydra.utils import call

from ca_moe.definitions.constants import CONFIG_PATH
from ca_moe.utils.tools import calc_channels, flatten, get_avg_pool_out_shape

logger = logging.getLogger(__name__)

_ = load_dotenv()

OmegaConf.register_new_resolver(name="flatten", resolver=flatten)
OmegaConf.register_new_resolver(name="calc_channels", resolver=calc_channels)
OmegaConf.register_new_resolver(name="calc_pool_out", resolver=get_avg_pool_out_shape)


@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(config: DictConfig):
    logger.info("Resolving Hydra Config")
    OmegaConf.resolve(cfg=config)

    builder = call(config.backend.builder)
    model = builder.build(config.model.layers)

    print(model.summary())


if __name__ == "__main__":
    main()
