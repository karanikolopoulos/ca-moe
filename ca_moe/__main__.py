from ca_moe.definitions.constants import CONFIG_PATH
import hydra
from hydra.core.hydra_config import DictConfig

import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(config: DictConfig):
    print(config)
    logger.info(config)
    
if __name__ == "__main__":
    main()