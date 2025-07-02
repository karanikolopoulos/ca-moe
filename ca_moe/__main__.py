from ca_moe.definitions.constants import CONFIG_PATH
import hydra
from hydra.core.hydra_config import DictConfig

@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(config: DictConfig):
    print(config)

if __name__ == "__main__":
    main()