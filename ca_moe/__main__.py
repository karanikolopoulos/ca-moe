from ca_moe.definitions.constants import CONFIG_PATH
import hydra
from hydra.core.hydra_config import OmegaConf, DictConfig

from ca_moe.data.load_and_preprocess_data import load_datasets
from ca_moe.model.tf.baseline_models import model_factory
from ca_moe.model.train_tools import fit_and_evaluate
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)

_ = load_dotenv()

@hydra.main(version_base=None, config_path=str(CONFIG_PATH), config_name="config")
def main(config: DictConfig):
    logging.info("Resolving Hydra Config")
    config = OmegaConf.to_container(cfg=config, resolve=True)

    # # load the dataset
    train_ds, valid_ds, _ = load_datasets(name=config["dataset"])    
    # # load the model
    model = model_factory(model=config["model"])
    
    print(config)
    history = fit_and_evaluate(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        backend=config["backend"]
    )
    
    return history