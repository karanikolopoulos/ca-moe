from ca_moe.load_and_preprocess_data import load_datasets
from baseline_models import *
from train_tools import *
import logging
from ca_moe.definitions.constants import CONFIG_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler(CONFIG_PATH.joinpath("my_logs.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

train_ds, valid_ds, _ = load_datasets(name="mnist")
lr = build_logreg()
nn1 = build_neural_net1()
nn2 = build_neural_net2()
my_callback = CustomCallback(logger=logger)


def my_main(baseline_model):
    fit_and_evaluate(
        model=baseline_model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        logger=logger)
    
my_main(baseline_model=lr)