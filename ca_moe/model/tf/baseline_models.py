from tensorflow import keras
import logging

logger = logging.getLogger(__name__)

def model_factory(model: str):
    if model == "LR":
        return build_logreg()
    
    if model == "NN1":
        return build_neural_net1()
    
    if model == "NN2":
        return build_neural_net2()

def build_logreg(seed=2025):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input([28, 28]),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ], name="Logistic_Regressor")
    return model

def build_neural_net1(seed=2025):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input([28, 28]),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ], name="Neural_Net_one_hidden")
    return model

def build_neural_net2(seed=2025):
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.Input([28, 28]),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ], name="Neural_Net_two_hidden")
    return model

