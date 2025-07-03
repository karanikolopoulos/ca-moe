import tensorflow as tf
from tensorflow import keras


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

