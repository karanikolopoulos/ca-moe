from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from hydra.utils import instantiate

import logging

logger = logging.getLogger(__name__)

class CustomCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.num_epochs = 0

    def on_train_begin(self, logs=None) -> None:
        self.start_time = datetime.now()
        
    def on_epoch_end(self, epoch: int, logs: dict=None) -> None:
        self.num_epochs += 1
        tl = logs["loss"]
        ta = logs["accuracy"]
        vl = logs["val_loss"]
        va = logs["val_accuracy"]

        msg = f"Epoch {epoch + 1} - loss: {tl:.4f} - val_loss: {vl:.4f} - accuracy: {ta:.2%} - val_accuracy: {va:.2%}"
        logger.info(msg)
        
    def on_train_end(self, logs=None) -> None:
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Duration: {duration} seconds.")


def fit_and_evaluate(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    backend: dict
    ): # TODO: add hint
    logger.info(f"Training {backend['model']} on {backend['device']}")
    
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=backend["patience"],
        restore_best_weights=True
    )
    
    optimizer = instantiate(backend["optimizer"])    
    model.compile(
        optimizer=optimizer,
        loss=backend["loss"],
        metrics=backend["metrics"]
    )
    
    batch_size = backend["batch_size"]
    with tf.device(backend["device"]):
        history = model.fit(
            train_ds.batch(batch_size),
            validation_data=valid_ds.batch(batch_size),
            epochs=backend["epochs"],
            verbose=False,
            callbacks=[CustomCallback(), early_stopping_cb])
        
    return history