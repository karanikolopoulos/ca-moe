from datetime import datetime
import tensorflow as tf
from tensorflow import keras

device_protos = tf.config.list_physical_devices()[0]

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.num_epochs = 0
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def on_train_begin(self, logs=None):
        self.start_time = datetime.now()
        device_proto = tf.config.list_physical_devices()[0].device_type
        if "GPU" in device_proto:
            self.logger.info(f"Training {self.model.name} on GPU")
        else:
            self.logger.info(f"Training {self.model.name} on CPU")

    def on_epoch_end(self, epoch, logs=None):
        self.num_epochs += 1
        tl = logs["loss"]
        ta = logs["accuracy"]
        vl = logs["val_loss"]
        va = logs["val_accuracy"]

        msg = f"Epoch {epoch + 1} - loss: {tl:.4f} - val_loss: {vl:.4f} - accuracy: {ta:.2%} - val_accuracy: {va:.2%}"
        self.logger.info(msg)
        
    def on_train_end(self, logs=None):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        self.logger.info(f"Duration: {duration} seconds.")


def fit_and_evaluate(model, train_ds, valid_ds, logger,
                     epochs=50, batch_size=32):
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    
    if "GPU" in device_protos.device_type:
        with tf.device("GPU: 0"):
            history = model.fit(
                train_ds.batch(batch_size).prefetch(1),
                validation_data=valid_ds.batch(batch_size),
                epochs=epochs,
                verbose=False,
                callbacks=[CustomCallback(logger=logger), early_stopping_cb])
    else:
        history = model.fit(
                train_ds.batch(batch_size).prefetch(1),
                validation_data=valid_ds.batch(batch_size),
                epochs=epochs,
                verbose=False,
                callbacks=[CustomCallback(logger=logger), early_stopping_cb])
    
    return history