import tensorflow as tf


def load_datasets(name : str) -> tuple[tf.data.Dataset]:
    if name == "mnist":
        data = tf.keras.datasets.mnist
    elif name == "fashion_mnist":
        data = tf.keras.datasets.fashion_mnist
    else:
        raise TypeError
    (X_train_full, y_train_full), (X_test, y_test) = data.load_data()
    X_train = X_train_full[:-10_000] / 255.
    y_train = y_train_full[:-10_000]
    X_valid = X_train_full[-10_000:] / 255.
    y_valid = y_train_full[-10_000:]
    X_test = X_test / 255.

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    return train_ds, valid_ds, test_ds
    