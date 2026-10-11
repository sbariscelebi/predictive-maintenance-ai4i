"""Build the model, load weights, and prepare input windows.

The model matches create_transformer_model in the training notebook.
One difference: GlobalAveragePooling1D replaces tf.reduce_mean(x, axis=1).
Both compute the same mean over the time axis and neither has parameters.
GlobalAveragePooling1D also works in Keras 2 and Keras 3.

This module imports TensorFlow. Only the CPU is visible to it.
"""
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model

import bench_config as cfg


class PositionalEncoding(tf.keras.layers.Layer):
    """Sinusoidal positional encoding, as in the training notebook."""

    def __init__(self, window_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(window_size, d_model)

    def positional_encoding(self, window_size, d_model):
        position = np.arange(window_size)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((window_size, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.cast(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:tf.shape(x)[1], :]


def build_model(window, features, classes, num_heads, key_dim):
    """Call: model = build_model(5, 10, 6, 2, 16)"""
    inputs = Input(shape=(window, features))
    x = PositionalEncoding(window, features)(inputs)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Dropout(cfg.DEFAULT_DROPOUT)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(units=classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


def read_config(path):
    """Call: read_config('data/bench_config.json'). Returns None if the file is missing."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def config_or_default(config):
    """Call: config_or_default(None) returns the default dimensions."""
    merged = {
        "window": cfg.DEFAULT_WINDOW,
        "features": cfg.DEFAULT_FEATURES,
        "classes": cfg.DEFAULT_CLASSES,
        "num_heads": cfg.DEFAULT_NUM_HEADS,
        "key_dim": cfg.DEFAULT_KEY_DIM,
    }
    if config is not None:
        merged.update(config)
    return merged


def load_weights_npz(model, path):
    """Call: load_weights_npz(model, 'data/bench_weights.npz'). Returns the number of arrays loaded."""
    with np.load(path) as data:
        names = sorted(data.files)
        weights = [data[name] for name in names]
    model.set_weights(weights)
    return len(weights)


def load_windows(path, count, window, features, seed):
    """Call: windows, is_real = load_windows('data/bench_windows.npy', 1000, 5, 10, 0)

    Reads real windows if the file exists. Otherwise it generates random windows.
    """
    if path is not None and os.path.isfile(path):
        return np.load(path).astype("float32"), True
    rng = np.random.default_rng(seed)
    return rng.random((count, window, features), dtype=np.float32), False


def check_window_shape(windows, config):
    """Call: check_window_shape(windows, config). Raises ValueError if the shape is wrong."""
    expected = (config["window"], config["features"])
    if windows.ndim != 3 or tuple(windows.shape[1:]) != expected:
        raise ValueError("Window array has shape " + str(windows.shape) + ", expected (N, "
                         + str(expected[0]) + ", " + str(expected[1]) + ").")


def load_reference(path):
    """Call: load_reference('data/bench_reference.npy'). Returns None if the file is missing."""
    if path is None or not os.path.isfile(path):
        return None
    return np.load(path).astype("float32")


def predict_parity(model, windows, reference):
    """Call: predict_parity(model, windows, reference)

    Returns the largest absolute difference between the class probabilities saved on
    Kaggle and the probabilities this model produces on the same windows.
    """
    count = min(len(windows), len(reference))
    probabilities = model.predict(windows[:count], verbose=0)
    return float(np.max(np.abs(probabilities - reference[:count])))


def count_parameters(model):
    """Call: count_parameters(model)"""
    return int(model.count_params())
