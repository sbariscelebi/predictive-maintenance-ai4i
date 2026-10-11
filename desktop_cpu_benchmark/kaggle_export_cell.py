"""ADD THIS AS A NEW CELL AT THE END OF THE KAGGLE NOTEBOOK. It does not touch existing code.

This cell exports the trained best_model so it can be benchmarked on a desktop PC.
Run it after training, once X_test (the 3D window array) exists.

Download the four files in /kaggle/working/bench/ and place them in
edge_benchmark/data/ on the PC:
    bench_weights.npz, bench_config.json, bench_windows.npy, bench_reference.npy

Weights are saved as plain numpy arrays. The file therefore loads even when the
Keras version on the PC differs from the one on Kaggle.
"""
import json
import os

import numpy as np
import tensorflow as tf


def find_attention_layer(model):
    """Call: find_attention_layer(best_model)"""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.MultiHeadAttention):
            return layer
    raise ValueError("The model has no MultiHeadAttention layer.")


def build_export_config(model):
    """Call: build_export_config(best_model) -> window, features, classes, num_heads, key_dim."""
    attention_config = find_attention_layer(model).get_config()
    return {
        "window": int(model.input_shape[1]),
        "features": int(model.input_shape[2]),
        "classes": int(model.output_shape[-1]),
        "num_heads": int(attention_config["num_heads"]),
        "key_dim": int(attention_config["key_dim"]),
    }


def save_weights_npz(model, path):
    """Call: save_weights_npz(best_model, '/kaggle/working/bench/bench_weights.npz')"""
    arrays = {"w{:03d}".format(index): weight for index, weight in enumerate(model.get_weights())}
    np.savez(path, **arrays)
    return len(arrays)


def save_json(data, path):
    """Call: save_json({'a': 1}, 'x.json')"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_windows_and_reference(model, windows_all, count, windows_path, reference_path):
    """Call: save_windows_and_reference(best_model, X_test, 1000, 'w.npy', 'r.npy')

    Saves the first `count` test windows and the class probabilities the model gives for them.
    If the PC reproduces the same probabilities, the weights were loaded correctly.
    """
    windows = np.asarray(windows_all[:count], dtype="float32")
    reference = model.predict(windows, verbose=0).astype("float32")
    np.save(windows_path, windows)
    np.save(reference_path, reference)
    return windows.shape


def export_all(model, windows_all, out_dir="/kaggle/working/bench", count=1000):
    """Call: export_all(best_model, X_test)"""
    os.makedirs(out_dir, exist_ok=True)
    config = build_export_config(model)
    save_json(config, os.path.join(out_dir, "bench_config.json"))
    arrays = save_weights_npz(model, os.path.join(out_dir, "bench_weights.npz"))
    shape = save_windows_and_reference(model, windows_all, count,
                                       os.path.join(out_dir, "bench_windows.npy"),
                                       os.path.join(out_dir, "bench_reference.npy"))
    print("Config:", config)
    print("Weight arrays:", arrays, "Window shape:", shape, "Parameters:", model.count_params())
    print("Files:", sorted(os.listdir(out_dir)))


export_all(best_model, X_test)
