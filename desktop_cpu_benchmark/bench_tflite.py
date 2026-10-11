"""Convert the model to TensorFlow Lite and report the file size.

This step is optional. A failure here does not affect the main measurement.
"""
import contextlib
import io
import os


def convert_to_tflite(model):
    """Call: blob = convert_to_tflite(model) -> bytes. Silences the converter's console output."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return converter.convert()


def size_kb(blob):
    """Call: size_kb(blob). 1 KB = 1000 bytes."""
    return len(blob) / 1000.0


def write_blob(blob, path):
    """Call: write_blob(blob, 'results/model_float32.tflite')"""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(blob)


def try_tflite(model, path):
    """Call: try_tflite(model, 'results/model_float32.tflite')

    On success it returns {'ok': True, 'bytes': ..., 'size_kb': ...}.
    On failure it returns {'ok': False, 'error': '...'}.
    """
    try:
        blob = convert_to_tflite(model)
        write_blob(blob, path)
        return {"ok": True, "bytes": len(blob), "size_kb": size_kb(blob)}
    except Exception as error:
        return {"ok": False, "error": str(error)[:300]}
