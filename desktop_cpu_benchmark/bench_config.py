"""Constants and default settings.

This file holds no logic. Other modules read from it.
"""
import os

# Model dimensions. bench_config.json exported from Kaggle overrides these.
DEFAULT_WINDOW = 5
DEFAULT_FEATURES = 10
DEFAULT_CLASSES = 6
DEFAULT_NUM_HEADS = 2
DEFAULT_KEY_DIM = 16
DEFAULT_DROPOUT = 0.4

# Measurement settings.
DEFAULT_WARMUP = 100
DEFAULT_RUNS = 1000
DEFAULT_RATE_HZ = 1.0
DEFAULT_RATE_SECONDS = 30.0
MIN_CPU_LOOP_SECONDS = 5.0
PARITY_TOLERANCE = 1e-4
RANDOM_SEED = 0
SYNTHETIC_WINDOW_COUNT = 1000

# Folders.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "results")

# File names of the files exported from Kaggle.
WEIGHTS_FILE = "bench_weights.npz"
CONFIG_FILE = "bench_config.json"
WINDOWS_FILE = "bench_windows.npy"
REFERENCE_FILE = "bench_reference.npy"

# Output file names.
RESULTS_FILE = "benchmark_results.json"
PARAGRAPH_FILE = "benchmark_paragraph.txt"
LATENCY_FILE = "latencies_ms.csv"
TFLITE_FILE = "model_float32.tflite"
