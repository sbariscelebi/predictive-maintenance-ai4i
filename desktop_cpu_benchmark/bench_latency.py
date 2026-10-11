"""Single-window latency and CPU percent measurement.

Each call is timed on its own with time.perf_counter(). Median and p95 come from that list.
"""
import time

import numpy as np

import bench_config as cfg
import bench_resources as res


def get_window(windows, index):
    """Call: get_window(windows, 7) -> one window with shape (1, window, features).

    Windows are served in order. An index beyond N wraps around to the start.
    """
    position = index % len(windows)
    return windows[position:position + 1]


def predict_call(model, window):
    """Direct model call. This is the fast path for a single window."""
    return model(window, training=False).numpy()


def predict_predict(model, window):
    """model.predict() call. About 7 to 8 times slower for a single window."""
    return model.predict(window, verbose=0)


PREDICTORS = {"call": predict_call, "predict": predict_predict}


def warm_up(predict_fn, model, windows, count):
    """Call: warm_up(predict_call, model, windows, 100)"""
    for index in range(count):
        predict_fn(model, get_window(windows, index))


def time_calls(predict_fn, model, windows, runs):
    """Call: time_calls(predict_call, model, windows, 1000) -> array of latencies in milliseconds."""
    times = np.empty(runs)
    for index in range(runs):
        window = get_window(windows, index)
        start = time.perf_counter()
        predict_fn(model, window)
        times[index] = (time.perf_counter() - start) * 1000.0
    return times


def summarize_latency(times):
    """Call: summarize_latency(times) -> median, p95, p99, mean, min, and max."""
    return {
        "runs": int(len(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def measure_loop(predict_fn, model, windows, warmup, runs):
    """Call: loop = measure_loop(predict_call, model, windows, 100, 1000)

    After the warm-up, it makes `runs` back-to-back predictions and derives the CPU
    percent from the same loop. A loop shorter than 5 seconds is marked as unreliable
    for CPU percent.
    """
    warm_up(predict_fn, model, windows, warmup)
    cpu_start = res.cpu_seconds()
    wall_start = time.perf_counter()
    times = time_calls(predict_fn, model, windows, runs)
    wall = time.perf_counter() - wall_start
    cpu_end = res.cpu_seconds()
    return {
        "latencies_ms": times,
        "wall_s": wall,
        "cpu_percent": res.cpu_percent_of_one_core(cpu_start, cpu_end, wall),
        "cpu_reliable": wall >= cfg.MIN_CPU_LOOP_SECONDS,
    }


def sleep_until(deadline):
    """Call: sleep_until(time.perf_counter() + 0.5)"""
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def measure_rate(predict_fn, model, windows, rate_hz, seconds):
    """Call: measure_rate(predict_call, model, windows, 1.0, 30.0)

    Feeds `rate_hz` windows per second for `seconds` seconds and measures the process
    CPU percent. This imitates a real monitoring scenario.
    """
    period = 1.0 / rate_hz
    count = 0
    cpu_start = res.cpu_seconds()
    start = time.perf_counter()
    deadline = start
    while time.perf_counter() - start < seconds:
        predict_fn(model, get_window(windows, count))
        count += 1
        deadline += period
        sleep_until(deadline)
    wall = time.perf_counter() - start
    cpu_end = res.cpu_seconds()
    return {
        "rate_hz": rate_hz,
        "windows": count,
        "wall_s": wall,
        "cpu_percent": res.cpu_percent_of_one_core(cpu_start, cpu_end, wall),
    }
