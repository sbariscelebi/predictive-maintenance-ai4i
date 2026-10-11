"""Desktop CPU inference benchmark. Main entry point.

Usage (Windows command line, inside this folder):

    python run_benchmark.py
    python run_benchmark.py --mode predict --runs 300
    python run_benchmark.py --rate-hz 0 --tflite

First put the four files exported from Kaggle into the data folder:
bench_weights.npz, bench_config.json, bench_windows.npy, bench_reference.npy
If the files are missing, the script does a dry run with random weights and random
windows, and it warns about it.

Run the measurement from the command line, not inside Jupyter. Jupyter inflates the
memory value.
"""
import argparse
import os
import sys

import bench_config as cfg
import bench_latency as lat
import bench_report as rep
import bench_resources as res
import bench_system as sysinfo
import bench_tflite as tfl


def use_utf8_console():
    """Call: use_utf8_console(). Lets the Windows console print the multiplication sign."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def parse_args():
    """Call: args = parse_args()"""
    parser = argparse.ArgumentParser(description="Desktop CPU inference benchmark")
    parser.add_argument("--data-dir", default=cfg.DEFAULT_DATA_DIR,
                        help="Folder with the files exported from Kaggle.")
    parser.add_argument("--out-dir", default=cfg.DEFAULT_OUT_DIR, help="Folder for the result files.")
    parser.add_argument("--mode", choices=sorted(lat.PREDICTORS), default="call",
                        help="'call' times model(x). 'predict' times model.predict(x).")
    parser.add_argument("--warmup", type=int, default=cfg.DEFAULT_WARMUP, help="Number of warm-up runs.")
    parser.add_argument("--runs", type=int, default=cfg.DEFAULT_RUNS, help="Number of timed predictions.")
    parser.add_argument("--rate-hz", type=float, default=cfg.DEFAULT_RATE_HZ,
                        help="Windows per second for the duty-cycle CPU test. 0 skips this test.")
    parser.add_argument("--rate-seconds", type=float, default=cfg.DEFAULT_RATE_SECONDS,
                        help="Duration of the duty-cycle CPU test in seconds.")
    parser.add_argument("--tflite", action="store_true", help="Also measure the TensorFlow Lite file size.")
    return parser.parse_args()


def validate_args(args):
    """Call: validate_args(args). Raises ValueError for invalid values."""
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("warmup must be at least 0 and runs must be at least 1.")
    if args.rate_hz < 0 or args.rate_seconds <= 0:
        raise ValueError("rate-hz cannot be negative and rate-seconds must be positive.")


def import_model_module():
    """Call: bm = import_model_module(). TensorFlow is imported here so memory can be tracked per stage."""
    import bench_model
    return bench_model


def prepare_model(bm, data_dir):
    """Call: model, config, loaded = prepare_model(bm, 'data')"""
    config = bm.config_or_default(bm.read_config(os.path.join(data_dir, cfg.CONFIG_FILE)))
    model = bm.build_model(config["window"], config["features"], config["classes"],
                           config["num_heads"], config["key_dim"])
    weights_path = os.path.join(data_dir, cfg.WEIGHTS_FILE)
    loaded = os.path.isfile(weights_path)
    if loaded:
        bm.load_weights_npz(model, weights_path)
    return model, config, loaded


def prepare_windows(bm, data_dir, config):
    """Call: windows, is_real = prepare_windows(bm, 'data', config)"""
    path = os.path.join(data_dir, cfg.WINDOWS_FILE)
    windows, is_real = bm.load_windows(path, cfg.SYNTHETIC_WINDOW_COUNT, config["window"],
                                       config["features"], cfg.RANDOM_SEED)
    bm.check_window_shape(windows, config)
    return windows, is_real


def run_parity(bm, model, windows, data_dir, weights_loaded):
    """Call: diff = run_parity(bm, model, windows, 'data', True). Returns None without a reference file."""
    reference = bm.load_reference(os.path.join(data_dir, cfg.REFERENCE_FILE))
    if reference is None or not weights_loaded:
        return None
    return bm.predict_parity(model, windows, reference)


def run_rate(args, model, windows):
    """Call: run_rate(args, model, windows). Returns None when --rate-hz is 0."""
    if args.rate_hz <= 0:
        return None
    print("Measuring CPU at {:g} windows per second for {:g} seconds...".format(args.rate_hz, args.rate_seconds))
    return lat.measure_rate(lat.PREDICTORS[args.mode], model, windows, args.rate_hz, args.rate_seconds)


def collect_warnings(args, weights_loaded, windows_real, loop, parity):
    """Call: collect_warnings(args, True, True, loop, parity) -> list of warning texts."""
    warnings = []
    if not weights_loaded:
        warnings.append("Trained weights were not loaded. The latency is valid, but this is not the trained model.")
    if not windows_real:
        warnings.append("No real test windows found. Random windows were used.")
    if not loop["cpu_reliable"]:
        warnings.append("The measurement loop lasted {:.1f} s. CPU percent needs at least {:g} s. "
                        "Increase --runs.".format(loop["wall_s"], cfg.MIN_CPU_LOOP_SECONDS))
    if parity is not None and parity > cfg.PARITY_TOLERANCE:
        warnings.append("Parity difference {:.2e} exceeds the tolerance {:.0e}. The weights or the model "
                        "structure do not match Kaggle.".format(parity, cfg.PARITY_TOLERANCE))
    return warnings


def build_results(args, system, config, parameters, loop, rate, memory, parity, tflite, flags):
    """Call: results = build_results(...). Collects all parts into one dictionary."""
    summary = lat.summarize_latency(loop["latencies_ms"])
    return {
        "system": system,
        "settings": {"mode": args.mode, "warmup": args.warmup, "runs": args.runs},
        "model": {"config": config, "parameters": parameters, "weights_loaded": flags["weights_loaded"],
                  "windows_real": flags["windows_real"], "parity_max_abs_diff": parity, "tflite": tflite},
        "latency": summary,
        "cpu_loop": {"cpu_percent": loop["cpu_percent"], "wall_s": loop["wall_s"],
                     "reliable": loop["cpu_reliable"]},
        "cpu_rate": rate,
        "memory": memory,
        "warnings": collect_warnings(args, flags["weights_loaded"], flags["windows_real"], loop, parity),
    }


def main():
    """Call: main(). Exit code 0 means the measurement finished."""
    use_utf8_console()
    args = parse_args()
    validate_args(args)
    rss_start = res.current_rss_mb()
    bm = import_model_module()
    rss_import = res.current_rss_mb()
    model, config, weights_loaded = prepare_model(bm, args.data_dir)
    windows, windows_real = prepare_windows(bm, args.data_dir, config)
    rss_build = res.current_rss_mb()
    print("Warm-up: {} runs. Measurement: {} runs ({} mode)...".format(args.warmup, args.runs, args.mode))
    loop = lat.measure_loop(lat.PREDICTORS[args.mode], model, windows, args.warmup, args.runs)
    rate = run_rate(args, model, windows)
    memory = {"rss_start_mb": rss_start, "rss_after_import_mb": rss_import,
              "rss_after_build_mb": rss_build, "peak_mb": res.peak_memory_mb()}
    parity = run_parity(bm, model, windows, args.data_dir, weights_loaded)
    tflite = None
    if args.tflite:
        tflite = tfl.try_tflite(model, os.path.join(args.out_dir, cfg.TFLITE_FILE))
    flags = {"weights_loaded": weights_loaded, "windows_real": windows_real}
    results = build_results(args, sysinfo.collect_system_info(), config, bm.count_parameters(model),
                            loop, rate, memory, parity, tflite, flags)
    names = {"results": cfg.RESULTS_FILE, "paragraph": cfg.PARAGRAPH_FILE, "latency": cfg.LATENCY_FILE}
    paragraph, extra = rep.write_outputs(results, loop["latencies_ms"], args.out_dir, names)
    print(rep.console_summary(results))
    print("\nARTICLE PARAGRAPH:\n" + paragraph)
    print("\nOPTIONAL SENTENCES:\n" + "\n".join(extra))
    print("\nFiles written to: " + os.path.abspath(args.out_dir))
    return 0


def run_guarded():
    """Call: sys.exit(run_guarded()). Prints a short message instead of a traceback for expected errors."""
    try:
        return main()
    except ValueError as error:
        print("ERROR: " + str(error))
        return 2


if __name__ == "__main__":
    sys.exit(run_guarded())
