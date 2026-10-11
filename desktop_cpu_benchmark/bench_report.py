"""Turn results into the English article paragraph, a console summary, and output files."""
import json
import os

import numpy as np


def indefinite_article(name):
    """Call: indefinite_article('Intel Core i5-10400') -> 'an'"""
    return "an" if name[:1].lower() in "aeiou" else "a"


def describe_call(mode):
    """Call: describe_call('call')"""
    if mode == "predict":
        return "one model.predict() call of the trained Keras model"
    return "one forward pass of the trained Keras model"


def describe_rate(rate_hz):
    """Call: describe_rate(1.0) -> 'one window per second'"""
    if rate_hz == 1.0:
        return "one window per second"
    return "{:g} windows per second".format(rate_hz)


def format_paragraph(results):
    """Call: format_paragraph(results) -> the article paragraph as plain text."""
    system = results["system"]
    settings = results["settings"]
    latency = results["latency"]
    template = (
        "We evaluated inference performance on a desktop computer with {article} {cpu} "
        "processor and {ram} GB of RAM. Inference used the CPU only. "
        "Each prediction was {call} on a single {window} \u00d7 {features} window "
        "(batch size 1) with TensorFlow {tf} under {os_name}. "
        "After {warmup} warm-up runs, we measured {runs:,} single-window predictions. "
        "Median inference latency was {median:.1f} ms, and the 95th-percentile latency was "
        "{p95:.1f} ms. Mean process CPU utilization was {cpu_pct:.0f}%, where 100% represented "
        "one fully utilized logical CPU core. Peak process memory usage was {memory:.0f} MB, "
        "measured as resident set size. These measurements describe desktop CPU performance. "
        "They do not establish performance on a resource-constrained edge device."
    )
    return template.format(
        article=indefinite_article(system["cpu_name"]),
        cpu=system["cpu_name"],
        ram=int(round(system["ram_gb"])),
        call=describe_call(settings["mode"]),
        window=results["model"]["config"]["window"],
        features=results["model"]["config"]["features"],
        tf=system["tensorflow"],
        os_name=system["os"],
        warmup=settings["warmup"],
        runs=latency["runs"],
        median=latency["median_ms"],
        p95=latency["p95_ms"],
        cpu_pct=results["cpu_loop"]["cpu_percent"],
        memory=results["memory"]["peak_mb"],
    )


def format_optional_sentences(results):
    """Call: format_optional_sentences(results) -> list of optional sentences."""
    sentences = []
    rate = results.get("cpu_rate")
    if rate is not None:
        sentences.append("At {}, mean process CPU utilization was {:.1f}%.".format(
            describe_rate(rate["rate_hz"]), rate["cpu_percent"]))
    parameters = results["model"]["parameters"]
    tflite = results["model"].get("tflite")
    if tflite is not None and tflite.get("ok"):
        sentences.append("The trained model has {:,} parameters and occupies {:.0f} KB as a "
                         "TensorFlow Lite file.".format(parameters, tflite["size_kb"]))
    else:
        sentences.append("The trained model has {:,} parameters.".format(parameters))
    return sentences


def console_summary(results):
    """Call: print(console_summary(results))"""
    system = results["system"]
    latency = results["latency"]
    memory = results["memory"]
    lines = [
        "",
        "=== RESULT ===",
        "Processor        : {} ({} logical cores)".format(system["cpu_name"], system["logical_cores"]),
        "RAM / OS         : {} GB / {}".format(system["ram_gb"], system["os"]),
        "TensorFlow       : {} / Keras {}".format(system["tensorflow"], system["keras"]),
        "Visible GPUs     : {}".format(system["visible_gpus"]),
        "Parameters       : {:,}".format(results["model"]["parameters"]),
        "Call mode        : {}".format(results["settings"]["mode"]),
        "Median latency   : {:.2f} ms".format(latency["median_ms"]),
        "p95 latency      : {:.2f} ms".format(latency["p95_ms"]),
        "CPU (back-to-back): {:.1f} %".format(results["cpu_loop"]["cpu_percent"]),
        "Peak memory      : {:.0f} MB".format(memory["peak_mb"]),
        "Memory stages    : start {:.0f} MB, after TensorFlow import {:.0f} MB, "
        "after model build {:.0f} MB".format(memory["rss_start_mb"], memory["rss_after_import_mb"],
                                             memory["rss_after_build_mb"]),
    ]
    rate = results.get("cpu_rate")
    if rate is not None:
        lines.append("CPU at {:g} Hz      : {:.2f} %".format(rate["rate_hz"], rate["cpu_percent"]))
    parity = results["model"].get("parity_max_abs_diff")
    if parity is not None:
        lines.append("Parity difference: {:.2e}".format(parity))
    for warning in results["warnings"]:
        lines.append("WARNING: " + warning)
    return "\n".join(lines)


def make_json_safe(value):
    """Call: make_json_safe(results). Converts numpy numbers to plain Python numbers."""
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(results, path):
    """Call: save_json(results, 'results/benchmark_results.json')"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(make_json_safe(results), handle, indent=2, ensure_ascii=False)


def save_text(text, path):
    """Call: save_text('...', 'results/benchmark_paragraph.txt')"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


def save_latencies(times, path):
    """Call: save_latencies(times, 'results/latencies_ms.csv')"""
    np.savetxt(path, times, fmt="%.6f", header="latency_ms", comments="")


def write_outputs(results, times, out_dir, file_names):
    """Call: write_outputs(results, times, 'results', {'results': ..., 'paragraph': ..., 'latency': ...})"""
    os.makedirs(out_dir, exist_ok=True)
    save_json(results, os.path.join(out_dir, file_names["results"]))
    paragraph = format_paragraph(results)
    extra = format_optional_sentences(results)
    save_text(paragraph + "\n\nOPTIONAL:\n" + "\n".join(extra), os.path.join(out_dir, file_names["paragraph"]))
    save_latencies(times, os.path.join(out_dir, file_names["latency"]))
    return paragraph, extra
