"""Unit tests. Run with: python -m unittest -v test_bench

Windows-only branches (winreg, peak_wset, getwindowsversion) are tested with fake objects.
"""
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

import bench_latency as lat
import bench_model as bm
import bench_report as rep
import bench_resources as res
import bench_system as sysinfo


def stub_predict(model, window):
    time.sleep(0.001)
    return window


def fake_open_key(root, path):
    return "key"


def fake_query_value(key, name):
    return ("Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz", 1)


def fake_close_key(key):
    return None


class FakeProcess:
    def memory_info(self):
        return types.SimpleNamespace(peak_wset=670_000_000, rss=500_000_000)


def make_fake_results():
    return {
        "system": {"cpu_name": "Intel Core i5-10400", "ram_gb": 15.8, "tensorflow": "2.15.0",
                   "os": "Windows 11", "keras": "2.15.0", "visible_gpus": 0, "logical_cores": 12},
        "settings": {"mode": "call", "warmup": 100, "runs": 1000},
        "model": {"config": {"window": 5, "features": 10}, "parameters": 1472,
                  "tflite": {"ok": True, "size_kb": 14.388}, "parity_max_abs_diff": 1e-7},
        "latency": {"runs": 1000, "median_ms": 6.04, "p95_ms": 8.26},
        "cpu_loop": {"cpu_percent": 96.4},
        "cpu_rate": {"rate_hz": 1.0, "cpu_percent": 0.74},
        "memory": {"peak_mb": 671.2, "rss_start_mb": 30.0, "rss_after_import_mb": 630.0,
                   "rss_after_build_mb": 640.0},
        "warnings": [],
    }


class TestSystem(unittest.TestCase):
    def test_clean_cpu_name(self):
        raw = "Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz"
        self.assertEqual(sysinfo.clean_cpu_name(raw), "Intel Core i5-10400")
        self.assertEqual(sysinfo.clean_cpu_name("AMD Ryzen 5 5600X 6-Core Processor"), "AMD Ryzen 5 5600X")
        self.assertEqual(sysinfo.clean_cpu_name("Intel(R) Xeon(R) Processor @ 2.80GHz"), "Intel Xeon")

    def test_windows_cpu_name_uses_registry(self):
        fake = types.SimpleNamespace(
            HKEY_LOCAL_MACHINE=1,
            OpenKey=fake_open_key,
            QueryValueEx=fake_query_value,
            CloseKey=fake_close_key,
        )
        with mock.patch.dict(sys.modules, {"winreg": fake}):
            with mock.patch.object(sys, "platform", "win32"):
                self.assertEqual(sysinfo.cpu_name_raw(), "Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz")

    def test_os_name_windows(self):
        with mock.patch.object(sys, "platform", "win32"):
            with mock.patch.object(sys, "getwindowsversion", create=True,
                                   return_value=types.SimpleNamespace(build=22631)):
                self.assertEqual(sysinfo.os_name(), "Windows 11")
            with mock.patch.object(sys, "getwindowsversion", create=True,
                                   return_value=types.SimpleNamespace(build=19045)):
                self.assertTrue(sysinfo.os_name().startswith("Windows "))
                self.assertNotEqual(sysinfo.os_name(), "Windows 11")

    def test_keras_version_variants(self):
        modern = types.SimpleNamespace(keras=types.SimpleNamespace(__version__="2.15.0"))
        self.assertEqual(sysinfo.keras_version(modern), "2.15.0")
        legacy = types.SimpleNamespace(keras=types.SimpleNamespace())
        with mock.patch.dict(os.environ, {"TF_USE_LEGACY_KERAS": "0"}):
            self.assertEqual(sysinfo.keras_version(legacy), "unknown")

    def test_collect_system_info_keys(self):
        info = sysinfo.collect_system_info()
        for key in ("cpu_name", "logical_cores", "ram_gb", "os", "tensorflow", "keras", "visible_gpus"):
            self.assertIn(key, info)
        self.assertEqual(info["visible_gpus"], 0)


class TestResources(unittest.TestCase):
    def test_cpu_percent(self):
        self.assertAlmostEqual(res.cpu_percent_of_one_core(1.0, 10.5, 10.0), 95.0)

    def test_peak_memory_windows_branch(self):
        with mock.patch.object(res, "PROCESS", FakeProcess()):
            self.assertAlmostEqual(res.peak_memory_mb(), 670.0)

    def test_peak_memory_real_process(self):
        self.assertGreater(res.peak_memory_mb(), 10.0)
        self.assertGreaterEqual(res.peak_memory_mb(), res.current_rss_mb() * 0.9)


class TestLatency(unittest.TestCase):
    def test_get_window_wraps(self):
        windows = np.arange(3 * 5 * 10, dtype="float32").reshape(3, 5, 10)
        self.assertEqual(lat.get_window(windows, 4).shape, (1, 5, 10))
        self.assertTrue(np.array_equal(lat.get_window(windows, 4), windows[1:2]))

    def test_summarize_latency(self):
        summary = lat.summarize_latency(np.arange(1, 101, dtype=float))
        self.assertAlmostEqual(summary["median_ms"], 50.5)
        self.assertAlmostEqual(summary["p95_ms"], 95.05)
        self.assertEqual(summary["runs"], 100)

    def test_measure_loop_short_loop_is_not_reliable(self):
        windows = np.zeros((4, 5, 10), dtype="float32")
        loop = lat.measure_loop(stub_predict, None, windows, 3, 20)
        self.assertEqual(len(loop["latencies_ms"]), 20)
        self.assertFalse(loop["cpu_reliable"])
        self.assertGreaterEqual(loop["latencies_ms"].min(), 1.0)

    def test_measure_rate_counts_windows(self):
        windows = np.zeros((4, 5, 10), dtype="float32")
        result = lat.measure_rate(stub_predict, None, windows, 20.0, 1.0)
        self.assertTrue(18 <= result["windows"] <= 21)
        self.assertLess(result["cpu_percent"], 30.0)


class TestModel(unittest.TestCase):
    def test_parameter_count_matches_notebook(self):
        model = bm.build_model(5, 10, 6, 2, 16)
        self.assertEqual(bm.count_parameters(model), 1472)

    def test_call_and_predict_agree(self):
        model = bm.build_model(5, 10, 6, 2, 16)
        window = np.random.default_rng(1).random((1, 5, 10), dtype=np.float32)
        out_call = lat.predict_call(model, window)
        out_predict = lat.predict_predict(model, window)
        self.assertEqual(out_call.shape, (1, 6))
        self.assertTrue(np.allclose(out_call, out_predict, atol=1e-6))
        self.assertAlmostEqual(float(out_call.sum()), 1.0, places=5)

    def test_weights_roundtrip_and_parity(self):
        source = bm.build_model(5, 10, 6, 2, 16)
        target = bm.build_model(5, 10, 6, 2, 16)
        windows = np.random.default_rng(2).random((20, 5, 10), dtype=np.float32)
        reference = source.predict(windows, verbose=0)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "w.npz")
            arrays = {"w{:03d}".format(i): w for i, w in enumerate(source.get_weights())}
            np.savez(path, **arrays)
            self.assertEqual(bm.load_weights_npz(target, path), len(arrays))
        self.assertLess(bm.predict_parity(target, windows, reference), 1e-5)
        untrained = bm.build_model(5, 10, 6, 2, 16)
        self.assertGreater(bm.predict_parity(untrained, windows, reference), 1e-4)

    def test_window_shape_check(self):
        config = bm.config_or_default(None)
        bm.check_window_shape(np.zeros((3, 5, 10), dtype="float32"), config)
        with self.assertRaises(ValueError):
            bm.check_window_shape(np.zeros((3, 4, 10), dtype="float32"), config)

    def test_load_windows_fallback_is_seeded(self):
        first, real_first = bm.load_windows(None, 8, 5, 10, 0)
        second, real_second = bm.load_windows(None, 8, 5, 10, 0)
        self.assertFalse(real_first or real_second)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.float32)


class TestReport(unittest.TestCase):
    def test_article(self):
        self.assertEqual(rep.indefinite_article("Intel Core i5"), "an")
        self.assertEqual(rep.indefinite_article("Ryzen 5"), "a")

    def test_paragraph_contains_measured_values(self):
        text = rep.format_paragraph(make_fake_results())
        for expected in ("an Intel Core i5-10400 processor", "16 GB of RAM", "TensorFlow 2.15.0",
                         "under Windows 11", "Median inference latency was 6.0 ms",
                         "95th-percentile latency was 8.3 ms", "utilization was 96%",
                         "usage was 671 MB", "1,000 single-window predictions"):
            self.assertIn(expected, text)

    def test_optional_sentences(self):
        sentences = rep.format_optional_sentences(make_fake_results())
        self.assertEqual(sentences[0], "At one window per second, mean process CPU utilization was 0.7%.")
        self.assertIn("1,472 parameters and occupies 14 KB", sentences[1])

    def test_console_summary_is_english(self):
        text = rep.console_summary(make_fake_results())
        self.assertIn("=== RESULT ===", text)
        self.assertIn("Median latency", text)

    def test_outputs_are_written(self):
        names = {"results": "r.json", "paragraph": "p.txt", "latency": "l.csv"}
        with tempfile.TemporaryDirectory() as folder:
            rep.write_outputs(make_fake_results(), np.array([1.0, 2.0, 3.0]), folder, names)
            self.assertEqual(sorted(os.listdir(folder)), ["l.csv", "p.txt", "r.json"])
            with open(os.path.join(folder, "p.txt"), encoding="utf-8") as handle:
                self.assertIn("5 \u00d7 10 window", handle.read())


if __name__ == "__main__":
    unittest.main()
