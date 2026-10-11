"""Process CPU time and memory measurements.

CPU percent: 100 means one logical core is fully busy.
Memory unit: 1 MB = 1,000,000 bytes.
"""
import sys

import psutil

PROCESS = psutil.Process()


def cpu_seconds():
    """Call: cpu_seconds(). Total user + system CPU time the process has used so far."""
    times = PROCESS.cpu_times()
    return times.user + times.system


def current_rss_mb():
    """Call: current_rss_mb(). Current resident set size."""
    return PROCESS.memory_info().rss / 1e6


def peak_memory_mb():
    """Call: peak_memory_mb()

    Windows: psutil peak_wset (peak working set).
    Linux: resource.ru_maxrss, reported in KB.
    macOS: resource.ru_maxrss, reported in bytes.
    """
    info = PROCESS.memory_info()
    if hasattr(info, "peak_wset"):
        return info.peak_wset / 1e6
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / 1e6
    return peak * 1024 / 1e6


def cpu_percent_of_one_core(cpu_start, cpu_end, wall_seconds):
    """Call: cpu_percent_of_one_core(0.0, 9.5, 10.0) -> 95.0"""
    return 100.0 * (cpu_end - cpu_start) / wall_seconds
