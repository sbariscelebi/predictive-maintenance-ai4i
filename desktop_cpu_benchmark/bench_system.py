"""Collect hardware and software information. TensorFlow is imported only when needed."""
import os
import platform
import re
import subprocess
import sys

import psutil


def clean_cpu_name(raw):
    """Call: clean_cpu_name('Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz') -> 'Intel Core i5-10400'

    For AMD: 'AMD Ryzen 5 5600X 6-Core Processor' -> 'AMD Ryzen 5 5600X'
    """
    name = re.sub(r"\((R|TM|r|tm)\)", "", raw)
    name = re.sub(r"\s*@.*$", "", name)
    name = re.sub(r"\bCPU\b", "", name)
    name = re.sub(r"\s*\d+-Core\s+Processor\s*$", "", name)
    name = re.sub(r"\s*Processor\s*$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def cpu_name_windows():
    """Read the processor name from the Windows registry."""
    import winreg
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    try:
        value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    finally:
        winreg.CloseKey(key)
    return value


def cpu_name_linux():
    """Read the processor name from /proc/cpuinfo."""
    with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def cpu_name_macos():
    """Read the processor name with sysctl."""
    result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                            capture_output=True, text=True)
    return result.stdout.strip()


def cpu_name_raw():
    """Call: cpu_name_raw(). Falls back to platform.processor() on any error."""
    try:
        if sys.platform.startswith("win"):
            return cpu_name_windows()
        if sys.platform == "darwin":
            return cpu_name_macos()
        return cpu_name_linux()
    except Exception:
        return platform.processor() or "unknown"


def os_name():
    """Call: os_name() -> 'Windows 11' and similar."""
    if sys.platform.startswith("win"):
        build = sys.getwindowsversion().build
        if build >= 22000:
            return "Windows 11"
        return "Windows " + platform.release()
    return platform.system() + " " + platform.release()


def ram_gb():
    """Call: ram_gb() -> total RAM in GiB."""
    return psutil.virtual_memory().total / (1024 ** 3)


def keras_version(tf):
    """Call: keras_version(tf)

    In TF 2.15 and in Keras 3, tf.keras.__version__ exists. With TF_USE_LEGACY_KERAS=1
    (tf-keras), the version is read from the tf_keras package.
    """
    version = getattr(tf.keras, "__version__", None)
    if version:
        return str(version)
    legacy = os.environ.get("TF_USE_LEGACY_KERAS", "").lower() in ("1", "true")
    if legacy:
        try:
            import tf_keras
            return tf_keras.__version__ + " (tf_keras)"
        except Exception:
            return "tf_keras"
    return "unknown"


def framework_info():
    """Call: framework_info() -> TensorFlow version, Keras version, and number of visible GPUs."""
    import tensorflow as tf
    return {
        "tensorflow": tf.__version__,
        "keras": keras_version(tf),
        "visible_gpus": len(tf.config.list_physical_devices("GPU")),
    }


def collect_system_info():
    """Call: collect_system_info() -> one dictionary."""
    raw = cpu_name_raw()
    info = {
        "cpu_name_raw": raw,
        "cpu_name": clean_cpu_name(raw),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(ram_gb(), 1),
        "os": os_name(),
        "python": platform.python_version(),
    }
    info.update(framework_info())
    return info
