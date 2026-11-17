# SPDX-License-Identifier: Apache-2.0

"""Local sensor reads (the node this runs on). All read-only."""

import glob
import shutil
import subprocess

CPUFREQ_AVG = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_avg_freq"))
CPUFREQ_CUR = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"))
THERMAL_ZONES = sorted(glob.glob("/sys/class/thermal/thermal_zone*"))
NVIDIA_SMI = shutil.which("nvidia-smi")


def smi_num(s):
    """Parse nvidia-smi CSV field; None for [N/A]/empty."""
    s = (s or "").strip()
    if s in ("", "[N/A]", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_sysfs_khz(path):
    """Read a sysfs kHz node. Binary I/O avoids text-codec races on sysfs."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        if not raw:
            return None
        s = raw.strip().decode("ascii", "ignore")
        if not s or not s.isdigit():
            return None
        return int(s)
    except (OSError, ValueError, TypeError):
        return None


def read_cpu_mhz():
    """Average ACTUAL CPU MHz via cpuinfo_avg_freq (CPPC). scaling_cur_freq is
    only a governor target and sits at cluster max on this host."""
    paths = CPUFREQ_AVG or CPUFREQ_CUR
    if not paths:
        return None
    vals = []
    for path in paths:
        khz = read_sysfs_khz(path)
        if khz is not None and khz > 0:
            vals.append(khz / 1000.0)
    if not vals and CPUFREQ_AVG and CPUFREQ_CUR:
        for path in CPUFREQ_CUR:
            khz = read_sysfs_khz(path)
            if khz is not None and khz > 0:
                vals.append(khz / 1000.0)
    return sum(vals) / len(vals) if vals else None


def read_gpu():
    """(util_pct, temp_c, sm_mhz) — fields may be None."""
    if not NVIDIA_SMI:
        return None, None, None
    try:
        out = subprocess.run(
            [NVIDIA_SMI,
             "--query-gpu=utilization.gpu,temperature.gpu,clocks.current.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()
        if not out:
            return None, None, None
        parts = [x.strip() for x in out[0].split(",")]
        while len(parts) < 3:
            parts.append("")
        return smi_num(parts[0]), smi_num(parts[1]), smi_num(parts[2])
    except Exception:
        return None, None, None


def read_zones():
    """[(label, temp_c)] from sysfs thermal zones."""
    zones = []
    for z in THERMAL_ZONES:
        try:
            with open(z + "/temp", "rb") as f:
                milli = int(f.read().strip())
            with open(z + "/type", "rb") as f:
                typ = f.read().strip().decode("ascii", "ignore").strip()
        except (OSError, ValueError):
            continue
        idx = z.rsplit("thermal_zone", 1)[-1]
        zones.append((f"{typ}{idx}", milli / 1000.0))
    return zones
