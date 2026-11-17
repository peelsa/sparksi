# SPDX-License-Identifier: Apache-2.0

"""Box-drawing primitives — identical feel to the legacy TUI."""

W = 40                  # visible content width inside the box
BAR_W = 14              # width of the system utilisation bars
DIM = "\033[2;36m"      # dim cyan for filled cells
FAINT = "\033[2;90m"    # faint grey for empty cells
HEAD = "\033[1;36m"     # bright cyan for section frames
RESET = "\033[0m"

NOMINAL_GB = 128.0      # GB10 nameplate; kernel vm.total is lower (carve-out)


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


def bar_segments(pct, w=BAR_W):
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100.0 * w))
    return [("█" * filled, DIM), ("░" * (w - filled), FAINT)]


def _vis(segs):
    return sum(len(t) for t, _ in segs)


def row(segs):
    body = "".join((c + t + RESET) if c else t for t, c in segs)
    pad = W - _vis(segs)
    if pad > 0:
        body += " " * pad
    return HEAD + "│" + RESET + " " + body + " " + HEAD + "│" + RESET


def top(title, right=""):
    label = f"─ {title} "
    tail = f" {right} ─" if right else "─"
    fill = "─" * max(0, (W + 2) - len(label) - len(tail))
    return HEAD + "┌" + label + fill + tail + "┐" + RESET


def divider(title, right=""):
    label = f"─ {title} "
    tail = f" {right} ─" if right else "─"
    fill = "─" * max(0, (W + 2) - len(label) - len(tail))
    return HEAD + "├" + label + fill + tail + "┤" + RESET


def bottom():
    return HEAD + "└" + "─" * (W + 2) + "┘" + RESET


def text(s, color=None):
    return row([(s, color)])


def label_bar(label, value_str, pct):
    left = f"{label}  {value_str}"
    gap = W - len(left) - BAR_W
    segs = [(left, None)]
    if gap > 0:
        segs.append((" " * gap, None))
    segs.extend(bar_segments(pct))
    return row(segs)


def system_lines(mem_used, mem_total, mem_pct, mem_avail,
                 cpu_pct, cpu_mhz, gpu_util, gpu_temp, gpu_mhz, zones):
    """Mem/Sys/CPU/GPU/Temp block for one unit. All sizes GiB, temps °C."""
    out = [label_bar("Mem", f"{mem_used:5.1f}/{mem_total:4.1f}G {mem_pct:4.1f}%", mem_pct)]
    reserved = max(0.0, NOMINAL_GB - mem_total)
    used128 = NOMINAL_GB - mem_avail
    pct128 = used128 / NOMINAL_GB * 100.0
    out.append(text(f"Sys  {used128:5.1f}/128.0G {pct128:4.1f}%  rsvd {reserved:.1f}G"))
    if cpu_mhz is not None:
        out.append(label_bar("CPU", f"{cpu_pct:3.0f}% {cpu_mhz:4.0f}M", cpu_pct))
    else:
        out.append(label_bar("CPU", f"{cpu_pct:3.0f}%", cpu_pct))
    if gpu_util is not None:
        if gpu_mhz is not None:
            gpu_val = f"{gpu_util:3.0f}% {gpu_mhz:4.0f}M"
        else:
            gpu_val = f"{gpu_util:3.0f}%"
        out.append(label_bar("GPU", gpu_val, gpu_util))
    else:
        out.append(text("GPU   n/a (nvidia-smi unavailable)"))
    gpu_t = f"{c_to_f(gpu_temp):3.0f}F" if gpu_temp is not None else "n/a"
    if zones:
        hot = max(zones, key=lambda z: z[1])
        out.append(text(f"Temp  GPU {gpu_t}   SoC {c_to_f(hot[1]):3.0f}F ({hot[0]})"))
        cells = [f"z{lbl.split('z')[-1]} {c_to_f(c):3.0f}" for lbl, c in zones]
        for i in range(0, len(cells), 4):
            out.append(text("  " + "  ".join(cells[i:i + 4]) + "F"))
    else:
        out.append(text(f"Temp  GPU {gpu_t}   SoC n/a"))
    return out
