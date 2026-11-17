# SPDX-License-Identifier: Apache-2.0

"""Whole-node power estimate, the same shape sparkDash uses for Fleet power.

GPU board watts come from the announcer. CPU watts scale from 5.2 W at idle
to 65 W at full utilization. Memory, network, and base add 23 W. The total
is clamped to the 240 W DGX Spark envelope. A missing reading stays blank.
"""

from __future__ import annotations

from typing import Any

CPU_IDLE_W = 5.2
CPU_FULL_W = 65.0
BASE_W = 23.0
ENVELOPE_W = 240.0


def node_watts(gpu_w: float | None, cpu_pct: float | None) -> float | None:
    if gpu_w is None or cpu_pct is None:
        return None
    bounded = max(0.0, min(100.0, float(cpu_pct)))
    cpu = CPU_IDLE_W + (CPU_FULL_W - CPU_IDLE_W) * bounded / 100.0
    total = float(gpu_w) + cpu + BASE_W
    return max(0.0, min(ENVELOPE_W, total))


def fleet_power(hosts: list[dict[str, Any]], configured_ids: list[str]) -> dict[str, Any]:
    """Sum fresh machines that reported watts. Any gap marks the total partial."""
    by_id = {str(host.get("id")): host for host in hosts if host.get("id")}
    total = 0.0
    counted = 0
    partial = False
    machines: list[dict[str, Any]] = []
    for mid in configured_ids:
        host = by_id.get(str(mid))
        fresh = bool(host) and bool(host.get("fresh")) and not host.get("unreachable")
        estimate = node_watts(host.get("power_w") if host else None, host.get("cpu_pct") if host else None) if fresh else None
        if estimate is None:
            partial = True
            machines.append({"id": str(mid), "watts": None})
            continue
        total += estimate
        counted += 1
        machines.append({"id": str(mid), "watts": round(estimate, 1)})
    return {
        "estimate": True,
        "partial": partial,
        "watts": None if counted == 0 else round(total, 1),
        "machines": machines,
    }
