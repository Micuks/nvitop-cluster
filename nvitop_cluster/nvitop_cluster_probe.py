#!/usr/bin/env python3
"""Single-node probe for nvitop-cluster: GPU stats + resolved process names.

Runs on each host (local or via SSH). share_l3 is mounted on all KML nodes.
Emits one JSON object on stdout.
"""
from __future__ import annotations

import json
import os
import pwd
import subprocess
import sys
from typing import Optional

TOOL = os.environ.get(
    "NVITOP_CLUSTER_HOME",
    os.path.dirname(os.path.abspath(__file__)),
)
# Allow `python path/to/nvitop_cluster_probe.py` (script mode) and package mode.
if TOOL not in sys.path:
    sys.path.insert(0, TOOL)


def _import_pid_map():
    """Load pid-map helper (package install or adjacent script layout)."""
    try:
        from nvitop_cluster import nvitop_pid_map as pm  # type: ignore

        return pm
    except Exception:
        pass
    try:
        import nvitop_pid_map as pm  # type: ignore

        return pm
    except Exception:
        return None


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            ["bash", "-lc", cmd],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:
        return ""


def _gpu_rows() -> list:
    q = (
        "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,"
        "memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits"
    )
    rows = []
    for line in _run(q).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            rows.append(
                {
                    "index": int(float(parts[0])),
                    "uuid": parts[1],
                    "name": parts[2],
                    "util": float(parts[3]),
                    "mem_used": float(parts[4]),
                    "mem_total": float(parts[5]),
                    "temp": float(parts[6]) if parts[6] else 0.0,
                    "power": float(parts[7]) if len(parts) > 7 and parts[7] else 0.0,
                }
            )
        except ValueError:
            continue
    return rows


def _username(pid: int) -> str:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("Uid:"):
                uid = int(line.split()[1])
                try:
                    return pwd.getpwuid(uid).pw_name
                except KeyError:
                    return str(uid)
    except Exception:
        pass
    return "?"


def _cmdline(pid: int) -> str:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
        s = raw.decode(errors="replace").strip()
        if s:
            return s
    except Exception:
        pass
    try:
        return open(f"/proc/{pid}/comm").read().strip() or "?"
    except Exception:
        return "?"


def _elapsed_from_proc_stat(stat: str, uptime: float, clock_ticks: int) -> float:
    """Return process age from Linux /proc data.

    ``comm`` (field 2) may contain spaces and closing parentheses, so split at
    the final ``)`` before indexing field 22 (starttime).
    """
    end_of_comm = stat.rfind(")")
    if end_of_comm < 0:
        raise ValueError("invalid /proc stat: missing comm terminator")
    fields_after_comm = stat[end_of_comm + 1 :].split()
    if len(fields_after_comm) <= 19:
        raise ValueError("invalid /proc stat: missing starttime")
    start_ticks = int(fields_after_comm[19])
    return max(0.0, float(uptime) - start_ticks / int(clock_ticks))


def _process_elapsed_seconds(pid: int) -> Optional[float]:
    """Read a process' monotonic running time from Linux procfs."""
    try:
        with open(f"/proc/{pid}/stat") as stat_file:
            stat = stat_file.read()
        with open("/proc/uptime") as uptime_file:
            uptime = float(uptime_file.read().split()[0])
        return _elapsed_from_proc_stat(stat, uptime, os.sysconf("SC_CLK_TCK"))
    except Exception:
        return None


def _proc_rows(uuid_to_index: dict) -> list:
    # Prefer pid-map resolution (host NVML pid -> container pid)
    pm = _import_pid_map()
    try:
        mapping = pm.build_host_to_local_map(force=True) if pm else {}
    except Exception:
        mapping = {}

    q = (
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory "
        "--format=csv,noheader,nounits"
    )
    out = []
    for line in _run(q).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
        try:
            host_pid = int(float(pid_s))
            mem_mib = float(mem_s)
        except ValueError:
            continue
        gpu_index = uuid_to_index.get(uuid, -1)
        local_pid = mapping.get(host_pid)
        if local_pid is None:
            # direct visibility or resolve helper
            if os.path.exists(f"/proc/{host_pid}"):
                local_pid = host_pid
            elif pm is not None:
                try:
                    local_pid = pm.resolve_local_pid(host_pid)
                except Exception:
                    local_pid = host_pid
            else:
                local_pid = host_pid

        if local_pid and os.path.exists(f"/proc/{local_pid}"):
            cmd = _cmdline(local_pid)
            user = _username(local_pid)
            elapsed = _process_elapsed_seconds(local_pid)
            shown_pid = local_pid
            ok = True
        else:
            cmd = "[No Such Process]"
            user = "?"
            elapsed = None
            shown_pid = host_pid
            ok = False

        out.append(
            {
                "gpu_index": gpu_index,
                "gpu_uuid": uuid,
                "host_pid": host_pid,
                "pid": shown_pid,
                "mem_mib": mem_mib,
                "user": user,
                "cmdline": cmd,
                "elapsed": elapsed,
                "resolved": ok,
            }
        )
    # stable: by gpu then mem desc
    out.sort(key=lambda r: (r["gpu_index"], -r["mem_mib"], r["pid"]))
    return out


def main() -> int:
    gpus = _gpu_rows()
    uuid_to_index = {g["uuid"]: g["index"] for g in gpus}
    procs = _proc_rows(uuid_to_index)
    payload = {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "gpus": gpus,
        "procs": procs,
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
