"""Map NVML host PIDs to container-local PIDs for nvitop inside K8s/KML pods."""
from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

_CACHE: Dict[int, int] = {}
_CACHE_TS = 0.0
_CACHE_TTL = 2.0
_PATCHED = False

_RANK_ENV = (
    "LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "SLURM_LOCALID",
    "RANK",
    "PMI_RANK",
)
_WORKER_HINT = re.compile(
    r"(python|torch|deepspeed|launch\.py|train|megatron|accelerate|torchrun)",
    re.I,
)


def _read_cmdline(pid: int) -> str:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")
        return raw.decode(errors="replace").strip()
    except Exception:
        return ""


def _read_comm(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except Exception:
        return ""


def _read_environ(pid: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
    except Exception:
        return out
    for item in raw:
        if not item or b"=" not in item:
            continue
        k, _, v = item.partition(b"=")
        try:
            out[k.decode()] = v.decode(errors="replace")
        except Exception:
            pass
    return out


def _rss(pid: int) -> int:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def _nvidia_devices(pid: int) -> Set[int]:
    devs: Set[int] = set()
    fd = f"/proc/{pid}/fd"
    try:
        ents = os.listdir(fd)
    except Exception:
        return devs
    for e in ents:
        try:
            t = os.readlink(f"{fd}/{e}")
        except OSError:
            continue
        if t.startswith("/dev/nvidia") and t[-1].isdigit():
            try:
                devs.add(int(t.rsplit("nvidia", 1)[1]))
            except ValueError:
                pass
    return devs


def _local_rank(cmdline: str, env: Dict[str, str]) -> Optional[int]:
    for k in _RANK_ENV:
        if k in env:
            try:
                return int(env[k])
            except ValueError:
                pass
    m = re.search(r"(?:--)?local[_-]rank(?:=|\s+)(\d+)", cmdline, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\bRANK=(\d+)\b", cmdline)
    if m:
        return int(m.group(1))
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and "," not in cvd and cvd.strip().isdigit():
        return int(cvd.strip())
    return None


def _is_candidate(pid: int, cmdline: str, comm: str, devs: Set[int]) -> bool:
    if not devs or pid <= 1:
        return False
    text = f"{comm} {cmdline}"
    if _WORKER_HINT.search(text):
        return True
    if len(devs) == 1:
        return True
    if _rss(pid) > 256 * 1024:
        return True
    return False


def _nvml_host_procs() -> List[Tuple[int, int, int]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        try:
            from nvitop.api import libnvml as pynvml  # type: ignore
        except Exception:
            return []
    try:
        pynvml.nvmlInit()
    except Exception:
        return []
    out: List[Tuple[int, int, int]] = []
    try:
        n = pynvml.nvmlDeviceGetCount()
    except Exception:
        return []
    for i in range(n):
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            continue
        procs = []
        for getter in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            fn = getattr(pynvml, getter, None)
            if fn is None:
                continue
            try:
                procs = fn(h) or []
                break
            except Exception:
                continue
        for p in procs:
            pid = int(getattr(p, "pid", 0) or 0)
            mem = int(getattr(p, "usedGpuMemory", 0) or 0)
            if pid > 0:
                out.append((pid, i, mem))
    return out


def _local_candidates() -> List[dict]:
    rows = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        devs = _nvidia_devices(pid)
        if not devs:
            continue
        cmdline = _read_cmdline(pid)
        comm = _read_comm(pid)
        if not _is_candidate(pid, cmdline, comm, devs):
            continue
        env = _read_environ(pid)
        rows.append(
            {
                "pid": pid,
                "comm": comm,
                "cmdline": cmdline,
                "devs": devs,
                "rss": _rss(pid),
                "rank": _local_rank(cmdline, env),
            }
        )
    return rows


def build_host_to_local_map(force: bool = False) -> Dict[int, int]:
    global _CACHE, _CACHE_TS
    now = time.time()
    if not force and _CACHE and (now - _CACHE_TS) < _CACHE_TTL:
        return _CACHE

    mapping: Dict[int, int] = {}
    host_procs = _nvml_host_procs()
    if not host_procs:
        _CACHE, _CACHE_TS = mapping, now
        return mapping

    for host_pid, _, _ in host_procs:
        if os.path.exists(f"/proc/{host_pid}"):
            mapping[host_pid] = host_pid

    missing = [(hp, gpu, mem) for hp, gpu, mem in host_procs if hp not in mapping]
    if not missing:
        _CACHE, _CACHE_TS = mapping, now
        return mapping

    locals_ = _local_candidates()
    used_local: Set[int] = set(mapping.values())

    by_gpu: Dict[int, List[Tuple[int, int]]] = {}
    for hp, gpu, mem in missing:
        by_gpu.setdefault(gpu, []).append((hp, mem))

    for gpu, items in by_gpu.items():
        items_sorted = sorted(items, key=lambda x: (-x[1], x[0]))
        rank_matches = [
            r for r in locals_
            if r["pid"] not in used_local and r["rank"] is not None and r["rank"] == gpu
        ]
        single = [
            r for r in locals_
            if r["pid"] not in used_local and r["devs"] == {gpu}
        ]
        pool = rank_matches or single
        pool_sorted = sorted(pool, key=lambda r: (-r["rss"], r["pid"]))
        for (hp, _), loc in zip(items_sorted, pool_sorted):
            mapping[hp] = loc["pid"]
            used_local.add(loc["pid"])

    missing2 = [(hp, gpu, mem) for hp, gpu, mem in missing if hp not in mapping]
    if missing2:
        remain_local = sorted(
            [r for r in locals_ if r["pid"] not in used_local],
            key=lambda r: (-r["rss"], r["pid"]),
        )
        remain_local_workers = [
            r for r in remain_local
            if _WORKER_HINT.search(r["cmdline"] or r["comm"] or "")
        ] or remain_local
        host_sorted = sorted(missing2, key=lambda x: (-x[2], x[0]))
        for (hp, _, _), loc in zip(host_sorted, remain_local_workers):
            if loc["pid"] in used_local:
                continue
            mapping[hp] = loc["pid"]
            used_local.add(loc["pid"])

    _CACHE, _CACHE_TS = mapping, now
    return mapping


def resolve_local_pid(pid: int) -> int:
    if pid is None:
        return pid
    try:
        pid = int(pid)
    except Exception:
        return pid
    if os.path.exists(f"/proc/{pid}"):
        return pid
    return build_host_to_local_map().get(pid, pid)


def patch_nvitop() -> None:
    """Monkey-patch nvitop so NVML host PIDs resolve to container PIDs."""
    global _PATCHED
    if _PATCHED:
        return

    # Patch GpuProcess.__new__
    try:
        from nvitop.api.process import GpuProcess
    except Exception:
        return

    if getattr(GpuProcess, "_kml_pid_patched", False):
        _PATCHED = True
        return

    _orig_new = GpuProcess.__new__

    def _new(cls, pid=None, device=None, *args, **kwargs):
        if pid is not None:
            try:
                pid = resolve_local_pid(int(pid))
            except Exception:
                pass
        # Support both positional and keyword forms used by nvitop
        if args or kwargs:
            return _orig_new(cls, pid, device, *args, **kwargs)
        return _orig_new(cls, pid, device, **kwargs)

    GpuProcess.__new__ = staticmethod(_new)  # type: ignore[method-assign]
    # staticmethod may break binding; use raw function (Python treats __new__ specially)
    GpuProcess.__new__ = _new  # type: ignore[method-assign]
    GpuProcess._kml_pid_patched = True

    # Also patch Device.processes construction path via libnvml wrapper if needed:
    # When HostProcess is created inside GpuProcess.__new__, pid is already remapped.

    # Patch HostProcess as well for direct use
    try:
        from nvitop.api.process import HostProcess
        _hp_new = HostProcess.__new__

        def _hp_new_wrap(cls, pid=None):
            if pid is not None:
                try:
                    pid = resolve_local_pid(int(pid))
                except Exception:
                    pass
            return _hp_new(cls, pid)

        HostProcess.__new__ = _hp_new_wrap  # type: ignore[method-assign]
        HostProcess._kml_pid_patched = True
    except Exception:
        pass

    _PATCHED = True
