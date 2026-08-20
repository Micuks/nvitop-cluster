#!/usr/bin/env python3
"""Multi-node GPU snapshot from /etc/mpi/hostfile (KML MPI jobs).

Native nvitop talks to local NVML only. This tool SSHes each hostfile peer
(and queries local without SSH) and prints a unified table of all GPUs,
including resolved process names (via nvitop_cluster_probe + pid map).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import select
import shutil
import socket
import subprocess
import sys
import termios
import time
import tty
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

HOSTFILE_CANDIDATES = (
    "/etc/mpi/hostfile",
    "/etc/mpi/mpi-hostfile",
    os.environ.get("NVITOP_HOSTFILE", ""),
    os.environ.get("PBS_NODEFILE", ""),
)

# Prefer package dir / env; fall back to common KML share path.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL = os.environ.get("NVITOP_CLUSTER_HOME", _PKG_DIR)
SSH_CFG = os.environ.get("NVITOP_CLUSTER_SSH_CONFIG", "/etc/kml/ssh/ssh_config")
PROBE = os.path.join(TOOL, "nvitop_cluster_probe.py")


def _probe_shell_cmd() -> str:
    """Portable shell snippet: run probe via installed module or script path.

    On multi-node jobs, peers need either:
      - the same package installed (``pip install nvitop-cluster`` in the image), or
      - the probe script at ``NVITOP_CLUSTER_HOME`` / package path (e.g. shared FS).
    """
    # Quote path for remote shells; keep ASCII-safe.
    probe_q = PROBE.replace("'", "'\"'\"'")
    return (
        "if [ -x /opt/conda/envs/py312/bin/python3 ]; then "
        "PY=/opt/conda/envs/py312/bin/python3; "
        "elif command -v python3 >/dev/null 2>&1; then PY=python3; "
        "else PY=python; fi; "
        "if $PY -c 'import nvitop_cluster.nvitop_cluster_probe' >/dev/null 2>&1; then "
        "$PY -m nvitop_cluster.nvitop_cluster_probe; "
        f"elif [ -f '{probe_q}' ]; then $PY '{probe_q}'; "
        "else echo 'nvitop-cluster: probe not found "
        "(pip install nvitop-cluster on each node, or set NVITOP_CLUSTER_HOME)' >&2; "
        "exit 127; fi"
    )


PROBE_CMD = _probe_shell_cmd()


def _local_ips() -> set:
    ips = {"127.0.0.1", "localhost"}
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, stderr=subprocess.DEVNULL)
        ips.update(out.split())
    except Exception:
        pass
    try:
        ips.add(socket.gethostname())
        ips.add(socket.getfqdn())
    except Exception:
        pass
    return ips


def parse_hostfile(path: str) -> List[str]:
    hosts: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()[0].split(":")[0]
            if tok and tok not in hosts:
                hosts.append(tok)
    return hosts


def find_hostfile(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    for p in HOSTFILE_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return None


def is_local(host: str, local_ips: set) -> bool:
    h = host.strip().lower()
    if h in {x.lower() for x in local_ips}:
        return True
    short = socket.gethostname().split(".")[0].lower()
    if h == short or h.startswith(short + "."):
        return True
    return False


def _ssh_base() -> List[str]:
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]
    if os.path.isfile(SSH_CFG):
        cmd.extend(["-F", SSH_CFG])
    return cmd


def run_probe(host: str, local: bool) -> Tuple[str, Optional[dict], Optional[str]]:
    """Return (host, payload_dict|None, error|None)."""
    try:
        if local:
            out = subprocess.check_output(
                ["bash", "-lc", PROBE_CMD],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        else:
            ssh = _ssh_base() + [f"root@{host}", PROBE_CMD]
            out = subprocess.check_output(ssh, text=True, stderr=subprocess.STDOUT, timeout=40)
        # last non-empty line should be JSON
        line = ""
        for ln in out.splitlines():
            s = ln.strip()
            if s.startswith("{"):
                line = s
        if not line:
            return host, None, f"empty probe output: {out[:180]!r}"
        return host, json.loads(line), None
    except subprocess.CalledProcessError as e:
        return host, None, f"exit {e.returncode}: {(e.output or '')[:200]}"
    except Exception as e:
        return host, None, str(e)[:200]


def bar(pct: float, width: int = 10, *, partial: bool = False) -> str:
    """Fixed-width bar using only full cells (█/░).

    Partial block glyphs (▏–▉) are intentionally disabled: many terminal fonts
    give them non-cell widths, which leaves uneven gaps between used/free.
    """
    pct = max(0.0, min(100.0, float(pct)))
    width = max(1, int(width))
    filled = int(round(pct / 100.0 * width))
    filled = min(width, max(0, filled))
    return "█" * filled + "░" * (width - filled)


def color(s: str, code: str, enable: bool) -> str:
    if not enable:
        return s
    return f"[{code}m{s}[0m"


def level_code(pct: float) -> str:
    if pct >= 75:
        return "91"  # red
    if pct >= 40:
        return "93"  # yellow
    if pct >= 10:
        return "92"  # green
    return "90"  # dim


def util_color(u: float, enable: bool) -> str:
    return color(f"{u:5.1f}%", level_code(u), enable)


def bar_color(pct: float, width: int, enable: bool, used_code: Optional[str] = None) -> str:
    """Color used (█) by level; free (░) always dim — clean used|free boundary."""
    pct = max(0.0, min(100.0, float(pct)))
    width = max(1, int(width))
    filled = int(round(pct / 100.0 * width))
    filled = min(width, max(0, filled))
    used = "█" * filled
    free = "░" * (width - filled)
    if not enable:
        return used + free
    return color(used, used_code or level_code(pct), True) + color(free, "90", True)


def util_level_code(pct: float) -> str:
    """Training-oriented utilization colors: busy is healthy, idle is not."""
    if pct >= 70:
        return "92"  # green
    if pct >= 20:
        return "93"  # yellow
    return "91"  # red


def memory_level_code(pct: float) -> str:
    if pct >= 90:
        return "91"
    if pct >= 75:
        return "93"
    return "96"  # cyan for normal allocated memory


def short_gpu_name(name: str) -> str:
    n = (name or "").replace("NVIDIA ", "")
    for key in ("H100", "H800", "A100", "A800", "L40", "V100", "T4", "A10", "A30", "A40"):
        if key in n:
            return key
    return (n.split()[0] if n else "?")[:5]


def term_size(cols_fallback: int = 200, rows_fallback: int = 40) -> Tuple[int, int]:
    try:
        size = shutil.get_terminal_size((cols_fallback, rows_fallback))
        return max(60, size.columns), max(12, size.lines)
    except Exception:
        return cols_fallback, rows_fallback


def term_cols(fallback: int = 200) -> int:
    return term_size(fallback, 40)[0]


def wrap_cmd(cmd: str, width: int) -> List[str]:
    """Full cmdline, wrapped to width (width<=0 => single uncut line)."""
    cmd = " ".join((cmd or "?").split())
    if width <= 0 or len(cmd) <= width:
        return [cmd]
    out: List[str] = []
    while len(cmd) > width:
        window = cmd[:width]
        sp = window.rfind(" ")
        sl = window.rfind("/")
        if sp >= width // 4:
            br = sp
            out.append(cmd[:br])
            cmd = cmd[br + 1 :]
        elif sl >= width // 4:
            br = sl + 1
            out.append(cmd[:br])
            cmd = cmd[br:]
        else:
            out.append(cmd[:width])
            cmd = cmd[width:]
    if cmd:
        out.append(cmd)
    return out


_PATH_PREFIXES = (
    "/share_l3/wuqingliu/envs/flash_gdn/bin/",
    "/share_l3/wuqingliu/",
    "/share_l3/",
    "/opt/conda/envs/py312/bin/",
    "/opt/conda/envs/",
    "/usr/bin/",
    "/usr/local/bin/",
)


def normalize_cmd(cmd: str) -> str:
    """Collapse whitespace and compress common long path prefixes."""
    cmd = " ".join((cmd or "?").split())
    if cmd in ("?", "[No Such Process]"):
        return cmd
    for pref in _PATH_PREFIXES:
        cmd = cmd.replace(pref, "")
    # drop leading bare python -u for density (path already stripped)
    toks = cmd.split()
    if toks and os.path.basename(toks[0]).startswith("python"):
        i = 1
        while i < len(toks) and toks[i] in ("-u", "-O", "-B"):
            i += 1
        cmd = " ".join(toks[i:]) if i < len(toks) else cmd
    return cmd


def fit_cmd(cmd: str, width: int, align: str = "left") -> str:
    """Fit cmdline to width like nvitop: left=head (C-a), right=tail (C-e)."""
    cmd = normalize_cmd(cmd)
    if width <= 0 or len(cmd) <= width:
        return cmd
    if width <= 2:
        return cmd[:width]
    if align == "right":
        # show end of command (config, fname, etc.)
        return "…" + cmd[-(width - 1) :]
    # left / head: show beginning (script path, rank)
    return cmd[: width - 1] + "…"


def _format_running_time(seconds: object) -> str:
    """Format elapsed seconds with the same thresholds as nvitop 1.6.x."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError, OverflowError):
        return "N/A"
    days, day_seconds = divmod(total, 86400)
    if days >= 4:
        return f"{days + day_seconds / 86400:.1f} days"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def collect(hosts: Sequence[str]) -> List[Tuple[str, Optional[dict], Optional[str]]]:
    local_ips = _local_ips()

    def one(h: str):
        loc = is_local(h, local_ips)
        host, payload, err = run_probe(h, loc)
        label = f"{h} (local)" if loc else h
        return label, payload, err

    results: List[Tuple[str, Optional[dict], Optional[str]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(hosts)))) as ex:
        futs = [ex.submit(one, h) for h in hosts]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    order = {h: i for i, h in enumerate(hosts)}

    def key(item):
        label = item[0].replace(" (local)", "")
        return order.get(label, 999)

    results.sort(key=key)
    return results


def _auto_bar_w(cols: int) -> int:
    """Widen bars on large terminals (nvitop-like)."""
    # aim: two bars share ~half the width after fixed columns (~50)
    room = max(20, cols - 55)
    # each bar gets roughly half of room, clamp 12..28
    return max(12, min(28, room // 2))


def render(
    results: List[Tuple[str, Optional[dict], Optional[str]]],
    color_on: bool,
    show_procs: bool,
    cmd_width: int,
    cols: int,
    verbose: bool,
    cmd_align: str = "left",
    bar_w: int = 0,
) -> str:
    """nvitop-inspired layout: wide bars, 2 lines/GPU (metrics + full-width CMD).

    cmd_align: 'left' (Ctrl-A head) or 'right' (Ctrl-E tail).
    """
    if bar_w <= 0:
        bar_w = _auto_bar_w(cols)

    lines: List[str] = []
    total_gpus = 0
    total_procs = 0
    sum_util = 0.0
    sum_mem = 0.0
    for _, payload, err in results:
        if err or not payload:
            continue
        gs = payload.get("gpus") or []
        total_gpus += len(gs)
        total_procs += len(payload.get("procs") or [])
        for g in gs:
            sum_util += g.get("util") or 0.0
            if g.get("mem_total"):
                sum_mem += (g["mem_used"] / g["mem_total"]) * 100.0

    avg_u = sum_util / total_gpus if total_gpus else 0.0
    avg_m = sum_mem / total_gpus if total_gpus else 0.0
    align_tag = "HEAD" if cmd_align == "left" else "TAIL"

    width = min(cols, 240)
    sep = "─" * width

    lines.append(
        color(
            f" nvitop-cluster │ hosts={len(results)}  gpus={total_gpus}  "
            f"procs={total_procs}  avg-util={avg_u:.0f}%  avg-mem={avg_m:.0f}%  "
            f"CMD[{align_tag}]  {time.strftime('%H:%M:%S')} ",
            "1;37;44",
            color_on,
        )
        if color_on
        else (
            f" nvitop-cluster │ hosts={len(results)}  gpus={total_gpus}  "
            f"procs={total_procs}  avg-util={avg_u:.0f}%  avg-mem={avg_m:.0f}%  "
            f"CMD[{align_tag}]  {time.strftime('%H:%M:%S')}"
        )
    )

    # Metric line layout (nvitop-like dual bars):
    # HOST GPU TYPE | GPU-Util [bar] pct | MEM [bar] used/tot | TEMP PWR
    host_w = 16
    # CMD on second line: indent + almost full width
    cmd_indent = "    │ "
    # Reserve the user column too.  The previous calculation only subtracted
    # cmd_indent, so long commands wrapped by a few cells on real terminals.
    cmd_prefix_w = len(cmd_indent) + 8 + 2
    if cmd_width < 0:
        cmd_budget = 10**9
    elif cmd_width > 0:
        cmd_budget = cmd_width
    else:
        cmd_budget = max(20, cols - cmd_prefix_w)

    elapsed_values = [
        _format_running_time(proc.get("elapsed"))
        for _, payload, err in results
        if not err and payload
        for proc in payload.get("procs") or []
    ]
    time_w = max(4, max((len(value) for value in elapsed_values), default=4))

    # header for metric row
    u_label = f"{'GPU-Util':^{bar_w + 7}}"
    m_label = f"{'Memory-Usage':^{bar_w + 10}}"
    lines.append(
        f"{'HOST':<{host_w}} {'GPU':>3} {'TYPE':<5} "
        f"{u_label}  {m_label}  {'TEMP':>5} {'PWR':>6}  {'TIME':>{time_w}} {'PID':>7}"
    )
    lines.append(sep)

    prev_host_key = None
    for label, payload, err in results:
        short_host = label.replace(" (local)", "*")
        host_key = short_host.rstrip("*")
        is_local = short_host.endswith("*")
        if len(short_host) > host_w:
            short_host = short_host[: host_w - 1] + "…"

        # host section banner when host changes
        if host_key != prev_host_key:
            prev_host_key = host_key
            tag = " local " if is_local else " remote "
            title = f"─ {short_host}{tag}"
            lines.append(color(title + "─" * max(0, width - len(title)), "36", color_on))

        if err:
            lines.append(color(f"  ERROR  {err}", "91", color_on))
            continue
        if not payload:
            lines.append("  (no data)")
            continue
        gpus = payload.get("gpus") or []
        procs = payload.get("procs") or []
        if not gpus:
            lines.append("  (no GPUs)")
            continue

        by_gpu: Dict[int, List[dict]] = {}
        for p in procs:
            by_gpu.setdefault(int(p.get("gpu_index", -1)), []).append(p)

        for g in gpus:
            mem_pct = (g["mem_used"] / g["mem_total"] * 100.0) if g["mem_total"] else 0.0
            mem_s = f"{g['mem_used']/1024:.1f}/{g['mem_total']/1024:.0f}Gi"
            gi = int(g["index"])
            gtype = short_gpu_name(g.get("name", ""))
            ub = bar_color(g["util"], bar_w, color_on)
            mb = bar_color(mem_pct, bar_w, color_on)
            util_s = util_color(g["util"], color_on)
            mem_pct_s = color(f"{mem_pct:4.0f}%", level_code(mem_pct), color_on)
            temp_pct = min(100.0, max(0.0, (g["temp"] - 30) / 50.0 * 100.0))
            temp_s = color(f"{g['temp']:3.0f}C", level_code(temp_pct), color_on)
            pwr_s = f"{g['power']:4.0f}W"

            # Line 1: metrics (nvitop-style dual gauges)
            metric = (
                f"{short_host:<{host_w}} {gi:>3} {gtype:<5} "
                f"{ub} {util_s}  "
                f"{mb} {mem_pct_s} {mem_s:<9}  "
                f"{temp_s} {pwr_s}"
            )

            plist = by_gpu.get(gi) or []
            if not show_procs:
                lines.append(metric)
                continue

            if not plist:
                lines.append(metric + f"  {'-':>{time_w}} {'-':>7}")
                lines.append(
                    cmd_indent + color("(no compute process)", "90", color_on)
                )
                continue

            for j, p in enumerate(plist):
                raw_cmd = p.get("cmdline") or "?"
                pid = p["pid"]
                user = (p.get("user") or "?")[:8]
                elapsed_s = _format_running_time(p.get("elapsed"))
                elapsed_s = color(f"{elapsed_s:>{time_w}}", "90", color_on)
                if p.get("resolved"):
                    pid_s = color(f"{pid:>7}", "96", color_on)
                else:
                    pid_s = color(f"{pid:>7}", "91", color_on)

                if j == 0:
                    lines.append(metric + f"  {elapsed_s} {pid_s}")
                else:
                    # extra process on same GPU
                    lines.append(
                        f"{'':<{host_w}} {'↳':>3} {'':<5} "
                        f"{'':<{bar_w}} {'':>6}  "
                        f"{'':<{bar_w}} {'':>5} {'':<9}  "
                        f"{'':>4} {'':>5}  {elapsed_s} {pid_s}"
                    )

                if verbose:
                    for chunk in wrap_cmd(normalize_cmd(raw_cmd), max(20, cols - len(cmd_indent))):
                        piece = chunk if p.get("resolved") else color(chunk, "91", color_on)
                        lines.append(cmd_indent + piece)
                else:
                    cmd_s = fit_cmd(raw_cmd, cmd_budget, align=cmd_align)
                    if not p.get("resolved"):
                        cmd_s = color(cmd_s, "91", color_on)
                    lines.append(
                        f"{cmd_indent}{color(user, '90', color_on)}  {cmd_s}"
                    )

    lines.append(sep)
    lines.append(
        color(" GPU-Util / Memory-Usage bars  │  ", "90", color_on)
        + color(f"C-a head  C-e tail  [{align_tag}]", "96", color_on)
        + color("  │  g overview  j/k host  p procs  v verbose  q quit", "90", color_on)
    )
    return "\n".join(lines)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _pad_visible(text: str, width: int) -> str:
    if _visible_len(text) > width:
        # Narrow fallback: preserve cell correctness even if it means dropping
        # inline colors from this one clipped row.
        text = _fit_plain(_ANSI_RE.sub("", text), width)
    return text + " " * max(0, width - _visible_len(text))


def _fit_plain(text: str, width: int, align: str = "left") -> str:
    text = " ".join((text or "?").split())
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    if align == "right":
        return "…" + text[-(width - 1) :]
    return text[: width - 1] + "…"


def _by_gpu(payload: Optional[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for proc in (payload or {}).get("procs") or []:
        grouped.setdefault(int(proc.get("gpu_index", -1)), []).append(proc)
    return grouped


def _attention_reasons(gpu: dict, procs: Sequence[dict]) -> List[str]:
    """Return compact reasons why a GPU deserves operator attention."""
    reasons: List[str] = []
    total = float(gpu.get("mem_total") or 0.0)
    used = float(gpu.get("mem_used") or 0.0)
    mem_pct = used / total * 100.0 if total else 0.0
    util = float(gpu.get("util") or 0.0)
    temp = float(gpu.get("temp") or 0.0)
    if temp >= 80:
        reasons.append("hot")
    if mem_pct >= 95:
        reasons.append("mem")
    if procs and mem_pct >= 10 and util < 5:
        reasons.append("idle")
    if not procs and mem_pct >= 5:
        reasons.append("no-proc")
    if any(not p.get("resolved", False) for p in procs):
        reasons.append("pid")
    return reasons


def _cluster_stats(
    results: Sequence[Tuple[str, Optional[dict], Optional[str]]],
) -> Tuple[int, int, float, float]:
    total_gpus = 0
    total_procs = 0
    util_sum = 0.0
    mem_sum = 0.0
    for _, payload, err in results:
        if err or not payload:
            continue
        gpus = payload.get("gpus") or []
        total_gpus += len(gpus)
        total_procs += len(payload.get("procs") or [])
        for gpu in gpus:
            util_sum += float(gpu.get("util") or 0.0)
            total = float(gpu.get("mem_total") or 0.0)
            if total:
                mem_sum += float(gpu.get("mem_used") or 0.0) / total * 100.0
    avg_util = util_sum / total_gpus if total_gpus else 0.0
    avg_mem = mem_sum / total_gpus if total_gpus else 0.0
    return total_gpus, total_procs, avg_util, avg_mem


def _overview_entries(
    results: Sequence[Tuple[str, Optional[dict], Optional[str]]],
    attention_only: bool,
) -> List[Tuple[int, str, Optional[dict], Optional[str]]]:
    entries: List[Tuple[int, str, Optional[dict], Optional[str]]] = []
    for index, (label, payload, err) in enumerate(results):
        if not attention_only or err or not payload:
            entries.append((index, label, payload, err))
            continue
        grouped = _by_gpu(payload)
        flagged = [
            gpu
            for gpu in payload.get("gpus") or []
            if _attention_reasons(gpu, grouped.get(int(gpu.get("index", -1)), []))
        ]
        if flagged:
            filtered = dict(payload)
            filtered["gpus"] = flagged
            entries.append((index, label, filtered, err))
    return entries


def _overview_geometry(
    entries: Sequence[Tuple[int, str, Optional[dict], Optional[str]]],
    cols: int,
    rows: int,
) -> Tuple[int, int, int, int, int]:
    # Pick a grid by usable detail, not width alone.  Cards stay readable down
    # to 72 columns.  Wide cards can fold 8 GPU rows into two table columns,
    # allowing common 8-host jobs to use a compact 2-column x 4-row grid.
    max_columns = min(max(1, len(entries)), max(1, min(8, cols // 72)))
    max_gpus = max(
        (len((payload or {}).get("gpus") or []) for _, _, payload, _ in entries),
        default=1,
    )

    # If every host fits, choose the grid with the tallest useful chart.  At a
    # tie prefer wider cards.  Graphs need at least five lines to be legible.
    # Design references (principles only; no GPL TUI code is copied):
    # - nvitop auto/full/compact: https://github.com/XuehaiPan/nvitop#monitor-mode
    # - btop resizable history graphs: https://github.com/aristocratos/btop#configurability
    choices = []
    for candidate in range(1, max_columns + 1):
        host_rows = max(1, (len(entries) + candidate - 1) // candidate)
        # title + separator + footer are the only fixed rows.  Any division
        # remainder is distributed by render_overview across host rows.
        usable = rows - 3 - (host_rows - 1)
        per_block = usable // host_rows
        gap_width = 3 if candidate > 1 else 0
        block_width = (cols - gap_width * (candidate - 1)) // candidate
        gpu_columns = 2 if block_width >= 140 and max_gpus >= 4 else 1
        gpu_rows = (max_gpus + gpu_columns - 1) // gpu_columns
        compact_height = max(4, gpu_rows + 4)
        if per_block < compact_height:
            continue
        surplus = per_block - compact_height
        graph_lines = min(7, surplus) if surplus >= 5 else 0
        exact_grid = int(len(entries) % candidate == 0)
        graph_quality = -abs(graph_lines - 7) if graph_lines else -7
        choices.append(
            (
                graph_quality,
                exact_grid,
                block_width,
                -candidate,
                graph_lines,
                candidate,
                host_rows,
                per_block,
            )
        )

    if choices:
        _, _, _, _, history_lines, columns, block_rows, block_height = max(
            choices, key=lambda item: item[:4]
        )
        page_size = columns * block_rows
        return columns, block_height, block_rows, page_size, history_lines

    # Compact pagination fallback when even the full host set cannot fit.
    fallback = []
    for candidate in range(1, max_columns + 1):
        gap_width = 3 if candidate > 1 else 0
        block_width = (cols - gap_width * (candidate - 1)) // candidate
        gpu_columns = 2 if block_width >= 140 and max_gpus >= 4 else 1
        gpu_rows = (max_gpus + gpu_columns - 1) // gpu_columns
        compact_height = max(4, gpu_rows + 4)
        block_rows = max(1, (rows - 2) // (compact_height + 1))
        page_size = candidate * block_rows
        fallback.append((page_size, block_width, candidate, block_rows, compact_height))
    page_size, _, columns, block_rows, compact_height = max(
        fallback, key=lambda item: (item[0], item[1])
    )
    return columns, compact_height, block_rows, page_size, 0


def _history_key(label: str, gpu_index: int, metric: str) -> Tuple[str, int, str]:
    return label.replace(" (local)", ""), gpu_index, metric


def _update_history(
    history: dict,
    results: Sequence[Tuple[str, Optional[dict], Optional[str]]],
    maxlen: int = 512,
) -> None:
    """Append per-GPU and host-average samples to bounded history windows."""
    for label, payload, err in results:
        if err or not payload:
            continue
        gpus = payload.get("gpus") or []
        host_values = {"util": [], "mem": []}
        for gpu in gpus:
            gi = int(gpu.get("index", -1))
            total = float(gpu.get("mem_total") or 0.0)
            used = float(gpu.get("mem_used") or 0.0)
            values = {
                "util": float(gpu.get("util") or 0.0),
                "mem": used / total * 100.0 if total else 0.0,
            }
            for metric, value in values.items():
                host_values[metric].append(value)
                key = _history_key(label, gi, metric)
                if key not in history:
                    history[key] = deque(maxlen=maxlen)
                history[key].append(value)
        for metric, values in host_values.items():
            if not values:
                continue
            key = _history_key(label, -1, metric)
            if key not in history:
                history[key] = deque(maxlen=maxlen)
            history[key].append(sum(values) / len(values))


_BRAILLE_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)


def _set_braille_pixel(canvas: List[List[int]], x: int, y: int) -> None:
    if y < 0 or y >= len(canvas) * 4 or x < 0 or x >= len(canvas[0]) * 2:
        return
    canvas[y // 4][x // 2] |= _BRAILLE_BITS[y % 4][x % 2]


def _draw_braille_segment(
    canvas: List[List[int]], x0: int, y0: int, x1: int, y1: int
) -> None:
    """Draw a connected segment on a 2x4 subpixel-per-cell Braille canvas."""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _set_braille_pixel(canvas, x0, y0)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _history_chart(
    values: Sequence[float],
    width: int,
    height: int,
    title: str,
    latest: float,
    code: str,
    color_on: bool,
) -> List[str]:
    """Render a connected 0–100 trace on a high-resolution Braille canvas."""
    width = max(12, width)
    height = max(5, height)
    plot_height = height - 2  # title + time axis
    plot_width = max(4, width - 4)
    canvas = [[0 for _ in range(plot_width)] for _ in range(plot_height)]
    pixel_width = plot_width * 2
    pixel_height = plot_height * 4
    recent = [max(0.0, min(100.0, float(value))) for value in list(values)[-pixel_width:]]
    offset = pixel_width - len(recent)
    points = [
        (offset + index, int(round((100.0 - value) / 100.0 * (pixel_height - 1))))
        for index, value in enumerate(recent)
    ]
    if len(points) == 1:
        _set_braille_pixel(canvas, *points[0])
    else:
        for first, second in zip(points, points[1:]):
            _draw_braille_segment(canvas, first[0], first[1], second[0], second[1])

    lines = [color(_fit_plain(f"{title}  now {latest:3.0f}%", width), code, color_on)]
    middle = (plot_height - 1) // 2
    for row, cells in enumerate(canvas):
        if row == 0:
            label = "100"
        elif row == plot_height - 1:
            label = "  0"
        elif row == middle:
            label = " 50"
        else:
            label = "   "
        trace = color(
            "".join(chr(0x2800 + bits) if bits else " " for bits in cells),
            code,
            color_on,
        )
        lines.append(f"{label}┤{trace}")
    lines.append("   └" + "─" * max(1, plot_width - 1) + "▶")
    return [_pad_visible(line, width) for line in lines[:height]]


def _host_history_lines(
    history: Optional[dict],
    label: str,
    width: int,
    history_lines: int,
    avg_util: float,
    avg_mem: float,
    color_on: bool,
) -> List[str]:
    if history_lines < 5:
        return []
    util_values = (history or {}).get(_history_key(label, -1, "util"), (avg_util,))
    mem_values = (history or {}).get(_history_key(label, -1, "mem"), (avg_mem,))
    inner_width = max(24, width - 2)
    gap = "   "
    chart_width = (inner_width - len(gap)) // 2
    util_chart = _history_chart(
        util_values, chart_width, history_lines, "UTIL HISTORY", avg_util,
        util_level_code(avg_util), color_on,
    )
    mem_chart = _history_chart(
        mem_values, chart_width, history_lines, "VRAM HISTORY", avg_mem,
        memory_level_code(avg_mem), color_on,
    )
    return [
        _pad_visible(left, chart_width) + gap + _pad_visible(right, chart_width)
        for left, right in zip(util_chart, mem_chart)
    ]


def _card_row(content: str, width: int) -> str:
    return "│" + _pad_visible(content, max(1, width - 2)) + "│"


def _card_top(left: str, right: str, width: int, code: str, color_on: bool) -> str:
    inner_width = max(1, width - 2)
    left = f"─{left} "
    right = f" {right}─"
    if len(left) + len(right) > inner_width:
        inside = _fit_plain(left + right, inner_width)
    else:
        inside = left + "─" * (inner_width - len(left) - len(right)) + right
    return color("╭" + inside + "╮", code, color_on)


def _card_bottom(summary: str, width: int, color_on: bool) -> str:
    summary_width = max(1, width - 6)
    if _visible_len(summary) > summary_width:
        summary = _fit_plain(_ANSI_RE.sub("", summary), summary_width)
    fill = max(1, width - _visible_len(summary) - 5)
    return (
        color("╰─ ", "36", color_on)
        + summary
        + color(" " + "─" * fill + "╯", "36", color_on)
    )


def _host_detail_lines(gpus: Sequence[dict], limit: int, color_on: bool) -> List[str]:
    """Use small vertical leftovers for useful aggregates, never blank filler."""
    if not gpus or limit <= 0:
        return []
    utils = [float(gpu.get("util") or 0.0) for gpu in gpus]
    mems = []
    temps = [float(gpu.get("temp") or 0.0) for gpu in gpus]
    powers = [float(gpu.get("power") or 0.0) for gpu in gpus]
    memory_used = 0.0
    memory_total = 0.0
    for gpu in gpus:
        total = float(gpu.get("mem_total") or 0.0)
        used = float(gpu.get("mem_used") or 0.0)
        memory_used += used
        memory_total += total
        mems.append(used / total * 100.0 if total else 0.0)
    lines = [
        (
            " UTIL RANGE  min "
            + color(f"{min(utils):>3.0f}%", util_level_code(min(utils)), color_on)
            + f"   avg {sum(utils) / len(utils):>3.0f}%   max {max(utils):>3.0f}%"
        ),
        (
            " VRAM RANGE  min "
            + color(f"{min(mems):>3.0f}%", memory_level_code(min(mems)), color_on)
            + f"   avg {sum(mems) / len(mems):>3.0f}%   max {max(mems):>3.0f}%"
        ),
        f" THERMAL  {min(temps):.0f}–{max(temps):.0f}C   POWER  {sum(powers) / 1000.0:.1f}kW total",
        (
            f" VRAM TOTAL  {memory_used / 1024:.1f}/{memory_total / 1024:.0f}G"
            f"   GPU BUSY  {sum(value >= 70 for value in utils)}/{len(utils)}"
        ),
    ]
    return lines[:limit]


def _command_summary(payload: Optional[dict], width: int, align: str, color_on: bool) -> str:
    procs = (payload or {}).get("procs") or []
    if not procs:
        return color("CMD", "1;93", color_on) + color("  — no compute process", "90", color_on)
    groups: Dict[str, List[dict]] = {}
    users = set()
    for proc in procs:
        cmd = normalize_cmd(proc.get("cmdline") or "?")
        groups.setdefault(cmd, []).append(proc)
        users.add((proc.get("user") or "?")[:8])
    lead_cmd, lead_procs = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[0]
    user = next(iter(users)) if len(users) == 1 else "mixed"
    group_note = "" if len(groups) == 1 else f"/{len(groups)}cmd"
    elapsed_values = [proc.get("elapsed") for proc in lead_procs if proc.get("elapsed") is not None]
    elapsed = _format_running_time(max(elapsed_values)) if elapsed_values else "N/A"
    badge = f"CMD ×{len(procs)}{group_note}"
    runtime = f"TIME {elapsed}"
    prefix = f"{badge}  {runtime}  {user} "
    command = _fit_plain(lead_cmd, max(1, width - len(prefix)), align)
    return (
        color(badge, "1;93", color_on)
        + "  "
        + color(runtime, "1;96", color_on)
        + "  "
        + color(user, "90", color_on)
        + " "
        + color(command, "97", color_on)
    )


def _gpu_table_cell(
    gpu: dict,
    procs: Sequence[dict],
    width: int,
    gauge_w: int,
    color_on: bool,
) -> str:
    gi = int(gpu.get("index", -1))
    total = float(gpu.get("mem_total") or 0.0)
    used = float(gpu.get("mem_used") or 0.0)
    mem_pct = used / total * 100.0 if total else 0.0
    util = float(gpu.get("util") or 0.0)
    temp = float(gpu.get("temp") or 0.0)
    power = float(gpu.get("power") or 0.0)
    reason = _attention_reasons(gpu, procs)
    reason_s = " " + color(f"⚠ {'+'.join(reason)}", "91", color_on) if reason else ""
    util_code = util_level_code(util)
    mem_code = memory_level_code(mem_pct)
    row = (
        f" {gi:>2} {short_gpu_name(gpu.get('name', '')):<5}  "
        + bar_color(util, gauge_w, color_on, util_code)
        + " "
        + color(f"{util:>3.0f}%", util_code, color_on)
        + "  "
        + bar_color(mem_pct, gauge_w, color_on, mem_code)
        + " "
        + color(f"{mem_pct:>3.0f}%", mem_code, color_on)
        + f"  {used/1024:>4.1f}/{total/1024:.0f}G "
        + color(
            f"{temp:>3.0f}C",
            level_code(min(100.0, max(0.0, (temp - 30) * 2))),
            color_on,
        )
        + f" {power:>4.0f}W{reason_s}"
    )
    return _pad_visible(row, width)


def _overview_block(
    entry: Tuple[int, str, Optional[dict], Optional[str]],
    width: int,
    height: int,
    color_on: bool,
    show_procs: bool,
    cmd_align: str,
    selected_host: int,
    history_lines: int,
    history: Optional[dict],
) -> List[str]:
    index, label, payload, err = entry
    local = label.endswith(" (local)")
    host = label.replace(" (local)", "")
    selected = index == selected_host
    gpus = (payload or {}).get("gpus") or []
    grouped = _by_gpu(payload)
    if gpus:
        avg_util = sum(float(g.get("util") or 0.0) for g in gpus) / len(gpus)
        avg_mem = sum(
            float(g.get("mem_used") or 0.0) / float(g.get("mem_total") or 1.0) * 100.0
            for g in gpus
        ) / len(gpus)
    else:
        avg_util = 0.0
        avg_mem = 0.0
    tag = "local" if local else "remote"
    header_code = "1;96" if selected else "36"
    gpu_types = sorted({short_gpu_name(gpu.get("name", "")) for gpu in gpus})
    gpu_tag = f"{len(gpus)}×{gpu_types[0]}" if len(gpu_types) == 1 else f"{len(gpus)} GPUs"
    host_title = f"▸ {host} {tag}" if selected else f" {host} {tag}"
    lines = [_card_top(host_title, gpu_tag, width, header_code, color_on)]

    if err:
        lines.append(_card_row(color(f" ERROR  {err}", "91", color_on), width))
    elif not payload:
        lines.append(_card_row(color(" (no data)", "91", color_on), width))
    elif not gpus:
        lines.append(_card_row(" (no GPUs)", width))
    else:
        attention_count = sum(
            bool(_attention_reasons(gpu, grouped.get(int(gpu.get("index", -1)), [])))
            for gpu in gpus
        )
        status = (
            color(f"ATTN {attention_count}", "91", color_on)
            if attention_count
            else color("OK", "92", color_on)
        )
        summary = (
            " AVG  UTIL "
            + color(f"{avg_util:>3.0f}%", util_level_code(avg_util), color_on)
            + "   VRAM "
            + color(f"{avg_mem:>3.0f}%", memory_level_code(avg_mem), color_on)
            + f"   PROCS {len(payload.get('procs') or []):>2}   "
            + status
        )
        lines.append(_card_row(summary, width))
        gpu_columns = 2 if width >= 140 and len(gpus) >= 4 else 1
        gpu_rows = (len(gpus) + gpu_columns - 1) // gpu_columns
        detail_lines = max(0, height - (gpu_rows + 4) - history_lines)
        lines.extend(
            _card_row(detail, width)
            for detail in _host_detail_lines(gpus, min(4, detail_lines), color_on)
        )
        lines.extend(
            _card_row(chart_line, width)
            for chart_line in _host_history_lines(
                history, label, width, history_lines, avg_util, avg_mem, color_on
            )
        )

        table_gap = " │ " if gpu_columns > 1 else ""
        table_inner_width = width - 2 - len(table_gap) * (gpu_columns - 1)
        cell_width, cell_remainder = divmod(table_inner_width, gpu_columns)
        cell_widths = [
            cell_width + (1 if index < cell_remainder else 0)
            for index in range(gpu_columns)
        ]
        gauge_w = max(6, min(12, (min(cell_widths) - 62) // 2))
        metric_width = gauge_w + 5
        header_cell = (
            f"  # TYPE   {'UTIL':<{metric_width}}  {'VRAM':<{metric_width}}  "
            f"{'USED/TOTAL':>10} {'TEMP':>4} {'PWR':>5}"
        )
        headers = [
            _pad_visible(color(header_cell, "2", color_on), cell_widths[index])
            for index in range(gpu_columns)
        ]
        lines.append(_card_row(table_gap.join(headers), width))
        for row_index in range(gpu_rows):
            cells = []
            for column in range(gpu_columns):
                gpu_index = row_index + column * gpu_rows
                if gpu_index >= len(gpus):
                    cells.append(" " * cell_widths[column])
                    continue
                gpu = gpus[gpu_index]
                gi = int(gpu.get("index", -1))
                cells.append(
                    _gpu_table_cell(
                        gpu,
                        grouped.get(gi) or [],
                        cell_widths[column],
                        gauge_w,
                        color_on,
                    )
                )
            lines.append(_card_row(table_gap.join(cells), width))

    command = (
        _command_summary(payload, width - 6, cmd_align, color_on)
        if show_procs
        else color("CMD hidden (p to show)", "90", color_on)
    )
    while len(lines) < height - 1:
        lines.append(_card_row("", width))
    lines.append(_card_bottom(command, width, color_on))
    return [_pad_visible(line, width) for line in lines[:height]]


def render_overview(
    results: List[Tuple[str, Optional[dict], Optional[str]]],
    color_on: bool,
    show_procs: bool,
    cols: int,
    rows: int,
    cmd_align: str = "left",
    selected_host: int = 0,
    attention_only: bool = False,
    history: Optional[dict] = None,
) -> str:
    """Render an all-host dashboard that adapts to terminal width and height."""
    width = cols
    entries = _overview_entries(results, attention_only)
    total_gpus, total_procs, avg_util, avg_mem = _cluster_stats(results)
    attention_tag = "  ATTENTION" if attention_only else ""
    if entries:
        columns, block_height, _block_rows, page_size, history_lines = _overview_geometry(
            entries, cols, rows
        )
    else:
        columns, block_height, _block_rows, page_size, history_lines = 1, 3, 1, 1, 0
    density = "COMPACT" if history_lines == 0 else ("RICH" if history_lines <= 7 else "FULL")
    title = (
        f" nvitop-cluster OVERVIEW/{density}{attention_tag} │ hosts={len(results)}  "
        f"gpus={total_gpus}  "
        f"procs={total_procs}  avg-util={avg_util:.0f}%  avg-mem={avg_mem:.0f}%  "
        f"{time.strftime('%H:%M:%S')}"
    )
    lines = [color(_fit_plain(title, width), "1;37;44", color_on)]
    if not entries:
        lines.extend(
            [
                "",
                color(" No GPUs currently require attention.", "92", color_on),
                "─" * width,
                color(
                    _fit_plain(
                        " x all GPUs  │  g overview  j/k host  Enter detail  p procs  q quit",
                        width,
                    ),
                    "90",
                    color_on,
                ),
            ]
        )
        return "\n".join(lines)

    selected_pos = next((i for i, item in enumerate(entries) if item[0] == selected_host), 0)
    page = selected_pos // page_size
    page_count = max(1, (len(entries) + page_size - 1) // page_size)
    visible = entries[page * page_size : (page + 1) * page_size]
    visible_rows = max(1, (len(visible) + columns - 1) // columns)
    available_card_height = rows - 3 - (visible_rows - 1)
    row_height, row_remainder = divmod(available_card_height, visible_rows)
    row_height = max(block_height, row_height)

    card_gap = 3
    block_width = width if columns == 1 else (width - card_gap * (columns - 1)) // columns
    for grid_row, offset in enumerate(range(0, len(visible), columns)):
        current_height = row_height + (1 if grid_row < row_remainder else 0)
        row_entries = visible[offset : offset + columns]
        row_columns = len(row_entries)
        blocks = [
            _overview_block(
                entry,
                block_width,
                current_height,
                color_on,
                show_procs,
                cmd_align,
                selected_host,
                history_lines,
                history,
            )
            for entry in row_entries
        ]
        if row_columns == columns and row_columns > 1:
            gap_total = width - block_width * row_columns
            gap_width, gap_remainder = divmod(gap_total, row_columns - 1)
            gaps = [gap_width + (1 if index < gap_remainder else 0) for index in range(row_columns - 1)]
            left_margin = right_margin = 0
        else:
            gaps = [card_gap] * max(0, row_columns - 1)
            used = block_width * row_columns + sum(gaps)
            left_margin = max(0, (width - used) // 2)
            right_margin = max(0, width - used - left_margin)
        for row in range(current_height):
            pieces = [" " * left_margin, blocks[0][row]]
            for index, block in enumerate(blocks[1:]):
                pieces.extend((" " * gaps[index], block[row]))
            pieces.append(" " * right_margin)
            lines.append("".join(pieces))
        if offset + columns < len(visible):
            lines.append("")

    lines.append("─" * width)
    page_tag = f"page {page + 1}/{page_count}" if page_count > 1 else "all hosts"
    lines.append(
        color(
            _fit_plain(
                f" {page_tag}  density={density.lower()}  │  g overview  j/k host  Enter detail  p procs  "
                "x attention  [/] page  C-a/C-e CMD  q quit",
                width,
            ),
            "90",
            color_on,
        )
    )
    return "\n".join(lines)


def _detail_line_estimate(
    results: Sequence[Tuple[str, Optional[dict], Optional[str]]], show_procs: bool
) -> int:
    lines = 4  # title, table header, separators/footer
    for _, payload, err in results:
        lines += 1  # host banner
        if err or not payload:
            lines += 1
            continue
        grouped = _by_gpu(payload)
        for gpu in payload.get("gpus") or []:
            lines += 1
            if show_procs:
                lines += max(1, len(grouped.get(int(gpu.get("index", -1)), [])))
    return lines


def render_dashboard(
    results: List[Tuple[str, Optional[dict], Optional[str]]],
    color_on: bool,
    show_procs: bool,
    cmd_width: int,
    cols: int,
    rows: int,
    verbose: bool,
    cmd_align: str,
    layout: str,
    selected_host: int,
    attention_only: bool,
    history: Optional[dict] = None,
) -> str:
    effective = layout
    if effective == "auto":
        effective = "overview" if _detail_line_estimate(results, show_procs) > rows else "detail"
    if effective == "overview":
        return render_overview(
            results,
            color_on=color_on,
            show_procs=show_procs,
            cols=cols,
            rows=rows,
            cmd_align=cmd_align,
            selected_host=selected_host,
            attention_only=attention_only,
            history=history,
        )
    detail_results = results
    if layout == "detail" and results:
        detail_results = [results[selected_host % len(results)]]
    return render(
        detail_results,
        color_on=color_on,
        show_procs=show_procs,
        cmd_width=cmd_width,
        cols=cols,
        verbose=verbose,
        cmd_align=cmd_align,
    )


class _CbreakTTY:
    """cbreak stdin for single-key reads (Ctrl-A / Ctrl-E)."""

    def __init__(self):
        self.fd = None
        self.old = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None and self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        return False


def _read_keys(timeout: float, state: dict) -> bool:
    """Wait up to timeout; handle keys. Returns True if need immediate redraw.

    State includes layout, selected_host, show_procs, and attention_only.
    """
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return False
    end = time.monotonic() + max(0.0, timeout)
    redraw = False
    while True:
        remain = end - time.monotonic()
        if remain <= 0:
            break
        r, _, _ = select.select([sys.stdin], [], [], min(0.2, remain))
        if not r:
            continue
        ch = os.read(sys.stdin.fileno(), 1)
        if not ch:
            break
        # bytes
        if ch == b"\x01":  # Ctrl-A — show command head (like nvitop)
            state["cmd_align"] = "left"
            redraw = True
            break
        if ch == b"\x05":  # Ctrl-E — show command tail
            state["cmd_align"] = "right"
            redraw = True
            break
        if ch in (b"q", b"Q"):
            state["quit"] = True
            break
        if ch == b"\x03":  # Ctrl-C
            state["quit"] = True
            break
        if ch in (b"v", b"V"):
            state["verbose"] = not state.get("verbose", False)
            redraw = True
            break
        if ch in (b"g", b"G", b"\x1b", b"\x7f"):
            state["layout"] = "overview"
            redraw = True
            break
        if ch in (b"d", b"D", b"\r", b"\n"):
            state["layout"] = "detail"
            redraw = True
            break
        if ch in (b"j", b"J"):
            state["host_move"] = 1
            redraw = True
            break
        if ch in (b"k", b"K"):
            state["host_move"] = -1
            redraw = True
            break
        if ch in (b"p", b"P"):
            state["show_procs"] = not state.get("show_procs", True)
            redraw = True
            break
        if ch in (b"x", b"X"):
            state["attention_only"] = not state.get("attention_only", False)
            state["layout"] = "overview"
            redraw = True
            break
        if ch == b"]":
            state["page_move"] = 1
            redraw = True
            break
        if ch == b"[":
            state["page_move"] = -1
            redraw = True
            break
        if ch == b"a":  # also accept plain a/e without ctrl for convenience
            state["cmd_align"] = "left"
            redraw = True
            break
        if ch == b"e":
            state["cmd_align"] = "right"
            redraw = True
            break
    return redraw



def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-node GPU view from MPI hostfile")
    ap.add_argument("-f", "--hostfile", default=None, help="hostfile path (default: /etc/mpi/hostfile)")
    ap.add_argument(
        "-w",
        "--watch",
        type=float,
        nargs="?",
        const=2.0,
        default=2.0,
        help="refresh interval seconds (default: 2). Use -1/--once for single shot.",
    )
    ap.add_argument(
        "-1",
        "--once",
        action="store_true",
        help="print once and exit (disable default watch)",
    )
    ap.add_argument(
        "-p",
        "--procs",
        action="store_true",
        default=True,
        help="show compute processes with resolved names (default: on)",
    )
    ap.add_argument(
        "--no-procs",
        action="store_true",
        help="hide process rows",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="multi-line full cmdline (default: one dense line per GPU)",
    )
    ap.add_argument(
        "--cmd-width",
        type=int,
        default=0,
        help="cmdline width: 0=fit terminal (default), -1=no truncate, >0=fixed",
    )
    ap.add_argument(
        "--cmd-align",
        choices=("left", "right", "head", "tail"),
        default="left",
        help="initial CMD truncate side: left/head=Ctrl-A, right/tail=Ctrl-E (default left)",
    )
    ap.add_argument(
        "--layout",
        choices=("auto", "overview", "detail"),
        default="auto",
        help="layout: auto fits all hosts to terminal height (default)",
    )
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--ascii", action="store_true", help="ASCII bars")
    args = ap.parse_args(argv)

    show_procs = args.procs and not args.no_procs
    watch = None if args.once else float(args.watch)
    align0 = args.cmd_align
    if align0 == "head":
        align0 = "left"
    if align0 == "tail":
        align0 = "right"

    hf = find_hostfile(args.hostfile)
    if not hf:
        print(
            "No hostfile found. Pass -f /path/to/hostfile or create /etc/mpi/hostfile",
            file=sys.stderr,
        )
        return 2
    hosts = parse_hostfile(hf)
    if not hosts:
        print(f"Empty hostfile: {hf}", file=sys.stderr)
        return 2

    probe_ok = os.path.isfile(PROBE)
    if not probe_ok:
        try:
            import importlib.util

            probe_ok = importlib.util.find_spec("nvitop_cluster.nvitop_cluster_probe") is not None
        except Exception:
            probe_ok = False
    if not probe_ok:
        print(
            f"Missing probe (script {PROBE} or package nvitop_cluster). "
            "Install with: pip install nvitop-cluster",
            file=sys.stderr,
        )
        return 2

    color_on = (not args.no_color) and sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"
    if args.ascii:
        global bar

        def bar(pct, width=10, partial=True):  # noqa: A001
            pct = max(0.0, min(100.0, pct))
            filled = int(round(pct / 100.0 * width))
            return "#" * filled + "-" * (width - filled)

    state = {
        "cmd_align": align0,
        "verbose": bool(args.verbose),
        "layout": args.layout,
        "selected_host": 0,
        "show_procs": show_procs,
        "attention_only": False,
        "host_move": 0,
        "page_move": 0,
        "quit": False,
    }
    # cache last probe results so C-a/C-e redraw is instant without re-SSH
    cache: dict = {"res": None, "history": {}}

    def once(force_collect: bool = True):
        if force_collect or cache["res"] is None:
            cache["res"] = collect(hosts)
            _update_history(cache["history"], cache["res"])
        results = cache["res"] or []
        cols, rows = term_size()

        # Apply navigation against the currently visible host set so attention
        # mode skips healthy hosts and page keys land on an actual card.
        entries = _overview_entries(results, state["attention_only"])
        visible_indices = [entry[0] for entry in entries]
        if not visible_indices:
            visible_indices = list(range(len(results)))
        if visible_indices:
            current = state["selected_host"]
            try:
                pos = visible_indices.index(current)
            except ValueError:
                pos = 0
            move = int(state.pop("host_move", 0) or 0)
            page_move = int(state.pop("page_move", 0) or 0)
            if page_move:
                _, _, _, page_size, _ = _overview_geometry(entries, cols, rows)
                move += page_move * page_size
            state["selected_host"] = visible_indices[(pos + move) % len(visible_indices)]
        else:
            state["host_move"] = 0
            state["page_move"] = 0

        text = render_dashboard(
            cache["res"],
            color_on=color_on,
            show_procs=state["show_procs"],
            cmd_width=args.cmd_width,
            cols=cols,
            rows=rows,
            verbose=state["verbose"],
            cmd_align=state["cmd_align"],
            layout=state["layout"],
            selected_host=state["selected_host"],
            attention_only=state["attention_only"],
            history=cache["history"],
        )
        if watch is not None:
            sys.stdout.write("\033[H\033[J")
            # The dashboard already occupies exactly ``rows`` lines.  A final
            # print newline would advance to row + 1, scroll away the title,
            # and leave an apparent blank line at the bottom.
            sys.stdout.write(text)
        else:
            print(text)
        sys.stdout.flush()

    if watch is None:
        once(True)
        return 0

    try:
        with _CbreakTTY():
            while not state["quit"]:
                once(force_collect=True)
                # poll keys for `watch` seconds; C-a/C-e redraw without re-probe
                deadline = time.monotonic() + watch
                while not state["quit"] and time.monotonic() < deadline:
                    remain = deadline - time.monotonic()
                    if _read_keys(remain, state):
                        if state["quit"]:
                            break
                        once(force_collect=False)  # instant align flip
                        # after key redraw, continue waiting rest of interval
    except KeyboardInterrupt:
        pass
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
