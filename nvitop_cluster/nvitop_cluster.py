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
PROBE_CMD = (
    "if [ -x /opt/conda/envs/py312/bin/python3 ]; then "
    "PY=/opt/conda/envs/py312/bin/python3; "
    "elif command -v python3 >/dev/null 2>&1; then PY=python3; "
    "else PY=python; fi; "
    f"$PY {PROBE}"
)


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


def bar_color(pct: float, width: int, enable: bool) -> str:
    """Color used (█) by level; free (░) always dim — clean used|free boundary."""
    pct = max(0.0, min(100.0, float(pct)))
    width = max(1, int(width))
    filled = int(round(pct / 100.0 * width))
    filled = min(width, max(0, filled))
    used = "█" * filled
    free = "░" * (width - filled)
    if not enable:
        return used + free
    return color(used, level_code(pct), True) + color(free, "90", True)


def short_gpu_name(name: str) -> str:
    n = (name or "").replace("NVIDIA ", "")
    for key in ("H100", "H800", "A100", "A800", "L40", "V100", "T4", "A10", "A30", "A40"):
        if key in n:
            return key
    return (n.split()[0] if n else "?")[:5]


def term_cols(fallback: int = 200) -> int:
    try:
        return max(60, shutil.get_terminal_size((fallback, 40)).columns)
    except Exception:
        return fallback


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
    if toks and toks[0] in ("python", "python3"):
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
    thin = "·" * width

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
    if cmd_width < 0:
        cmd_budget = 10**9
    elif cmd_width > 0:
        cmd_budget = cmd_width
    else:
        cmd_budget = max(40, cols - len(cmd_indent))

    # header for metric row
    u_label = f"{'GPU-Util':^{bar_w + 7}}"
    m_label = f"{'Memory-Usage':^{bar_w + 10}}"
    lines.append(
        f"{'HOST':<{host_w}} {'GPU':>3} {'TYPE':<5} "
        f"{u_label}  {m_label}  {'TEMP':>5} {'PWR':>6}  {'PID':>7}"
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
                lines.append(metric + f"  {'-':>7}")
                lines.append(
                    cmd_indent + color("(no compute process)", "90", color_on)
                )
                continue

            for j, p in enumerate(plist):
                raw_cmd = p.get("cmdline") or "?"
                pid = p["pid"]
                user = (p.get("user") or "?")[:8]
                if p.get("resolved"):
                    pid_s = color(f"{pid:>7}", "96", color_on)
                else:
                    pid_s = color(f"{pid:>7}", "91", color_on)

                if j == 0:
                    lines.append(metric + f"  {pid_s}")
                else:
                    # extra process on same GPU
                    lines.append(
                        f"{'':<{host_w}} {'↳':>3} {'':<5} "
                        f"{'':<{bar_w}} {'':>6}  "
                        f"{'':<{bar_w}} {'':>5} {'':<9}  "
                        f"{'':>4} {'':>5}  {pid_s}"
                    )

                if verbose:
                    for chunk in wrap_cmd(normalize_cmd(raw_cmd), max(40, cols - len(cmd_indent))):
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
        + color("  │  v verbose  q quit  -1 once", "90", color_on)
    )
    return "\n".join(lines)


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

    state keys: cmd_align ('left'|'right'), verbose (bool), quit (bool)
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

    if not os.path.isfile(PROBE):
        print(f"Missing probe script: {PROBE}", file=sys.stderr)
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
        "quit": False,
    }
    # cache last probe results so C-a/C-e redraw is instant without re-SSH
    cache: dict = {"res": None}

    def once(force_collect: bool = True):
        if force_collect or cache["res"] is None:
            cache["res"] = collect(hosts)
        cols = term_cols()
        text = render(
            cache["res"],
            color_on=color_on,
            show_procs=show_procs,
            cmd_width=args.cmd_width,
            cols=cols,
            verbose=state["verbose"],
            cmd_align=state["cmd_align"],
        )
        if watch is not None:
            sys.stdout.write("\033[H\033[J")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
