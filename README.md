# nvitop-cluster

Multi-node GPU monitor for training clusters: read MPI/OpenMPI **hostfile**, query each host (local NVML / remote SSH), show **nvitop-style** UTIL/MEM bars, temperatures, power, and **resolved process command lines**.

Designed for multi-node jobs (e.g. DeepSpeed / torchrun / MPI) where plain `nvitop` only sees the current machine.

## Screenshot

![nvitop-cluster sample](https://raw.githubusercontent.com/Micuks/nvitop-cluster/main/docs/assets/nvitop-cluster-sample.png)

*Sample from a 2-node × 8-GPU job (`/etc/mpi/hostfile`). Small jobs use the detailed bar view. Large jobs automatically switch to a compact multi-column overview so every host remains visible.*

## Features

- **Hostfile discovery**: `/etc/mpi/hostfile`, `/etc/mpi/mpi-hostfile`, or `-f PATH`
- **Local + remote**: local probe without SSH; peers via `ssh` (optional KML-style `ssh_config`)
- **Wide dual bars**: GPU-Util and Memory-Usage (width scales with terminal)
- **Adaptive density**: compact per-GPU tables expand into host-average UTIL/VRAM line charts when terminal area leaves room
- **Uniform grid**: common 8-host and 16-host jobs use equal 2-column×4-row / 4×4 cards; partial final rows stay equal and centered
- **Command + runtime**: repeated rank commands collapse to one `CMD ×N` row per host with nvitop-style `TIME`; detail view shows each PID's runtime
- **Process names**: optional host-PID → container-PID remap for Kubernetes/container jobs
- **Watch mode by default** (2s); keys:
  - **Ctrl-A** / `a` — CMD head (like nvitop)
  - **Ctrl-E** / `e` — CMD tail
  - **j** / **k** — select next / previous host
  - **Enter** / **d** — detail view for the selected host
  - **g** — return to the all-host overview
  - **p** — show / hide process information
  - **x** — attention-only view (stalled, hot, full-memory, missing/unresolved process)
  - **[** / **]** — previous / next host page on smaller terminals
  - **v** — verbose full cmdline
  - **q** — quit

## Install

Zero third-party Python deps (stdlib only). Needs **Python ≥ 3.9**, **`nvidia-smi`** on each node, and **SSH to peers** for multi-node.

### pip / uv (recommended)

```bash
pip install nvitop-cluster
# or
uv pip install nvitop-cluster

# one-shot without installing into the env
uvx nvitop-cluster

# from GitHub (latest main)
pip install "git+https://github.com/Micuks/nvitop-cluster.git"
```

Then:

```bash
nvitop-cluster
# or
python -m nvitop_cluster
```

**Multi-node tip:** either install the package in the **same image/env on every rank**, or put the repo on a shared path and set `NVITOP_CLUSTER_HOME` so remote SSH probes can find the script.

### From source (no install)

```bash
git clone https://github.com/Micuks/nvitop-cluster.git
cd nvitop-cluster
python3 -m nvitop_cluster
# or
./nvitop-cluster
```

Optional (local interactive nvitop + container PID fix):

```bash
pip install 'nvitop-cluster[nvitop]'
# then: nvitop_cluster/nvitop --cluster
```

### KML shared-path activation

When this repository is deployed at
`/share_l3/wuqingliu/tools/nvitop-container`, sourcing the shared tmux config
activates the PID mapper and installs stable `nvitop` / `nvitop-cluster`
launchers:

```bash
tmux source-file /mmu_mllm_hdd_3/wuqingliu/.tmux.conf
```

The accompanying config can bind `Prefix + N` to open the adaptive dashboard:

```tmux
bind-key N new-window -n nvitop-cluster -c "#{pane_current_path}" \
  "/share_l3/wuqingliu/bin/nvitop-cluster"
```

## Usage

```bash
# multi-node table (default: watch 2s)
nvitop-cluster

# once
nvitop-cluster -1

# force a particular layout (default: auto)
nvitop-cluster --layout overview
nvitop-cluster --layout detail

# custom hostfile / SSH config
nvitop-cluster -f /path/to/hostfile
export NVITOP_CLUSTER_SSH_CONFIG=~/.ssh/config

# via nvitop-compatible wrapper (optional extra)
python3 -m nvitop_cluster  # same as nvitop-cluster
```

### Hostfile format

```text
10.0.0.1 slots=8
10.0.0.2 slots=8
```

(Also accepts bare hostnames / `host:slots`.)

### Environment

| Variable | Meaning |
|----------|---------|
| `NVITOP_CLUSTER_HOME` | Directory containing probe scripts (default: package dir) |
| `NVITOP_CLUSTER_SSH_CONFIG` | SSH config for peers (default: `/etc/kml/ssh/ssh_config` if present) |
| `NVITOP_HOSTFILE` | Extra hostfile path candidate |
| `NVITOP_CLUSTER=1` | Force cluster mode when using the `nvitop` wrapper |

### Adaptive layouts

`--layout auto` estimates the detailed view's physical height using the current
terminal dimensions. If it would overflow, nvitop-cluster switches to host
cards, then calculates detail density from the available terminal area per GPU:

- **COMPACT** — one line/GPU with current utilization, memory, temperature, and power
- **RICH** — adds compact host-average utilization and VRAM time-series charts
- **FULL** — renders connected Braille traces (2×4 subpixels/cell) while keeping each plot vertically bounded

If all hosts do not fit at COMPACT density, the view paginates. History is
retained in a bounded in-memory window and survives terminal resizes during the
same monitor session. A partial final row keeps the same card dimensions as
preceding rows and is centered instead of stretching a few cards. Identical
process commands remain collapsed to `CMD ×N`; their footer shows the longest
runtime in that command group, while detail view reports `TIME` per PID.

Press `Enter` for the selected host's full bars and command lines, then `g` to
return. Press `x` to temporarily show only GPUs that need attention.

## Container / Kubernetes note

On many GPU pods, NVML reports **host PIDs** while `/proc` only has **namespace PIDs**, so process names become `N/A`. `nvitop_pid_map.py` remaps host→local PIDs (heuristic: rank / device / RSS). Works on each node when `share` storage holds the same scripts (or install the package everywhere).

## Layout (vs plain nvitop)

| | nvitop | nvitop-cluster |
|--|--------|----------------|
| Scope | Current host | All hostfile hosts |
| UI | Full interactive TUI | Watch table + keys |
| Remote | — | SSH probe |
| Processes | Yes | Yes (resolved when possible) |

## License

Apache-2.0 (see [LICENSE](LICENSE)).

## Related

- [nvitop](https://github.com/XuehaiPan/nvitop) — single-node interactive viewer
- [all-smi](https://github.com/lablup/all-smi) — multi-host accelerator TUI
- [gpustat-web](https://github.com/wookayin/gpustat-web) — multi-node web dashboard
