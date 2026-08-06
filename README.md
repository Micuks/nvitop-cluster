# nvitop-cluster

Multi-node GPU monitor for training clusters: read MPI/OpenMPI **hostfile**, query each host (local NVML / remote SSH), show **nvitop-style** UTIL/MEM bars, temperatures, power, and **resolved process command lines**.

Designed for multi-node jobs (e.g. DeepSpeed / torchrun / MPI) where plain `nvitop` only sees the current machine.

## Screenshot

![nvitop-cluster sample](https://raw.githubusercontent.com/Micuks/nvitop-cluster/main/docs/assets/nvitop-cluster-sample.png)

*Sample from a 2-node × 8-GPU job (`/etc/mpi/hostfile`). Bars = GPU util (left) / memory (right). Second line per GPU is the process command (Ctrl-A / Ctrl-E switch head vs tail when truncated).*

## Features

- **Hostfile discovery**: `/etc/mpi/hostfile`, `/etc/mpi/mpi-hostfile`, or `-f PATH`
- **Local + remote**: local probe without SSH; peers via `ssh` (optional KML-style `ssh_config`)
- **Wide dual bars**: GPU-Util and Memory-Usage (width scales with terminal)
- **Process names**: optional host-PID → container-PID remap for Kubernetes/container jobs
- **Watch mode by default** (2s); keys:
  - **Ctrl-A** / `a` — CMD head (like nvitop)
  - **Ctrl-E** / `e` — CMD tail
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

## Usage

```bash
# multi-node table (default: watch 2s)
nvitop-cluster

# once
nvitop-cluster -1

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
