#!/usr/bin/env bash
# Idempotent: wire nvitop host-PID→container-PID fix into this KML pod.
# Safe to call from tmux `run-shell` on every `source-file` of the share tmux.conf.
#
# After this runs, bare `nvitop` works even in already-open panes, because we
# replace common entrypoint paths in-place (not only PATH prepend).
set -euo pipefail

SHARE_BIN="/share_l3/wuqingliu/bin"
TOOL="/share_l3/wuqingliu/tools/nvitop-container"
WRAPPER_PY="$TOOL/nvitop"
MARKER="nvitop-container"
SHARE_LAUNCHER="$SHARE_BIN/nvitop"
SHARE_CLUSTER_LAUNCHER="$SHARE_BIN/nvitop-cluster"

log() { :; }
if [ -n "${NVITOP_PID_MAP_DEBUG:-}" ]; then
  log() { printf '[nvitop-container] %s\n' "$*" >&2; }
fi

[ -f "$WRAPPER_PY" ] || exit 0
[ -f "$TOOL/nvitop_pid_map.py" ] || exit 0

mkdir -p "$SHARE_BIN"

# 1) Ensure share bin launcher exists (and is the smart python picker).
#    activate.sh is the source of truth on new pods; rewrite if missing or stale.
need_launcher=0
if [ ! -x "$SHARE_LAUNCHER" ]; then
  need_launcher=1
elif ! grep -q 'nvitop-container launcher' "$SHARE_LAUNCHER" 2>/dev/null; then
  need_launcher=1
elif ! grep -q '_pick_python' "$SHARE_LAUNCHER" 2>/dev/null; then
  need_launcher=1
fi

if [ "$need_launcher" -eq 1 ]; then
  cat > "$SHARE_LAUNCHER" << 'WRAP'
#!/bin/bash
# nvitop-container launcher: host-PID → container-PID remap (KML/K8s)
TOOL="/share_l3/wuqingliu/tools/nvitop-container/nvitop"

_candidates() {
  local c
  for c in \
    /usr/bin/python3 \
    /usr/local/bin/python3 \
    python3 \
    /opt/conda/envs/py312/bin/python3 \
    /opt/conda/bin/python3 \
    "$HOME/.local/share/mamba/envs/py312/bin/python3" \
    /share_l3/wuqingliu/envs/flash_gdn/bin/python
  do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then
      printf '%s\n' "$c"
    fi
  done
}

_pick_python() {
  local py
  if [ -n "${NVITOP_PYTHON:-}" ]; then
    printf '%s\n' "$NVITOP_PYTHON"
    return 0
  fi
  while IFS= read -r py; do
    if "$py" -c 'import nvitop' >/dev/null 2>&1; then
      printf '%s\n' "$py"
      return 0
    fi
  done < <(_candidates)
  return 1
}

py="$(_pick_python)" || {
  echo "[nvitop-container] no python with nvitop installed; tried common paths" >&2
  echo "[nvitop-container] install nvitop or set NVITOP_PYTHON=/path/to/python" >&2
  exit 127
}

exec "$py" "$TOOL" "$@"
WRAP
  chmod +x "$SHARE_LAUNCHER"
  log "wrote $SHARE_LAUNCHER"
fi

# Stable cluster entrypoint used by tmux bindings and interactive shells.
# Rewrite it on every activation: it is tiny and must follow the current wrapper.
cat > "$SHARE_CLUSTER_LAUNCHER" << EOFCLUSTER
#!/bin/bash
# nvitop-container cluster launcher
exec ${SHARE_LAUNCHER} --cluster "\$@"
EOFCLUSTER
chmod +x "$SHARE_CLUSTER_LAUNCHER"

# 2) In-place wrap every nvitop entry we can find so bare `nvitop` works
#    without relying on PATH order or reopening panes.
wrap_entry() {
  local target="$1"
  [ -e "$target" ] || [ -L "$target" ] || return 0

  # already our thin wrapper?
  if [ -f "$target" ] && grep -q "$MARKER" "$target" 2>/dev/null; then
    # refresh if it does not exec share launcher (old style hardcoded py)
    if grep -q "$SHARE_LAUNCHER\|${TOOL}/nvitop" "$target" 2>/dev/null; then
      # If it hardcodes a bad python, rewrite to share launcher
      if grep -q 'exec .*python' "$target" 2>/dev/null && ! grep -q "$SHARE_LAUNCHER" "$target" 2>/dev/null; then
        :
      else
        if grep -q "$SHARE_LAUNCHER" "$target" 2>/dev/null; then
          return 0
        fi
      fi
    fi
  fi

  local dir
  dir="$(dirname "$target")"
  if [ ! -w "$dir" ] && [ ! -w "$target" ]; then
    log "skip unwritable $target"
    return 0
  fi

  # backup original once
  if [ ! -e "${target}.orig-before-pidmap" ]; then
    cp -a "$target" "${target}.orig-before-pidmap" 2>/dev/null || true
  fi

  cat > "$target" << EOFINNER
#!/bin/bash
# ${MARKER}: host PID → container PID remap for KML/K8s
# Installed by activate.sh; always go through share launcher (smart python pick).
exec ${SHARE_LAUNCHER} "\$@"
EOFINNER
  chmod +x "$target" 2>/dev/null || true
  log "wrapped $target"
}

# Known fixed paths
wrap_entry /usr/local/bin/nvitop
wrap_entry /usr/bin/nvitop
wrap_entry /opt/conda/envs/py312/bin/nvitop
wrap_entry /opt/conda/bin/nvitop
wrap_entry "${HOME}/.local/bin/nvitop"
wrap_entry /root/.local/bin/nvitop

# Discover any other nvitop on PATH (best-effort, no fail)
while IFS= read -r p; do
  [ -n "$p" ] || continue
  case "$p" in
    "$SHARE_LAUNCHER") continue ;;
  esac
  wrap_entry "$p"
done < <(type -a nvitop 2>/dev/null | awk '/is \//{print $NF}' | sort -u || true)

# Also scan common bin dirs without requiring them to be on PATH yet
for d in /usr/local/bin /usr/bin /opt/conda/envs/*/bin /opt/conda/bin "$HOME/.local/bin"; do
  for f in "$d"/nvitop; do
    [ -e "$f" ] || continue
    [ "$f" = "$SHARE_LAUNCHER" ] && continue
    wrap_entry "$f"
  done
done 2>/dev/null || true

# 3) Shell helper: PATH + alias (for new interactive shells)
cat > /share_l3/wuqingliu/.nvitop_container.sh << 'SHELLEOF'
# nvitop container PID fix (KML/K8s host PID vs local PID)
case ":${PATH}:" in
  *":/share_l3/wuqingliu/bin:"*) ;;
  *) export PATH="/share_l3/wuqingliu/bin:${PATH}" ;;
esac
# unalias first in case a stale alias points elsewhere
unalias nvitop 2>/dev/null || true
alias nvitop='/share_l3/wuqingliu/bin/nvitop' 2>/dev/null || true
alias nvitop-cluster='/share_l3/wuqingliu/bin/nvitop-cluster' 2>/dev/null || true
# drop bash hash cache so bare nvitop re-resolves after activate
hash -d nvitop 2>/dev/null || hash -r 2>/dev/null || true
SHELLEOF

# 4) Persist into bashrc / profile on this pod
for rc in "${HOME}/.bashrc" "${HOME}/.bash_profile" /root/.bashrc /root/.profile; do
  # create bashrc if missing so next login gets it
  if [ "$rc" = "${HOME}/.bashrc" ] || [ "$rc" = "/root/.bashrc" ]; then
    [ -f "$rc" ] || touch "$rc" 2>/dev/null || true
  fi
  [ -f "$rc" ] || continue
  if ! grep -q 'nvitop_container\|nvitop-container' "$rc" 2>/dev/null; then
    {
      echo ''
      echo '# nvitop-container: host PID → local PID fix'
      echo '[ -f /share_l3/wuqingliu/.nvitop_container.sh ] && . /share_l3/wuqingliu/.nvitop_container.sh'
    } >> "$rc" 2>/dev/null || true
    log "wired $rc"
  fi
done

# 5) tmux global PATH for new panes/windows
if command -v tmux >/dev/null 2>&1; then
  if tmux list-sessions >/dev/null 2>&1; then
    cur="$(tmux show-environment -g PATH 2>/dev/null | sed 's/^PATH=//' || true)"
    if [ -z "$cur" ]; then
      cur="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
    fi
    case ":$cur:" in
      *":${SHARE_BIN}:"*) ;;
      *)
        tmux set-environment -g PATH "${SHARE_BIN}:${cur}" 2>/dev/null || true
        log "tmux PATH prepended"
        ;;
    esac
    # Also export into existing sessions' environments when possible
    while IFS= read -r s; do
      [ -n "$s" ] || continue
      tmux set-environment -t "$s" PATH "${SHARE_BIN}:${cur}" 2>/dev/null || true
    done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)
  fi
fi

# 6) If invoked from an interactive shell (not only tmux run-shell), fix *this* shell
#    when user runs: bash activate.sh
if [ -n "${BASH_VERSION:-}" ] && [[ "${BASH_SOURCE[0]:-}" != "$0" || -n "${PS1:-}" ]]; then
  # sourced
  # shellcheck disable=SC1091
  . /share_l3/wuqingliu/.nvitop_container.sh 2>/dev/null || true
fi

log "activate complete"
exit 0
