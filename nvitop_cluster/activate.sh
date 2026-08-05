#!/usr/bin/env bash
# Idempotent: wire nvitop host-PID→container-PID fix into this KML pod.
# Safe to call from tmux `run-shell` on every `source-file ~/.tmux.conf`.
set -euo pipefail

SHARE_BIN="/share_l3/wuqingliu/bin"
TOOL="/share_l3/wuqingliu/tools/nvitop-container"
WRAPPER_PY="$TOOL/nvitop"
MARKER="nvitop-container"

log() { :; }  # silent by default; set NVITOP_PID_MAP_DEBUG=1 to see
if [ -n "${NVITOP_PID_MAP_DEBUG:-}" ]; then
  log() { printf '[nvitop-container] %s\n' "$*" >&2; }
fi

[ -x "$WRAPPER_PY" ] || [ -f "$WRAPPER_PY" ] || exit 0
[ -f "$TOOL/nvitop_pid_map.py" ] || exit 0

# 1) Ensure share bin wrapper exists
mkdir -p "$SHARE_BIN"
if [ ! -x "$SHARE_BIN/nvitop" ]; then
  cat > "$SHARE_BIN/nvitop" << 'WRAP'
#!/bin/bash
if [ -x /opt/conda/envs/py312/bin/python3 ]; then
  exec /opt/conda/envs/py312/bin/python3 /share_l3/wuqingliu/tools/nvitop-container/nvitop "$@"
fi
if [ -x /share_l3/wuqingliu/envs/flash_gdn/bin/python ]; then
  exec /share_l3/wuqingliu/envs/flash_gdn/bin/python /share_l3/wuqingliu/tools/nvitop-container/nvitop "$@"
fi
exec python3 /share_l3/wuqingliu/tools/nvitop-container/nvitop "$@"
WRAP
  chmod +x "$SHARE_BIN/nvitop"
  log "created $SHARE_BIN/nvitop"
fi

# 2) Wrap common nvitop entrypoints so bare `nvitop` works even if PATH order is wrong
wrap_entry() {
  local target="$1"
  [ -e "$target" ] || return 0
  # skip if already our wrapper
  if grep -q "$MARKER" "$target" 2>/dev/null; then
    return 0
  fi
  # only wrap scripts/binaries we can replace
  if [ -f "$target" ] && [ -w "$target" ] || [ -w "$(dirname "$target")" ]; then
    if [ ! -e "${target}.orig-before-pidmap" ]; then
      cp -a "$target" "${target}.orig-before-pidmap" 2>/dev/null || true
    fi
    # Prefer python shebang wrapper that always goes through our module
    local py=""
    if [ -x /opt/conda/envs/py312/bin/python3 ]; then
      py=/opt/conda/envs/py312/bin/python3
    elif [ -x /usr/bin/python3 ]; then
      py=/usr/bin/python3
    else
      py=python3
    fi
    cat > "$target" << EOFINNER
#!/bin/bash
# ${MARKER}: host PID → container PID remap for KML/K8s
exec ${py} ${TOOL}/nvitop "\$@"
EOFINNER
    chmod +x "$target" 2>/dev/null || true
    log "wrapped $target"
  fi
}

wrap_entry /opt/conda/envs/py312/bin/nvitop
# also wrap if a user-local copy exists
wrap_entry "${HOME}/.local/bin/nvitop"

# 3) Persist PATH for login shells on this pod
for rc in "${HOME}/.bashrc" "${HOME}/.bash_profile" /root/.bashrc; do
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

# 4) Ensure shell helper exists
cat > /share_l3/wuqingliu/.nvitop_container.sh << 'SHELLEOF'
# nvitop container PID fix (KML/K8s host PID vs local PID)
case ":${PATH}:" in
  *":/share_l3/wuqingliu/bin:"*) ;;
  *) export PATH="/share_l3/wuqingliu/bin:${PATH}" ;;
esac
alias nvitop='/share_l3/wuqingliu/bin/nvitop' 2>/dev/null || true
SHELLEOF

# 5) Push PATH into tmux global environment when a tmux server is reachable.
#    (run-shell from source-file may not export TMUX; still talk to the server.)
if command -v tmux >/dev/null 2>&1; then
  if tmux list-sessions >/dev/null 2>&1; then
    cur="$(tmux show-environment -g PATH 2>/dev/null | sed "s/^PATH=//" || true)"
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
  fi
fi

exit 0
