#!/usr/bin/env bash
# measure_on_jetson.sh -- the whole Jetson measurement session in one go (about 50 minutes).
#
#   tmux new -s measure            # keeps running if the SSH connection drops
#                                  #   (detach: Ctrl+B then D, back: tmux attach -t measure)
#   sudo su                        # optional: as root no password is asked during the run
#   bash measure_on_jetson.sh
#
# Put this script, inference_logger_template.py (with setup/infer filled in) and
# parse_tegrastats.py in one folder. Every run writes into a new folder run_YYYYmmdd_HHMMSS,
# so old logs are never overwritten or appended to.
#
# Python: the logger must run with the interpreter that has your torch / TensorRT. Under
# `sudo su`, `python3` can be a different environment (venv, conda and ~/.local packages are
# not seen by root). The script checks this before measuring; if it fails, pass the path:
#   PY=/home/<user>/venv/bin/python bash measure_on_jetson.sh
#
# Options (environment variables): PY, FPS (30), IDLE_SEC (720), MINUTES (11), WARMUP (60),
# SKIP_HEAD (120, seconds dropped at the start of every power log by the parser),
# TWIST_KIND (fp32 | xai: what the fallback path runs), LOGGER_ARGS (extra logger flags, tests only).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
FPS="${FPS:-30}"
IDLE_SEC="${IDLE_SEC:-720}"
MINUTES="${MINUTES:-11}"
WARMUP="${WARMUP:-60}"
SKIP_HEAD="${SKIP_HEAD:-120}"
TWIST_KIND="${TWIST_KIND:-fp32}"
LOGGER_ARGS="${LOGGER_ARGS:-}"
RUN="$DIR/run_$(date +%Y%m%d_%H%M%S)"
LOGGER="$DIR/inference_logger_template.py"
PARSER="$DIR/parse_tegrastats.py"
OWNER="$(stat -c %U "$DIR")"
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi
TS_MODE=""; TS_PID=""; LOGGER_PID=""; KEEPALIVE_PID=""

mkdir -p "$RUN"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/session.txt"; }

# ---------------------------------------------------------------- power log control
start_ts() {   # $1 = absolute log path (a daemon may not keep our working directory)
  if $SUDO tegrastats --interval 1000 --logfile "$1" --start >/dev/null 2>&1; then
    sleep 3
    if [ -s "$1" ]; then TS_MODE="daemon"; return; fi
    $SUDO tegrastats --stop >/dev/null 2>&1 || true
  fi
  # fallback for tegrastats without --start/--stop: run it in the background ourselves
  $SUDO tegrastats --interval 1000 --logfile "$1" >/dev/null 2>&1 &
  TS_PID=$!; TS_MODE="bg"; sleep 3
  [ -s "$1" ] || { log "ERROR: tegrastats is not writing $1"; exit 1; }
}
stop_ts() {
  if [ "$TS_MODE" = "daemon" ]; then
    $SUDO tegrastats --stop >/dev/null 2>&1 || true
  elif [ "$TS_MODE" = "bg" ] && [ -n "$TS_PID" ]; then
    $SUDO kill "$TS_PID" 2>/dev/null || true
    wait "$TS_PID" 2>/dev/null || true
  fi
  TS_MODE=""; TS_PID=""
  sleep 3
}

cleanup() {
  stop_ts
  $SUDO tegrastats --stop >/dev/null 2>&1 || true
  if [ -n "$LOGGER_PID" ]; then kill "$LOGGER_PID" 2>/dev/null || true; wait "$LOGGER_PID" 2>/dev/null || true; fi
  [ -n "$KEEPALIVE_PID" ] && kill "$KEEPALIVE_PID" 2>/dev/null || true
  rm -f "$RUN"/ready_*.flag
  if [ "$(id -u)" -eq 0 ] && [ "$OWNER" != "root" ]; then chown -R "$OWNER": "$RUN" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'log "interrupted -- power logging stopped"; exit 130' INT TERM

# ---------------------------------------------------------------- phases
idle_phase() {   # $1 = name
  log "$1: idle ${IDLE_SEC}s -- do not use the board"
  start_ts "$RUN/$1.log"
  sleep "$IDLE_SEC"
  stop_ts
}

infer_phase() {  # $1 = light | twist
  local mode="$1" flag="$RUN/ready_$1.flag"
  rm -f "$flag"
  log "$mode: loading model + ${WARMUP}s warm-up (power log starts after the warm-up)"
  ( cd "$RUN" && exec $PY "$LOGGER" --mode "$mode" --twist-kind "$TWIST_KIND" --fps "$FPS" \
        --minutes "$MINUTES" --warmup "$WARMUP" --ready-file "$flag" $LOGGER_ARGS ) &
  LOGGER_PID=$!
  while [ ! -f "$flag" ]; do
    if ! kill -0 "$LOGGER_PID" 2>/dev/null; then
      log "ERROR: the $mode logger stopped before its warm-up finished (see messages above)"; exit 1
    fi
    sleep 1
  done
  start_ts "$RUN/$mode.log"
  log "$mode: power log running, ${MINUTES} counted minutes"
  wait "$LOGGER_PID"; LOGGER_PID=""
  stop_ts
}

# ---------------------------------------------------------------- pre-flight
command -v tegrastats >/dev/null || { echo "tegrastats not found (run this on the Jetson)"; exit 1; }
[ -f "$LOGGER" ] || { echo "missing $LOGGER"; exit 1; }
if [ -n "$SUDO" ]; then              # not root: ask once, keep the sudo ticket alive for the whole run
  sudo -v
  ( while true; do sudo -n true; sleep 60; done ) >/dev/null 2>&1 &
  KEEPALIVE_PID=$!
fi
$SUDO tegrastats --stop >/dev/null 2>&1 || true     # no leftover instance from an earlier run

log "session folder: $RUN"
{
  echo "date: $(date)"
  echo "shell user: $(id -un)   folder owner: $OWNER"
  echo "python: $PY -> $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])' 2>&1)"
  echo "twist kind: $TWIST_KIND   target fps: $FPS   logger args: ${LOGGER_ARGS:-none}"
  cat /etc/nv_tegra_release 2>/dev/null || true
  $SUDO nvpmodel -q 2>/dev/null || true
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 || true
  $PY -c "import tensorrt; print('tensorrt', tensorrt.__version__)" 2>&1 || true
} > "$RUN/env_info.txt"
cat "$RUN/env_info.txt"

log "pre-flight: load both paths and run 5 s each (catches a wrong Python environment early)"
( cd "$RUN" && $PY "$LOGGER" --mode light --fps "$FPS" --minutes 0 --warmup 5 $LOGGER_ARGS )
( cd "$RUN" && $PY "$LOGGER" --mode twist --twist-kind "$TWIST_KIND" --fps "$FPS" --minutes 0 --warmup 5 $LOGGER_ARGS )
rm -f "$RUN"/inference_counts_*.csv
log "pre-flight OK -- measurement starts, about $(( (2*IDLE_SEC + 2*(WARMUP + 60*MINUTES)) / 60 + 3 )) min"

# ---------------------------------------------------------------- measurement
idle_phase idle1
infer_phase light
infer_phase twist
idle_phase idle2
log "measurement finished"

# ---------------------------------------------------------------- parameters
if [ -f "$PARSER" ] && $PY -c "import numpy" >/dev/null 2>&1; then
  ( cd "$RUN" && $PY "$PARSER" build --idle idle1.log idle2.log --light light.log --twist twist.log \
        --counts-light inference_counts_light.csv --counts-twist inference_counts_twist.csv \
        --skip-head "$SKIP_HEAD" ) 2>&1 | tee -a "$RUN/session.txt" \
    || log "parse step failed -- run parse_tegrastats.py manually in $RUN"
else
  log "numpy or parse_tegrastats.py not available here -- copy $RUN to the PC and run the parser there"
fi
log "done. Files in $RUN (copy the whole folder to the PC)"
