#!/usr/bin/env bash
#
# Run one phase of the study on a rented pod, then terminate the instance.
#
# The point of this script is that a forgotten pod costs about £6 a night --
# your entire budget. Monitoring does not protect you from that; the instance
# ending itself does. Everything here serves that: a hard wall-clock ceiling, an
# EXIT trap so termination fires on crash and on Ctrl-C as well as on success,
# and results pushed to HuggingFace *before* anything is torn down.
#
# One rule that overrides termination: if the results push fails, the pod stays
# up. Burning another hour of credit is recoverable. Losing four hours of
# extraction because the upload 401'd is not.
#
# Usage:
#   ./scripts/pod_session.sh pilot
#   ./scripts/pod_session.sh extract
#   ./scripts/pod_session.sh grid --tier 1
#   ./scripts/pod_session.sh eval
#
# First time, run with --no-terminate and watch it. Trust it after that.
#
# Environment:
#   HF_REPO          HuggingFace dataset repo, e.g. jacobrhys/xjepa   (required)
#   HF_TOKEN         HuggingFace write token                          (required)
#   RUNPOD_API_KEY   RunPod API key, for self-termination             (recommended)
#   RUNPOD_POD_ID    set automatically inside a RunPod pod
#   MAX_HOURS        hard ceiling for the whole session; default per phase
#   REPO_URL         git repo to clone if not already present

set -euo pipefail

# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #

PHASE="${1:-}"; shift || true
TERMINATE=1
HEARTBEAT=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-terminate) TERMINATE=0; shift ;;
    --no-heartbeat) HEARTBEAT=0; shift ;;
    --max-hours)    MAX_HOURS="$2"; shift 2 ;;
    *)              EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$PHASE" in
  pilot)   DEFAULT_HOURS=2  ;;
  extract) DEFAULT_HOURS=6  ;;
  grid)    DEFAULT_HOURS=12 ;;
  eval)    DEFAULT_HOURS=4  ;;
  *)
    cat >&2 <<'USAGE'
usage: pod_session.sh {pilot|extract|grid|eval} [--no-terminate] [--max-hours N] [args...]

  pilot    measure throughput, settle bucket policy and head count (~1-2 h)
  extract  ESM-IF1 -> PCA cache -> push to HF (~3-4 h, needs 60 GB disk)
  grid     the condition grid, tiered and spend-capped (~8-11 h)
  eval     probes and aggregation (~3 h)
USAGE
    exit 2 ;;
esac

MAX_HOURS="${MAX_HOURS:-$DEFAULT_HOURS}"
MAX_SECONDS=$(python3 -c "print(int(float('$MAX_HOURS')*3600))")
WORKDIR="${WORKDIR:-/workspace/protein}"
REPO_URL="${REPO_URL:-}"
STARTED_AT=$(date -u +%s)

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
elapsed_min() { echo $(( ( $(date -u +%s) - STARTED_AT ) / 60 )); }

# --------------------------------------------------------------------------- #
# teardown -- runs on success, failure, timeout and interrupt
# --------------------------------------------------------------------------- #

PUSH_OK=0
HEARTBEAT_PID=""
WATCHDOG_PID=""
#: Set once setup succeeds. Before that there is nothing to push and no pod
#: state worth protecting, so cleanup exits quietly instead of making network
#: calls for directories that do not exist.
PHASE_STARTED=0

push_results() {
  # Always attempted, even when the phase failed: a failed run's logs are
  # exactly what you need to work out why, and they die with the pod.
  [[ -z "${HF_REPO:-}" ]] && { log "HF_REPO unset -- cannot push results"; return 1; }
  log "pushing results to ${HF_REPO} ..."
  local ok=1
  for dir in runs results report data/corpus; do
    [[ -d "$WORKDIR/$dir" ]] || continue
    if huggingface-cli upload "$HF_REPO" "$WORKDIR/$dir" "$dir" \
         --repo-type dataset --quiet 2>&1 | tail -2 >&2; then
      log "  pushed $dir"
    else
      log "  FAILED to push $dir"
      ok=0
    fi
  done
  return $(( ok == 1 ? 0 : 1 ))
}

terminate_pod() {
  local pod_id="${RUNPOD_POD_ID:-}"
  if [[ -z "$pod_id" ]]; then
    log "RUNPOD_POD_ID unset -- not on a RunPod pod, or the variable is missing."
    log "TERMINATE THE POD YOURSELF from the dashboard."
    return 1
  fi
  log "terminating pod ${pod_id} ..."
  if command -v runpodctl >/dev/null 2>&1; then
    runpodctl remove pod "$pod_id" && return 0
    log "runpodctl failed; trying the API directly"
  fi
  if [[ -n "${RUNPOD_API_KEY:-}" ]]; then
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${pod_id}\\\"}) }\"}" \
      >&2 && return 0
  fi
  log "COULD NOT TERMINATE. Kill the pod from the dashboard NOW -- it is still billing."
  return 1
}

cleanup() {
  local rc=$?
  set +e
  # Kill both children first. A live background job holds the script's stdout
  # open, so anything piping this script's output would hang forever.
  [[ -n "$HEARTBEAT_PID" ]] && kill "$HEARTBEAT_PID" 2>/dev/null
  [[ -n "$WATCHDOG_PID" ]] && kill "$WATCHDOG_PID" 2>/dev/null

  if [[ "$PHASE_STARTED" -eq 0 ]]; then
    # Reaching cleanup before the phase begins is always a failure, whatever $?
    # says. Bash does not reliably set a non-zero status before the EXIT trap
    # for some expansion errors, and a preflight abort that reports success
    # would let a wrapper script march on to the next phase.
    [[ "$rc" -eq 0 ]] && rc=1
    log "exited during preflight (code ${rc}); nothing produced, nothing to push"
    log "pod left running -- you never got as far as doing work on it"
    exit "$rc"
  fi

  log "=============================================================="
  log "phase '${PHASE}' finished with code ${rc} after $(elapsed_min) min"

  if push_results; then
    PUSH_OK=1
    log "results are safe on HuggingFace"
  else
    PUSH_OK=0
    log "RESULTS PUSH FAILED"
  fi

  if [[ "$TERMINATE" -eq 0 ]]; then
    log "--no-terminate: pod left running. It is still billing you."
  elif [[ "$PUSH_OK" -eq 1 ]]; then
    terminate_pod
  else
    log "NOT terminating: the results push failed, and the work on this disk"
    log "is worth more than the credit it costs to keep the pod alive."
    log "Fix HF_TOKEN / HF_REPO, re-run push_results, then kill the pod."
  fi
  log "=============================================================="
  exit "$rc"
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# watchdog -- hard ceiling on the whole session
# --------------------------------------------------------------------------- #

start_watchdog() {
  # Redirect the child's stdout so it cannot hold a pipe open, and start it only
  # once real work is about to begin.
  ( sleep "$MAX_SECONDS"
    log "WATCHDOG: ${MAX_HOURS}h ceiling reached, stopping the session"
    kill -TERM $$ 2>/dev/null ) >/dev/null &
  WATCHDOG_PID=$!
}

# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

log "=============================================================="
log "phase: ${PHASE}   ceiling: ${MAX_HOURS}h"
if [[ "$TERMINATE" -eq 1 ]]; then
  log "THIS POD WILL TERMINATE ITSELF when the phase ends."
else
  log "self-termination DISABLED (--no-terminate)"
fi
log "=============================================================="

if [[ -z "${HF_REPO:-}" ]]; then
  log "HF_REPO is unset. Results would have nowhere to go, and the pod would"
  log "terminate with the work still on its disk. Refusing to start."
  log "  export HF_REPO=youruser/xjepa"
  exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  log "HF_TOKEN is unset -- the results push would fail. Refusing to start."
  exit 1
fi
if [[ "$TERMINATE" -eq 1 && -z "${RUNPOD_POD_ID:-}" ]]; then
  log "RUNPOD_POD_ID is unset, so this script cannot terminate the instance."
  log "Either run on a RunPod pod, or pass --no-terminate and kill it yourself."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  log "no nvidia-smi: this is not a GPU instance. Refusing to run."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >&2

AVAIL_GB=$(df -BG --output=avail /workspace 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
log "disk available: ${AVAIL_GB} GB"
if [[ "$PHASE" == "extract" && "${AVAIL_GB:-0}" -lt 40 ]]; then
  log "extract needs ~40 GB free (the raw ESM-IF1 bank alone is 12.8 GB)."
  log "Redeploy with a 60 GB container disk. Stopping before wasting the hour."
  exit 1
fi

# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #

if [[ ! -d "$WORKDIR" ]]; then
  [[ -z "$REPO_URL" ]] && { log "no $WORKDIR and REPO_URL unset"; exit 1; }
  log "cloning ${REPO_URL}"
  git clone --depth 1 "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

log "installing"
pip install -q -e ".[dev]" 2>&1 | tail -3 >&2

log "running the test suite before spending GPU time on a broken tree"
if ! python -m pytest -q -x 2>&1 | tail -5 >&2; then
  log "TESTS FAILED. Not running the phase."
  exit 1
fi

# Pull the cached corpus for phases that consume it.
if [[ "$PHASE" == "grid" || "$PHASE" == "eval" ]]; then
  if [[ ! -d data/corpus ]]; then
    log "pulling corpus from ${HF_REPO}"
    huggingface-cli download "$HF_REPO" --repo-type dataset \
      --include 'data/corpus/*' --local-dir . --quiet
  fi
  [[ -f data/corpus/targets.npy ]] || { log "no corpus after pull -- run 'extract' first"; exit 1; }
fi

# --------------------------------------------------------------------------- #
# heartbeat -- lets you watch progress without an SSH session
# --------------------------------------------------------------------------- #

PHASE_STARTED=1
start_watchdog

if [[ "$HEARTBEAT" -eq 1 ]]; then
  ( while true; do
      sleep 600
      {
        echo "phase=${PHASE} elapsed_min=$(elapsed_min) ceiling_h=${MAX_HOURS}"
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
        ls -1 runs 2>/dev/null | tail -5
      } > /tmp/heartbeat.txt 2>/dev/null
      huggingface-cli upload "$HF_REPO" /tmp/heartbeat.txt heartbeat.txt \
        --repo-type dataset --quiet >/dev/null 2>&1 || true
      for d in runs results; do
        [[ -d "$d" ]] && huggingface-cli upload "$HF_REPO" "$d" "$d" \
          --repo-type dataset --quiet >/dev/null 2>&1 || true
      done
    done ) >/dev/null 2>&1 &
  HEARTBEAT_PID=$!
  log "heartbeat every 10 min -> ${HF_REPO}/heartbeat.txt"
fi

# --------------------------------------------------------------------------- #
# the phase
# --------------------------------------------------------------------------- #

log "starting phase '${PHASE}'"

case "$PHASE" in
  pilot)
    python scripts/pilot.py --out runs/pilot "${EXTRA_ARGS[@]}"
    log "READ runs/pilot/pilot.md BEFORE THE NEXT PHASE."
    log "It settles the bucket policy and head count, and recomputes the budget."
    ;;

  extract)
    log "pilot pass over 500 chains first, to measure before committing hours"
    python scripts/extract_esmif1.py --shards data/structures \
      --allowlist data/splits/pretrain_accessions.txt --out data/raw_pilot --limit 500
    log "full extraction"
    python scripts/extract_esmif1.py --shards data/structures \
      --allowlist data/splits/pretrain_accessions.txt --out data/raw "${EXTRA_ARGS[@]}"
    python -m xjepa.data.build_cache --embeddings data/raw/esmif1_512.npy \
      --tokens data/raw/tokens.npy --offsets data/raw/offsets.npy \
      --out data/corpus --dim 128
    log "target bank RankMe (the H1b ceiling) is in data/corpus/meta.json -- READ IT."
    log "A low value caps what C3 can learn, and it is a go/no-go before the grid."
    cat data/corpus/meta.json >&2 || true
    ;;

  grid)
    python scripts/run_grid.py --corpus data/corpus --out runs/ "${EXTRA_ARGS[@]}"
    ;;

  eval)
    python scripts/run_eval.py --runs runs/ --eval-data data/eval --out results/ "${EXTRA_ARGS[@]}"
    python scripts/aggregate_results.py --results results/ --out report/
    ;;
esac

log "phase '${PHASE}' complete"
