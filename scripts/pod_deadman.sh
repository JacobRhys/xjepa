#!/usr/bin/env bash
#
# Terminate this pod after N hours, whatever else is happening.
#
# pod_session.sh can only end the pod from inside a phase. It cannot see an
# idle shell, a queue that never started, or a phase run with --no-terminate,
# and on 6-7 Oct one of those left a pod billing idle for seven hours. This
# timer is detached from the terminal, so closing the browser or tmux does not
# stop it. Run it once, first thing, on every pod.
#
# Usage:
#   ./scripts/pod_deadman.sh 8        # terminate 8 hours from now
#   ./scripts/pod_deadman.sh status   # show the pending deadline
#   ./scripts/pod_deadman.sh cancel   # call it off
#
# Needs RUNPOD_API_KEY exported (RUNPOD_POD_ID is set by RunPod).

set -euo pipefail

STATE=/tmp/pod_deadman

case "${1:-}" in
  status)
    if [[ -f "$STATE.pid" ]] && kill -0 "$(cat "$STATE.pid")" 2>/dev/null; then
      echo "deadman armed: terminates at $(cat "$STATE.at") UTC"
    else
      echo "no deadman running"
    fi
    exit 0 ;;
  cancel)
    if [[ -f "$STATE.pid" ]] && kill "$(cat "$STATE.pid")" 2>/dev/null; then
      echo "deadman cancelled"
    else
      echo "no deadman running"
    fi
    rm -f "$STATE.pid" "$STATE.at"
    exit 0 ;;
  ''|-h|--help)
    sed -n 2,17p "$0"; exit 1 ;;
esac

HOURS="$1"
[[ "$HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "hours must be a number, got '$HOURS'"; exit 1; }
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID unset -- not on a RunPod pod}"
: "${RUNPOD_API_KEY:?export RUNPOD_API_KEY first, or the timer cannot terminate anything}"

# Prove the key works now, not at 4am.
if ! curl -fsS -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
     -H 'Content-Type: application/json' -d '{"query":"query { myself { id } }"}' \
     | grep -q '"myself"'; then
  echo "RunPod API rejected the key. Fix RUNPOD_API_KEY; the deadman is NOT armed."
  exit 1
fi

if [[ -f "$STATE.pid" ]] && kill -0 "$(cat "$STATE.pid")" 2>/dev/null; then
  kill "$(cat "$STATE.pid")" 2>/dev/null || true
  echo "replaced the previous deadman"
fi

SECONDS_LEFT=$(python3 -c "print(int(float('$HOURS') * 3600))")
date -u -d "+${SECONDS_LEFT} seconds" '+%Y-%m-%d %H:%M' > "$STATE.at"

setsid nohup bash -c "
  sleep ${SECONDS_LEFT}
  for attempt in 1 2 3 4 5; do
    curl -s -X POST 'https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}' \
      -H 'Content-Type: application/json' \
      -d '{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}'
    sleep 60
  done
" > /tmp/pod_deadman.log 2>&1 < /dev/null &
echo $! > "$STATE.pid"

echo "deadman armed: pod ${RUNPOD_POD_ID} terminates at $(cat "$STATE.at") UTC"
echo "  (${HOURS} h from now; worst case ~\$$(python3 -c "print(round(float('$HOURS') * 0.75, 2))") at \$0.75/h)"
