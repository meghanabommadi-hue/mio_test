#!/usr/bin/env bash
# One-shot, throwaway test: launch a vllm + codec server instance, run exactly
# one test batch against it, then always tear that instance down again --
# nothing this script started is left running afterward, pass or fail.
#
# NOTE: this does NOT kill any pre-existing vllm/codec servers first. If one
# is already running (e.g. on the default ports), the new vllm/codec launch
# below will fail (port already bound / insufficient free GPU memory) -- stop
# other instances yourself first (pkill -9 -f "vllm serve"; pkill -9 -f
# "\.venv_vllm/bin/python3"; pkill -9 -f "miotts.codec_server") if that happens.
#
# This is deliberately separate from run.sh, which is meant for long-lived
# servers you keep running across multiple test invocations. Use this script
# instead when you want a clean, isolated, single measurement -- e.g. right
# after changing config.py and wanting one definitive answer, not a server
# you have to remember to stop later.
#
# Usage:
#   ./test_ephemeral.sh                                  # one config_test batch, defaults
#   ./test_ephemeral.sh --benchmark                       # run miotts.benchmark instead
#   ./test_ephemeral.sh -- --categories collections        # forward args to config_test
#   ./test_ephemeral.sh --benchmark -- --batch-sizes 100   # forward args to benchmark
#
# Env overrides: MAX_MODEL_LEN, GPU_MEM_UTIL, PORT, CODEC_PORT (same defaults as run.sh).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2560}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
PORT="${PORT:-8000}"
CODEC_PORT="${CODEC_PORT:-8001}"
LOG_FILE="$(mktemp /tmp/test_ephemeral_vllm.XXXXXX.log)"
CODEC_LOG_FILE="$(mktemp /tmp/test_ephemeral_codec.XXXXXX.log)"

MODE="config-test"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) MODE="benchmark"; shift ;;
    --config-test) MODE="config-test"; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

VLLM_PID=""
CODEC_PID=""

cleanup() {
  echo "Tearing down ephemeral instance ..."
  [[ -n "$VLLM_PID" ]] && kill -9 "$VLLM_PID" 2>/dev/null
  [[ -n "$CODEC_PID" ]] && kill -9 "$CODEC_PID" 2>/dev/null
  # vllm's engine core runs as a multiprocessing worker with no "vllm serve"
  # in its argv, so it survives a plain kill of the parent PID and keeps
  # holding GPU memory -- kill anything from this venv's interpreter too.
  pkill -9 -f "\.venv_vllm/bin/python3" 2>/dev/null
  pkill -9 -f "miotts.codec_server --port $CODEC_PORT" 2>/dev/null
  rm -f "$LOG_FILE" "$CODEC_LOG_FILE"
  echo "Done -- no processes left running from this test."
}
trap cleanup EXIT

echo "Starting ephemeral vllm server (max-model-len=$MAX_MODEL_LEN, gpu-memory-utilization=$GPU_MEM_UTIL, port=$PORT) ..."
(
  source .venv_vllm/bin/activate
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  exec vllm serve SPRINGLab/Indic-Mio \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT"
) > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "Waiting for vllm server startup (log: $LOG_FILE) ..."
until grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; do
  if grep -qE "Traceback|EngineCore failed to start" "$LOG_FILE" 2>/dev/null; then
    echo "vllm server failed to start:" >&2
    tail -30 "$LOG_FILE" >&2
    exit 1
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vllm server process died unexpectedly:" >&2
    tail -30 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 2
done
echo "vllm server is up on port $PORT (pid $VLLM_PID)."

echo "Starting ephemeral codec server (port=$CODEC_PORT) ..."
(
  source .venv/bin/activate
  exec python3 -m miotts.codec_server --port "$CODEC_PORT"
) > "$CODEC_LOG_FILE" 2>&1 &
CODEC_PID=$!

echo "Waiting for codec server startup (log: $CODEC_LOG_FILE) ..."
until grep -q "Application startup complete" "$CODEC_LOG_FILE" 2>/dev/null; do
  if grep -qE "Traceback" "$CODEC_LOG_FILE" 2>/dev/null; then
    echo "codec server failed to start:" >&2
    tail -30 "$CODEC_LOG_FILE" >&2
    exit 1
  fi
  if ! kill -0 "$CODEC_PID" 2>/dev/null; then
    echo "codec server process died unexpectedly:" >&2
    tail -30 "$CODEC_LOG_FILE" >&2
    exit 1
  fi
  sleep 2
done
echo "codec server is up on port $CODEC_PORT (pid $CODEC_PID)."

echo "Running $MODE against the ephemeral instance ..."
(
  source .venv/bin/activate
  if [[ "$MODE" == "benchmark" ]]; then
    if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
      EXTRA_ARGS=(--batch-sizes 50 --languages english hindi telugu)
    fi
    # Always force --backend vllm (argparse takes the LAST occurrence of a
    # flag, so this intentionally overrides anything in EXTRA_ARGS) -- this
    # script's whole point is testing the vllm instance it just started, so
    # silently falling back to benchmark.py's own default (in-process
    # transformers, ignoring the vllm instance entirely) would defeat the
    # purpose. --concurrency can still be tuned via the CONCURRENCY env var.
    exec python3 -m miotts.benchmark "${EXTRA_ARGS[@]}" \
      --backend vllm --concurrency "${CONCURRENCY:-10}" \
      --vllm-base-url "http://localhost:$PORT" --codec-base-url "http://localhost:$CODEC_PORT"
  else
    exec python3 -m miotts.config_test "${EXTRA_ARGS[@]}" \
      --backend vllm --vllm-base-url "http://localhost:$PORT" --codec-base-url "http://localhost:$CODEC_PORT"
  fi
)
TEST_EXIT=$?

exit "$TEST_EXIT"
