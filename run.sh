#!/usr/bin/env bash
# Start the vLLM-hosted Indic-Mio server + a persistent codec server, and
# optionally run a benchmark against them.
#
# Two venvs are involved (see requirements-vllm.txt / requirements.txt for why):
#   .venv_vllm (Python 3.10) -- runs the vllm server itself.
#   .venv      (Python 3.12) -- runs the codec server (miotts.codec_server) and
#                                miotts.benchmark / miotts.cli as HTTP clients
#                                against both servers (needs miocodec, which
#                                requires Python >=3.12 and cannot be
#                                installed into .venv_vllm).
#
# The codec server exists so the ~3s codec-load cost is paid ONCE, not on
# every benchmark/CLI process start -- point clients at it with
# --codec-base-url so repeated/scripted runs don't reload it each time.
#
# Usage:
#   ./run.sh                              # start both servers, block in foreground
#   ./run.sh --benchmark                  # start both servers, wait for them, run a
#                                          # default benchmark sweep against both,
#                                          # then leave the servers running
#   ./run.sh --benchmark --stop-after     # same, but stop both servers when the
#                                          # benchmark finishes
#   ./run.sh --config-test                # start both servers, wait for them, run
#                                          # miotts.config_test (cross-language audio
#                                          # sanity check), then leave servers running
#   ./run.sh --no-codec-server            # only start vllm; benchmark/CLI will load
#                                          # the codec locally per-invocation instead
#   ./run.sh --ws                         # also start the WebSocket TTS server
#                                          # (miotts.ws_server) for low-latency
#                                          # repeated-request use
#   ./run.sh --restart --config-test      # kill any already-running servers first,
#                                          # then start fresh instances and run the
#                                          # test -- use this after changing config.py
#                                          # or any server code, since a running
#                                          # process won't pick up code changes
#
# Extra args after `--` are forwarded to `python3 -m miotts.benchmark` (or
# miotts.config_test, if --config-test was given), e.g.:
#   ./run.sh --benchmark -- --batch-sizes 50 100 --languages hindi telugu
#   ./run.sh --restart --config-test -- --categories collections --reference-audio ref_audio/avira.mp3

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2560}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
PORT="${PORT:-8000}"
CODEC_PORT="${CODEC_PORT:-8001}"
WS_PORT="${WS_PORT:-8765}"
LOG_FILE="${LOG_FILE:-/tmp/vllm_server.log}"
CODEC_LOG_FILE="${CODEC_LOG_FILE:-/tmp/codec_server.log}"
WS_LOG_FILE="${WS_LOG_FILE:-/tmp/ws_server.log}"

RUN_BENCHMARK=false
RUN_CONFIG_TEST=false
STOP_AFTER=false
USE_CODEC_SERVER=true
RUN_WS_SERVER=false
RESTART=false
BENCHMARK_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) RUN_BENCHMARK=true; shift ;;
    --config-test) RUN_CONFIG_TEST=true; shift ;;
    --stop-after) STOP_AFTER=true; shift ;;
    --no-codec-server) USE_CODEC_SERVER=false; shift ;;
    --ws) RUN_WS_SERVER=true; shift ;;
    --restart) RESTART=true; shift ;;
    --) shift; BENCHMARK_ARGS=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "$RESTART" == true ]]; then
  echo "Restarting: stopping any existing vllm/codec/ws servers ..."
  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "miotts.codec_server" 2>/dev/null || true
  pkill -9 -f "miotts.ws_server" 2>/dev/null || true
  # vllm's engine core runs as a multiprocessing worker subprocess that does
  # NOT have "vllm serve" in its argv, so the pkill above misses it -- it
  # survives holding GPU memory, the wait-loop below times out without ever
  # seeing memory free, and the new instance fails with "No available memory
  # for the cache blocks." Kill anything running out of .venv_vllm's
  # interpreter too (safe: nothing else should run from that venv).
  pkill -9 -f "\.venv_vllm/bin/python3" 2>/dev/null || true
  # Wait for GPU memory to actually free before starting a new vllm instance,
  # otherwise --gpu-memory-utilization can be computed against memory that's
  # still being released by the just-killed process(es).
  for _ in $(seq 1 30); do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [[ "${used_mib:-0}" -lt 500 ]]; then
      break
    fi
    sleep 1
  done
  used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [[ "${used_mib:-0}" -ge 500 ]]; then
    echo "WARNING: GPU still shows ${used_mib}MiB used after stopping servers -- a stray process may remain (check: nvidia-smi, ps aux | grep venv_vllm)." >&2
  fi
  echo "Stopped."
fi

if pgrep -f "vllm serve" > /dev/null; then
  echo "A vllm server is already running (pid $(pgrep -f 'vllm serve' | head -1)). Not starting another."
else
  echo "Starting vllm server (max-model-len=$MAX_MODEL_LEN, gpu-memory-utilization=$GPU_MEM_UTIL, port=$PORT) ..."
  source .venv_vllm/bin/activate
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  setsid nohup vllm serve SPRINGLab/Indic-Mio \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port "$PORT" \
    > "$LOG_FILE" 2>&1 < /dev/null &
  disown -a
  deactivate

  echo "Waiting for vllm server startup (log: $LOG_FILE) ..."
  until grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; do
    if grep -qE "Traceback|EngineCore failed to start" "$LOG_FILE" 2>/dev/null; then
      echo "vllm server failed to start -- see $LOG_FILE" >&2
      exit 1
    fi
    sleep 2
  done
  echo "vllm server is up on port $PORT."
fi

if [[ "$USE_CODEC_SERVER" == true ]]; then
  if pgrep -f "miotts.codec_server" > /dev/null; then
    echo "A codec server is already running (pid $(pgrep -f 'miotts.codec_server' | head -1)). Not starting another."
  else
    echo "Starting codec server (port=$CODEC_PORT) ..."
    source .venv/bin/activate
    setsid nohup python3 -m miotts.codec_server --port "$CODEC_PORT" \
      > "$CODEC_LOG_FILE" 2>&1 < /dev/null &
    disown -a
    deactivate

    echo "Waiting for codec server startup (log: $CODEC_LOG_FILE) ..."
    until grep -q "Application startup complete" "$CODEC_LOG_FILE" 2>/dev/null; do
      if grep -qE "Traceback" "$CODEC_LOG_FILE" 2>/dev/null; then
        echo "codec server failed to start -- see $CODEC_LOG_FILE" >&2
        exit 1
      fi
      sleep 2
    done
    echo "codec server is up on port $CODEC_PORT."
  fi
fi

if [[ "$RUN_WS_SERVER" == true ]]; then
  if pgrep -f "miotts.ws_server" > /dev/null; then
    echo "A WS server is already running (pid $(pgrep -f 'miotts.ws_server' | head -1)). Not starting another."
  else
    echo "Starting WS server (port=$WS_PORT) ..."
    source .venv/bin/activate
    setsid nohup python3 -m miotts.ws_server --port "$WS_PORT" \
      --vllm-base-url "http://localhost:$PORT" --codec-base-url "http://localhost:$CODEC_PORT" \
      > "$WS_LOG_FILE" 2>&1 < /dev/null &
    disown -a
    deactivate

    echo "Waiting for WS server startup (log: $WS_LOG_FILE) ..."
    until grep -q "TTS WebSocket server ready" "$WS_LOG_FILE" 2>/dev/null; do
      if grep -qE "Traceback" "$WS_LOG_FILE" 2>/dev/null; then
        echo "WS server failed to start -- see $WS_LOG_FILE" >&2
        exit 1
      fi
      sleep 2
    done
    echo "WS server is up on port $WS_PORT."
  fi
fi

if [[ "$RUN_BENCHMARK" == true ]]; then
  echo "Running benchmark (from .venv) ..."
  source .venv/bin/activate
  if [[ ${#BENCHMARK_ARGS[@]} -eq 0 ]]; then
    BENCHMARK_ARGS=(--batch-sizes 50 100 --languages english hindi telugu --backend vllm --concurrency 10 --vllm-base-url "http://localhost:$PORT")
    if [[ "$USE_CODEC_SERVER" == true ]]; then
      BENCHMARK_ARGS+=(--codec-base-url "http://localhost:$CODEC_PORT")
    fi
  fi
  python3 -m miotts.benchmark "${BENCHMARK_ARGS[@]}"
  deactivate

  if [[ "$STOP_AFTER" == true ]]; then
    echo "Stopping servers ..."
    pkill -f "vllm serve" || true
    pkill -f "miotts.codec_server" || true
    pkill -f "miotts.ws_server" || true
  fi
fi

if [[ "$RUN_CONFIG_TEST" == true ]]; then
  echo "Running config_test (from .venv) ..."
  source .venv/bin/activate
  CONFIG_TEST_ARGS=("${BENCHMARK_ARGS[@]}")
  if [[ ${#CONFIG_TEST_ARGS[@]} -eq 0 ]]; then
    CONFIG_TEST_ARGS=(--backend vllm --vllm-base-url "http://localhost:$PORT")
    if [[ "$USE_CODEC_SERVER" == true ]]; then
      CONFIG_TEST_ARGS+=(--codec-base-url "http://localhost:$CODEC_PORT")
    fi
  fi
  python3 -m miotts.config_test "${CONFIG_TEST_ARGS[@]}"
  deactivate

  if [[ "$STOP_AFTER" == true ]]; then
    echo "Stopping servers ..."
    pkill -f "vllm serve" || true
    pkill -f "miotts.codec_server" || true
    pkill -f "miotts.ws_server" || true
  fi
fi
