#!/usr/bin/env bash
# One-time environment bootstrap: creates both venvs this repo needs and
# installs their pinned dependencies. See HOSTING.md / TRAINING.md for what
# each venv is used for.
#
#   .venv       (Python 3.12, required by miocodec) -- CLI, benchmark, the
#               simple API server, the codec server, the WS server.
#   .venv_vllm  (Python 3.10, pinned vllm/torch/transformers -- see
#               requirements-vllm.txt for why) -- runs only `vllm serve`.
#
# Usage:
#   ./setup.sh              # create/update both venvs
#   ./setup.sh --vllm-only  # skip .venv, only set up .venv_vllm
#   ./setup.sh --no-vllm    # skip .venv_vllm, only set up .venv

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SETUP_VENV=true
SETUP_VLLM=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-only) SETUP_VENV=false; shift ;;
    --no-vllm) SETUP_VLLM=false; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ "$SETUP_VENV" == true ]]; then
  if [[ ! -x .venv/bin/python3 ]]; then
    if ! command -v python3.12 >/dev/null; then
      echo "python3.12 not found. Install it first, e.g. (Debian/Ubuntu):" >&2
      echo "  sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update && sudo apt install -y python3.12 python3.12-venv" >&2
      exit 1
    fi
    echo "Creating .venv (python3.12) ..."
    python3.12 -m venv .venv
  fi
  echo "Installing requirements.txt into .venv ..."
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt
  echo ".venv ready."
fi

if [[ "$SETUP_VLLM" == true ]]; then
  if [[ ! -x .venv_vllm/bin/python3 ]]; then
    if ! command -v python3.10 >/dev/null; then
      echo "python3.10 not found. Install it first, e.g. (Debian/Ubuntu):" >&2
      echo "  sudo apt update && sudo apt install -y python3.10 python3.10-venv" >&2
      exit 1
    fi
    echo "Creating .venv_vllm (python3.10) ..."
    python3.10 -m venv .venv_vllm
  fi
  echo "Installing requirements-vllm.txt into .venv_vllm ..."
  .venv_vllm/bin/pip install --upgrade pip -q
  .venv_vllm/bin/pip install -r requirements-vllm.txt
  # Ships a prebuilt .so with an ABI mismatch in this environment; vllm falls
  # back to its native sampler fine without it (see requirements-vllm.txt).
  .venv_vllm/bin/pip uninstall -y flashinfer-python 2>/dev/null || true
  echo ".venv_vllm ready."
fi

echo ""
echo "Setup complete. Next steps:"
echo "  - Hosting the API:      see HOSTING.md"
echo "  - Fine-tuning the model: see TRAINING.md"
echo "  - cp .env.example .env  and adjust ports/paths if the defaults don't fit this host"
