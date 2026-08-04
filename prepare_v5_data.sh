#!/usr/bin/env bash
# Runs all 9 language prepare-hf steps for the v5 training data in one shot.
# A failed language is SKIPPED (not fatal) -- the script keeps going and
# prints a summary of which languages succeeded/failed at the end, so one
# bad dataset/token doesn't block preparing the rest.
#
# Requires HF_TOKEN to be exported before running (these are gated datasets):
#   export HF_TOKEN=<your Shubhangi7-scoped token>
#   export HF_TOKEN_KANNADA=<token with access to MeghanaKap/kannada_dataset>  # different owner, may need a different token
#   ./prepare_v5_data.sh                # prep only
#   ./prepare_v5_data.sh --then-train   # prep, then auto-launch v5 training (only if ALL 9 succeeded)
#
# Re-running is safe: prepare-hf skips files that already exist in the cache
# dir unless --overwrite is passed (add OVERWRITE=1 ./prepare_v5_data.sh to
# force re-encoding everything).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate

THEN_TRAIN=0
if [[ "${1:-}" == "--then-train" ]]; then
  THEN_TRAIN=1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set -- export it first (these are gated datasets)." >&2
  exit 1
fi

# miotts/__init__.py sets HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 by default
# (via os.environ.setdefault, for fully-offline inference) -- override that
# here since prepare-hf needs actual network access to stream these datasets.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

declare -A LANG_DATASETS=(
  [assamese]="Shubhangi7/assamese-tts-elevenlabs"
  [gujarati]="Shubhangi7/gujarati-tts-elevenlabs"
  [hindi]="Shubhangi7/hindi_english_combined_hf_dataset"
  [kannada]="MeghanaKap/kannada_dataset"
  [malayalam]="Shubhangi7/malayalam-tts-elevenlabs"
  [marathi]="Shubhangi7/marathi-tts-elevenlabs"
  [punjabi]="Shubhangi7/punjabi-tts-elevenlabs"
  [tamil]="Shubhangi7/tamil-tts-elevenlabs"
  [telugu]="Shubhangi7/telugu-tts-elevenlabs"
)

# MeghanaKap/kannada_dataset is under a different HF account than the
# Shubhangi7/* datasets, so it may need its own token. Set HF_TOKEN_KANNADA
# if $HF_TOKEN doesn't have access to it; falls back to HF_TOKEN otherwise.
declare -A LANG_TOKEN_VARS=(
  [kannada]="HF_TOKEN_KANNADA"
)

# Fixed order so output is deterministic (bash assoc arrays don't preserve
# insertion order).
LANGS=(assamese gujarati hindi kannada malayalam marathi punjabi tamil telugu)

FAILED_LANGS=()
OK_LANGS=()

for lang in "${LANGS[@]}"; do
  dataset="${LANG_DATASETS[$lang]}"
  token_var="${LANG_TOKEN_VARS[$lang]:-}"
  lang_token="$HF_TOKEN"
  if [[ -n "$token_var" && -n "${!token_var:-}" ]]; then
    lang_token="${!token_var}"
    echo ""
    echo "=== [$lang] $dataset -> data/cache_v5/$lang (using \$$token_var) ==="
  else
    echo ""
    echo "=== [$lang] $dataset -> data/cache_v5/$lang ==="
  fi

  HF_TOKEN="$lang_token" python3 -m miotts.train prepare-hf \
    --dataset "$dataset" \
    --cache-dir "data/cache_v5/$lang" \
    --id-prefix "$lang" \
    $OVERWRITE_FLAG
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "!!! [$lang] prepare-hf failed (exit $status) -- skipping, continuing with remaining languages." >&2
    FAILED_LANGS+=("$lang")
  else
    OK_LANGS+=("$lang")
  fi
done

echo ""
echo "Cache counts:"
CACHE_DIRS=()
for lang in "${LANGS[@]}"; do
  count=$(find "data/cache_v5/$lang" -name "*.pt" 2>/dev/null | wc -l)
  echo "  $lang: $count cached examples"
  if [[ "$count" -gt 0 ]]; then
    CACHE_DIRS+=("data/cache_v5/$lang")
  fi
done

echo ""
if [[ ${#FAILED_LANGS[@]} -eq 0 ]]; then
  echo "All ${#LANGS[@]} languages prepared successfully."
else
  echo "Failed/skipped languages (${#FAILED_LANGS[@]}): ${FAILED_LANGS[*]}"
  echo "Succeeded (${#OK_LANGS[@]}): ${OK_LANGS[*]}"
fi

if [[ "$THEN_TRAIN" -eq 0 ]]; then
  exit 0
fi

if [[ ${#FAILED_LANGS[@]} -ne 0 ]]; then
  echo "" >&2
  echo "Refusing to auto-launch training: ${FAILED_LANGS[*]} failed/produced no data." >&2
  echo "Fix the failing language(s) above, then re-run with --then-train." >&2
  exit 1
fi

echo ""
echo "=== All languages have cached data -- launching v5 training ==="
# Restore offline mode for training itself (loads the model from already-
# cached weights; no network needed here -- prepare-hf above needed it off,
# training doesn't).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 -m miotts.train train \
  --manifest data/train.jsonl \
  --cache-dir "${CACHE_DIRS[@]}" \
  --init-model runs/full-ft-v3 \
  --output-dir runs/full-ft-v5 \
  --full-finetune \
  --epochs 3 \
  --push-to-hub-repo MeghanaKap/miomio_cp5_public \
  --push-every-steps 5000 \
  --sweep-out-dir outputs/full-ft-v5-sweeps
