# Training / fine-tuning Indic-Mio

Indic-Mio (`SPRINGLab/Indic-Mio`) is a `Qwen3ForCausalLM` speech-LM: speech is
just another token vocabulary appended to the text vocabulary
(ids `151669..164468`, one FSQ-quantized MioCodec token per 25Hz codec frame).
Fine-tuning it is therefore plain instruction-tuning SFT — no custom modeling
code — once audio has been converted to speech-code tokens. See
`miotts/train.py`'s module docstring for the full reasoning (chat template,
token-id mapping, why the codec's global/speaker embedding never enters
training).

Full flag reference for every command below is in `commands.md`'s
"Fine-tune (LoRA)" section and `miotts/train.py --help`. This doc covers the
setup + workflow.

## 0. Environment

```bash
cd /home/jovyan/miotts
./setup.sh              # one-time, creates .venv (see HOSTING.md)
source .venv/bin/activate
pip install -e ".[train]"    # installs peft, needed only for training
```

Training needs a GPU (`MioConfig.device`, default `cuda`; verified on an H100
80GB) for both the codec-encode ("prepare") step and the SFT loop itself.

## 1. Get data into a manifest

A JSONL manifest, one example per line:

```json
{"text": "Hello, how are you today?", "audio_path": "data/wavs/0001.wav"}
{"text": "नमस्ते, आप कैसे हैं?", "audio_path": "data/wavs/0002.wav"}
```

- `text` should already include any emotion tag the clip expresses (e.g.
  `"... <happy>"`) — these are plain text, not special tokens.
- `audio_path` is resolved relative to the manifest file's directory.

If your data lives in a Hugging Face dataset repo instead of local wavs, skip
the manifest and use `prepare-hf` (step 2b) directly against the repo id.

## 2. Prepare: encode audio to cached codec tokens

Encoding audio through MioCodec is the expensive, one-time step — cache its
output so re-running/resuming training doesn't re-encode audio. Each cached
`.pt` is just `{text, content_token_indices}` (a few KB); raw audio is never
kept on disk for the HF-streaming path.

### 2a. From a local manifest

```bash
python3 -m miotts.train prepare \
  --manifest data/train.jsonl \
  --cache-dir data/cache
```

### 2b. From a Hugging Face audio dataset (streamed, no local wavs)

```bash
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0   # miotts defaults to offline; override to stream
export HF_TOKEN=<token, only if the dataset repo is gated>

python3 -m miotts.train prepare-hf \
  --dataset Shubhangi7/marathi-tts-elevenlabs \
  --cache-dir data/cache_v5/marathi \
  --id-prefix marathi
```

Re-running is safe: existing cache entries are skipped unless `--overwrite`
is passed.

### 2c. Multiple languages in one shot

`prepare_v5_data.sh` runs all 9 language `prepare-hf` steps back to back,
skips (not fails) on a bad language/token, and prints a per-language success
summary:

```bash
export HF_TOKEN=<Shubhangi7-scoped token>
export HF_TOKEN_KANNADA=<token with access to MeghanaKap/kannada_dataset>  # different owner
./prepare_v5_data.sh                  # prep only
./prepare_v5_data.sh --then-train     # prep, then auto-launch training (only if all 9 succeeded)
```

## 3. Train

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # restore offline mode; training loads only cached weights

python3 -m miotts.train train \
  --manifest data/train.jsonl \
  --cache-dir data/cache \
  --output-dir runs/lora-v1
```

LoRA (default) vs. full fine-tune:

```bash
# LoRA adapter (default) -- lr 2e-4, r=16
python3 -m miotts.train train --manifest data/train.jsonl --cache-dir data/cache --output-dir runs/lora-v1

# full fine-tune -- lr 2e-5, updates every weight
python3 -m miotts.train train --manifest data/train.jsonl --cache-dir data/cache \
  --output-dir runs/full-ft-v1 --full-finetune
```

Merge multiple cache dirs into one training run (e.g. multi-language):

```bash
python3 -m miotts.train train --manifest data/train.jsonl \
  --cache-dir data/cache_v5/hindi data/cache_v5/telugu data/cache_v5/marathi \
  --output-dir runs/full-ft-v5 --full-finetune
```

Continue from a prior checkpoint instead of the base model:

```bash
python3 -m miotts.train train --manifest data/train.jsonl --cache-dir data/cache \
  --output-dir runs/full-ft-v4 \
  --init-model MeghanaKap/miomio_cp1_public --init-subfolder checkpoint-2740 \
  --full-finetune
```

Resume an interrupted run from a trainer checkpoint:

```bash
python3 -m miotts.train train --manifest data/train.jsonl --cache-dir data/cache \
  --output-dir runs/full-ft-v4 --resume-from-checkpoint runs/full-ft-v4/checkpoint-15000
```

If you hit shape/CUBLAS errors from Qwen3's QK-norm + LoRA on `q_proj`/`k_proj`,
add `--no-lora-qk` to fall back to `v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`
only.

### Pushing checkpoints to the Hub during training

For long runs, push+validate each checkpoint instead of only saving locally
(`runs/` is git-ignored and disk fills up fast with full-finetune checkpoints):

```bash
python3 -m miotts.train train --manifest data/train.jsonl --cache-dir data/cache \
  --output-dir runs/full-ft-v5 --full-finetune --epochs 3 \
  --push-to-hub-repo MeghanaKap/miomio_cp5_public \
  --push-every-steps 5000 \
  --sweep-out-dir outputs/full-ft-v5-sweeps
```

Each `on_save`: the checkpoint is uploaded to the given HF model repo under
`checkpoint-<step>`, a background "simran sweep" (`miotts/simran_sweep.py`)
generates sample audio from it into `<sweep-out-dir>_step<N>` for a quick
listen/regression check, and the local checkpoint copy is deleted afterward
either way (`save_total_limit=1` in `train.py` keeps at most one on disk at a
time regardless).

## 4. Validate a checkpoint

Point `MioConfig.model_name` (or `--init-model`) at the checkpoint and run the
usual synthesis checks:

```bash
python3 -m miotts.simran_sweep --model-path runs/full-ft-v5/checkpoint-20000 --out-dir outputs/manual-check
```

Or spin up the full stack against it and run `config_test`/`benchmark` — see
`commands.md` and `HOSTING.md`.

## Data layout reference

```
data/
  train.jsonl          # manifest (text, audio_path pairs) -- not committed, see .gitignore
  cache/                # prepare output for a single manifest
  cache_v5/<lang>/       # prepare-hf output, one dir per language
runs/
  <run-name>/            # train output-dir: adapter or full weights + tokenizer
  <run-name>/checkpoint-<step>.sweep.log   # background sweep logs (PushCheckpointToHubCallback)
```

`data/` and `runs/` are git-ignored (training caches and model weights are
large/regenerable) — push checkpoints to the Hub (above) rather than to git.
