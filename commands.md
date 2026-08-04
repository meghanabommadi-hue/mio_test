# miotts commands

All commands assume you're in the repo root with the venv active:

```bash
cd /home/jovyan/miotts
source .venv/bin/activate
```

## Synthesize speech (CLI)

```bash
python3 -m miotts.cli "Hello, how are you today?" -o outputs/hello.wav
```

Voice selection (in priority order): `--reference-audio` > `--voice-preset` > hardcoded
default (`ref_audio/avira.mp3`, set in `miotts/config.py`) > bundled `en_female` preset.

```bash
# clone a voice from a reference clip (wav/mp3, any sample rate)
python3 -m miotts.cli "नमस्ते ! यह call आपके loan के बारे में हैं ।" \
    --reference-audio ref_audio/avira.mp3 \
    -o outputs/avira_test.wav

# use a bundled preset instead (en_female, en_male, jp_female, jp_male)
python3 -m miotts.cli "Hello there" --voice-preset en_male -o outputs/male.wav

# emotion tag: append at the end of the sentence (Indic: happy/sad/angry/disgust/fear/surprise;
# English adds enunciated/confused/whisper)
python3 -m miotts.cli "This is a final notice on your loan. <angry>" -o outputs/angry.wav
```

Other flags: `--max-new-tokens` (default 1024), `--temperature` (0.9), `--top-p` (0.9).

## Benchmark

```bash
python3 -m miotts.benchmark --batch-sizes 10 --languages english hindi telugu --categories generic
```

Each run writes to its own timestamped folder by default:
`outputs/runs/<YYYYMMDD_HHMMSS>/benchmark_results.json` + `outputs/runs/<YYYYMMDD_HHMMSS>/wavs/`.
Override with `--run-dir`, or independently override `--output` / `--wavs-dir`.

- `--categories generic` sentences exist for: `english`, `hindi`, `telugu`.
- `--categories collections` (debt-collection call style) sentences exist for:
  `english`, `hindi`, `telugu`, `punjabi`, `marathi`.
- Passing a language that doesn't exist for a category raises `KeyError` — don't mix
  `punjabi`/`marathi` into `--categories generic`.

```bash
# generic category, its 3 supported languages
python3 -m miotts.benchmark --batch-sizes 10 --languages english hindi telugu --categories generic

# collections category, all 5 supported languages
python3 -m miotts.benchmark --batch-sizes 10 --languages english hindi telugu punjabi marathi --categories collections

# quick smoke test, no wav saving
python3 -m miotts.benchmark --batch-sizes 5 --languages english --no-save-wavs

# save every wav instead of one-per-unique-sentence
python3 -m miotts.benchmark --batch-sizes 20 --languages hindi --save-all-wavs
```

Reported metrics per language: avg/p50/p95 latency, throughput (req/s), and **RTF**
(wall time ÷ audio duration — lower is better; <1.0 is faster than real-time).

Note: benchmark requests use whatever voice `IndicMioSynthesizer.synthesize()` defaults
to (currently the hardcoded `ref_audio/avira.mp3` reference) since no preset/reference
override is threaded through `run_benchmark()`.

## Faster inference with vLLM

vLLM replaces the in-process transformers `.generate()` call with a served, continuously-
batched LLM backend. Measured on this repo's setup (H100, `SPRINGLab/Indic-Mio`): single-request
latency drops from ~1.0-1.2s to ~0.32s (~3.5x), and 20 concurrent requests hit ~35 req/s
throughput vs. ~1 req/s sequential with plain transformers.

A second, persistent **codec server** (`miotts/codec_server.py`) is also available so the
MioCodec decode step doesn't reload (~3s) on every single benchmark/CLI process start — both
servers load once and stay resident; clients talk to both over HTTP.

### Quick start: `run.sh`

```bash
cd /home/jovyan/miotts
./run.sh                          # start vllm server + codec server, block in foreground
./run.sh --benchmark              # start both (if not already up), run a default benchmark, leave them running
./run.sh --benchmark --stop-after # same, but tear both servers down afterward
./run.sh --benchmark -- --batch-sizes 50 100 --languages hindi telugu   # custom benchmark args
```

`run.sh` detects already-running servers and won't double-start them. Override
`MAX_MODEL_LEN` / `GPU_MEM_UTIL` / `PORT` / `CODEC_PORT` as env vars if needed, e.g.
`GPU_MEM_UTIL=0.8 ./run.sh`.

### Manual setup (what `run.sh` automates)

vLLM needs its own venv: it has strict torch/transformers pins that conflict with `.venv`'s.
The versions below (also captured in `requirements-vllm.txt`) are what was actually verified to
work in this environment (driver 550.127, CUDA 12.4 max) — vLLM's latest release pulls a torch
build requiring newer CUDA than this driver supports, so don't just `pip install vllm` on its own.

```bash
# one-time setup
cd /home/jovyan/miotts
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate
pip install -r requirements-vllm.txt
pip uninstall -y flashinfer-python  # ships a prebuilt .so with an ABI mismatch in this env; vllm falls back to its native sampler fine without it
```

Start the vLLM server (in `.venv_vllm`):

```bash
source .venv_vllm/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # fully offline -- weights are already cached locally
vllm serve SPRINGLab/Indic-Mio --max-model-len 2560 --gpu-memory-utilization 0.6 --port 8000
```

`--max-model-len` must exceed `config.max_new_tokens` (2048) plus your longest prompt's token
count, or requests 400 with "maximum context length" errors. Wait for `Application startup
complete` in the logs, then confirm it's up: `curl -s http://localhost:8000/v1/models`.

Start the codec server (in the **regular** `.venv` — needs `miocodec`, Python >=3.12, which
cannot be installed into `.venv_vllm`, Python 3.10):

```bash
source .venv/bin/activate
python3 -m miotts.codec_server --port 8001
```

Confirm it's up: `curl -s http://localhost:8001/health`.

### Using both from your own code

```bash
source .venv/bin/activate
python3 -c "
from miotts.vllm_synthesizer import VLLMIndicMioSynthesizer
import soundfile as sf

synth = VLLMIndicMioSynthesizer(codec_base_url='http://localhost:8001').load()
waveform = synth.synthesize('Hello, how are you today?')
sf.write('outputs/vllm_test.wav', waveform.numpy(), synth.config.sample_rate)
"
```

`VLLMIndicMioSynthesizer` has the same `synthesize(text, voice_preset=, reference_audio=,
global_embedding=)` signature and voice-priority order as `IndicMioSynthesizer` — it's a
drop-in swap. Pass `base_url=` for a non-default vLLM server location. Omit `codec_base_url`
to load the codec locally instead (simpler for one-off scripts, pays the ~3s load cost).

`miotts.benchmark --backend vllm` accepts `--vllm-base-url` and `--codec-base-url` the same way.

`synthesize()` also accepts `return_timing=True` (both backends), returning
`(waveform, SynthesisTiming)` with `.ttft` (time to first speech-token chunk),
`.chunk_times`, `.llm_time`, `.codec_time`, `.total_time` — useful for per-request
latency breakdowns without instrumenting your own timers.

### Generation stability: `repetition_penalty`

The model sometimes fails to emit its end token and keeps sampling speech tokens
until `max_new_tokens` (a ~6s, `max_new_tokens/25Hz`-length runaway instead of a
normal ~0.2-0.6s response). Empirically swept on 40+ trials across
english/hindi/telugu/punjabi, plain and code-mixed: `repetition_penalty=1.0` (off)
and `>=1.15` both hit this 10-100% of the time depending on text;
**`repetition_penalty=1.05-1.1`** (config default: `1.1`) eliminated it in every
trial. This is not a "more is safer" knob — `1.3` made it worse (8/8 runaways
in testing) — don't tune past ~1.1 without re-sweeping the way this was found
(`miotts/config.py`'s `repetition_penalty` field has the full note).

## Low-latency WebSocket server

For repeated/live-call-style use where per-request HTTP overhead and per-process
model loading matter, `miotts/ws_server.py` keeps a persistent WebSocket
connection open and proxies to the already-running vLLM + codec servers (it
loads no model weights itself, starts instantly).

**Important limitation**: this does NOT stream audio out in chunks as it's
generated. MioCodec's `decode()` has no incremental/causal mode — it's one
forward pass over the whole token sequence, and an empirical test found
decoding a token prefix produces meaningfully *different* audio for those same
tokens than decoding the full sequence (max abs waveform diff ~0.66 on a
~[-1,1] scale in the overlapping region) — not just a seam artifact, but
audibly different/re-warped audio. So unlike architectures with causal/windowed
codecs (e.g. FlowTTS's Mira backend), true chunked-audio streaming isn't safe
here without a different codec. This server streams the *text→speech-token*
step (hence TTFT is meaningful) but returns one complete WAV per request.

```bash
source .venv/bin/activate
python3 -m miotts.ws_server --port 8765 \
    --vllm-base-url http://localhost:8000 --codec-base-url http://localhost:8001
```

Or via `run.sh --ws` (starts/detects vllm + codec servers too):
```bash
./run.sh --ws
```

Protocol: send one JSON message per request over a persistent connection,
receive a JSON metadata message (ttft_ms/llm_ms/codec_time_ms/total_ms) followed
by the raw WAV as a binary frame:

```python
import asyncio, json, websockets

async def test():
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({"text": "Hello, how are you today?", "call_id": "1"}))
        meta = json.loads(await ws.recv())
        wav_bytes = await ws.recv()
        print(meta)

asyncio.run(test())
```

Health check: `GET http://localhost:8765/health`.

Stop all three servers when done (they hold GPU memory / stay resident the whole
time they're running):
```bash
pkill -f "vllm serve"
pkill -f "miotts.codec_server"
pkill -f "miotts.ws_server"
```

## Fine-tune (LoRA)

Two-stage: cache codec tokens once, then train against the cache.

```bash
# 1. encode a JSONL manifest of {"text": ..., "audio_path": ...} pairs to cached codec tokens
python3 -m miotts.train prepare --manifest data/train.jsonl --cache-dir data/cache

# 2. run LoRA SFT over the cache
python3 -m miotts.train train --manifest data/train.jsonl --cache-dir data/cache \
    --output-dir runs/lora-v1
```

`prepare` flags: `--device` (default cuda), `--overwrite` (re-encode existing cache entries).

`train` flags: `--epochs` (3.0), `--batch-size` (4), `--grad-accum` (4),
`--learning-rate` (2e-4), `--max-length` (1024), `--lora-r` (16), `--lora-alpha` (32),
`--lora-dropout` (0.05), `--no-lora-qk` (drop q_proj/k_proj from LoRA targets — workaround
if you hit shape/CUBLAS errors from Qwen3's QK-norm).

See `miotts/train.py` module docstring for manifest format details and the reasoning
behind the masking/token-mapping approach.

## Tests

```bash
pytest tests/
```
