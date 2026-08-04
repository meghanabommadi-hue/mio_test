# Hosting Indic-Mio

Two ways to host the TTS model, in order of setup effort:

1. **Simple API server** (`miotts.api_server`) — one process, in-process
   transformers backend, no vLLM required. Good for low/occasional traffic or
   a quick demo.
2. **vLLM stack** (`vllm serve` + `miotts.codec_server` + optional
   `miotts.ws_server`) — three processes, ~3.5x lower latency and much higher
   throughput under concurrency. Use this for production/call-center-scale
   traffic.

Both paths need the model weights and `MioCodec` reachable (see `miotts/config.py`
for `model_name`/`codec_name`) and a GPU (`config.device`, default `cuda`;
verified on an H100 80GB).

See `commands.md` for the full CLI flag reference for every command below.

## 0. One-time environment setup

Run once per machine:

```bash
cd /home/jovyan/miotts
./setup.sh
```

This creates both venvs this repo needs and installs their pinned
dependencies:

- `.venv` (Python **3.12** — required by `miocodec`) — runs the CLI,
  benchmark, the simple API server, the codec server, and the WS server.
- `.venv_vllm` (Python **3.10**, pinned `vllm==0.8.5`/`transformers==4.51.3` —
  see `requirements-vllm.txt` for exactly why those pins) — runs only the
  `vllm serve` process.

If `python3.12` or `python3.10` aren't installed system-wide, `setup.sh` prints
the `apt`/`deadsnakes` command to install them and exits — install those first,
then re-run.

Copy `.env.example` to `.env` and adjust ports/paths if the defaults don't fit
your host:

```bash
cp .env.example .env
```

## 1. Simple API server

```bash
source .venv/bin/activate
python3 -m miotts.api_server --port 8080
```

Endpoints:

| Method | Path      | Description                                                        |
|--------|-----------|----------------------------------------------------------------------|
| GET    | `/`       | Demo web page (text box + Speak button)                              |
| GET    | `/health` | `{"status": "ok", "loaded": bool}`                                    |
| POST   | `/tts`    | `{text, voice_preset?, reference_audio?}` -> `audio/wav` bytes        |

```bash
curl -s -X POST http://localhost:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, how are you today?"}' \
  -o out.wav
```

Model + codec load once at startup (a minute or two); the process then serves
requests sequentially in-process (no batching).

### Run it as a systemd service

```bash
sudo cp deploy/systemd/mio-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mio-api
sudo journalctl -u mio-api -f   # logs / startup progress
```

## 2. vLLM stack (production)

Three long-lived processes: `vllm serve` (LLM: text -> speech tokens),
`miotts.codec_server` (MioCodec: speech tokens -> waveform, loaded once so
its ~3s load cost isn't paid per request), and optionally `miotts.ws_server`
(persistent low-latency WebSocket front door for repeated/live-call use).

### Quickest: `run.sh`

```bash
cd /home/jovyan/miotts
./run.sh              # start vllm + codec servers, block in foreground
./run.sh --ws          # also start the WebSocket server
```

`run.sh` detects already-running servers (won't double start), and honors
`MAX_MODEL_LEN` / `GPU_MEM_UTIL` / `PORT` / `CODEC_PORT` / `WS_PORT` env vars.
See `commands.md`'s "Faster inference with vLLM" section for the full flag
set, manual (non-`run.sh`) startup commands, and the `repetition_penalty`
stability note.

### Production: systemd

For a host that should keep these processes running across reboots/crashes,
use the provided unit files instead of `run.sh` (systemd, not `run.sh`'s
`nohup`/`disown`, owns process supervision/restart here):

```bash
sudo cp deploy/systemd/mio-vllm.service deploy/systemd/mio-codec.service deploy/systemd/mio-ws.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mio-vllm mio-codec
sudo systemctl enable --now mio-ws     # optional, needs mio-vllm + mio-codec up first
```

Defaults (`MAX_MODEL_LEN=2560`, `GPU_MEM_UTIL=0.6`, `PORT=8000`,
`CODEC_PORT=8001`, `WS_PORT=8765`) are baked into the unit files and can be
overridden by uncommenting the matching line in `/home/jovyan/miotts/.env`
(loaded via `EnvironmentFile=`) — re-run `daemon-reload` + `restart` after
editing.

Verify:

```bash
curl -s http://localhost:8000/v1/models   # vllm
curl -s http://localhost:8001/health      # codec server
curl -s http://localhost:8765/health      # ws server, if enabled
```

Stop / restart:

```bash
sudo systemctl stop mio-ws mio-vllm mio-codec
sudo systemctl restart mio-vllm   # e.g. after changing config.py -- a running
                                   # process won't pick up code changes
```

### Talking to the stack from code

```python
from miotts.vllm_synthesizer import VLLMIndicMioSynthesizer
import soundfile as sf

synth = VLLMIndicMioSynthesizer(codec_base_url="http://localhost:8001").load()
waveform = synth.synthesize("Hello, how are you today?")
sf.write("out.wav", waveform.numpy(), synth.config.sample_rate)
```

`VLLMIndicMioSynthesizer` has the same `synthesize(text, voice_preset=,
reference_audio=, global_embedding=)` signature as the in-process
`IndicMioSynthesizer` used by the simple API server — swapping backends
doesn't change caller code.

### WebSocket protocol (`mio-ws` / `miotts.ws_server`)

One JSON request per message over a persistent connection; response is a JSON
metadata message followed by the raw WAV as a binary frame. **Not**
chunked-audio streaming — see `miotts/ws_server.py`'s module docstring for why
MioCodec's decoder rules that out. Full protocol + example client in
`commands.md`'s "Low-latency WebSocket server" section.

## Choosing between the two paths

| | Simple API server | vLLM stack |
|---|---|---|
| Processes | 1 | 2-3 |
| Setup | `./setup.sh` + `.venv` only | `./setup.sh` (both venvs) |
| Single-request latency | ~1.0-1.2s | ~0.32s |
| Concurrency | sequential | ~35 req/s @ 20 concurrent |
| Best for | demos, low traffic | production, call-center scale |

## Troubleshooting

- **vLLM won't start / "maximum context length" 400s**: `--max-model-len` must
  exceed `config.max_new_tokens` (1024) plus your longest prompt's token
  count.
- **GPU memory still held after stopping servers**: vLLM's engine core runs
  as a subprocess with no `vllm serve` in its argv, so a plain `kill`/`systemctl
  stop` of the parent can leave it running. `run.sh --restart` and
  `test_ephemeral.sh` both handle this (kill anything from `.venv_vllm`'s
  interpreter); do the same manually if needed:
  `pkill -9 -f "\.venv_vllm/bin/python3"`.
- **Runaway ~6s responses instead of ~0.2-0.6s**: the model occasionally fails
  to emit its end token. Don't change `config.py`'s `repetition_penalty` (1.1)
  without re-reading its inline comment — this was empirically swept and is
  not a "more is safer" knob.
