"""WebSocket TTS server for low-latency, repeated-request use (e.g. live calls).

Modeled on FlowTTS's server.py pattern (persistent per-call WebSocket, warmup on
startup, TTFT/latency logging) but scoped down for this repo's actual
capabilities:
  - Full-response synthesis only. MioCodec's decode() runs one forward pass
    over the whole token sequence with no incremental/causal mode -- an
    empirical test showed decoding a token prefix produces meaningfully
    different audio for those same tokens than decoding the full sequence
    (max abs waveform diff ~0.66 on a ~[-1,1] scale in the "overlapping"
    region). So unlike FlowTTS's Mira backend, there is no safe chunked-audio
    streaming path here -- don't try to bolt one on without re-verifying the
    codec's behavior first.
  - No OOM auto-recovery, no Prometheus metrics, no per-voice wav caching, no
    dynamic multi-port binding, no connection reaping. Add these later if the
    need actually shows up; they aren't free and this server doesn't have
    FlowTTS's production history to justify them yet.

Uses VLLMIndicMioSynthesizer against an already-running vLLM server + codec
server (see commands.md / run.sh) -- this process itself loads no model
weights, so it stays lightweight and starts instantly.

Protocol (one JSON message in, one JSON message + binary wav out, per request):
  Client -> {"text": "...", "call_id"?: "...", "voice_preset"?: "...",
             "reference_audio"?: "..."}
  Server -> {"type": "audio", "call_id", "sample_rate", "wav_bytes": N,
             "ttft_ms", "llm_ms", "codec_time_ms", "total_ms", "n_chunks"}
            followed immediately by the raw WAV bytes as a binary frame.
  On error -> {"type": "error", "call_id", "error": "..."}

Usage:
    python3 -m miotts.ws_server --port 8765 \
        --vllm-base-url http://localhost:8000 --codec-base-url http://localhost:8001
"""

import argparse
import asyncio
import io
import json
import time

import soundfile as sf
import websockets

from .config import MioConfig
from .vllm_synthesizer import VLLMIndicMioSynthesizer

_WARMUP_SENTENCES = [
    "Hello, how are you today?",
    "नमस्ते, आप कैसे हैं?",
    "నమస్కారం, మీరు ఎలా ఉన్నారు?",
]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _waveform_to_wav_bytes(waveform, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, waveform.numpy(), sample_rate, format="WAV")
    return buf.getvalue()


class TTSServer:
    def __init__(self, synth: VLLMIndicMioSynthesizer, log_chunks: bool = False):
        self.synth = synth
        self.log_chunks = log_chunks

    async def warmup(self):
        print(f"[{_ts()}] warming up ({len(_WARMUP_SENTENCES)} sentences)...", flush=True)
        t0 = time.perf_counter()
        for sentence in _WARMUP_SENTENCES:
            try:
                await asyncio.to_thread(self.synth.synthesize, sentence)
            except Exception as e:
                print(f"[{_ts()}] warmup failed for {sentence!r}: {e}", flush=True)
        print(f"[{_ts()}] warmup done ({(time.perf_counter() - t0) * 1000:.0f}ms)", flush=True)

    async def handle_request(self, ws, data: dict):
        call_id = data.get("call_id") or "unknown"
        text = (data.get("text") or "").strip()
        if not text:
            await ws.send(json.dumps({"type": "error", "call_id": call_id, "error": "Missing text"}))
            return

        # `language` (e.g. "english"/"hindi"/"telugu") is accepted purely for
        # client-side bookkeeping/logging, same non-functional role it plays in
        # benchmark.py/config_test.py (SAMPLE_TEXTS[category][language]) --
        # Indic-Mio has no language parameter or in-text tag; it infers
        # language from the text's own script (see its model card's
        # "Prompting" section, which documents only emotion tags like
        # <happy>/<angry>, not a language mechanism). Passing it here does
        # NOT change synthesis; it's echoed back so callers can correlate
        # requests/responses when running mixed-language batches.
        language = data.get("language")

        voice_preset = data.get("voice_preset")
        reference_audio = data.get("reference_audio")

        on_chunk = None
        if self.log_chunks:
            def on_chunk(chunk_index, elapsed):
                print(f"[{_ts()}] {call_id}  chunk {chunk_index}: {elapsed * 1000:.1f}ms", flush=True)

        t0 = time.perf_counter()
        try:
            waveform, timing = await asyncio.to_thread(
                self.synth.synthesize,
                text,
                voice_preset=voice_preset,
                reference_audio=reference_audio,
                return_timing=True,
                on_chunk=on_chunk,
            )
        except Exception as e:
            print(f"[{_ts()}] {call_id}  ERROR: {e}", flush=True)
            await ws.send(json.dumps({"type": "error", "call_id": call_id, "error": str(e)}))
            return

        wav_bytes = await asyncio.to_thread(_waveform_to_wav_bytes, waveform, self.synth.config.sample_rate)
        total_ms = round((time.perf_counter() - t0) * 1000)

        await ws.send(json.dumps({
            "type": "audio",
            "call_id": call_id,
            "language": language,
            "sample_rate": self.synth.config.sample_rate,
            "wav_bytes": len(wav_bytes),
            "ttft_ms": round(timing.ttft * 1000, 1),
            "llm_ms": round(timing.llm_time * 1000, 1),
            "codec_time_ms": round(timing.codec_time * 1000, 1),
            "total_ms": total_ms,
            "n_chunks": len(timing.chunk_times),
        }))
        await ws.send(wav_bytes)

        print(
            f"[{_ts()}] {call_id}  language={language}  ttft={timing.ttft*1000:.1f}ms "
            f"llm={timing.llm_time*1000:.1f}ms codec={timing.codec_time*1000:.1f}ms "
            f"total={total_ms}ms n_chunks={len(timing.chunk_times)} "
            f"text={text[:50]!r}",
            flush=True,
        )

    async def handle_connection(self, ws):
        peer = ws.remote_address
        print(f"[{_ts()}] connected  peer={peer}", flush=True)
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send(json.dumps({"type": "error", "error": "Invalid JSON"}))
                    continue
                await self.handle_request(ws, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"[{_ts()}] disconnected  peer={peer}", flush=True)


async def _process_request(connection, request):
    if request.path == "/health":
        from websockets.http11 import Response, Headers

        body = json.dumps({"status": "ok"}).encode()
        headers = Headers([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return Response(200, "OK", headers, body)
    return None


async def run_server(host: str, port: int, vllm_base_url: str, codec_base_url: str, log_chunks: bool = True):
    config = MioConfig()
    synth = VLLMIndicMioSynthesizer(config, base_url=vllm_base_url, codec_base_url=codec_base_url).load()
    server = TTSServer(synth, log_chunks=log_chunks)
    await server.warmup()

    async with websockets.serve(
        server.handle_connection,
        host,
        port,
        process_request=_process_request,
        ping_interval=30,
        ping_timeout=30,
    ):
        print(f"[{_ts()}] TTS WebSocket server ready: ws://{host}:{port}", flush=True)
        await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="miotts WebSocket TTS server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--vllm-base-url", default="http://localhost:8000")
    parser.add_argument("--codec-base-url", default="http://localhost:8001")
    parser.add_argument(
        "--no-log-chunks",
        action="store_true",
        help="Disable per-chunk arrival-time logging (on by default; verbose under heavy concurrent load)",
    )
    args = parser.parse_args()

    asyncio.run(run_server(args.host, args.port, args.vllm_base_url, args.codec_base_url, log_chunks=not args.no_log_chunks))


if __name__ == "__main__":
    main()
