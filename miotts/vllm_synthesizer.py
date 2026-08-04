"""vLLM-backed synthesis client for Indic-Mio.

Talks to a `vllm serve` OpenAI-compatible endpoint for the LLM token-generation
step (typically 3-4x lower single-request latency and much higher concurrent
throughput than the in-process transformers path in synthesizer.py, thanks to
vLLM's continuous batching).

For the codec (speech tokens + voice embedding -> waveform) step, two modes:
  - `codec_base_url` given: talk to a persistent miotts.codec_server over HTTP
    too -- no local torch/miocodec import at all, no per-invocation codec
    load. Use this for repeated/scripted runs (benchmarks, batch jobs) where
    paying the ~3s codec load cost on every process start is wasteful.
  - `codec_base_url` omitted (default): load the codec locally via
    VoiceResolver, same as before -- simplest for one-off scripts.

Requires a running vLLM server, in its OWN venv (its torch/transformers pins
are incompatible with the rest of this project -- see commands.md for the
exact versions and setup steps that were verified to work):

    source .venv_vllm/bin/activate
    vllm serve SPRINGLab/Indic-Mio --max-model-len 2560 --gpu-memory-utilization 0.6

Then, from the *regular* project venv:

    from miotts.vllm_synthesizer import VLLMIndicMioSynthesizer
    synth = VLLMIndicMioSynthesizer().load()
    waveform = synth.synthesize("Hello there")

Or, with the persistent codec server also running (`python3 -m miotts.codec_server`):

    synth = VLLMIndicMioSynthesizer(codec_base_url="http://localhost:8001").load()
    waveform = synth.synthesize("Hello there")
"""

import base64
import json
import re
import time
from dataclasses import dataclass, field

import numpy as np
import requests
import torch

from .config import MioConfig
from .postprocess import smooth_glitches
from .voice import VoiceResolver

SPEECH_TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")


@dataclass
class SynthesisTiming:
    """Per-request streaming timing, in seconds, measured from just before the
    HTTP request is sent."""

    ttft: float  # time to first speech-token chunk from vLLM
    chunk_times: list[float] = field(default_factory=list)  # arrival time of each chunk
    llm_time: float = 0.0  # time from request start to the last streamed chunk
    codec_time: float = 0.0  # time spent in embedding-resolve + decode
    total_time: float = 0.0  # end-to-end, request start to waveform returned


class VLLMIndicMioSynthesizer:
    """Same public interface as IndicMioSynthesizer, backed by a vLLM HTTP server
    (and optionally a persistent codec HTTP server -- see module docstring)."""

    def __init__(
        self,
        config: MioConfig | None = None,
        base_url: str = "http://localhost:8000",
        codec_base_url: str | None = None,
    ):
        self.config = config or MioConfig()
        self.base_url = base_url.rstrip("/")
        self.codec_base_url = codec_base_url.rstrip("/") if codec_base_url else None
        self._voice = None if self.codec_base_url else VoiceResolver(self.config)

    def load(self):
        if self._voice is not None:
            self._voice.load_codec()
        return self

    @property
    def is_loaded(self) -> bool:
        return self.codec_base_url is not None or self._voice._codec is not None

    def load_preset_embedding(self, preset_id: str):
        if self._voice is not None:
            return self._voice.load_preset_embedding(preset_id)
        raise RuntimeError("load_preset_embedding() requires a local codec; use resolve_embedding() with codec_base_url set")

    def embedding_from_reference_audio(self, audio_path: str):
        if self._voice is not None:
            return self._voice.embedding_from_reference_audio(audio_path)
        raise RuntimeError("embedding_from_reference_audio() requires a local codec; use resolve_embedding() with codec_base_url set")

    def _resolve_embedding_remote(self, voice_preset, reference_audio, timeout):
        response = requests.post(
            f"{self.codec_base_url}/resolve_embedding",
            json={"voice_preset": voice_preset, "reference_audio": reference_audio},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def _decode_remote(self, audio_codes, embedding, timeout):
        response = requests.post(
            f"{self.codec_base_url}/decode",
            json={"audio_codes": audio_codes, "embedding": embedding},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        waveform_bytes = base64.b64decode(payload["waveform_b64"])
        waveform = np.frombuffer(waveform_bytes, dtype=np.float32).copy()
        return torch.from_numpy(waveform)

    def _stream_speech_tokens(self, text: str, timeout: float):
        """POSTs a streaming chat completion and yields (chunk_text, elapsed_s)
        for each SSE chunk as it arrives, elapsed_s measured from just before
        the request is sent."""
        cfg = self.config
        start = time.perf_counter()
        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": cfg.model_name,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "repetition_penalty": cfg.repetition_penalty,
                "stream": True,
            },
            timeout=timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content, time.perf_counter() - start

    def synthesize(
        self,
        text: str,
        voice_preset: str | None = None,
        reference_audio: str | None = None,
        global_embedding=None,
        timeout: float = 60.0,
        smooth: bool | None = None,
        return_timing: bool = False,
        on_chunk=None,
    ):
        """Generate speech audio for `text` via the vLLM server, streaming the
        response so time-to-first-token (TTFT) and per-chunk arrival times can
        be measured. Returns a 1-D float waveform tensor at cfg.sample_rate,
        or (waveform, SynthesisTiming) if `return_timing=True`. Voice selection
        matches IndicMioSynthesizer.synthesize() (see its docstring for priority
        order).

        `smooth` (default `config.smooth_glitches`) crossfades over isolated codec
        glitches after decode -- see postprocess.py.

        `on_chunk`, if given, is called as `on_chunk(chunk_index, elapsed_s)` the
        moment each speech-token chunk arrives from vLLM -- e.g. for live
        per-chunk logging. Called synchronously from this (blocking) method.
        """
        if not self.is_loaded:
            self.load()

        cfg = self.config
        request_start = time.perf_counter()
        content = ""
        ttft = None
        chunk_times = []
        for chunk_text, elapsed in self._stream_speech_tokens(text, timeout):
            if ttft is None:
                ttft = elapsed
            chunk_times.append(elapsed)
            if on_chunk is not None:
                on_chunk(len(chunk_times) - 1, elapsed)
            content += chunk_text
        llm_time = time.perf_counter() - request_start

        audio_codes = [int(m) for m in SPEECH_TOKEN_RE.findall(content)]
        if not audio_codes:
            raise RuntimeError(
                "No speech tokens were generated for the given text; try again or "
                "increase max_new_tokens."
            )

        codec_start = time.perf_counter()
        if self.codec_base_url:
            if global_embedding is not None:
                embedding_list = global_embedding.tolist() if hasattr(global_embedding, "tolist") else list(global_embedding)
            else:
                embedding_list = self._resolve_embedding_remote(voice_preset, reference_audio, timeout)
            waveform = self._decode_remote(audio_codes, embedding_list, timeout)
        else:
            embedding = self._voice.resolve(voice_preset, reference_audio, global_embedding)
            waveform = self._voice.decode(audio_codes, embedding)
        codec_time = time.perf_counter() - codec_start

        do_smooth = cfg.smooth_glitches if smooth is None else smooth
        if do_smooth:
            waveform = torch.from_numpy(smooth_glitches(waveform.numpy(), cfg.sample_rate))

        if not return_timing:
            return waveform

        timing = SynthesisTiming(
            ttft=ttft or 0.0,
            chunk_times=chunk_times,
            llm_time=llm_time,
            codec_time=codec_time,
            total_time=time.perf_counter() - request_start,
        )
        return waveform, timing
