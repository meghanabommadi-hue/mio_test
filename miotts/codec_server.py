"""Persistent MioCodec server so the codec loads once and stays resident,
instead of every benchmark/CLI invocation paying its ~3s load cost.

Complements the vLLM server: vLLM hosts the LLM (text -> speech tokens),
this hosts the codec (speech tokens + voice embedding -> waveform). Both are
started once and left running; VLLMIndicMioSynthesizer talks to both over
HTTP with no local torch/miocodec loading of its own.

Usage:
    source .venv/bin/activate   # needs miocodec -> Python >=3.12, i.e. NOT .venv_vllm
    python3 -m miotts.codec_server --port 8001

Endpoints:
    GET  /health
    POST /resolve_embedding {voice_preset?, reference_audio?} -> {embedding: [float]}
    POST /decode {audio_codes: [int], embedding: [float]}
         -> {waveform_b64: base64 float32 bytes, sample_rate: int}
"""

import argparse
import base64

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .config import MioConfig
from .voice import VoiceResolver

app = FastAPI()
_state = {}


class ResolveEmbeddingRequest(BaseModel):
    voice_preset: str | None = None
    reference_audio: str | None = None


class DecodeRequest(BaseModel):
    audio_codes: list[int]
    embedding: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "codec_loaded": _state["voice"]._codec is not None}


@app.post("/resolve_embedding")
def resolve_embedding(req: ResolveEmbeddingRequest):
    voice: VoiceResolver = _state["voice"]
    embedding = voice.resolve(req.voice_preset, req.reference_audio, None)
    return {"embedding": embedding.tolist()}


@app.post("/decode")
def decode(req: DecodeRequest):
    import torch

    voice: VoiceResolver = _state["voice"]
    embedding = torch.tensor(req.embedding, dtype=torch.float32, device=voice.codec_device())
    waveform = voice.decode(req.audio_codes, embedding)
    waveform_bytes = waveform.numpy().astype(np.float32).tobytes()
    return {
        "waveform_b64": base64.b64encode(waveform_bytes).decode("ascii"),
        "sample_rate": _state["config"].sample_rate,
    }


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Persistent MioCodec server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    config = MioConfig()
    voice = VoiceResolver(config)
    print(f"Loading codec ({config.codec_name}) ...")
    voice.load_codec()
    print("Codec loaded, starting server.")

    _state["config"] = config
    _state["voice"] = voice

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
