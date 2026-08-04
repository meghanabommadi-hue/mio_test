"""Simple HTTP TTS API for external requests.

Loads IndicMioSynthesizer once at startup (in-process transformers backend --
no vLLM/codec server setup required) and exposes a single synthesis endpoint.

Usage:
    source .venv/bin/activate
    python3 -m miotts.api_server --port 8080

Endpoints:
    GET  /          -> demo web page (text box + Speak button)
    GET  /health
    POST /tts {text, voice_preset?, reference_audio?} -> audio/wav bytes
"""

import argparse
import io
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .config import MioConfig
from .synthesizer import IndicMioSynthesizer

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
_state = {}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


class TTSRequest(BaseModel):
    text: str
    voice_preset: str | None = None
    reference_audio: str | None = None


@app.get("/health")
def health():
    synth: IndicMioSynthesizer = _state["synth"]
    return {"status": "ok", "loaded": synth.is_loaded}


@app.post("/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")

    synth: IndicMioSynthesizer = _state["synth"]
    try:
        waveform = synth.synthesize(
            text,
            voice_preset=req.voice_preset,
            reference_audio=req.reference_audio,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buf = io.BytesIO()
    sf.write(buf, waveform.numpy(), synth.config.sample_rate, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="miotts HTTP TTS API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    config = MioConfig()
    synth = IndicMioSynthesizer(config)
    print("Loading model + codec ...")
    synth.load()
    print("Loaded, starting server.")

    _state["synth"] = synth

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
