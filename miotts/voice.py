"""Shared codec loading + voice-embedding resolution, used by both the in-process
(synthesizer.py) and vLLM-client (vllm_synthesizer.py) synthesis backends -- the
MioCodec decode step is identical regardless of which backend generated the
speech tokens.
"""

from pathlib import Path

import torch

from .config import MioConfig

PRESETS_DIR = Path(__file__).parent / "presets"
REPO_ROOT = Path(__file__).parent.parent


class VoiceResolver:
    """Loads MioCodec and resolves a `global_embedding` from a preset id, a
    reference audio path, or the configured hardcoded default."""

    def __init__(self, config: MioConfig):
        self.config = config
        self._codec = None
        self._preset_cache: dict[str, torch.Tensor] = {}

    def load_codec(self):
        from miocodec import MioCodecModel

        self._codec = MioCodecModel.from_pretrained(self.config.codec_name).eval().to(self.config.device)
        return self._codec

    @property
    def codec(self):
        if self._codec is None:
            self.load_codec()
        return self._codec

    def codec_device(self) -> torch.device:
        return next(self.codec.parameters()).device

    def load_preset_embedding(self, preset_id: str) -> torch.Tensor:
        if preset_id in self._preset_cache:
            return self._preset_cache[preset_id]
        path = PRESETS_DIR / f"{preset_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Voice preset '{preset_id}' not found at {path}")
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        embedding = embedding.squeeze().to(self.codec_device())
        self._preset_cache[preset_id] = embedding
        return embedding

    def embedding_from_reference_audio(self, audio_path: str) -> torch.Tensor:
        from miocodec.util import load_audio

        waveform = load_audio(audio_path, sample_rate=self.codec.config.sample_rate)
        waveform = waveform.to(self.codec_device())
        with torch.no_grad():
            features = self.codec.encode(waveform, return_content=False, return_global=True)
        return features.global_embedding

    def resolve(
        self,
        voice_preset: str | None,
        reference_audio: str | None,
        global_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Voice is selected via one of (in priority order): `global_embedding`
        (precomputed), `reference_audio` (path to clone), `voice_preset` (bundled
        preset id), `config.default_reference_audio` (hardcoded reference clip),
        or `config.default_preset` if none given.
        """
        if global_embedding is not None:
            return global_embedding.squeeze()

        cfg = self.config
        if reference_audio is not None:
            embedding = self.embedding_from_reference_audio(reference_audio)
        elif voice_preset is None and cfg.default_reference_audio is not None:
            ref_path = Path(cfg.default_reference_audio)
            if not ref_path.is_absolute():
                ref_path = REPO_ROOT / ref_path
            embedding = self.embedding_from_reference_audio(str(ref_path))
        else:
            embedding = self.load_preset_embedding(voice_preset or cfg.default_preset)

        return embedding.squeeze()

    def decode(self, audio_codes: list[int], global_embedding: torch.Tensor) -> torch.Tensor:
        content_token_indices = torch.tensor(audio_codes, dtype=torch.long, device=self.codec_device())
        with torch.no_grad():
            waveform = self.codec.decode(
                global_embedding=global_embedding,
                content_token_indices=content_token_indices,
            )
        return waveform.squeeze().float().cpu()


def extract_speech_tokens(token_ids: list[int], config: MioConfig) -> list[int]:
    codes = []
    for token_id in token_ids:
        if config.speech_token_offset <= token_id < config.speech_token_offset + config.speech_vocab_size:
            codes.append(token_id - config.speech_token_offset)
    return codes
