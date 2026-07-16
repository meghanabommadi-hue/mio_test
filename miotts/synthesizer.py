from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MioConfig

PRESETS_DIR = Path(__file__).parent / "presets"


class IndicMioSynthesizer:
    """Wraps SPRINGLab/Indic-Mio (LLM) + MioCodec (audio decoder) for text-to-speech."""

    def __init__(self, config: MioConfig | None = None):
        self.config = config or MioConfig()
        self._tokenizer = None
        self._model = None
        self._codec = None
        self._preset_cache: dict[str, torch.Tensor] = {}

    def load(self):
        cfg = self.config
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map=cfg.device,
        )

        from miocodec import MioCodecModel

        self._codec = MioCodecModel.from_pretrained(cfg.codec_name).eval().to(cfg.device)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._codec is not None

    def _codec_device(self) -> torch.device:
        return next(self._codec.parameters()).device

    def load_preset_embedding(self, preset_id: str) -> torch.Tensor:
        if preset_id in self._preset_cache:
            return self._preset_cache[preset_id]
        path = PRESETS_DIR / f"{preset_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Voice preset '{preset_id}' not found at {path}")
        embedding = torch.load(path, map_location="cpu", weights_only=True)
        embedding = embedding.squeeze().to(self._codec_device())
        self._preset_cache[preset_id] = embedding
        return embedding

    def embedding_from_reference_audio(self, audio_path: str) -> torch.Tensor:
        from miocodec.util import load_audio

        waveform = load_audio(audio_path, sample_rate=self._codec.config.sample_rate)
        waveform = waveform.to(self._codec_device())
        with torch.no_grad():
            features = self._codec.encode(waveform, return_content=False, return_global=True)
        return features.global_embedding

    def _extract_speech_tokens(self, generated_ids: torch.Tensor) -> list[int]:
        cfg = self.config
        codes = []
        for token_id in generated_ids.tolist():
            if cfg.speech_token_offset <= token_id < cfg.speech_token_offset + cfg.speech_vocab_size:
                codes.append(token_id - cfg.speech_token_offset)
        return codes

    def synthesize(
        self,
        text: str,
        voice_preset: str | None = None,
        reference_audio: str | None = None,
        global_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate speech audio for `text`. Returns a 1-D float waveform tensor at cfg.sample_rate.

        Voice is selected via one of (in priority order): `global_embedding` (precomputed),
        `reference_audio` (wav path to clone), `voice_preset` (bundled preset id), or
        `config.default_preset` if none given.
        """
        if not self.is_loaded:
            self.load()

        cfg = self.config
        messages = [{"role": "user", "content": text}]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        audio_codes = self._extract_speech_tokens(generated)
        if not audio_codes:
            raise RuntimeError(
                "No speech tokens were generated for the given text; try again or "
                "increase max_new_tokens."
            )

        if global_embedding is None:
            if reference_audio is not None:
                global_embedding = self.embedding_from_reference_audio(reference_audio)
            else:
                global_embedding = self.load_preset_embedding(voice_preset or cfg.default_preset)

        global_embedding = global_embedding.squeeze()

        content_token_indices = torch.tensor(
            audio_codes, dtype=torch.long, device=self._codec_device()
        )

        with torch.no_grad():
            waveform = self._codec.decode(
                global_embedding=global_embedding,
                content_token_indices=content_token_indices,
            )

        return waveform.squeeze().float().cpu()
