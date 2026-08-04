import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MioConfig
from .postprocess import smooth_glitches
from .voice import VoiceResolver, extract_speech_tokens


class IndicMioSynthesizer:
    """Wraps SPRINGLab/Indic-Mio (LLM) + MioCodec (audio decoder) for text-to-speech."""

    def __init__(self, config: MioConfig | None = None):
        self.config = config or MioConfig()
        self._tokenizer = None
        self._model = None
        self._voice = VoiceResolver(self.config)

    def load(self):
        cfg = self.config
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, subfolder=cfg.model_subfolder or "", trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            subfolder=cfg.model_subfolder or "",
            torch_dtype=torch.bfloat16,
            device_map=cfg.device,
        )
        self._voice.load_codec()
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._voice._codec is not None

    def load_preset_embedding(self, preset_id: str) -> torch.Tensor:
        return self._voice.load_preset_embedding(preset_id)

    def embedding_from_reference_audio(self, audio_path: str) -> torch.Tensor:
        return self._voice.embedding_from_reference_audio(audio_path)

    def synthesize(
        self,
        text: str,
        voice_preset: str | None = None,
        reference_audio: str | None = None,
        global_embedding: torch.Tensor | None = None,
        smooth: bool | None = None,
    ) -> torch.Tensor:
        """Generate speech audio for `text`. Returns a 1-D float waveform tensor at cfg.sample_rate.

        Voice is selected via one of (in priority order): `global_embedding` (precomputed),
        `reference_audio` (wav path to clone), `voice_preset` (bundled preset id),
        `config.default_reference_audio` (hardcoded reference clip), or
        `config.default_preset` if none given.

        `smooth` (default `config.smooth_glitches`) crossfades over isolated codec
        glitches after decode -- see postprocess.py. Adds ~0.6ms per second of
        audio; pass `smooth=False` to disable per-call.
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
                repetition_penalty=cfg.repetition_penalty,
                do_sample=True,
            )

        generated = output[0][inputs["input_ids"].shape[1]:]
        audio_codes = extract_speech_tokens(generated.tolist(), cfg)
        if not audio_codes:
            raise RuntimeError(
                "No speech tokens were generated for the given text; try again or "
                "increase max_new_tokens."
            )

        global_embedding = self._voice.resolve(voice_preset, reference_audio, global_embedding)
        waveform = self._voice.decode(audio_codes, global_embedding)

        do_smooth = cfg.smooth_glitches if smooth is None else smooth
        if do_smooth:
            waveform = torch.from_numpy(smooth_glitches(waveform.numpy(), cfg.sample_rate))
        return waveform
