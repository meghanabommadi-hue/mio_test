from dataclasses import dataclass


@dataclass(frozen=True)
class MioConfig:
    model_name: str = "/home/jovyan/miotts/runs/full-ft-v3"
    model_subfolder: str | None = None
    codec_name: str = "Aratako/MioCodec-25Hz-44.1kHz-v2"
    sample_rate: int = 44100
    default_preset: str = "en_female"
    default_reference_audio: str | None = "ref_audio/friendly_simran.wav"
    speech_token_offset: int = 151669
    speech_vocab_size: int = 12800
    max_new_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9
    # Guards against runaway generation (model fails to emit <|im_end|> and
    # keeps sampling speech tokens until max_new_tokens). Empirically swept:
    # 1.0 (off) and 1.15+ both hit this ~10-100% of the time depending on
    # text; 1.05-1.1 eliminated it across 40+ trials on diverse sentences
    # (english/hindi/telugu/punjabi, plain and code-mixed). Values >=1.15
    # get rapidly worse, not better -- this is not a monotonic "more is
    # safer" knob, don't tune past ~1.1 without re-sweeping.
    repetition_penalty: float = 1.1
    device: str = "cuda"
    smooth_glitches: bool = True
