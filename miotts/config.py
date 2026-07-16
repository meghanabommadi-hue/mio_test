from dataclasses import dataclass


@dataclass(frozen=True)
class MioConfig:
    model_name: str = "SPRINGLab/Indic-Mio"
    codec_name: str = "Aratako/MioCodec-25Hz-44.1kHz-v2"
    sample_rate: int = 44100
    default_preset: str = "en_female"
    speech_token_offset: int = 151669
    speech_vocab_size: int = 12800
    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.9
    device: str = "cuda"
