import os

# Model/tokenizer/codec weights are already cached locally -- force fully offline
# loading so `from_pretrained()` never phones home for an etag/revision check.
# Must be set before transformers/huggingface_hub is imported anywhere below.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .synthesizer import IndicMioSynthesizer

__all__ = ["IndicMioSynthesizer"]
