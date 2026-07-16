import os

import pytest
import soundfile as sf

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

ENGLISH_SENTENCES = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "This is a test of the Indic Mio text to speech system.",
]


@pytest.mark.parametrize("text", ENGLISH_SENTENCES)
def test_synthesize_english_produces_nonempty_audio(synth, text):
    waveform = synth.synthesize(text)
    assert waveform.ndim == 1
    assert waveform.shape[0] > 0


def test_synthesize_english_writes_valid_wav(synth, tmp_path):
    waveform = synth.synthesize("Good morning, welcome to the demonstration.")
    out_path = tmp_path / "english_test.wav"
    sf.write(str(out_path), waveform.numpy(), synth.config.sample_rate)

    data, sr = sf.read(str(out_path))
    assert sr == synth.config.sample_rate
    assert len(data) > 0


def test_synthesize_with_emotion_tag(synth):
    waveform = synth.synthesize("I am so excited about this! <happy>")
    assert waveform.shape[0] > 0
