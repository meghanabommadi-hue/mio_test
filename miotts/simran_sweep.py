"""Generate the standard 7-language 'collections' sentence sweep with the default
(friendly_simran) reference voice, for a given model checkpoint.

Usage:
    python -m miotts.simran_sweep --model-path runs/full-ft-v1 --out-dir outputs/simran_sweep_full-ft-v1
"""

import argparse
from pathlib import Path

import soundfile as sf

from .config import MioConfig
from .sample_texts import SAMPLE_TEXTS
from .synthesizer import IndicMioSynthesizer

LANGUAGES = ["english", "hindi", "telugu", "punjabi", "marathi", "assamese", "gujarati"]


def main():
    parser = argparse.ArgumentParser(description="Run the simran-voice sentence sweep against a checkpoint")
    parser.add_argument("--model-path", default=None, help="Local checkpoint dir or HF repo id")
    parser.add_argument(
        "--model-subfolder",
        default=None,
        help="Subfolder within --model-path's HF repo to load (e.g. checkpoint-5000)",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write <language>/sentence_N.wav into")
    args = parser.parse_args()

    defaults = MioConfig()
    if args.model_path is not None:
        config = MioConfig(model_name=args.model_path, model_subfolder=args.model_subfolder)
    else:
        config = MioConfig(model_subfolder=args.model_subfolder or defaults.model_subfolder)
    synth = IndicMioSynthesizer(config).load()

    out_dir = Path(args.out_dir)
    for lang in LANGUAGES:
        lang_dir = out_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(SAMPLE_TEXTS["collections"][lang]):
            waveform = synth.synthesize(text)
            out_path = lang_dir / f"sentence_{i}.wav"
            sf.write(out_path, waveform.numpy(), config.sample_rate)
            print(f"[{lang}] wrote {out_path}")

    print(f"Done. Sweep written to {out_dir}")


if __name__ == "__main__":
    main()
