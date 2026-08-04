import argparse

import soundfile as sf

from .config import MioConfig
from .synthesizer import IndicMioSynthesizer


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech with Indic-Mio")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="outputs/output.wav", help="Output wav path")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--voice-preset", default=None, help="Bundled voice preset id (e.g. en_female)")
    parser.add_argument("--reference-audio", default=None, help="Reference wav to clone voice from")
    parser.add_argument(
        "--model-path", default=None, help="Local checkpoint dir or HF repo id to load instead of the default model"
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable post-decode glitch smoothing (see miotts/postprocess.py)",
    )
    args = parser.parse_args()

    defaults = MioConfig()
    config = MioConfig(
        model_name=args.model_path if args.model_path is not None else defaults.model_name,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else defaults.max_new_tokens,
        temperature=args.temperature if args.temperature is not None else defaults.temperature,
        top_p=args.top_p if args.top_p is not None else defaults.top_p,
    )
    synth = IndicMioSynthesizer(config).load()
    waveform = synth.synthesize(
        args.text,
        voice_preset=args.voice_preset,
        reference_audio=args.reference_audio,
        smooth=not args.no_smooth,
    )
    sf.write(args.output, waveform.numpy(), config.sample_rate)
    print(f"Wrote {args.output} ({waveform.shape[0] / config.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
