import argparse

import soundfile as sf

from .config import MioConfig
from .synthesizer import IndicMioSynthesizer


def main():
    parser = argparse.ArgumentParser(description="Synthesize speech with Indic-Mio")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="outputs/output.wav", help="Output wav path")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--voice-preset", default=None, help="Bundled voice preset id (e.g. en_female)")
    parser.add_argument("--reference-audio", default=None, help="Reference wav to clone voice from")
    args = parser.parse_args()

    config = MioConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    synth = IndicMioSynthesizer(config).load()
    waveform = synth.synthesize(
        args.text,
        voice_preset=args.voice_preset,
        reference_audio=args.reference_audio,
    )
    sf.write(args.output, waveform.numpy(), config.sample_rate)
    print(f"Wrote {args.output} ({waveform.shape[0] / config.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
