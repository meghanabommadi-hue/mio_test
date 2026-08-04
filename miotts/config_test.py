"""Quick cross-language sanity check after changing MioConfig values (temperature,
repetition_penalty, reference voice, etc.) or the reference-audio clip itself.

Synthesizes one sentence per language (or --per-language N), saves audio into a
folder named after the config values used, and reports an objective glitch/
runaway check per file so you know what to actually listen to first.

Not a timing benchmark (see benchmark.py / ws_loadtest.py for that) -- this is
for "does the current config sound right, across every language" after a
config or voice change.

Usage:
    python3 -m miotts.config_test                      # collections category, every language
    python3 -m miotts.config_test --languages hindi telugu
    python3 -m miotts.config_test --categories generic
    python3 -m miotts.config_test --per-language 3 --backend vllm --codec-base-url http://localhost:8001
    python3 -m miotts.config_test --reference-audio ref_audio/romantic_anvi.mp3
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MioConfig
from .sample_texts import SAMPLE_TEXTS


def _count_glitches(data: np.ndarray, percentile: float = 99.9, min_jump: float = 0.15) -> int:
    diff = np.abs(np.diff(data))
    if diff.size == 0:
        return 0
    threshold = max(np.percentile(diff, percentile), min_jump)
    return int(np.sum(diff > threshold))


def _config_tag(cfg: MioConfig, reference_audio: str | None) -> str:
    ref = reference_audio or cfg.default_reference_audio or cfg.default_preset
    ref_name = Path(ref).stem if "/" in ref or "." in ref else ref
    return f"temp{cfg.temperature}_rp{cfg.repetition_penalty}_topp{cfg.top_p}_{ref_name}"


def main():
    parser = argparse.ArgumentParser(description="Cross-language config sanity check")
    parser.add_argument("--languages", nargs="+", default=None, help="Default: every language in the selected categories")
    parser.add_argument("--categories", nargs="+", default=["collections"], choices=list(SAMPLE_TEXTS.keys()))
    parser.add_argument("--per-language", type=int, default=1, help="Sentences to synthesize per language (cycles through the category's list)")
    parser.add_argument("--reference-audio", default=None, help="Override config.default_reference_audio for this run")
    parser.add_argument("--voice-preset", default=None)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000")
    parser.add_argument("--codec-base-url", default="http://localhost:8001")
    parser.add_argument("--run-dir", default=None, help="Default: outputs/runs/<timestamp>_<config_tag>")
    args = parser.parse_args()

    config = MioConfig()

    if args.backend == "vllm":
        from .vllm_synthesizer import VLLMIndicMioSynthesizer

        synth = VLLMIndicMioSynthesizer(config, base_url=args.vllm_base_url, codec_base_url=args.codec_base_url).load()
    else:
        from .synthesizer import IndicMioSynthesizer

        synth = IndicMioSynthesizer(config).load()

    tag = _config_tag(config, args.reference_audio)
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir_abs = run_dir.resolve()

    print(f"Config: temperature={config.temperature} top_p={config.top_p} "
          f"repetition_penalty={config.repetition_penalty} "
          f"reference_audio={args.reference_audio or config.default_reference_audio}")
    print(f"Backend: {args.backend}")
    print(f"Output dir: {run_dir_abs}\n")

    results = []
    for category in args.categories:
        languages = args.languages or list(SAMPLE_TEXTS[category].keys())
        for lang in languages:
            texts = SAMPLE_TEXTS[category][lang]
            for i in range(args.per_language):
                text = texts[i % len(texts)]
                lang_dir = run_dir / category / lang
                lang_dir.mkdir(parents=True, exist_ok=True)
                out_path = lang_dir / f"sentence_{i}.wav"

                try:
                    waveform = synth.synthesize(
                        text,
                        voice_preset=args.voice_preset,
                        reference_audio=args.reference_audio,
                    )
                    data = waveform.numpy()
                    sf.write(str(out_path), data, config.sample_rate)

                    duration = len(data) / config.sample_rate
                    glitches = _count_glitches(data)
                    # 25Hz codec: max_new_tokens/25 is the runaway-generation duration ceiling
                    is_runaway = duration >= (config.max_new_tokens / 25) * 0.95

                    status = "RUNAWAY" if is_runaway else ("GLITCHY" if glitches > 0 else "ok")
                    print(f"  [{category}/{lang}] sentence_{i}: {duration:.2f}s glitches={glitches} -> {status}  {out_path}")
                    results.append({"category": category, "lang": lang, "i": i, "duration": duration,
                                     "glitches": glitches, "runaway": is_runaway, "ok": True})
                except Exception as e:  # noqa: BLE001
                    print(f"  [{category}/{lang}] sentence_{i}: FAILED - {e}")
                    results.append({"category": category, "lang": lang, "i": i, "ok": False, "error": str(e)})

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    runaways = [r for r in ok if r["runaway"]]
    glitchy = [r for r in ok if r["glitches"] > 0 and not r["runaway"]]

    print(f"\n{'=' * 60}")
    print(f"Total: {len(results)}  ok: {len(ok)}  failed: {len(failed)}  "
          f"runaway: {len(runaways)}  glitchy: {len(glitchy)}")
    print(f"Audio saved under: {run_dir_abs}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
