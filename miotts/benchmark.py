import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf
import torch

from .config import MioConfig
from .synthesizer import IndicMioSynthesizer

SAMPLE_TEXTS = {
    "generic": {
        "english": [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Welcome to the demonstration of the text to speech system.",
            "This model supports many Indian languages and English.",
            "Artificial intelligence is transforming how we communicate.",
        ],
        "hindi": [
            "नमस्ते, आप कैसे हैं?",
            "आज मौसम बहुत अच्छा है।",
            "भारत एक विविधतापूर्ण देश है।",
            "यह एक पाठ से वाक् प्रणाली का परीक्षण है।",
            "मुझे हिंदी में बात करना पसंद है।",
        ],
        "telugu": [
            "నమస్కారం, మీరు ఎలా ఉన్నారు?",
            "ఈ రోజు వాతావరణం చాలా బాగుంది.",
            "తెలుగు ఒక అందమైన భాష.",
            "ఇది టెక్స్ట్ టు స్పీచ్ వ్యవస్థ యొక్క పరీక్ష.",
            "నాకు తెలుగులో మాట్లాడటం ఇష్టం.",
        ],
    },
    "collections": {
        "english": [
            "This is a reminder that your EMI payment of two thousand rupees is overdue since the fifth of this month.",
            "We have not received your loan installment for this month. Please clear the due amount at the earliest to avoid a late fee.",
            "Your account is now thirty days past due. Kindly make the payment today to prevent further action on your loan.",
            "We understand things can get difficult. Can we help you set up a revised payment plan for your pending EMI?",
            "This is a final notice regarding your outstanding balance. Please contact our office within forty eight hours.",
        ],
        "hindi": [
            "यह एक अनुस्मारक है कि आपकी ईएमआई का भुगतान इस महीने की पांच तारीख से बकाया है।",
            "हमें इस महीने आपकी लोन किस्त प्राप्त नहीं हुई है। कृपया जल्द से जल्द बकाया राशि जमा करें।",
            "आपका खाता अब तीस दिनों से अधिक समय से बकाया है। कृपया आज ही भुगतान करें।",
            "क्या हम आपकी बकाया ईएमआई के लिए एक नई भुगतान योजना बनाने में मदद कर सकते हैं?",
            "यह आपके बकाया राशि के संबंध में अंतिम सूचना है। कृपया अड़तालीस घंटों के भीतर हमसे संपर्क करें।",
        ],
        "telugu": [
            "మీ ఈఎంఐ చెల్లింపు ఈ నెల ఐదవ తేదీ నుండి బాకీ ఉందని ఇది ఒక రిమైండర్.",
            "ఈ నెల మీ లోన్ వాయిదా మాకు అందలేదు. దయచేసి వీలైనంత త్వరగా బాకీ మొత్తాన్ని చెల్లించండి.",
            "మీ ఖాతా ఇప్పుడు ముప్పై రోజులు దాటి బాకీ ఉంది. దయచేసి ఈరోజే చెల్లింపు చేయండి.",
            "మీ బాకీ ఈఎంఐ కోసం కొత్త చెల్లింపు ప్రణాళికను ఏర్పాటు చేయడంలో మేము సహాయం చేయవచ్చా?",
            "మీ బాకీ మొత్తానికి సంబంధించి ఇది చివరి నోటీసు. దయచేసి నలభై ఎనిమిది గంటల్లో మా కార్యాలయాన్ని సంప్రదించండి.",
        ],
    },
}


@dataclass
class BenchmarkResult:
    category: str
    language: str
    batch_size: int
    num_requests: int
    num_failures: int
    total_wall_time_s: float
    total_audio_duration_s: float
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    throughput_req_per_s: float
    rtf: float  # real-time factor: wall_time / audio_duration (lower is better)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(pct / 100 * (len(values) - 1))))
    return values[idx]


def run_benchmark(
    synth: IndicMioSynthesizer,
    language: str,
    num_requests: int,
    category: str = "generic",
    wavs_dir: Path | None = None,
    save_all: bool = False,
) -> BenchmarkResult:
    texts = SAMPLE_TEXTS[category][language]
    latencies = []
    audio_durations = []
    failures = 0
    saved_texts = set()

    start = time.perf_counter()
    for i in range(num_requests):
        text_idx = i % len(texts)
        text = texts[text_idx]
        req_start = time.perf_counter()
        try:
            waveform = synth.synthesize(text)
            latency = time.perf_counter() - req_start
            latencies.append(latency)
            audio_durations.append(waveform.shape[0] / synth.config.sample_rate)

            if wavs_dir is not None:
                should_save = save_all or text_idx not in saved_texts
                if should_save:
                    saved_texts.add(text_idx)
                    suffix = f"_{i:04d}" if save_all else f"_sample{text_idx}"
                    out_path = wavs_dir / f"{category}_{language}{suffix}.wav"
                    sf.write(str(out_path), waveform.numpy(), synth.config.sample_rate)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [{language}] request {i} failed: {exc}")
    total_wall_time = time.perf_counter() - start

    total_audio = sum(audio_durations)
    return BenchmarkResult(
        category=category,
        language=language,
        batch_size=num_requests,
        num_requests=num_requests,
        num_failures=failures,
        total_wall_time_s=round(total_wall_time, 3),
        total_audio_duration_s=round(total_audio, 3),
        avg_latency_s=round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        p50_latency_s=round(_percentile(latencies, 50), 4),
        p95_latency_s=round(_percentile(latencies, 95), 4),
        throughput_req_per_s=round(len(latencies) / total_wall_time, 4) if total_wall_time else 0.0,
        rtf=round(total_wall_time / total_audio, 4) if total_audio else float("inf"),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Indic-Mio TTS across languages and batch sizes")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[50, 100, 300, 500])
    parser.add_argument("--languages", nargs="+", default=["english", "hindi", "telugu"])
    parser.add_argument("--output", default="outputs/benchmark_results.json")
    parser.add_argument("--wavs-dir", default="outputs/wavs", help="Directory to save synthesized wavs")
    parser.add_argument(
        "--save-all-wavs",
        action="store_true",
        help="Save every request's wav (default: one wav per unique sample text)",
    )
    parser.add_argument("--no-save-wavs", action="store_true", help="Disable saving wavs")
    args = parser.parse_args()

    config = MioConfig()
    synth = IndicMioSynthesizer(config).load()

    wavs_dir = None
    if not args.no_save_wavs:
        wavs_dir = Path(args.wavs_dir)
        wavs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for language in args.languages:
        for batch_size in args.batch_sizes:
            print(f"Running {language} batch_size={batch_size} ...")
            result = run_benchmark(
                synth, language, batch_size, wavs_dir=wavs_dir, save_all=args.save_all_wavs
            )
            results.append(asdict(result))
            print(
                f"  -> throughput={result.throughput_req_per_s} req/s, "
                f"rtf={result.rtf}, p95={result.p95_latency_s}s, failures={result.num_failures}"
            )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
