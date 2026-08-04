import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch

from .config import MioConfig
from .sample_texts import SAMPLE_TEXTS
from .synthesizer import IndicMioSynthesizer


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
    reference_audio: str | None = None,
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
            waveform = synth.synthesize(text, reference_audio=reference_audio)
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


def run_benchmark_concurrent(
    synth,
    language: str,
    num_requests: int,
    concurrency: int,
    category: str = "generic",
    wavs_dir: Path | None = None,
    save_all: bool = False,
    reference_audio: str | None = None,
) -> BenchmarkResult:
    """Same as run_benchmark but fires requests from a thread pool. Only useful for
    backends that release the GIL while waiting (e.g. VLLMIndicMioSynthesizer, which
    blocks on HTTP I/O) -- an in-process transformers model can't actually run
    concurrent GPU forward passes this way, so use run_benchmark for that backend.
    """
    texts = SAMPLE_TEXTS[category][language]
    latencies = [None] * num_requests
    audio_durations = [None] * num_requests
    failures = 0
    lock_saved = set()

    def one_request(i):
        text_idx = i % len(texts)
        text = texts[text_idx]
        req_start = time.perf_counter()
        waveform = synth.synthesize(text, reference_audio=reference_audio)
        latency = time.perf_counter() - req_start
        return i, text_idx, latency, waveform

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one_request, i): i for i in range(num_requests)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                i, text_idx, latency, waveform = future.result()
                latencies[i] = latency
                audio_durations[i] = waveform.shape[0] / synth.config.sample_rate

                if wavs_dir is not None:
                    should_save = save_all or text_idx not in lock_saved
                    if should_save:
                        lock_saved.add(text_idx)
                        suffix = f"_{i:04d}" if save_all else f"_sample{text_idx}"
                        out_path = wavs_dir / f"{category}_{language}{suffix}.wav"
                        sf.write(str(out_path), waveform.numpy(), synth.config.sample_rate)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  [{language}] request {i} failed: {exc}")
    total_wall_time = time.perf_counter() - start

    latencies = [l for l in latencies if l is not None]
    audio_durations = [d for d in audio_durations if d is not None]
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
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["generic"],
        choices=list(SAMPLE_TEXTS.keys()),
        help="Sentence categories to benchmark (generic, collections)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory for this run's output.json + wavs/ (default: outputs/runs/<timestamp>)",
    )
    parser.add_argument("--output", default=None, help="Override path for the results JSON")
    parser.add_argument("--wavs-dir", default=None, help="Override directory to save synthesized wavs")
    parser.add_argument(
        "--save-all-wavs",
        action="store_true",
        help="Save every request's wav (default: one wav per unique sample text)",
    )
    parser.add_argument("--no-save-wavs", action="store_true", help="Disable saving wavs")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="transformers",
        help="transformers: in-process model (sequential only). "
        "vllm: HTTP client to a running `vllm serve` instance (supports --concurrency).",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://localhost:8000",
        help="Base URL of the vLLM server (only used with --backend vllm)",
    )
    parser.add_argument(
        "--codec-base-url",
        default=None,
        help="Base URL of a persistent miotts.codec_server (only used with --backend vllm). "
        "If omitted, the codec loads locally in this process instead.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent in-flight requests (only meaningful with --backend vllm)",
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        help="Reference wav/mp3 to clone the voice from for every request "
        "(overrides config.default_reference_audio / voice presets)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else run_dir / "benchmark_results.json"

    config = MioConfig()
    if args.backend == "vllm":
        from .vllm_synthesizer import VLLMIndicMioSynthesizer

        synth = VLLMIndicMioSynthesizer(
            config, base_url=args.vllm_base_url, codec_base_url=args.codec_base_url
        ).load()
    else:
        if args.concurrency > 1:
            raise ValueError("--concurrency > 1 requires --backend vllm; the transformers "
                              "backend runs one GPU forward pass at a time.")
        synth = IndicMioSynthesizer(config).load()

    wavs_dir = None
    if not args.no_save_wavs:
        wavs_dir = Path(args.wavs_dir) if args.wavs_dir else run_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for category in args.categories:
        for language in args.languages:
            for batch_size in args.batch_sizes:
                print(f"Running {category}/{language} batch_size={batch_size} "
                      f"(backend={args.backend}, concurrency={args.concurrency}) ...")
                if args.backend == "vllm" and args.concurrency > 1:
                    result = run_benchmark_concurrent(
                        synth,
                        language,
                        batch_size,
                        args.concurrency,
                        category=category,
                        wavs_dir=wavs_dir,
                        save_all=args.save_all_wavs,
                        reference_audio=args.reference_audio,
                    )
                else:
                    result = run_benchmark(
                        synth,
                        language,
                        batch_size,
                        category=category,
                        wavs_dir=wavs_dir,
                        save_all=args.save_all_wavs,
                        reference_audio=args.reference_audio,
                    )
                results.append(asdict(result))
                print(
                    f"  -> throughput={result.throughput_req_per_s} req/s, "
                    f"rtf={result.rtf}, p95={result.p95_latency_s}s, failures={result.num_failures}"
                )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
