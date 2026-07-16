import argparse
import json


def format_report(results: list[dict]) -> str:
    lines = []
    lines.append("Indic-Mio TTS Benchmark Results")
    lines.append("=" * 70)
    lines.append("")

    by_language: dict[str, list[dict]] = {}
    for r in results:
        by_language.setdefault(r["language"], []).append(r)

    for language, rows in by_language.items():
        lines.append(f"Language: {language}")
        lines.append("-" * 70)
        header = (
            f"{'batch':>6} {'ok':>6} {'fail':>5} {'wall_s':>9} {'audio_s':>9} "
            f"{'avg_lat':>8} {'p50_lat':>8} {'p95_lat':>8} {'req/s':>7} {'rtf':>6}"
        )
        lines.append(header)
        for r in sorted(rows, key=lambda x: x["batch_size"]):
            ok = r["num_requests"] - r["num_failures"]
            lines.append(
                f"{r['batch_size']:>6} {ok:>6} {r['num_failures']:>5} "
                f"{r['total_wall_time_s']:>9.2f} {r['total_audio_duration_s']:>9.2f} "
                f"{r['avg_latency_s']:>8.3f} {r['p50_latency_s']:>8.3f} {r['p95_latency_s']:>8.3f} "
                f"{r['throughput_req_per_s']:>7.3f} {r['rtf']:>6.3f}"
            )
        lines.append("")

    lines.append("Columns: batch=requests sent, ok=succeeded, fail=errored,")
    lines.append("wall_s=total wall time, audio_s=total generated audio duration,")
    lines.append("avg/p50/p95_lat=per-request latency in seconds, req/s=throughput,")
    lines.append("rtf=wall_time/audio_duration (lower is better, <1.0 is faster than real-time).")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark JSON results to a text report")
    parser.add_argument("--input", default="outputs/benchmark_results.json")
    parser.add_argument("--output", default="outputs/benchmark_results.txt")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    report = format_report(results)
    with open(args.output, "w") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nWrote text report to {args.output}")


if __name__ == "__main__":
    main()
