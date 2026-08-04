"""Load test for miotts.ws_server: open N concurrent WebSocket connections,
each sending one request, and report aggregate TTFT/latency stats.

This exercises the WS server the way many simultaneous live calls would --
each connection is independent (its own asyncio task), unlike
miotts.benchmark's --concurrency (which fans out HTTP requests via a thread
pool against the vLLM/codec HTTP servers directly, not through ws_server).

Usage:
    python3 -m miotts.ws_loadtest --connections 100
    python3 -m miotts.ws_loadtest --connections 100 --languages hindi telugu --categories collections
    python3 -m miotts.ws_loadtest --connections 20 --requests-per-connection 5
"""

import argparse
import asyncio
import json
import statistics
import time

import websockets

from .sample_texts import SAMPLE_TEXTS


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(pct / 100 * (len(values) - 1))))
    return values[idx]


async def _one_connection(conn_id: int, uri: str, texts: list[str], requests_per_connection: int, timeout: float):
    """Opens one WS connection, sends requests_per_connection requests over it
    sequentially (matching real call behavior -- one call, several utterances),
    and returns a list of per-request result dicts."""
    results = []
    try:
        async with websockets.connect(uri, open_timeout=timeout) as ws:
            for i in range(requests_per_connection):
                text = texts[(conn_id * requests_per_connection + i) % len(texts)]
                call_id = f"conn{conn_id}-req{i}"
                t0 = time.perf_counter()
                try:
                    await ws.send(json.dumps({"text": text, "call_id": call_id}))
                    meta = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                    if meta.get("type") == "error":
                        results.append({"conn_id": conn_id, "ok": False, "error": meta.get("error")})
                        continue
                    wav_bytes = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    client_total = time.perf_counter() - t0
                    results.append({
                        "conn_id": conn_id,
                        "ok": True,
                        "ttft_ms": meta["ttft_ms"],
                        "total_ms": meta["total_ms"],
                        "client_total_ms": round(client_total * 1000, 1),
                        "n_chunks": meta["n_chunks"],
                        "wav_bytes": len(wav_bytes),
                    })
                except Exception as e:  # noqa: BLE001
                    results.append({"conn_id": conn_id, "ok": False, "error": str(e)})
    except Exception as e:  # noqa: BLE001
        results.append({"conn_id": conn_id, "ok": False, "error": f"connect failed: {e}"})
    return results


async def run_loadtest(
    uri: str,
    n_connections: int,
    requests_per_connection: int,
    texts: list[str],
    timeout: float,
    ramp_up_s: float,
):
    tasks = []
    start = time.perf_counter()
    for conn_id in range(n_connections):
        tasks.append(asyncio.create_task(_one_connection(conn_id, uri, texts, requests_per_connection, timeout)))
        if ramp_up_s > 0:
            await asyncio.sleep(ramp_up_s / n_connections)

    all_results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - start

    flat = [r for conn_results in all_results for r in conn_results]
    ok = [r for r in flat if r["ok"]]
    failed = [r for r in flat if not r["ok"]]

    print(f"\n{'=' * 60}")
    print(f"Load test: {n_connections} connections x {requests_per_connection} req/conn = {len(flat)} total requests")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Succeeded: {len(ok)}/{len(flat)}  Failed: {len(failed)}/{len(flat)}")
    print(f"{'=' * 60}")

    if ok:
        ttfts = [r["ttft_ms"] for r in ok]
        totals = [r["total_ms"] for r in ok]
        client_totals = [r["client_total_ms"] for r in ok]

        print("\nTTFT (ms):")
        print(f"  avg={statistics.mean(ttfts):.1f}  p50={_percentile(ttfts, 50):.1f}  "
              f"p95={_percentile(ttfts, 95):.1f}  max={max(ttfts):.1f}")

        print("\nServer-reported total latency (ms):")
        print(f"  avg={statistics.mean(totals):.1f}  p50={_percentile(totals, 50):.1f}  "
              f"p95={_percentile(totals, 95):.1f}  max={max(totals):.1f}")

        print("\nClient-observed total latency (ms, includes network round-trip):")
        print(f"  avg={statistics.mean(client_totals):.1f}  p50={_percentile(client_totals, 50):.1f}  "
              f"p95={_percentile(client_totals, 95):.1f}  max={max(client_totals):.1f}")

        print(f"\nThroughput: {len(ok) / wall_time:.2f} req/s")

        runaways = [r for r in ok if r["n_chunks"] >= 2000]
        if runaways:
            print(f"\nWARNING: {len(runaways)}/{len(ok)} requests hit likely runaway generation (n_chunks>=2000)")

    if failed:
        print(f"\nFailures ({len(failed)}):")
        for r in failed[:20]:
            print(f"  conn {r['conn_id']}: {r['error']}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    return {"ok": ok, "failed": failed, "wall_time": wall_time}


def main():
    parser = argparse.ArgumentParser(description="Load test miotts.ws_server with concurrent WS connections")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--connections", type=int, default=100, help="Number of concurrent WS connections")
    parser.add_argument("--requests-per-connection", type=int, default=1, help="Sequential requests sent per connection")
    parser.add_argument("--languages", nargs="+", default=["english", "hindi", "telugu"])
    parser.add_argument("--categories", nargs="+", default=["generic"], choices=list(SAMPLE_TEXTS.keys()))
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument("--ramp-up", type=float, default=0.0, help="Seconds to spread connection starts over (0 = all at once)")
    args = parser.parse_args()

    texts = []
    for category in args.categories:
        for lang in args.languages:
            texts.extend(SAMPLE_TEXTS[category][lang])
    if not texts:
        raise ValueError("No texts selected -- check --languages/--categories")

    uri = f"ws://{args.host}:{args.port}"
    print(f"Connecting to {uri}, {len(texts)} candidate texts loaded")
    asyncio.run(run_loadtest(uri, args.connections, args.requests_per_connection, texts, args.timeout, args.ramp_up))


if __name__ == "__main__":
    main()
