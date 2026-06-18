#!/usr/bin/env python
"""Sequential single-session latency benchmark.

Unlike ``stress_test.py`` (which fires N sessions concurrently to probe
throughput), this runs N sessions strictly one-at-a-time across a spread of
queries and reports the end-to-end latency distribution. Running sequentially
avoids the free-tier shared-queue throttling, so every session completes and
the p50/p95/p99 numbers reflect genuine pipeline latency rather than rate
limiting.

Usage
-----
    uv run python scripts/latency_bench.py --n 5
    uv run python scripts/latency_bench.py --n 8 --base-url http://localhost:8000

Prerequisites: API + worker running, at least one ticker indexed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

try:
    import httpx
    import websockets
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {e}. Run: uv add --dev httpx websockets") from e


# Diverse queries spanning single-ticker, comparative, and figure-lookup paths.
QUERIES: list[tuple[str, list[str]]] = [
    ("What was NVIDIA's total revenue in fiscal year 2024?", ["NVDA"]),
    ("How does NVIDIA describe its data center growth strategy?", ["NVDA"]),
    ("What is Apple's largest revenue segment?", ["AAPL"]),
    ("How did AMD's Embedded segment perform in fiscal 2023?", ["AMD"]),
    ("Compare the gross margins of NVIDIA and AMD.", ["NVDA", "AMD"]),
    ("What export controls does NVIDIA face for its China business?", ["NVDA"]),
    ("Describe Apple's Services segment and its strategic importance.", ["AAPL"]),
    ("What antitrust risks does Apple disclose?", ["AAPL"]),
]


async def _run_one(client: httpx.AsyncClient, base_url: str, query: str,
                   tickers: list[str], ws_timeout_s: float) -> tuple[bool, float, str | None]:
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{base_url}/api/sessions",
            json={"query": query, "tickers": tickers},
            timeout=15.0,
        )
        resp.raise_for_status()
        session_id = resp.json()["session_id"]

        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/sessions/{session_id}/stream"

        async with websockets.connect(ws_url, open_timeout=10) as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=ws_timeout_s)
                event = json.loads(raw)
                etype = event.get("event_type", "")
                if etype in ("session_done", "error"):
                    ok = etype == "session_done"
                    err = None if ok else event.get("payload", {}).get("message")
                    return ok, time.perf_counter() - t0, err
    except Exception as exc:  # noqa: BLE001
        return False, time.perf_counter() - t0, str(exc)


def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


async def main(base_url: str, n: int, ws_timeout_s: float) -> None:
    print(f"\nSequential latency benchmark: {n} sessions → {base_url}")
    print("─" * 64)

    latencies: list[float] = []
    n_ok = 0
    async with httpx.AsyncClient(base_url=base_url) as client:
        for i in range(n):
            query, tickers = QUERIES[i % len(QUERIES)]
            ok, latency, err = await _run_one(client, base_url, query, tickers, ws_timeout_s)
            latencies.append(latency)
            n_ok += int(ok)
            status = "✓" if ok else "✗"
            print(f"  {status} {latency:6.2f}s  {query[:55]}"
                  + (f"  ERROR: {err}" if err else ""))

    print("─" * 64)
    succ = [latencies[i] for i in range(n)]  # all latencies (failures kept for wall)
    ok_latencies = [latencies[i] for i in range(n)]
    print(f"  Sessions    : {n}")
    print(f"  Succeeded   : {n_ok}  ({n_ok / n * 100:.1f}%)")
    print(f"  Mean latency: {statistics.mean(ok_latencies):.2f} s")
    print(f"  p50 latency : {_pct(ok_latencies, 50):.2f} s")
    print(f"  p95 latency : {_pct(ok_latencies, 95):.2f} s")
    print(f"  p99 latency : {_pct(ok_latencies, 99):.2f} s")
    print(f"  Min / Max   : {min(ok_latencies):.2f} s / {max(ok_latencies):.2f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sequential single-session latency benchmark.")
    ap.add_argument("--n", type=int, default=5, help="Number of sequential sessions (default: 5)")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--ws-timeout", type=float, default=240.0)
    args = ap.parse_args()
    asyncio.run(main(args.base_url, args.n, args.ws_timeout))
