#!/usr/bin/env python
"""Phase 9 stress test — concurrent session throughput benchmark.

Fires N concurrent query sessions against the running API, subscribes to each
session's WebSocket stream, and waits for SESSION_DONE or ERROR. Reports wall-
clock latency per session and an aggregate summary with p50/p95/p99 statistics.

Usage
-----
    # 10 sessions (default) against local stack
    python scripts/stress_test.py

    # 20 sessions against a deployed instance
    python scripts/stress_test.py --sessions 20 --base-url https://your-hf-spaces-url

    # Makefile shortcut
    make stress-test sessions=15

Prerequisites
-------------
- pip install httpx websockets (already in dev deps)
- The API stack must be running: `make up && make worker`
- At least one ticker must be indexed: `make ingest ticker=NVDA && make index ticker=NVDA`

Output example
--------------
    ┌─ Stress Test Results ──────────────────────────────────────────┐
    │  Sessions    : 10                                              │
    │  Succeeded   : 10  (100.0%)                                   │
    │  Failed      : 0                                              │
    │  p50 latency : 4.31 s                                         │
    │  p95 latency : 8.47 s                                         │
    │  p99 latency : 9.12 s                                         │
    │  Total wall  : 11.2 s                                         │
    └────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import textwrap
from dataclasses import dataclass, field
from typing import Optional
import uuid

try:
    import httpx
    import websockets
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}. "
        "Run: uv add --dev httpx websockets"
    ) from e


# ---------------------------------------------------------------------------
# Sample queries spread across NVDA / AMD / AAPL to exercise multiple paths
# ---------------------------------------------------------------------------
SAMPLE_QUERIES: list[tuple[str, list[str]]] = [
    ("What was NVIDIA's total revenue in fiscal year 2024?", ["NVDA"]),
    ("How does NVIDIA describe its data center growth strategy?", ["NVDA"]),
    ("What are NVIDIA's primary supply chain risks?", ["NVDA"]),
    ("How did AMD's Embedded segment perform in fiscal 2023?", ["AMD"]),
    ("Compare the gross margins of NVIDIA and AMD.", ["NVDA", "AMD"]),
    ("What is Apple's largest revenue segment?", ["AAPL"]),
    ("How does NVIDIA's R&D spending compare to Apple's?", ["NVDA", "AAPL"]),
    ("What export controls does NVIDIA face for its China business?", ["NVDA"]),
    ("What was AMD's Data Center revenue in fiscal 2023?", ["AMD"]),
    ("Describe Apple's Services segment and its strategic importance.", ["AAPL"]),
    ("What was NVIDIA's net income for fiscal 2024?", ["NVDA"]),
    ("How do NVIDIA and AMD describe competition in AI accelerators?", ["NVDA", "AMD"]),
    ("What was Apple's gross margin in fiscal 2023?", ["AAPL"]),
    ("What is NVIDIA's Blackwell GPU architecture?", ["NVDA"]),
    ("What antitrust risks does Apple disclose?", ["AAPL"]),
    ("How did NVIDIA's Data Center growth drive gross margin expansion?", ["NVDA"]),
    ("Compare revenue diversification across NVIDIA, AMD, and Apple.", ["NVDA", "AMD", "AAPL"]),
    ("What was NVIDIA's free cash flow in fiscal 2024?", ["NVDA"]),
    ("How does AMD's ROCm software compare to NVIDIA's CUDA?", ["AMD", "NVDA"]),
    ("What was Apple's Services revenue growth in fiscal 2023?", ["AAPL"]),
]


@dataclass
class SessionResult:
    session_id: str
    query: str
    success: bool
    latency_s: float
    error: Optional[str] = None
    event_count: int = 0


async def _run_one_session(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    tickers: list[str],
    ws_timeout_s: float,
) -> SessionResult:
    """POST a session, then drain the WebSocket until done or error."""
    t0 = time.perf_counter()
    session_id: str = ""

    try:
        # ── Create session ──────────────────────────────────────────────────
        resp = await client.post(
            f"{base_url}/api/sessions",
            json={"query": query, "tickers": tickers},
            timeout=15.0,
        )
        resp.raise_for_status()
        body = resp.json()
        session_id = body["session_id"]

        # ── Subscribe to WebSocket stream ───────────────────────────────────
        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/sessions/{session_id}/stream"

        event_count = 0
        async with websockets.connect(ws_url, open_timeout=10) as ws:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=ws_timeout_s)
                    event = json.loads(raw)
                    event_count += 1
                    etype = event.get("event_type", "")
                    if etype in ("session_done", "error"):
                        success = etype == "session_done"
                        err = event.get("payload", {}).get("message") if not success else None
                        return SessionResult(
                            session_id=session_id,
                            query=query,
                            success=success,
                            latency_s=time.perf_counter() - t0,
                            error=err,
                            event_count=event_count,
                        )
                except asyncio.TimeoutError:
                    return SessionResult(
                        session_id=session_id,
                        query=query,
                        success=False,
                        latency_s=time.perf_counter() - t0,
                        error=f"WebSocket timeout after {ws_timeout_s}s",
                        event_count=event_count,
                    )

    except Exception as exc:  # noqa: BLE001
        return SessionResult(
            session_id=session_id or "(no id)",
            query=query,
            success=False,
            latency_s=time.perf_counter() - t0,
            error=str(exc),
        )


async def run_stress_test(
    base_url: str,
    n_sessions: int,
    ws_timeout_s: float,
    verbose: bool,
) -> None:
    """Fire n_sessions concurrently and print a summary report."""

    # Cycle through sample queries if n_sessions > len(SAMPLE_QUERIES)
    tasks_input = [
        SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        for i in range(n_sessions)
    ]

    print(f"\nStress test: {n_sessions} concurrent sessions → {base_url}")
    print(f"WebSocket timeout per session: {ws_timeout_s}s")
    print("─" * 64)

    wall_start = time.perf_counter()

    async with httpx.AsyncClient(base_url=base_url) as client:
        coros = [
            _run_one_session(client, base_url, query, tickers, ws_timeout_s)
            for query, tickers in tasks_input
        ]
        results: list[SessionResult] = await asyncio.gather(*coros)

    wall_elapsed = time.perf_counter() - wall_start

    # ── Per-session output (verbose) ─────────────────────────────────────────
    if verbose:
        for r in results:
            status = "✓" if r.success else "✗"
            print(
                f"  {status} {r.session_id[:8]}  "
                f"{r.latency_s:5.2f}s  "
                f"events={r.event_count:3d}  "
                f"{r.query[:50]}"
                + (f"  ERROR: {r.error}" if r.error else "")
            )
        print()

    # ── Aggregate stats ───────────────────────────────────────────────────────
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_s for r in results]

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    mean = statistics.mean(latencies) if latencies else 0.0

    success_pct = len(succeeded) / len(results) * 100 if results else 0.0

    width = 64
    bar = "─" * (width - 2)
    print(f"┌─ Stress Test Results {bar[21:]}┐")
    print(f"│  Sessions    : {n_sessions:<{width-18}}│")
    print(f"│  Succeeded   : {len(succeeded)}  ({success_pct:.1f}%){' ' * (width - 22 - len(str(len(succeeded))))}│")
    print(f"│  Failed      : {len(failed):<{width-18}}│")
    print(f"│  Mean latency: {mean:<.2f} s{' ' * (width - 22)}│")
    print(f"│  p50 latency : {p50:<.2f} s{' ' * (width - 22)}│")
    print(f"│  p95 latency : {p95:<.2f} s{' ' * (width - 22)}│")
    print(f"│  p99 latency : {p99:<.2f} s{' ' * (width - 22)}│")
    print(f"│  Total wall  : {wall_elapsed:<.1f} s{' ' * (width - 22)}│")
    print(f"└{'─' * (width - 2)}┘")

    if failed:
        print("\nFailed sessions:")
        for r in failed:
            print(f"  • {r.session_id[:8]} — {r.error}")

    # Exit non-zero if >20% failure rate (useful for CI)
    if success_pct < 80:
        raise SystemExit(
            f"\n[FAIL] Success rate {success_pct:.1f}% is below 80% threshold."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent session stress test for the MIA API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/stress_test.py
              python scripts/stress_test.py --sessions 20 --base-url http://localhost:8000
              python scripts/stress_test.py --sessions 5 --ws-timeout 120 --verbose
            """
        ),
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Number of concurrent sessions to fire (default: 10)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--ws-timeout",
        type=float,
        default=180.0,
        help="Per-message WebSocket timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-session results in addition to the summary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        run_stress_test(
            base_url=args.base_url,
            n_sessions=args.sessions,
            ws_timeout_s=args.ws_timeout,
            verbose=args.verbose,
        )
    )
