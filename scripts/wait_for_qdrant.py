"""Block until the Qdrant container is accepting requests, then exit 0.

Used by ``make run``/``make ingest`` so the ≤3-command flow is robust against the
container still booting. Host/port come from env (falls back to localhost:6333),
matching the config-driven design.

Usage: ``python scripts/wait_for_qdrant.py [--timeout 60]``
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx


def wait_for_qdrant(host: str, port: int, timeout: float) -> bool:
    """Poll ``/readyz`` until ready or ``timeout`` seconds elapse. Return success."""
    url = f"http://{host}:{port}/readyz"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for Qdrant readiness.")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    host = os.environ.get("RAG__QDRANT__HOST", "localhost")
    port = int(os.environ.get("RAG__QDRANT__PORT", "6333"))

    if wait_for_qdrant(host, port, args.timeout):
        print(f"Qdrant ready at {host}:{port}")
        return 0
    print(f"Qdrant not ready at {host}:{port} after {args.timeout}s", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
