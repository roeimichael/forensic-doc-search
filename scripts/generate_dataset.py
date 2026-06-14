"""Standalone shim so the generator runs without the console script installed.

Equivalent to ``rag generate``. Lets a reviewer do ``python scripts/generate_dataset.py``.
"""

from __future__ import annotations

import argparse

from ragforce.dataset import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic forensic corpus.")
    parser.add_argument("--n", type=int, default=120, help="Number of documents (50–200).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument("--out", default="data/generated", help="Output directory.")
    args = parser.parse_args()
    out = generate(n=args.n, seed=args.seed, out_dir=args.out)
    print(f"Generated dataset → {out}")


if __name__ == "__main__":
    main()
