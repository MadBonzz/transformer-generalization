from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.job_runner import aggregate_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed manifest job results into a summary CSV.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--summary-out", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_results(args.manifest, args.summary_out)


if __name__ == "__main__":
    main()
