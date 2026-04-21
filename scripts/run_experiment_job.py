from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.job_runner import read_manifest, run_job_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single experiment job from a manifest.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--job-index", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = read_manifest(args.manifest)
    run_job_spec(jobs[args.job_index])


if __name__ == "__main__":
    main()
