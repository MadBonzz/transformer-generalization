from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.plotting import generate_run_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate metric plots for completed experiment runs.")
    parser.add_argument("--root", required=True, type=str, help="Run directory or parent directory containing runs.")
    return parser.parse_args()


def _iter_run_dirs(root: Path) -> list[Path]:
    if (root / "metrics.jsonl").exists():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_dir() and (path / "metrics.jsonl").exists())


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_dirs = _iter_run_dirs(root)
    for run_dir in run_dirs:
        plot_paths = generate_run_plots(run_dir)
        print(f"{run_dir}: {len(plot_paths)} plot files")


if __name__ == "__main__":
    main()
