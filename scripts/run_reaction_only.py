from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "reaction_cloud_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch only the 6 chemistry reaction runs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallel-workers", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--checkpoint-schedule", choices=["staged", "fixed", "none"], default="staged")
    parser.add_argument("--checkpoint-every-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--num-rows", type=int, default=100000)
    parser.add_argument("--max-scale", type=int, default=12)
    parser.add_argument("--element-max-scale", type=int, default=1)
    parser.add_argument("--element-synthesis-fraction", type=float, default=0.50)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT_DIR / "reaction-combination" / "run_base_experiment.py"),
        "--output-root",
        str(args.output_root),
        "--parallel-workers",
        str(args.parallel_workers),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--log-every",
        str(args.log_every),
        "--checkpoint-schedule",
        args.checkpoint_schedule,
        "--checkpoint-every-steps",
        str(args.checkpoint_every_steps),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--dataset-seed",
        str(args.dataset_seed),
        "--num-rows",
        str(args.num_rows),
        "--max-scale",
        str(args.max_scale),
        "--element-max-scale",
        str(args.element_max_scale),
        "--element-synthesis-fraction",
        str(args.element_synthesis_fraction),
        "--poll-interval-sec",
        str(args.poll_interval_sec),
    ]


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT_DIR))


if __name__ == "__main__":
    main()
