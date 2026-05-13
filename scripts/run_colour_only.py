from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "colour_cloud_bundle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch only the 12 colour runs: 4 layer depths x 3 seeds.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallel-workers", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--checkpoint-schedule", choices=["staged", "fixed", "none"], default="fixed")
    parser.add_argument("--checkpoint-every-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--num-base-colors", type=int, default=2000)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT_DIR / "colour-combination" / "run_base_experiment.py"),
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
        "--num-base-colors",
        str(args.num_base_colors),
        "--poll-interval-sec",
        str(args.poll_interval_sec),
    ]


def validate_requested_device(device: str) -> None:
    if device != "cuda":
        return
    if importlib.util.find_spec("torch") is None:
        raise SystemExit("CUDA was requested, but PyTorch is not installed in this environment.")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested with --device cuda, but torch.cuda.is_available() is false. "
            "Check `nvidia-smi`, the cloud instance GPU attachment, and whether this environment has a CUDA PyTorch build."
        )


def main() -> None:
    args = parse_args()
    validate_requested_device(args.device)
    command = build_command(args)
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.call(command, cwd=ROOT_DIR))


if __name__ == "__main__":
    main()
