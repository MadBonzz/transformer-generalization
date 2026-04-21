from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three grokking studies sequentially.")
    parser.add_argument("--profile", choices=("pilot", "full10"), default="pilot")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--study1-max-steps", type=int, default=10_000_000)
    parser.add_argument("--study2-max-steps", type=int, default=10_000_000)
    parser.add_argument("--study3-max-steps", type=int, default=10_000_000)
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-vram-mb", type=float, default=500.0)
    parser.add_argument("--safety-margin-mb", type=float, default=500.0)
    parser.add_argument("--per-process-overhead-mb", type=float, default=256.0)
    parser.add_argument("--min-free-system-ram-mb", type=float, default=0.0)
    parser.add_argument("--system-ram-safety-margin-mb", type=float, default=0.0)
    parser.add_argument("--per-process-ram-mb", type=float, default=0.0)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--launch-settle-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    studies = [
        {
            "script": root / "run_loss_vs_rl.py",
            "output_root": Path(args.output_root) / "study1_loss_vs_rl",
            "max_steps": args.study1_max_steps,
        },
        {
            "script": root / "run_fake_labels.py",
            "output_root": Path(args.output_root) / "study2_fake_labels",
            "max_steps": args.study2_max_steps,
        },
        {
            "script": root / "run_range_transfer.py",
            "output_root": Path(args.output_root) / "study3_range_transfer",
            "max_steps": args.study3_max_steps,
        },
    ]

    for study in studies:
        if args.parallel_workers == 1:
            command = [
                sys.executable,
                str(study["script"]),
                "--profile",
                args.profile,
                "--output-root",
                str(study["output_root"]),
                "--device",
                args.device,
                "--max-steps",
                str(study["max_steps"]),
            ]
            subprocess.run(command, check=True)
            continue

        manifest_path = Path(study["output_root"]) / "manifest.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(study["script"]),
                "--profile",
                args.profile,
                "--output-root",
                str(study["output_root"]),
                "--device",
                args.device,
                "--max-steps",
                str(study["max_steps"]),
                "--manifest-out",
                str(manifest_path),
                "--manifest-only",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(root / "launch_manifest_parallel.py"),
                "--manifest",
                str(manifest_path),
                "--max-parallel",
                str(args.parallel_workers),
                "--gpu-index",
                str(args.gpu_index),
                "--min-free-vram-mb",
                str(args.min_free_vram_mb),
                "--safety-margin-mb",
                str(args.safety_margin_mb),
                "--per-process-overhead-mb",
                str(args.per_process_overhead_mb),
                "--min-free-system-ram-mb",
                str(args.min_free_system_ram_mb),
                "--system-ram-safety-margin-mb",
                str(args.system_ram_safety_margin_mb),
                "--per-process-ram-mb",
                str(args.per_process_ram_mb),
                "--poll-interval-sec",
                str(args.poll_interval_sec),
                "--launch-settle-sec",
                str(args.launch_settle_sec),
                "--summary-out",
                str(Path(study["output_root"]) / "summary.csv"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
