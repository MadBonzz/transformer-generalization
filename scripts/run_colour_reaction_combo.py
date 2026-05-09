from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "colour_reaction_12runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the 6 colour runs and 6 chemistry runs into one downloadable output bundle."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallel-workers", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--colour-num-base-colors", type=int, default=2000)
    parser.add_argument("--reaction-num-rows", type=int, default=100000)
    parser.add_argument("--reaction-max-scale", type=int, default=12)
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    return parser.parse_args()


def split_workers(total_workers: int) -> tuple[int, int]:
    if total_workers <= 0:
        return 6, 6
    colour_workers = min(6, max(1, total_workers // 2))
    reaction_workers = min(6, max(1, total_workers - colour_workers))
    if total_workers >= 12:
        return 6, 6
    return colour_workers, reaction_workers


def colour_command(args: argparse.Namespace, workers: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT_DIR / "colour-combination" / "run_base_experiment.py"),
        "--output-root",
        str(args.output_root / "colour"),
        "--parallel-workers",
        str(workers),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--log-every",
        str(args.log_every),
        "--checkpoint-schedule",
        "fixed",
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
        str(args.colour_num_base_colors),
    ]


def chemistry_command(args: argparse.Namespace, workers: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT_DIR / "reaction-combination" / "run_base_experiment.py"),
        "--output-root",
        str(args.output_root / "chemistry"),
        "--parallel-workers",
        str(workers),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--log-every",
        str(args.log_every),
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
        str(args.reaction_num_rows),
        "--max-scale",
        str(args.reaction_max_scale),
    ]


def write_manifest(args: argparse.Namespace, colour_workers: int, chemistry_workers: int) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "output_root": str(args.output_root),
        "subfolders": {
            "colour": str(args.output_root / "colour"),
            "chemistry": str(args.output_root / "chemistry"),
        },
        "total_runs": 12,
        "colour_runs": 6,
        "chemistry_runs": 6,
        "requested_parallel_workers": args.parallel_workers,
        "colour_parallel_workers": colour_workers,
        "chemistry_parallel_workers": chemistry_workers,
        "max_steps": args.max_steps,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "dataset_seed": args.dataset_seed,
    }
    (args.output_root / "combined_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def launch(name: str, command: list[str], log_path: Path) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(" ".join(command) + "\n\n")
    log_handle.flush()
    process = subprocess.Popen(command, cwd=ROOT_DIR, stdout=log_handle, stderr=subprocess.STDOUT)
    print(f"[LAUNCH] {name}: pid={process.pid} log={log_path}")
    return process, log_handle


def main() -> None:
    args = parse_args()
    colour_workers, chemistry_workers = split_workers(args.parallel_workers)
    write_manifest(args, colour_workers, chemistry_workers)

    colour_process, colour_log = launch(
        "colour",
        colour_command(args, colour_workers),
        args.output_root / "launcher_logs" / "colour.log",
    )
    chemistry_process, chemistry_log = launch(
        "chemistry",
        chemistry_command(args, chemistry_workers),
        args.output_root / "launcher_logs" / "chemistry.log",
    )

    processes = {"colour": (colour_process, colour_log), "chemistry": (chemistry_process, chemistry_log)}
    failed: list[str] = []
    while processes:
        finished: list[str] = []
        for name, (process, log_handle) in processes.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            finished.append(name)
            if return_code != 0:
                failed.append(name)
                print(f"[FAILED] {name}: return_code={return_code}")
            else:
                print(f"[DONE] {name}")
        for name in finished:
            del processes[name]
        if processes:
            active = ", ".join(f"{name}=pid{process.pid}" for name, (process, _) in processes.items())
            print(f"[ACTIVE] {active}")
            time.sleep(args.poll_interval_sec)

    if failed:
        raise SystemExit(f"failed suites: {', '.join(failed)}")
    print(f"[DONE] combined output bundle: {args.output_root}")


if __name__ == "__main__":
    main()
