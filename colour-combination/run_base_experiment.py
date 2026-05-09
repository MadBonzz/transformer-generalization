from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from grokking_transformer.experiment_utils import (  # noqa: E402
    RunConfig,
    run_training,
    transformer_architecture_name,
    transformer_run_prefix,
)
from grokking_transformer.tasks import DatasetInfo, TaskDataset  # noqa: E402


DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "mixbox_base_case"
DEFAULT_NUM_BASE_COLORS = 2000


def dataset_name(num_base_colors: int) -> str:
    return f"colour_mixing_mixbox_100k_{num_base_colors}base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 6-run color-combination baseline: 2 transformer depths x 3 seeds."
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-schedule",
        type=str,
        default="fixed",
        choices=["staged", "fixed", "none"],
        help="staged: 1k to 10k, 5k to 50k, 10k after 50k; fixed: use --checkpoint-every-steps.",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--num-base-colors", type=int, default=DEFAULT_NUM_BASE_COLORS)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--launch-settle-sec", type=float, default=1.0)
    parser.add_argument("--run-single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-layer", type=int, choices=[1, 2], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-seed", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def staged_checkpoint_steps(max_steps: int) -> tuple[int, ...]:
    steps = set(range(1000, min(max_steps, 10000) + 1, 1000))
    if max_steps > 10000:
        steps.update(range(15000, min(max_steps, 50000) + 1, 5000))
    if max_steps > 50000:
        steps.update(range(60000, max_steps + 1, 10000))
    return tuple(sorted(steps))


def checkpoint_steps_for_args(args: argparse.Namespace) -> tuple[int, ...] | None:
    if args.checkpoint_schedule == "none":
        return None
    if args.checkpoint_schedule == "staged":
        return staged_checkpoint_steps(args.max_steps)
    return None


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir
    return args.output_root / "dataset" / dataset_name(args.num_base_colors)


def run_dir_for(output_root: Path, *, layers: int, seed: int, learning_rate: float, weight_decay: float, batch_size: int) -> Path:
    prefix = transformer_run_prefix(layers)
    run_name = f"{prefix}_mixbox_color_mix_seed{seed}_lr{learning_rate}_wd{weight_decay}_bs{batch_size}"
    return output_root / "runs" / run_name


def generate_dataset(args: argparse.Namespace, dataset_dir: Path) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "dataset_generation.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(THIS_DIR / "generate_mixbox_dataset.py"),
            "--output-dir",
            str(dataset_dir),
            "--seed",
            str(args.dataset_seed),
            "--num-base-colors",
            str(args.num_base_colors),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args, output=completed.stdout)


def load_split_dataset(dataset_dir: Path, split: str) -> TaskDataset:
    inputs: list[list[int]] = []
    targets: list[int] = []
    with (dataset_dir / "tokenized_examples.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] != split:
                continue
            inputs.append(
                [
                    int(row["input_0_hex_1_id"]),
                    int(row["input_1_hex_2_id"]),
                    int(row["input_2_t_id"]),
                ]
            )
            targets.append(int(row["target_hex_id"]))
    if not inputs:
        raise ValueError(f"dataset split {split!r} is empty in {dataset_dir}")
    return TaskDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))


def load_dataset_metadata(dataset_dir: Path) -> dict[str, object]:
    with (dataset_dir / "metadata.json").open(encoding="utf-8") as file:
        return json.load(file)


def build_dataset_info(metadata: dict[str, object]) -> DatasetInfo:
    return DatasetInfo(
        vocab_size=int(metadata["vocab_size"]),
        target_vocab_size=int(metadata["target_vocab_size"]),
        seq_len=int(metadata["sequence_length"]),
        eq_token_id=-1,
        operator_token_ids={},
    )


def run_one(args: argparse.Namespace, *, layers: int, seed: int) -> dict[str, object]:
    output_dir = run_dir_for(
        args.output_root,
        layers=layers,
        seed=seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )
    metadata = load_dataset_metadata(args.dataset_dir)
    config = RunConfig(
        study_name="colour_combination_base",
        model_type="transformer",
        objective="ce",
        seed=seed,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        output_dir=str(output_dir),
        device=resolve_device(args.device),
        transformer_n_layers=layers,
        checkpoint_every_steps=(
            args.checkpoint_every_steps if args.checkpoint_schedule == "fixed" else None
        ),
        checkpoint_steps=checkpoint_steps_for_args(args),
        metadata={
            "dataset": str(metadata["name"]),
            "dataset_kind": str(metadata["dataset_kind"]),
            "dataset_seed": args.dataset_seed,
            "architecture": transformer_architecture_name(layers),
            "mixing_rule": str(metadata["mixing_rule"]),
            "mixing_model": str(metadata["mixing_model"]),
            "num_hex_tokens": int(metadata["num_hex_tokens"]),
            "num_t_tokens": int(metadata["num_t_tokens"]),
            "split_fractions": metadata["split_fractions"],
        },
    )
    train_dataset = load_split_dataset(args.dataset_dir, "train")
    val_dataset = load_split_dataset(args.dataset_dir, "val")
    test_dataset = load_split_dataset(args.dataset_dir, "test")
    return run_training(
        config=config,
        info=build_dataset_info(metadata),
        train_dataset=train_dataset,
        eval_datasets={"val": val_dataset, "test": test_dataset},
        summary_csv_path=output_dir / "summary_row.csv",
    )


def build_jobs(args: argparse.Namespace) -> list[tuple[int, int]]:
    return [(layers, seed) for layers in args.layers for seed in args.seeds]


def write_manifest(args: argparse.Namespace, jobs: list[tuple[int, int]], dataset_dir: Path) -> None:
    manifest = {
        "dataset_dir": str(dataset_dir),
        "output_root": str(args.output_root),
        "dataset_kind": "mixbox_pigment_like",
        "dataset_seed": args.dataset_seed,
        "num_rows": 100000,
        "num_base_colours": args.num_base_colors,
        "split_fractions": {"train": 0.5, "val": 0.25, "test": 0.25},
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "checkpoint_schedule": args.checkpoint_schedule,
        "checkpoint_every_steps": (
            args.checkpoint_every_steps if args.checkpoint_schedule == "fixed" else None
        ),
        "checkpoint_steps": list(staged_checkpoint_steps(args.max_steps))
        if args.checkpoint_schedule == "staged"
        else None,
        "dataset_generation_log": str(args.output_root / "dataset_generation.log"),
        "jobs": [
            {
                "layers": layers,
                "seed": seed,
                "output_dir": str(
                    run_dir_for(
                        args.output_root,
                        layers=layers,
                        seed=seed,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        batch_size=args.batch_size,
                    )
                ),
            }
            for layers, seed in jobs
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def aggregate_summary(output_root: Path, jobs: list[tuple[int, int]], args: argparse.Namespace) -> None:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for layers, seed in jobs:
        summary_path = run_dir_for(
            output_root,
            layers=layers,
            seed=seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
        ) / "summary_row.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing per-run summary: {summary_path}")
        with summary_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rows.append(row)
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def single_run_command(args: argparse.Namespace, *, layers: int, seed: int, dataset_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(THIS_DIR / "run_base_experiment.py"),
        "--run-single",
        "--single-layer",
        str(layers),
        "--single-seed",
        str(seed),
        "--dataset-dir",
        str(dataset_dir),
        "--output-root",
        str(args.output_root),
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
    ]
    return command


def read_progress_state(output_dir: Path) -> dict[str, object] | None:
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return None
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def format_running_status(item: dict[str, object]) -> str:
    output_dir = item["output_dir"]
    assert isinstance(output_dir, Path)
    progress_state = read_progress_state(output_dir)
    label = f"layers={item['layers']} seed={item['seed']}"
    if progress_state is None:
        return f"{label} starting"
    step = int(progress_state.get("step", 0))
    max_steps = int(progress_state.get("max_steps", 0))
    if max_steps <= 0:
        return f"{label} step={step}"
    return f"{label} {step}/{max_steps} ({100.0 * step / max_steps:.1f}%)"


def run_parallel(args: argparse.Namespace, jobs: list[tuple[int, int]], dataset_dir: Path) -> None:
    max_parallel = len(jobs) if args.parallel_workers <= 0 else args.parallel_workers
    pending = list(jobs)
    running: list[dict[str, object]] = []
    logs_dir = args.output_root / "launcher_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    last_active_report_at = 0.0

    progress = tqdm(total=len(jobs), desc="colour-combination runs", unit="run", dynamic_ncols=True)
    tqdm.write(
        f"[SCHEDULER START] total_runs={len(jobs)} | max_parallel={max_parallel} | output={args.output_root}"
    )
    while pending or running:
        still_running: list[dict[str, object]] = []
        for item in running:
            process = item["process"]
            assert isinstance(process, subprocess.Popen)
            return_code = process.poll()
            if return_code is None:
                still_running.append(item)
                continue
            log_handle = item["log_handle"]
            assert not isinstance(log_handle, subprocess.Popen)
            log_handle.close()
            if return_code != 0:
                progress.close()
                raise subprocess.CalledProcessError(return_code, process.args)
            completed += 1
            progress.update(1)
            tqdm.write(f"[DONE] layers={item['layers']} seed={item['seed']} | completed={completed}/{len(jobs)}")
        running = still_running

        while pending and len(running) < max_parallel:
            layers, seed = pending.pop(0)
            log_path = logs_dir / f"layers{layers}_seed{seed}.log"
            output_dir = run_dir_for(
                args.output_root,
                layers=layers,
                seed=seed,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
            )
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                single_run_command(args, layers=layers, seed=seed, dataset_dir=dataset_dir),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "process": process,
                    "log_handle": log_handle,
                    "layers": layers,
                    "seed": seed,
                    "output_dir": output_dir,
                }
            )
            tqdm.write(f"[LAUNCH] layers={layers} seed={seed} pid={process.pid} log={log_path}")
            time.sleep(args.launch_settle_sec)

        if pending or running:
            progress.set_postfix_str(f"running={len(running)}, pending={len(pending)}")
        if running and time.monotonic() - last_active_report_at >= max(5.0, args.poll_interval_sec):
            active_status = "; ".join(format_running_status(item) for item in running[:4])
            if len(running) > 4:
                active_status += "; ..."
            tqdm.write(f"[ACTIVE] completed={completed}/{len(jobs)} | {active_status}")
            last_active_report_at = time.monotonic()
            time.sleep(args.poll_interval_sec)
    progress.close()
    tqdm.write(f"[SCHEDULER DONE] completed={completed}/{len(jobs)}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.dataset_dir = resolve_dataset_dir(args)

    if args.run_single:
        if args.single_layer is None or args.single_seed is None:
            raise ValueError("--run-single requires --single-layer and --single-seed")
        run_one(args, layers=args.single_layer, seed=args.single_seed)
        return

    print(f"Generating deterministic dataset in {args.dataset_dir}")
    generate_dataset(args, args.dataset_dir)
    jobs = build_jobs(args)
    write_manifest(args, jobs, args.dataset_dir)

    if args.parallel_workers == 1:
        for layers, seed in jobs:
            print(f"Running color-combination baseline: layers={layers}, seed={seed}")
            run_one(args, layers=layers, seed=seed)
    else:
        run_parallel(args, jobs, args.dataset_dir)

    aggregate_summary(args.output_root, jobs, args)
    print(f"Completed {len(jobs)} runs. Bundle: {args.output_root}")
    print(f"Summary: {args.output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
