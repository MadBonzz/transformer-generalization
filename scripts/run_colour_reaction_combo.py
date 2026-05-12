from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "colour_reaction_12runs"
REQUIRED_MODULES = {
    "torch": "torch",
    "tqdm": "tqdm",
    "numpy": "numpy",
    "pandas": "pandas",
    "mixbox": "pymixbox",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the 6 colour runs and 6 chemistry runs into one downloadable output bundle."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--parallel-workers", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument(
        "--checkpoint-schedule",
        type=str,
        default="staged",
        choices=["staged", "fixed", "none"],
        help="staged saves every 1k through 25k, then every --checkpoint-every-steps.",
    )
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
    parser.add_argument("--reaction-element-max-scale", type=int, default=1)
    parser.add_argument("--reaction-element-synthesis-fraction", type=float, default=0.50)
    parser.add_argument("--reaction-no-reaction-fraction", type=float, default=0.25)
    parser.add_argument("--reaction-split-strategy", choices=["generalization", "random"], default="generalization")
    parser.add_argument("--poll-interval-sec", type=float, default=5.0)
    parser.add_argument("--progress-report-every-sec", type=float, default=30.0)
    return parser.parse_args()


def split_workers(total_workers: int) -> tuple[int, int]:
    if total_workers <= 0:
        return 6, 6
    colour_workers = min(6, max(1, total_workers // 2))
    reaction_workers = min(6, max(1, total_workers - colour_workers))
    if total_workers >= 12:
        return 6, 6
    return colour_workers, reaction_workers


def check_required_modules() -> None:
    missing_packages = [
        package_name
        for module_name, package_name in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_packages:
        packages = " ".join(dict.fromkeys(missing_packages))
        raise SystemExit(
            "Missing required Python packages: "
            f"{packages}\n"
            "Install dependencies in the active environment, then rerun:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
            "If PyTorch is already installed and you only need dataset dependencies:\n"
            f"  {sys.executable} -m pip install pandas numpy pymixbox matplotlib scikit-learn"
        )


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
        str(args.reaction_num_rows),
        "--max-scale",
        str(args.reaction_max_scale),
        "--element-max-scale",
        str(args.reaction_element_max_scale),
        "--element-synthesis-fraction",
        str(args.reaction_element_synthesis_fraction),
        "--no-reaction-fraction",
        str(args.reaction_no_reaction_fraction),
        "--split-strategy",
        args.reaction_split_strategy,
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
        "full_train_eval_logged": True,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "checkpoint_schedule": args.checkpoint_schedule,
        "poll_interval_sec": args.poll_interval_sec,
        "progress_report_every_sec": args.progress_report_every_sec,
        "dataset_seed": args.dataset_seed,
        "reaction_element_synthesis_fraction": args.reaction_element_synthesis_fraction,
        "reaction_no_reaction_fraction": args.reaction_no_reaction_fraction,
        "reaction_split_strategy": args.reaction_split_strategy,
        "reaction_element_max_scale": args.reaction_element_max_scale,
    }
    (args.output_root / "combined_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def read_text_tail(path: Path, *, max_lines: int = 100) -> str:
    if not path.exists():
        return f"<missing log: {path}>"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return "\n".join(lines[-max_lines:])


def read_json_safely(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def discover_run_progress(output_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for suite_name in ("colour", "chemistry"):
        runs_dir = output_root / suite_name / "runs"
        if not runs_dir.exists():
            continue
        for progress_path in sorted(runs_dir.glob("*/progress.json")):
            payload = read_json_safely(progress_path)
            if payload is None:
                continue
            step = int(payload.get("step", 0))
            max_steps = int(payload.get("max_steps", 0))
            status = str(payload.get("status", "unknown"))
            records.append(
                {
                    "suite": suite_name,
                    "run_name": progress_path.parent.name,
                    "status": status,
                    "step": step,
                    "max_steps": max_steps,
                    "progress_fraction": 1.0 if max_steps <= 0 else min(max(step / max_steps, 0.0), 1.0),
                    "path": progress_path,
                    "train_label_accuracy": payload.get("train_label_accuracy"),
                    "train_exact_match_accuracy": payload.get("train_exact_match_accuracy"),
                    "test_label_accuracy": payload.get("test_label_accuracy"),
                    "test_exact_match_accuracy": payload.get("test_exact_match_accuracy"),
                }
            )
    return records


def format_run_progress(record: dict[str, object]) -> str:
    step = int(record["step"])
    max_steps = int(record["max_steps"])
    status = str(record["status"])
    run_name = str(record["run_name"])
    suite = str(record["suite"])
    metric_parts: list[str] = []
    for key, label in (
        ("train_label_accuracy", "train_acc"),
        ("train_exact_match_accuracy", "train_exact"),
        ("test_label_accuracy", "test_acc"),
        ("test_exact_match_accuracy", "test_exact"),
    ):
        value = record.get(key)
        if isinstance(value, (int, float)):
            metric_parts.append(f"{label}={value:.3f}")
    if max_steps <= 0:
        base = f"{suite}/{run_name} {status} step={step}"
        return f"{base} {' '.join(metric_parts)}".rstrip()
    percent = 100.0 * step / max_steps
    base = f"{suite}/{run_name} {status} {step}/{max_steps} ({percent:.1f}%)"
    return f"{base} {' '.join(metric_parts)}".rstrip()


def print_progress_snapshot(output_root: Path, *, total_runs: int, show_completed: bool = False) -> int:
    records = discover_run_progress(output_root)
    completed = sum(1 for record in records if str(record["status"]) == "completed")
    failed = sum(1 for record in records if str(record["status"]) == "failed")
    running = [record for record in records if str(record["status"]) not in {"completed", "failed"}]
    tqdm.write(
        f"[RUN PROGRESS] discovered={len(records)}/{total_runs} | "
        f"completed={completed}/{total_runs} | failed={failed} | running={len(running)}"
    )
    displayed = records if show_completed else [record for record in records if str(record["status"]) != "completed"]
    for record in sorted(displayed, key=lambda item: (str(item["suite"]), str(item["run_name"]))):
        tqdm.write(f"  {format_run_progress(record)}")
    return completed


def launch(name: str, command: list[str], log_path: Path) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write(" ".join(command) + "\n\n")
    log_handle.flush()
    process = subprocess.Popen(command, cwd=ROOT_DIR, stdout=log_handle, stderr=subprocess.STDOUT)
    print(f"[LAUNCH] {name}: pid={process.pid} log={log_path}")
    return process, log_handle


def terminate_remaining(processes: dict[str, tuple[subprocess.Popen, object]]) -> None:
    for name, (process, _) in processes.items():
        if process.poll() is None:
            tqdm.write(f"[TERMINATE] {name}: pid={process.pid}")
            process.terminate()
    deadline = time.monotonic() + 15.0
    for name, (process, _) in processes.items():
        if process.poll() is not None:
            continue
        remaining = max(deadline - time.monotonic(), 0.0)
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            tqdm.write(f"[KILL] {name}: pid={process.pid}")
            process.kill()


def main() -> None:
    args = parse_args()
    check_required_modules()
    colour_workers, chemistry_workers = split_workers(args.parallel_workers)
    write_manifest(args, colour_workers, chemistry_workers)
    total_runs = 12

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
    progress = tqdm(total=total_runs, desc="combined runs", unit="run", dynamic_ncols=True)
    last_completed = 0
    last_progress_report_at = 0.0
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
                tqdm.write(f"[FAILED] {name}: return_code={return_code}")
                suite_log_path = args.output_root / "launcher_logs" / f"{name}.log"
                tqdm.write(f"[FAILED SUITE LOG TAIL] {suite_log_path}\n{read_text_tail(suite_log_path)}")
                remaining = {key: value for key, value in processes.items() if key != name}
                terminate_remaining(remaining)
            else:
                tqdm.write(f"[DONE] {name}")
        for name in finished:
            del processes[name]

        now = time.monotonic()
        should_report = bool(finished) or now - last_progress_report_at >= args.progress_report_every_sec
        if should_report:
            completed = print_progress_snapshot(args.output_root, total_runs=total_runs)
            if completed > last_completed:
                progress.update(completed - last_completed)
                last_completed = completed
            last_progress_report_at = now
        if processes:
            active = ", ".join(f"{name}=pid{process.pid}" for name, (process, _) in processes.items())
            tqdm.write(f"[ACTIVE SUITES] {active}")
            time.sleep(args.poll_interval_sec)

    completed = print_progress_snapshot(args.output_root, total_runs=total_runs, show_completed=True)
    if completed > last_completed:
        progress.update(completed - last_completed)
    progress.close()
    if failed:
        raise SystemExit(f"failed suites: {', '.join(failed)}")
    print(f"[DONE] combined output bundle: {args.output_root}")


if __name__ == "__main__":
    main()
