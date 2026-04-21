from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.experiment_utils import RunConfig
from grokking_transformer.job_runner import job_run_name, job_study_name, run_job_spec, write_manifest
from grokking_transformer.logging_utils import ensure_dir


def parse_csv_list(raw: str, cast) -> list:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study 3: mixed-task range transfer between modular addition and multiplication.")
    parser.add_argument("--profile", choices=("pilot", "full10"), default="pilot")
    parser.add_argument("--output-root", type=str, default="outputs/study3_range_transfer")
    parser.add_argument("--modulus", type=int, default=251, help="Numeric input range size for the mixed-task study.")
    parser.add_argument("--output-modulus", type=int, default=None, help="Optional output modulus. Defaults to the input modulus.")
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--train-fractions", type=str, default="")
    parser.add_argument("--add-offset", type=int, default=0)
    parser.add_argument("--mul-offset", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--manifest-out", type=str, default="")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def defaults(profile: str) -> dict[str, object]:
    if profile == "full10":
        return {
            "seeds": list(range(10)),
            "lrs": [1e-3, 3e-4],
            "weight_decays": [0.5, 1.0],
            "batch_sizes": [128, 256],
            "train_fractions": [0.1, 0.3, 0.5],
        }
    return {
        "seeds": [0, 1],
        "lrs": [1e-3],
        "weight_decays": [1.0],
        "batch_sizes": [128],
        "train_fractions": [0.3],
    }


def build_job_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    preset = defaults(args.profile)
    output_modulus = args.modulus if args.output_modulus is None else args.output_modulus
    seeds = parse_csv_list(args.seeds, int) if args.seeds else preset["seeds"]
    if args.train_fractions:
        train_fractions = parse_csv_list(args.train_fractions, float)
    elif args.train_fraction is not None:
        train_fractions = [args.train_fraction]
    else:
        train_fractions = preset["train_fractions"]
    output_root = ensure_dir(args.output_root)
    summary_csv = output_root / "summary.csv"
    manifest_mode = bool(args.manifest_out)
    jobs: list[dict[str, object]] = []

    for train_fraction in train_fractions:
        for seed, lr, weight_decay, batch_size in itertools.product(
            seeds,
            preset["lrs"],
            preset["weight_decays"],
            preset["batch_sizes"],
        ):
            run_name = f"transformer_tf{train_fraction}_ce_seed{seed}_lr{lr}_wd{weight_decay}_bs{batch_size}"
            config = RunConfig(
                study_name="study3_range_transfer",
                model_type="transformer",
                objective="ce",
                seed=seed,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_steps=args.max_steps,
                eval_every=args.eval_every,
                log_every=args.log_every,
                device=args.device,
                output_dir=str(output_root / run_name),
                metadata={
                    "input_modulus": args.modulus,
                    "output_modulus": output_modulus,
                    "train_fraction": train_fraction,
                    "data_seed": 0,
                    "add_offset": args.add_offset,
                    "mul_offset": args.mul_offset,
                },
            )
            jobs.append(
                make_job_spec(
                    config,
                    task_spec={
                        "kind": "range_transfer",
                        "modulus": args.modulus,
                        "output_modulus": output_modulus,
                        "train_fraction": train_fraction,
                        "seed": 0,
                        "add_offset": args.add_offset,
                        "mul_offset": args.mul_offset,
                        "include_zero": True,
                    },
                    summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
                )
            )

    return jobs


def make_job_spec(config: RunConfig, task_spec: dict[str, object], summary_csv_path: Path) -> dict[str, object]:
    return {
        "task": task_spec,
        "run_config": {
            "study_name": config.study_name,
            "model_type": config.model_type,
            "objective": config.objective,
            "seed": config.seed,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "max_steps": config.max_steps,
            "eval_every": config.eval_every,
            "log_every": config.log_every,
            "device": config.device,
            "output_dir": config.output_dir,
            "full_batch": config.full_batch,
            "mlp_hidden_dim": config.mlp_hidden_dim,
            "metadata": config.metadata,
            "grpo": None,
            "ppo": None,
        },
        "summary_csv_path": str(summary_csv_path),
    }


def _summary_target(output_root: Path, run_name: str, manifest_mode: bool, summary_csv: Path) -> Path:
    if not manifest_mode:
        return summary_csv
    summary_dir = output_root / "_summary_rows"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / f"{run_name}.csv"


def main() -> None:
    args = parse_args()
    jobs = build_job_specs(args)
    study_name = job_study_name(jobs[0]) if jobs else "study3_range_transfer"
    if args.manifest_out:
        write_manifest(args.manifest_out, jobs)
        tqdm.write(f"[MANIFEST] {study_name}: wrote {len(jobs)} runs to {args.manifest_out}")
        if args.manifest_only:
            return
    tqdm.write(f"[STUDY START] {study_name}: {len(jobs)} runs -> {args.output_root}")
    run_progress = tqdm(total=len(jobs), desc=f"{study_name} runs", unit="run", dynamic_ncols=True)
    for run_index, job in enumerate(jobs, start=1):
        tqdm.write(f"[RUN {run_index}/{len(jobs)} START] {job_run_name(job)}")
        run_job_spec(job)
        run_progress.update(1)
        tqdm.write(f"[RUN {run_index}/{len(jobs)} DONE] {job_run_name(job)}")
    run_progress.close()
    tqdm.write(f"[STUDY DONE] {study_name}: {len(jobs)}/{len(jobs)} runs complete")


if __name__ == "__main__":
    main()
