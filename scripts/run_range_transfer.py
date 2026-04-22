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


BASELINE_ABLATION_SCENARIO = "contiguous_partitioned_ops"
SCENARIO_CHOICES = (
    "contiguous_partitioned_ops",
    "contiguous_both_ops_within_set",
    "contiguous_both_ops_all_pairs",
    "contiguous_partitioned_ops_no_task_token",
    "interleaved10_partitioned_ops",
    "interleaved20_partitioned_ops",
)
PROFILE_DEFAULTS = {
    "full10": {
        "seeds": [0, 1, 2],
        "lrs": [1e-3, 3e-4],
        "weight_decays": [0.5, 1.0],
        "batch_sizes": [128, 256],
        "train_fractions": [0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
    },
    "pilot": {
        "seeds": [0, 1],
        "lrs": [1e-3],
        "weight_decays": [1.0],
        "batch_sizes": [256],
        "train_fractions": [0.5],
    },
}
SCENARIO_SPECS = {
    "contiguous_partitioned_ops": {
        "task_scenario": "partitioned_ops",
        "use_task_token": True,
        "interleave_chunk_size": None,
    },
    "contiguous_both_ops_within_set": {
        "task_scenario": "both_ops_within_set",
        "use_task_token": True,
        "interleave_chunk_size": None,
    },
    "contiguous_both_ops_all_pairs": {
        "task_scenario": "both_ops_all_pairs",
        "use_task_token": True,
        "interleave_chunk_size": None,
    },
    "contiguous_partitioned_ops_no_task_token": {
        "task_scenario": "partitioned_ops",
        "use_task_token": False,
        "interleave_chunk_size": None,
    },
    "interleaved10_partitioned_ops": {
        "task_scenario": "interleaved_partitioned_ops",
        "use_task_token": True,
        "interleave_chunk_size": 10,
    },
    "interleaved20_partitioned_ops": {
        "task_scenario": "interleaved_partitioned_ops",
        "use_task_token": True,
        "interleave_chunk_size": 20,
    },
}


def parse_csv_list(raw: str, cast) -> list:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study 3: partitioned-number multitask transfer experiments.")
    parser.add_argument("--profile", choices=("pilot", "full10"), default="pilot")
    parser.add_argument("--output-root", type=str, default="outputs/study3_range_transfer")
    parser.add_argument("--sweep-mode", choices=("grid", "baseline_ablation"), default="grid")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIO_CHOICES))
    parser.add_argument("--modulus", type=int, default=131)
    parser.add_argument("--output-modulus", type=int, default=None)
    parser.add_argument("--add-set-size", type=int, default=66)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--train-fractions", type=str, default="")
    parser.add_argument("--lrs", type=str, default="")
    parser.add_argument("--weight-decays", type=str, default="")
    parser.add_argument("--batch-sizes", type=str, default="")
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--baseline-weight-decay", type=float, default=1.0)
    parser.add_argument("--baseline-batch-size", type=int, default=256)
    parser.add_argument("--baseline-train-fraction", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--manifest-out", type=str, default="")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def defaults(profile: str) -> dict[str, object]:
    return PROFILE_DEFAULTS[profile]


def _scenario_spec(name: str) -> dict[str, object]:
    try:
        return SCENARIO_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported Study 3 scenario={name}") from exc


def _resolve_train_fractions(args: argparse.Namespace, preset: dict[str, object]) -> list[float]:
    if args.train_fractions:
        return parse_csv_list(args.train_fractions, float)
    if args.train_fraction is not None:
        return [args.train_fraction]
    return list(preset["train_fractions"])


def _resolve_sweep_values(args: argparse.Namespace, preset: dict[str, object]) -> tuple[list[float], list[float], list[int], list[float]]:
    lrs = parse_csv_list(args.lrs, float) if args.lrs else list(preset["lrs"])
    weight_decays = parse_csv_list(args.weight_decays, float) if args.weight_decays else list(preset["weight_decays"])
    batch_sizes = parse_csv_list(args.batch_sizes, int) if args.batch_sizes else list(preset["batch_sizes"])
    train_fractions = _resolve_train_fractions(args, preset)
    return lrs, weight_decays, batch_sizes, train_fractions


def _baseline_point(args: argparse.Namespace) -> tuple[float, float, int, float]:
    return (
        args.baseline_lr,
        args.baseline_weight_decay,
        args.baseline_batch_size,
        args.baseline_train_fraction,
    )


def _candidate_points(
    *,
    args: argparse.Namespace,
    preset: dict[str, object],
) -> list[tuple[float, float, int, float]]:
    lrs, weight_decays, batch_sizes, train_fractions = _resolve_sweep_values(args, preset)

    if args.sweep_mode == "baseline_ablation":
        points: list[tuple[float, float, int, float]] = []
        seen: set[tuple[float, float, int, float]] = set()

        def add_point(lr: float, weight_decay: float, batch_size: int, train_fraction: float) -> None:
            point = (lr, weight_decay, batch_size, train_fraction)
            if point not in seen:
                seen.add(point)
                points.append(point)

        add_point(*_baseline_point(args))
        for lr in lrs:
            add_point(lr, args.baseline_weight_decay, args.baseline_batch_size, args.baseline_train_fraction)
        for weight_decay in weight_decays:
            add_point(args.baseline_lr, weight_decay, args.baseline_batch_size, args.baseline_train_fraction)
        for batch_size in batch_sizes:
            add_point(args.baseline_lr, args.baseline_weight_decay, batch_size, args.baseline_train_fraction)
        for train_fraction in train_fractions:
            add_point(args.baseline_lr, args.baseline_weight_decay, args.baseline_batch_size, train_fraction)
        return points

    return [
        (lr, weight_decay, batch_size, train_fraction)
        for train_fraction in train_fractions
        for lr, weight_decay, batch_size in itertools.product(lrs, weight_decays, batch_sizes)
    ]


def _scenario_candidate_points(
    scenario_name: str,
    *,
    args: argparse.Namespace,
    preset: dict[str, object],
) -> list[tuple[float, float, int, float]]:
    if scenario_name == BASELINE_ABLATION_SCENARIO:
        return _candidate_points(args=args, preset=preset)
    return [_baseline_point(args)]


def _metadata_payload(
    *,
    scenario_name: str,
    scenario_spec: dict[str, object],
    input_modulus: int,
    output_modulus: int,
    train_fraction: float,
    add_set_size: int,
) -> dict[str, object]:
    return {
        "scenario_name": scenario_name,
        "task_scenario": scenario_spec["task_scenario"],
        "input_modulus": input_modulus,
        "output_modulus": output_modulus,
        "train_fraction": train_fraction,
        "data_seed": 0,
        "add_set_size": add_set_size,
        "use_task_token": scenario_spec["use_task_token"],
        "interleave_chunk_size": scenario_spec["interleave_chunk_size"],
    }


def _study3_task_spec(
    *,
    args: argparse.Namespace,
    output_modulus: int,
    train_fraction: float,
    scenario_spec: dict[str, object],
) -> dict[str, object]:
    return {
        "kind": "study3_variant",
        "modulus": args.modulus,
        "output_modulus": output_modulus,
        "train_fraction": train_fraction,
        "seed": 0,
        "scenario": scenario_spec["task_scenario"],
        "use_task_token": scenario_spec["use_task_token"],
        "add_set_size": args.add_set_size,
        "interleave_chunk_size": scenario_spec["interleave_chunk_size"],
    }


def build_job_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    preset = defaults(args.profile)
    seeds = parse_csv_list(args.seeds, int) if args.seeds else preset["seeds"]
    scenarios = parse_str_list(args.scenarios)
    invalid = [scenario for scenario in scenarios if scenario not in SCENARIO_CHOICES]
    if invalid:
        raise ValueError(f"unsupported Study 3 scenarios: {', '.join(invalid)}")

    output_modulus = args.modulus if args.output_modulus is None else args.output_modulus
    output_root = ensure_dir(args.output_root)
    summary_csv = output_root / "summary.csv"
    manifest_mode = bool(args.manifest_out)
    jobs: list[dict[str, object]] = []

    for scenario_name in scenarios:
        scenario_spec = _scenario_spec(scenario_name)
        scenario_root = output_root / scenario_name
        candidate_points = _scenario_candidate_points(scenario_name, args=args, preset=preset)
        for lr, weight_decay, batch_size, train_fraction in candidate_points:
            for seed in seeds:
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
                    output_dir=str(scenario_root / run_name),
                    metadata=_metadata_payload(
                        scenario_name=scenario_name,
                        scenario_spec=scenario_spec,
                        input_modulus=args.modulus,
                        output_modulus=output_modulus,
                        train_fraction=train_fraction,
                        add_set_size=args.add_set_size,
                    ),
                )
                jobs.append(
                    make_job_spec(
                        config,
                        task_spec=_study3_task_spec(
                            args=args,
                            output_modulus=output_modulus,
                            train_fraction=train_fraction,
                            scenario_spec=scenario_spec,
                        ),
                        summary_csv_path=_summary_target(output_root, scenario_name, run_name, manifest_mode, summary_csv),
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


def _summary_target(output_root: Path, scenario_name: str, run_name: str, manifest_mode: bool, summary_csv: Path) -> Path:
    if not manifest_mode:
        return summary_csv
    summary_dir = output_root / "_summary_rows" / scenario_name
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
