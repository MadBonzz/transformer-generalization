from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.experiment_utils import (
    RunConfig,
    run_config_payload,
    transformer_architecture_name,
    transformer_run_prefix,
)
from grokking_transformer.job_runner import job_run_name, job_study_name, run_job_spec, write_manifest
from grokking_transformer.logging_utils import ensure_dir
from grokking_transformer.rl import GRPOConfig, PPOConfig
from grokking_transformer.tasks import build_single_operator_task


def parse_csv_list(raw: str, cast) -> list:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study 2: fake-label robustness and inversion.")
    parser.add_argument("--profile", choices=("pilot", "full10"), default="pilot")
    parser.add_argument("--output-root", type=str, default="outputs/study2_fake_labels")
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--operators", type=str, default="add,sub,mul,poly")
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--train-fractions", type=str, default="")
    parser.add_argument("--corruption-levels", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--transformer-layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--manifest-out", type=str, default="")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def defaults(profile: str) -> dict[str, object]:
    if profile == "full10":
        return {
            "seeds": list(range(10)),
            "corruption_levels": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "transformer_lrs": [1e-3],
            "transformer_weight_decays": [0.5, 1.0],
            "transformer_batch_sizes": [128, 256],
            "mlp_lrs": [1e-3, 3e-3],
            "mlp_weight_decays": [0.0, 1e-4],
            "grpo_n_samples": [4, 8],
            "train_fractions": [0.1, 0.3, 0.5],
        }
    return {
        "seeds": [0, 1],
        "corruption_levels": [0.0, 0.2, 0.5],
        "transformer_lrs": [1e-3],
        "transformer_weight_decays": [1.0],
        "transformer_batch_sizes": [128],
        "mlp_lrs": [1e-3],
        "mlp_weight_decays": [0.0],
        "grpo_n_samples": [4],
        "train_fractions": [0.3],
    }


def build_job_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    preset = defaults(args.profile)
    seeds = parse_csv_list(args.seeds, int) if args.seeds else preset["seeds"]
    if args.train_fractions:
        train_fractions = parse_csv_list(args.train_fractions, float)
    elif args.train_fraction is not None:
        train_fractions = [args.train_fraction]
    else:
        train_fractions = preset["train_fractions"]
    corruption_levels = (
        parse_csv_list(args.corruption_levels, float) if args.corruption_levels else preset["corruption_levels"]
    )
    operators = parse_csv_list(args.operators, str)
    output_root = ensure_dir(args.output_root)
    summary_csv = output_root / "summary.csv"
    manifest_mode = bool(args.manifest_out)
    jobs: list[dict[str, object]] = []
    transformer_prefix = transformer_run_prefix(args.transformer_layers)
    transformer_architecture = transformer_architecture_name(args.transformer_layers)

    for operator, corruption_level, train_fraction in itertools.product(operators, corruption_levels, train_fractions):
        task = build_single_operator_task(
            modulus=args.modulus,
            operator=operator,
            train_fraction=train_fraction,
            seed=0,
            corruption_fraction=corruption_level,
            corruption_seed=1234,
            include_zero=operator != "div",
        )
        train_size = len(task.train)
        for seed, lr, batch_size, weight_decay in itertools.product(
            seeds,
            preset["transformer_lrs"],
            preset["transformer_batch_sizes"],
            preset["transformer_weight_decays"],
        ):
            run_name = f"{transformer_prefix}_tf{train_fraction}_ce_{operator}_corr{corruption_level}_seed{seed}_lr{lr}_wd{weight_decay}_bs{batch_size}"
            config = RunConfig(
                study_name="study2_fake_labels",
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
                transformer_n_layers=args.transformer_layers,
                metadata={
                    "operator": operator,
                    "modulus": args.modulus,
                    "train_fraction": train_fraction,
                    "data_seed": 0,
                    "corruption_level": corruption_level,
                    "corruption_seed": 1234,
                    "include_zero": operator != "div",
                    "reward_mode": "",
                    "n_samples": 0,
                    "transformer_n_layers": args.transformer_layers,
                    "transformer_architecture": transformer_architecture,
                },
            )
            jobs.append(
                make_job_spec(
                    config,
                    task_spec={
                        "kind": "single_operator",
                        "modulus": args.modulus,
                        "operator": operator,
                        "train_fraction": train_fraction,
                        "seed": 0,
                        "corruption_fraction": corruption_level,
                        "corruption_seed": 1234,
                        "include_zero": operator != "div",
                    },
                    summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
                )
            )

        for seed, lr, weight_decay, objective in itertools.product(
            seeds,
            preset["mlp_lrs"],
            preset["mlp_weight_decays"],
            ("ce", "mse", "mae"),
        ):
            run_name = f"mlp_tf{train_fraction}_{objective}_{operator}_corr{corruption_level}_seed{seed}_lr{lr}_wd{weight_decay}"
            config = RunConfig(
                study_name="study2_fake_labels",
                model_type="mlp",
                objective=objective,
                seed=seed,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=train_size,
                max_steps=args.max_steps,
                eval_every=args.eval_every,
                log_every=args.log_every,
                device=args.device,
                output_dir=str(output_root / run_name),
                full_batch=True,
                mlp_hidden_dim=args.mlp_hidden_dim,
                metadata={
                    "operator": operator,
                    "modulus": args.modulus,
                    "train_fraction": train_fraction,
                    "data_seed": 0,
                    "corruption_level": corruption_level,
                    "corruption_seed": 1234,
                    "include_zero": operator != "div",
                    "reward_mode": "",
                    "n_samples": 0,
                },
            )
            jobs.append(
                make_job_spec(
                    config,
                    task_spec={
                        "kind": "single_operator",
                        "modulus": args.modulus,
                        "operator": operator,
                        "train_fraction": train_fraction,
                        "seed": 0,
                        "corruption_fraction": corruption_level,
                        "corruption_seed": 1234,
                        "include_zero": operator != "div",
                    },
                    summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
                )
            )

        for seed, n_samples, reward_mode in itertools.product(
            seeds,
            preset["grpo_n_samples"],
            ("binary", "binary_plus_partial_absolute"),
        ):
            run_name = f"{transformer_prefix}_tf{train_fraction}_grpo_{reward_mode}_{operator}_corr{corruption_level}_seed{seed}_ns{n_samples}"
            config = RunConfig(
                study_name="study2_fake_labels",
                model_type="transformer",
                objective="grpo",
                seed=seed,
                lr=5e-4,
                weight_decay=1.0,
                batch_size=64,
                max_steps=args.max_steps,
                eval_every=args.eval_every,
                log_every=args.log_every,
                device=args.device,
                output_dir=str(output_root / run_name),
                transformer_n_layers=args.transformer_layers,
                metadata={
                    "operator": operator,
                    "modulus": args.modulus,
                    "train_fraction": train_fraction,
                    "data_seed": 0,
                    "corruption_level": corruption_level,
                    "corruption_seed": 1234,
                    "include_zero": operator != "div",
                    "reward_mode": reward_mode,
                    "n_samples": n_samples,
                    "transformer_n_layers": args.transformer_layers,
                    "transformer_architecture": transformer_architecture,
                },
                grpo=GRPOConfig(n_samples=n_samples, reward_mode=reward_mode, clip_eps=0.2, policy_epochs=4, entropy_coef=1e-3),
            )
            jobs.append(
                make_job_spec(
                    config,
                    task_spec={
                        "kind": "single_operator",
                        "modulus": args.modulus,
                        "operator": operator,
                        "train_fraction": train_fraction,
                        "seed": 0,
                        "corruption_fraction": corruption_level,
                        "corruption_seed": 1234,
                        "include_zero": operator != "div",
                    },
                    summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
                )
            )

            mlp_run_name = f"mlp_tf{train_fraction}_ppo_{reward_mode}_{operator}_corr{corruption_level}_seed{seed}_ns{n_samples}"
            mlp_config = RunConfig(
                study_name="study2_fake_labels",
                model_type="mlp",
                objective="ppo",
                seed=seed,
                lr=1e-3,
                weight_decay=0.0,
                batch_size=train_size,
                max_steps=args.max_steps,
                eval_every=args.eval_every,
                log_every=args.log_every,
                device=args.device,
                output_dir=str(output_root / mlp_run_name),
                full_batch=True,
                mlp_hidden_dim=args.mlp_hidden_dim,
                metadata={
                    "operator": operator,
                    "modulus": args.modulus,
                    "train_fraction": train_fraction,
                    "data_seed": 0,
                    "corruption_level": corruption_level,
                    "corruption_seed": 1234,
                    "include_zero": operator != "div",
                    "reward_mode": reward_mode,
                    "n_samples": n_samples,
                },
                ppo=PPOConfig(n_samples=n_samples, reward_mode=reward_mode, clip_eps=0.2, policy_epochs=4, entropy_coef=1e-3, value_coef=0.5),
            )
            jobs.append(
                make_job_spec(
                    mlp_config,
                    task_spec={
                        "kind": "single_operator",
                        "modulus": args.modulus,
                        "operator": operator,
                        "train_fraction": train_fraction,
                        "seed": 0,
                        "corruption_fraction": corruption_level,
                        "corruption_seed": 1234,
                        "include_zero": operator != "div",
                    },
                    summary_csv_path=_summary_target(output_root, mlp_run_name, manifest_mode, summary_csv),
                )
            )

    return jobs


def make_job_spec(config: RunConfig, task_spec: dict[str, object], summary_csv_path: Path) -> dict[str, object]:
    return {
        "task": task_spec,
        "run_config": run_config_payload(config),
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
    study_name = job_study_name(jobs[0]) if jobs else "study2_fake_labels"
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
