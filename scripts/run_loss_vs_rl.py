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
    parser = argparse.ArgumentParser(description="Study 1: compare supervised losses vs RL/GRPO on grokking tasks.")
    parser.add_argument("--profile", choices=("pilot", "full10"), default="pilot")
    parser.add_argument("--output-root", type=str, default="outputs/study1_loss_vs_rl")
    parser.add_argument("--modulus", type=int, default=113)
    parser.add_argument("--operator", type=str, default="add")
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--train-fractions", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--transformer-layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ce-lrs", type=str, default="")
    parser.add_argument("--ce-batch-sizes", type=str, default="")
    parser.add_argument("--ce-weight-decays", type=str, default="")
    parser.add_argument("--grpo-lrs", type=str, default="")
    parser.add_argument("--grpo-batch-sizes", type=str, default="")
    parser.add_argument("--grpo-weight-decays", type=str, default="")
    parser.add_argument("--grpo-n-samples", type=str, default="")
    parser.add_argument("--mlp-lrs", type=str, default="")
    parser.add_argument("--mlp-weight-decays", type=str, default="")
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--manifest-out", type=str, default="")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def defaults(profile: str) -> dict[str, object]:
    if profile == "full10":
        return {
            "seeds": list(range(10)),
            "ce_lrs": [1e-3],
            "ce_batch_sizes": [128, 256],
            "ce_weight_decays": [0.5, 1.0],
            "grpo_lrs": [5e-4],
            "grpo_batch_sizes": [32, 64],
            "grpo_weight_decays": [0.5, 1.0],
            "grpo_n_samples": [4, 8],
            "mlp_lrs": [1e-3, 3e-3],
            "mlp_weight_decays": [0.0, 1e-4],
            "train_fractions": [0.1, 0.3, 0.5],
        }
    return {
        "seeds": [0, 1],
        "ce_lrs": [1e-3],
        "ce_batch_sizes": [128],
        "ce_weight_decays": [1.0],
        "grpo_lrs": [5e-4],
        "grpo_batch_sizes": [32],
        "grpo_weight_decays": [1.0],
        "grpo_n_samples": [4],
        "mlp_lrs": [1e-3],
        "mlp_weight_decays": [0.0],
        "train_fractions": [0.3],
    }


def build_job_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    preset = defaults(args.profile)
    seeds = parse_csv_list(args.seeds, int) if args.seeds else preset["seeds"]
    ce_lrs = parse_csv_list(args.ce_lrs, float) if args.ce_lrs else preset["ce_lrs"]
    ce_batch_sizes = parse_csv_list(args.ce_batch_sizes, int) if args.ce_batch_sizes else preset["ce_batch_sizes"]
    ce_weight_decays = parse_csv_list(args.ce_weight_decays, float) if args.ce_weight_decays else preset["ce_weight_decays"]
    grpo_lrs = parse_csv_list(args.grpo_lrs, float) if args.grpo_lrs else preset["grpo_lrs"]
    grpo_batch_sizes = parse_csv_list(args.grpo_batch_sizes, int) if args.grpo_batch_sizes else preset["grpo_batch_sizes"]
    grpo_weight_decays = parse_csv_list(args.grpo_weight_decays, float) if args.grpo_weight_decays else preset["grpo_weight_decays"]
    grpo_n_samples = parse_csv_list(args.grpo_n_samples, int) if args.grpo_n_samples else preset["grpo_n_samples"]
    mlp_lrs = parse_csv_list(args.mlp_lrs, float) if args.mlp_lrs else preset["mlp_lrs"]
    mlp_weight_decays = parse_csv_list(args.mlp_weight_decays, float) if args.mlp_weight_decays else preset["mlp_weight_decays"]
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
    transformer_prefix = transformer_run_prefix(args.transformer_layers)
    transformer_architecture = transformer_architecture_name(args.transformer_layers)

    for train_fraction in train_fractions:
        task = build_single_operator_task(
            modulus=args.modulus,
            operator=args.operator,
            train_fraction=train_fraction,
            seed=0,
        )
        train_size = len(task.train)
        for model_type in ("transformer", "mlp"):
            if model_type == "transformer":
                for seed, lr, batch_size, weight_decay in itertools.product(seeds, ce_lrs, ce_batch_sizes, ce_weight_decays):
                    run_name = f"{transformer_prefix}_tf{train_fraction}_ce_seed{seed}_lr{lr}_wd{weight_decay}_bs{batch_size}"
                    config = RunConfig(
                        study_name="study1_loss_vs_rl",
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
                            "operator": args.operator,
                            "modulus": args.modulus,
                            "train_fraction": train_fraction,
                            "data_seed": 0,
                            "corruption_seed": None,
                            "include_zero": True,
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
                                "operator": args.operator,
                                "train_fraction": train_fraction,
                                "seed": 0,
                                "corruption_fraction": 0.0,
                                "corruption_seed": None,
                                "include_zero": True,
                            },
                            summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
                        )
                    )

                for seed, lr, batch_size, weight_decay, n_samples, reward_mode in itertools.product(
                    seeds,
                    grpo_lrs,
                    grpo_batch_sizes,
                    grpo_weight_decays,
                    grpo_n_samples,
                    ("binary", "binary_plus_partial_absolute"),
                ):
                    run_name = (
                        f"{transformer_prefix}_tf{train_fraction}_grpo_{reward_mode}_seed{seed}_lr{lr}_wd{weight_decay}"
                        f"_bs{batch_size}_ns{n_samples}"
                    )
                    config = RunConfig(
                        study_name="study1_loss_vs_rl",
                        model_type="transformer",
                        objective="grpo",
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
                            "operator": args.operator,
                            "modulus": args.modulus,
                            "train_fraction": train_fraction,
                            "data_seed": 0,
                            "corruption_seed": None,
                            "include_zero": True,
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
                                "operator": args.operator,
                                "train_fraction": train_fraction,
                                "seed": 0,
                                "corruption_fraction": 0.0,
                                "corruption_seed": None,
                                "include_zero": True,
                            },
                            summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
                        )
                    )
            else:
                for seed, lr, weight_decay, objective in itertools.product(seeds, mlp_lrs, mlp_weight_decays, ("ce", "mse", "mae")):
                    run_name = f"mlp_tf{train_fraction}_{objective}_seed{seed}_lr{lr}_wd{weight_decay}"
                    config = RunConfig(
                        study_name="study1_loss_vs_rl",
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
                            "operator": args.operator,
                            "modulus": args.modulus,
                            "train_fraction": train_fraction,
                            "data_seed": 0,
                            "corruption_seed": None,
                            "include_zero": True,
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
                                "operator": args.operator,
                                "train_fraction": train_fraction,
                                "seed": 0,
                                "corruption_fraction": 0.0,
                                "corruption_seed": None,
                                "include_zero": True,
                            },
                            summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
                        )
                    )

                for seed, lr, weight_decay, n_samples, reward_mode in itertools.product(
                    seeds,
                    mlp_lrs,
                    mlp_weight_decays,
                    grpo_n_samples,
                    ("binary", "binary_plus_partial_absolute"),
                ):
                    run_name = f"mlp_tf{train_fraction}_ppo_{reward_mode}_seed{seed}_lr{lr}_wd{weight_decay}_ns{n_samples}"
                    config = RunConfig(
                        study_name="study1_loss_vs_rl",
                        model_type="mlp",
                        objective="ppo",
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
                            "operator": args.operator,
                            "modulus": args.modulus,
                            "train_fraction": train_fraction,
                            "data_seed": 0,
                            "corruption_seed": None,
                            "include_zero": True,
                            "reward_mode": reward_mode,
                            "n_samples": n_samples,
                        },
                        ppo=PPOConfig(n_samples=n_samples, reward_mode=reward_mode, clip_eps=0.2, policy_epochs=4, entropy_coef=1e-3, value_coef=0.5),
                    )
                    jobs.append(
                        make_job_spec(
                            config,
                            task_spec={
                                "kind": "single_operator",
                                "modulus": args.modulus,
                                "operator": args.operator,
                                "train_fraction": train_fraction,
                                "seed": 0,
                                "corruption_fraction": 0.0,
                                "corruption_seed": None,
                                "include_zero": True,
                            },
                            summary_csv_path=_summary_target(output_root, run_name, manifest_mode, summary_csv),
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
    study_name = job_study_name(jobs[0]) if jobs else "study1_loss_vs_rl"
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
