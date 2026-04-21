from __future__ import annotations

import csv
import json
from pathlib import Path

from .experiment_utils import RunConfig, run_training
from .rl import GRPOConfig, PPOConfig
from .tasks import build_range_transfer_task, build_single_operator_task


def run_job_spec(job_spec: dict[str, object]) -> dict[str, object]:
    task_spec = job_spec["task"]
    run_config = _run_config_from_dict(job_spec["run_config"])
    summary_csv_path = Path(job_spec["summary_csv_path"])

    if task_spec["kind"] == "single_operator":
        task = build_single_operator_task(
            modulus=int(task_spec["modulus"]),
            operator=str(task_spec["operator"]),
            train_fraction=float(task_spec["train_fraction"]),
            seed=int(task_spec.get("seed", 0)),
            corruption_fraction=float(task_spec.get("corruption_fraction", 0.0)),
            corruption_seed=int(task_spec["corruption_seed"]) if task_spec.get("corruption_seed") is not None else None,
            include_zero=bool(task_spec.get("include_zero", True)),
        )
        return run_training(
            config=run_config,
            info=task.info,
            train_dataset=task.train,
            eval_datasets={"train": task.train, "test": task.test},
            summary_csv_path=summary_csv_path,
        )

    if task_spec["kind"] == "range_transfer":
        task = build_range_transfer_task(
            modulus=int(task_spec["modulus"]),
            train_fraction=float(task_spec["train_fraction"]),
            seed=int(task_spec.get("seed", 0)),
            add_offset=int(task_spec["add_offset"]),
            mul_offset=int(task_spec["mul_offset"]),
            include_zero=bool(task_spec.get("include_zero", True)),
        )
        return run_training(
            config=run_config,
            info=task.info,
            train_dataset=task.train,
            eval_datasets={"train": task.train, "test": task.test},
            final_only_eval_datasets={"cross_add": task.cross_add, "cross_mul": task.cross_mul},
            summary_csv_path=summary_csv_path,
        )

    raise ValueError(f"unsupported task kind={task_spec['kind']}")


def estimate_vram_mb(job_spec: dict[str, object], per_process_overhead_mb: float = 350.0) -> float:
    run_config = job_spec["run_config"]
    task_spec = job_spec["task"]
    output_dir = Path(run_config["output_dir"])
    result_path = output_dir / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            peak_reserved = float(result.get("cuda_peak_reserved_mb", 0.0))
            if peak_reserved > 0.0:
                return peak_reserved + per_process_overhead_mb
        except Exception:
            pass

    model_type = run_config["model_type"]
    objective = run_config["objective"]
    batch_size = int(run_config["batch_size"])
    full_batch = bool(run_config.get("full_batch", False))

    if task_spec["kind"] == "single_operator":
        modulus = int(task_spec["modulus"])
        seq_len = 3
        target_vocab = modulus
        train_size = int((modulus * modulus) * float(task_spec["train_fraction"]))
    else:
        modulus = int(task_spec["modulus"])
        seq_len = 4
        target_vocab = max(int(task_spec["add_offset"]) + modulus - 1, int(task_spec["mul_offset"]) + modulus - 1) + 1
        train_size = int((2 * modulus * modulus) * float(task_spec["train_fraction"]))

    effective_batch = train_size if full_batch else min(batch_size, train_size)

    if model_type == "transformer":
        d_model = 128
        d_mlp = 512
        activation_mb = 4.0 * effective_batch * seq_len * (6 * d_model + 2 * d_mlp + target_vocab) / (1024 ** 2)
        rl_extra = 0.0
        if objective == "grpo":
            n_samples = int(run_config["grpo"]["n_samples"])
            rl_extra = 4.0 * effective_batch * n_samples * 4 / (1024 ** 2)
        return per_process_overhead_mb + max(150.0, activation_mb + rl_extra + 64.0)

    hidden_dim = int(run_config.get("mlp_hidden_dim", 512))
    activation_mb = 4.0 * effective_batch * (3 * modulus + hidden_dim) / (1024 ** 2)
    rl_extra = 0.0
    if objective == "ppo":
        n_samples = int(run_config["ppo"]["n_samples"])
        rl_extra = 4.0 * effective_batch * n_samples * 6 / (1024 ** 2)
    return per_process_overhead_mb + max(150.0, activation_mb + rl_extra + 64.0)


def aggregate_results(manifest_path: str | Path, output_csv_path: str | Path) -> None:
    jobs = read_manifest(manifest_path)
    rows: list[dict[str, object]] = []
    all_keys: set[str] = set()

    for job in jobs:
        output_dir = Path(job["run_config"]["output_dir"])
        result_path = output_dir / "result.json"
        if not result_path.exists():
            continue
        row = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(row)
        all_keys.update(row.keys())

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(all_keys)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(path: str | Path, jobs: list[dict[str, object]]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job) + "\n")


def read_manifest(path: str | Path) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def _run_config_from_dict(payload: dict[str, object]) -> RunConfig:
    grpo = GRPOConfig(**payload["grpo"]) if payload.get("grpo") is not None else None
    ppo = PPOConfig(**payload["ppo"]) if payload.get("ppo") is not None else None
    return RunConfig(
        study_name=str(payload["study_name"]),
        model_type=str(payload["model_type"]),
        objective=str(payload["objective"]),
        seed=int(payload["seed"]),
        lr=float(payload["lr"]),
        weight_decay=float(payload["weight_decay"]),
        batch_size=int(payload["batch_size"]),
        max_steps=int(payload["max_steps"]),
        eval_every=int(payload["eval_every"]),
        log_every=int(payload["log_every"]),
        device=str(payload["device"]),
        output_dir=str(payload["output_dir"]),
        full_batch=bool(payload.get("full_batch", False)),
        mlp_hidden_dim=int(payload.get("mlp_hidden_dim", 512)),
        metadata=payload.get("metadata"),
        grpo=grpo,
        ppo=ppo,
    )
