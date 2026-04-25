from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .logging_utils import append_csv, append_csv_stable, append_jsonl, ensure_dir, write_json
from .mlp import MLPConfig, ModularMLP
from .model import GrokkingTransformer, TransformerConfig
from .rl import GRPOConfig, PPOConfig, create_reference_model, grpo_update, ppo_update
from .tasks import DatasetInfo, TaskDataset
from .train_utils import evaluate, train_step


@dataclass(frozen=True)
class RunConfig:
    study_name: str
    model_type: str
    objective: str
    seed: int
    lr: float
    weight_decay: float
    batch_size: int
    max_steps: int
    eval_every: int
    log_every: int
    device: str
    output_dir: str
    full_batch: bool = False
    mlp_hidden_dim: int = 512
    transformer_n_layers: int = 1
    checkpoint_every_steps: int | None = None
    checkpoint_steps: tuple[int, ...] | None = None
    metadata: dict[str, object] | None = None
    grpo: GRPOConfig | None = None
    ppo: PPOConfig | None = None


def transformer_run_prefix(transformer_n_layers: int) -> str:
    if transformer_n_layers == 1:
        return "transformer"
    if transformer_n_layers == 2:
        return "transformer2"
    raise ValueError("transformer_n_layers must be 1 or 2")


def transformer_architecture_name(transformer_n_layers: int) -> str:
    if transformer_n_layers == 1:
        return "neel_nanda_1layer"
    if transformer_n_layers == 2:
        return "power_etal_2layer"
    raise ValueError("transformer_n_layers must be 1 or 2")


def run_config_payload(config: RunConfig) -> dict[str, object]:
    return {
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
        "transformer_n_layers": config.transformer_n_layers,
        "checkpoint_every_steps": config.checkpoint_every_steps,
        "checkpoint_steps": config.checkpoint_steps,
        "metadata": config.metadata,
        "grpo": config.grpo.to_dict() if config.grpo is not None else None,
        "ppo": config.ppo.to_dict() if config.ppo is not None else None,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: RunConfig, info: DatasetInfo) -> nn.Module:
    if config.model_type == "transformer":
        if config.transformer_n_layers == 1:
            transformer_config = TransformerConfig.neel_nanda(
                vocab_size=info.vocab_size,
                seq_len=info.seq_len,
            )
        elif config.transformer_n_layers == 2:
            transformer_config = TransformerConfig.power_grokking(
                vocab_size=info.vocab_size,
                seq_len=info.seq_len,
            )
        else:
            raise ValueError("transformer_n_layers must be 1 or 2")
        return GrokkingTransformer(transformer_config)
    if config.model_type == "mlp":
        if info.seq_len != 3:
            raise ValueError("the MLP runner only supports single-operator tasks with seq_len=3")
        return ModularMLP(MLPConfig(prime=info.target_vocab_size, hidden_dim=config.mlp_hidden_dim))
    raise ValueError(f"unsupported model_type={config.model_type}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_progress_state(
    *,
    path: Path,
    config: RunConfig,
    run_name: str,
    status: str,
    step: int,
    train_size: int,
    steps_per_epoch: int,
    train_loss: float | None,
    train_metrics: dict[str, float] | None,
    eval_metrics: dict[str, dict[str, float]] | None,
) -> None:
    progress_fraction = 1.0 if config.max_steps <= 0 else min(max(step / config.max_steps, 0.0), 1.0)
    payload: dict[str, object] = {
        "updated_at_utc": _timestamp_utc(),
        "status": status,
        "study_name": config.study_name,
        "run_name": run_name,
        "model_type": config.model_type,
        "objective": config.objective,
        "seed": config.seed,
        "step": step,
        "max_steps": config.max_steps,
        "progress_fraction": progress_fraction,
        "train_size": train_size,
        "steps_per_epoch": steps_per_epoch,
        "output_dir": str(path.parent),
    }
    if train_loss is not None:
        payload["train_update_loss"] = float(train_loss)
    if train_metrics is not None:
        for key, value in train_metrics.items():
            payload[f"train_{key}"] = float(value)
    if eval_metrics is not None:
        for split_name, split_metrics in eval_metrics.items():
            for key, value in split_metrics.items():
                payload[f"{split_name}_{key}"] = float(value)
    write_json(path, payload)


def _cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_peak_allocated_mb": 0.0,
            "cuda_peak_reserved_mb": 0.0,
        }

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {
        "cuda_allocated_mb": float(allocated),
        "cuda_reserved_mb": float(reserved),
        "cuda_peak_allocated_mb": float(peak_allocated),
        "cuda_peak_reserved_mb": float(peak_reserved),
    }


def _make_dataloader(dataset: TaskDataset, batch_size: int, shuffle: bool, full_batch: bool) -> DataLoader:
    loader_batch_size = len(dataset) if full_batch else min(batch_size, len(dataset))
    return DataLoader(dataset, batch_size=loader_batch_size, shuffle=shuffle, drop_last=False)


def evaluate_dataset(
    *,
    model: nn.Module,
    dataset: TaskDataset,
    device: torch.device,
    batch_size: int,
    full_batch: bool,
    target_vocab_size: int,
    objective: str,
) -> dict[str, float]:
    dataloader = _make_dataloader(dataset, batch_size=batch_size, shuffle=False, full_batch=full_batch)
    loss_type = _loss_type_for_objective(objective)
    metrics = evaluate(
        model,
        dataloader,
        device,
        target_vocab_size=target_vocab_size,
        loss_type=loss_type,
    )

    predictions = []
    model.eval()
    with torch.no_grad():
        for tokens, _ in dataloader:
            logits = model(tokens.to(device))[:, :target_vocab_size]
            predictions.append(logits.argmax(dim=-1).cpu())
    predicted = torch.cat(predictions, dim=0)
    true_targets = dataset.true_targets.cpu()
    label_targets = dataset.targets.cpu()
    corrupted_mask = dataset.corrupted_mask.cpu()
    clean_mask = ~corrupted_mask

    results: dict[str, float] = {
        "loss": metrics.loss,
        "label_accuracy": float((predicted == label_targets).float().mean().item()),
        "true_accuracy": float((predicted == true_targets).float().mean().item()),
    }

    if corrupted_mask.any():
        results["corrupted_true_accuracy"] = float((predicted[corrupted_mask] == true_targets[corrupted_mask]).float().mean().item())
        results["corrupted_label_accuracy"] = float((predicted[corrupted_mask] == label_targets[corrupted_mask]).float().mean().item())
    else:
        results["corrupted_true_accuracy"] = float("nan")
        results["corrupted_label_accuracy"] = float("nan")

    if clean_mask.any():
        results["clean_true_accuracy"] = float((predicted[clean_mask] == true_targets[clean_mask]).float().mean().item())
    else:
        results["clean_true_accuracy"] = float("nan")

    return results


def export_dataset_snapshot(
    *,
    path: str | Path,
    train_dataset: TaskDataset,
    eval_datasets: dict[str, TaskDataset],
    final_only_eval_datasets: dict[str, TaskDataset] | None = None,
) -> None:
    payload: dict[str, dict[str, torch.Tensor]] = {
        "train": {
            "inputs": train_dataset.inputs.cpu(),
            "targets": train_dataset.targets.cpu(),
            "true_targets": train_dataset.true_targets.cpu(),
            "corrupted_mask": train_dataset.corrupted_mask.cpu(),
        }
    }
    for split_name, dataset in eval_datasets.items():
        payload[split_name] = {
            "inputs": dataset.inputs.cpu(),
            "targets": dataset.targets.cpu(),
            "true_targets": dataset.true_targets.cpu(),
            "corrupted_mask": dataset.corrupted_mask.cpu(),
        }
    if final_only_eval_datasets is not None:
        for split_name, dataset in final_only_eval_datasets.items():
            payload[split_name] = {
                "inputs": dataset.inputs.cpu(),
                "targets": dataset.targets.cpu(),
                "true_targets": dataset.true_targets.cpu(),
                "corrupted_mask": dataset.corrupted_mask.cpu(),
            }
    torch.save(payload, Path(path))


def export_prediction_table(
    *,
    path: str | Path,
    model: nn.Module,
    dataset: TaskDataset,
    device: torch.device,
    batch_size: int,
    full_batch: bool,
    target_vocab_size: int,
) -> None:
    dataloader = _make_dataloader(dataset, batch_size=batch_size, shuffle=False, full_batch=full_batch)
    rows: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        offset = 0
        for tokens, _ in dataloader:
            batch_tokens = tokens.to(device)
            logits = model(batch_tokens)[:, :target_vocab_size]
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = probabilities.max(dim=-1)
            top_k = min(2, probabilities.size(-1))
            top_values, top_indices = probabilities.topk(k=top_k, dim=-1)
            second_best = top_values[:, 1] if top_k > 1 else torch.zeros_like(top_values[:, 0])
            margin = top_values[:, 0] - second_best

            batch_size_now = tokens.size(0)
            for i in range(batch_size_now):
                row_index = offset + i
                rows.append(
                    {
                        "index": row_index,
                        "input_tokens": " ".join(str(int(x)) for x in dataset.inputs[row_index].tolist()),
                        "target_label": int(dataset.targets[row_index].item()),
                        "true_target": int(dataset.true_targets[row_index].item()),
                        "prediction": int(predictions[i].item()),
                        "is_label_correct": int(predictions[i].item() == dataset.targets[row_index].item()),
                        "is_true_correct": int(predictions[i].item() == dataset.true_targets[row_index].item()),
                        "is_corrupted": int(dataset.corrupted_mask[row_index].item()),
                        "confidence": float(confidences[i].item()),
                        "margin": float(margin[i].item()),
                        "top1_prob": float(top_values[i, 0].item()),
                        "top1_class": int(top_indices[i, 0].item()),
                        "top2_prob": float(second_best[i].item()),
                        "top2_class": int(top_indices[i, 1].item()) if top_k > 1 else -1,
                    }
                )
            offset += batch_size_now

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def save_checkpoint(
    *,
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    value_head: nn.Module | None,
    step: int,
    result: dict[str, object],
) -> None:
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "value_head_state_dict": value_head.state_dict() if value_head is not None else None,
        "result": result,
    }
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def _loss_type_for_objective(objective: str) -> str:
    if objective == "ce":
        return "cross_entropy"
    if objective == "mse":
        return "mse_one_hot"
    if objective == "mae":
        return "mae_one_hot"
    if objective == "grpo":
        return "cross_entropy"
    if objective == "ppo":
        return "cross_entropy"
    raise ValueError(f"unsupported objective={objective}")


def _should_save_periodic_checkpoint(config: RunConfig, step: int) -> bool:
    if config.checkpoint_steps is not None:
        return step in config.checkpoint_steps
    return config.checkpoint_every_steps is not None and step % config.checkpoint_every_steps == 0


def _make_optimizer(parameters, config: RunConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))


def _metrics_fieldnames(
    *,
    eval_split_names: list[str],
    objective: str,
) -> list[str]:
    fieldnames = [
        "step",
        "train_update_loss",
        "lr",
        "param_norm",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
    ]
    if objective in {"grpo", "ppo"}:
        fieldnames.extend(
            [
                "train_reward_mean",
                "train_reward_std",
                "train_entropy",
                "train_clip_fraction",
            ]
        )
    if objective == "grpo":
        fieldnames.append("train_kl")
    if objective == "ppo":
        fieldnames.append("train_value_loss")

    split_suffixes = [
        "loss",
        "label_accuracy",
        "true_accuracy",
        "corrupted_true_accuracy",
        "corrupted_label_accuracy",
        "clean_true_accuracy",
    ]
    for split_name in eval_split_names:
        for suffix in split_suffixes:
            fieldnames.append(f"{split_name}_{suffix}")
    return fieldnames


def run_training(
    *,
    config: RunConfig,
    info: DatasetInfo,
    train_dataset: TaskDataset,
    eval_datasets: dict[str, TaskDataset],
    final_only_eval_datasets: dict[str, TaskDataset] | None = None,
    summary_csv_path: str | Path,
) -> dict[str, object]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    run_name = output_dir.name
    device = torch.device(config.device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config, info).to(device)
    value_head: nn.Module | None = None
    if config.objective == "ppo":
        value_head = nn.Linear(info.target_vocab_size, 1).to(device)
        optimizer = _make_optimizer(list(model.parameters()) + list(value_head.parameters()), config)
    else:
        optimizer = _make_optimizer(model.parameters(), config)
    train_loader = _make_dataloader(train_dataset, config.batch_size, shuffle=True, full_batch=config.full_batch)
    train_iterator = iter(train_loader)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_csv_path = output_dir / "metrics.csv"
    progress_path = output_dir / "progress.json"
    metrics_fieldnames = _metrics_fieldnames(
        eval_split_names=sorted(eval_datasets.keys()),
        objective=config.objective,
    )

    write_json(
        output_dir / "config.json",
        _serialize_run_config(
            config,
            info,
            model,
            value_head,
            train_size=len(train_dataset),
            eval_sizes={
                **{split_name: len(dataset) for split_name, dataset in eval_datasets.items()},
                **(
                    {split_name: len(dataset) for split_name, dataset in final_only_eval_datasets.items()}
                    if final_only_eval_datasets is not None
                    else {}
                ),
            },
            steps_per_epoch=len(train_loader),
        ),
    )
    export_dataset_snapshot(
        path=output_dir / "dataset_snapshot.pt",
        train_dataset=train_dataset,
        eval_datasets=eval_datasets,
        final_only_eval_datasets=final_only_eval_datasets,
    )

    reference_model = create_reference_model(model) if config.objective == "grpo" and config.grpo and config.grpo.kl_coef > 0.0 else None

    progress = tqdm(total=config.max_steps, desc=run_name, leave=False, dynamic_ncols=True)
    final_eval: dict[str, dict[str, float]] = {}
    step = 0
    train_loss = float("nan")
    train_metrics: dict[str, float] = {}
    last_progress_write_at = 0.0
    progress_write_interval_sec = 2.0
    _write_progress_state(
        path=progress_path,
        config=config,
        run_name=run_name,
        status="starting",
        step=0,
        train_size=len(train_dataset),
        steps_per_epoch=len(train_loader),
        train_loss=None,
        train_metrics=None,
        eval_metrics=None,
    )

    try:
        for step in range(1, config.max_steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            if config.objective == "grpo":
                assert config.grpo is not None
                train_metrics = grpo_update(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    device=device,
                    target_vocab_size=info.target_vocab_size,
                    config=config.grpo,
                    reference_model=reference_model,
                )
                train_loss = train_metrics["loss"]
            elif config.objective == "ppo":
                assert config.ppo is not None
                assert value_head is not None
                train_metrics = ppo_update(
                    model=model,
                    value_head=value_head,
                    batch=batch,
                    optimizer=optimizer,
                    device=device,
                    target_vocab_size=info.target_vocab_size,
                    config=config.ppo,
                )
                train_loss = train_metrics["loss"]
            else:
                train_loss = train_step(
                    model,
                    batch,
                    optimizer,
                    device,
                    target_vocab_size=info.target_vocab_size,
                    loss_type=_loss_type_for_objective(config.objective),
                )
                train_metrics = {"loss": train_loss}

            if step == 1 or step % config.log_every == 0 or step % config.eval_every == 0 or step == config.max_steps:
                current_lr = float(optimizer.param_groups[0]["lr"])
                param_norm = float(torch.sqrt(sum(parameter.detach().float().pow(2).sum() for parameter in model.parameters())).item())
                record: dict[str, object] = {
                    "step": step,
                    "train_update_loss": train_loss,
                    "lr": current_lr,
                    "param_norm": param_norm,
                }
                record.update(_cuda_memory_stats(device))
                for key, value in train_metrics.items():
                    record[f"train_{key}"] = value

                if step == 1 or step % config.eval_every == 0 or step == config.max_steps:
                    final_eval = {}
                    for split_name, dataset in eval_datasets.items():
                        split_metrics = evaluate_dataset(
                            model=model,
                            dataset=dataset,
                            device=device,
                            batch_size=config.batch_size,
                            full_batch=config.full_batch,
                            target_vocab_size=info.target_vocab_size,
                            objective=config.objective,
                        )
                        final_eval[split_name] = split_metrics
                        for key, value in split_metrics.items():
                            record[f"{split_name}_{key}"] = value

                append_jsonl(metrics_path, record)
                append_csv_stable(metrics_csv_path, metrics_fieldnames, record)

                postfix = {"loss": f"{train_loss:.4f}"}
                if "test" in final_eval and "true_accuracy" in final_eval["test"]:
                    postfix["test_acc"] = f"{final_eval['test']['true_accuracy']:.3f}"
                elif "train" in final_eval and "true_accuracy" in final_eval["train"]:
                    postfix["train_acc"] = f"{final_eval['train']['true_accuracy']:.3f}"
                progress.set_postfix(postfix, refresh=False)

            if (
                step == 1
                or step == config.max_steps
                or step % config.log_every == 0
                or step % config.eval_every == 0
                or time.monotonic() - last_progress_write_at >= progress_write_interval_sec
            ):
                _write_progress_state(
                    path=progress_path,
                    config=config,
                    run_name=run_name,
                    status="running",
                    step=step,
                    train_size=len(train_dataset),
                    steps_per_epoch=len(train_loader),
                    train_loss=train_loss,
                    train_metrics=train_metrics,
                    eval_metrics=final_eval or None,
                )
                last_progress_write_at = time.monotonic()

            if _should_save_periodic_checkpoint(config, step):
                checkpoint_result = {
                    "study_name": config.study_name,
                    "model_type": config.model_type,
                    "objective": config.objective,
                    "seed": config.seed,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "batch_size": config.batch_size,
                    "max_steps": config.max_steps,
                    "completed_steps": step,
                    "output_dir": str(output_dir),
                    "checkpoint_path": str(output_dir / "checkpoints" / f"step_{step:06d}.pt"),
                }
                if config.metadata is not None:
                    checkpoint_result.update(config.metadata)
                save_checkpoint(
                    path=output_dir / "checkpoints" / f"step_{step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    value_head=value_head,
                    step=step,
                    result=checkpoint_result,
                )

            progress.update(1)
    except Exception:
        _write_progress_state(
            path=progress_path,
            config=config,
            run_name=run_name,
            status="failed",
            step=step,
            train_size=len(train_dataset),
            steps_per_epoch=len(train_loader),
            train_loss=train_loss if not math.isnan(train_loss) else None,
            train_metrics=train_metrics or None,
            eval_metrics=final_eval or None,
        )
        progress.close()
        raise

    progress.close()
    if final_only_eval_datasets is not None:
        for split_name, dataset in final_only_eval_datasets.items():
            split_metrics = evaluate_dataset(
                model=model,
                dataset=dataset,
                device=device,
                batch_size=config.batch_size,
                full_batch=config.full_batch,
                target_vocab_size=info.target_vocab_size,
                objective=config.objective,
            )
            final_eval[split_name] = split_metrics

    result = {
        "study_name": config.study_name,
        "model_type": config.model_type,
        "objective": config.objective,
        "seed": config.seed,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "full_batch": config.full_batch,
        "max_steps": config.max_steps,
        "completed_steps": step,
        "output_dir": str(output_dir),
        "checkpoint_path": str(output_dir / "final_checkpoint.pt"),
        "train_size": len(train_dataset),
        "steps_per_epoch": len(train_loader),
        "parameter_count": parameter_count(model) + (parameter_count(value_head) if value_head is not None else 0),
    }
    result.update(_cuda_memory_stats(device))
    if config.metadata is not None:
        result.update(config.metadata)
    for split_name, split_metrics in final_eval.items():
        for key, value in split_metrics.items():
            result[f"{split_name}_{key}"] = value

    append_csv(summary_csv_path, result)
    write_json(output_dir / "result.json", result)
    _write_progress_state(
        path=progress_path,
        config=config,
        run_name=run_name,
        status="completed",
        step=step,
        train_size=len(train_dataset),
        steps_per_epoch=len(train_loader),
        train_loss=train_loss,
        train_metrics=train_metrics,
        eval_metrics=final_eval,
    )
    for split_name, dataset in eval_datasets.items():
        export_prediction_table(
            path=output_dir / f"{split_name}_predictions.csv",
            model=model,
            dataset=dataset,
            device=device,
            batch_size=config.batch_size,
            full_batch=config.full_batch,
            target_vocab_size=info.target_vocab_size,
        )
    if final_only_eval_datasets is not None:
        for split_name, dataset in final_only_eval_datasets.items():
            export_prediction_table(
                path=output_dir / f"{split_name}_predictions.csv",
                model=model,
                dataset=dataset,
                device=device,
                batch_size=config.batch_size,
                full_batch=config.full_batch,
                target_vocab_size=info.target_vocab_size,
            )
    save_checkpoint(
        path=output_dir / "final_checkpoint.pt",
        model=model,
        optimizer=optimizer,
        value_head=value_head,
        step=step,
        result=result,
    )
    return result


def _serialize_run_config(
    config: RunConfig,
    info: DatasetInfo,
    model: nn.Module,
    value_head: nn.Module | None,
    *,
    train_size: int,
    eval_sizes: dict[str, int],
    steps_per_epoch: int,
) -> dict[str, object]:
    payload = asdict(config)
    if config.grpo is not None:
        payload["grpo"] = config.grpo.to_dict()
    if config.ppo is not None:
        payload["ppo"] = config.ppo.to_dict()
    payload["dataset_info"] = {
        "vocab_size": info.vocab_size,
        "target_vocab_size": info.target_vocab_size,
        "seq_len": info.seq_len,
        "eq_token_id": info.eq_token_id,
        "operator_token_ids": info.operator_token_ids,
    }
    payload["metadata"] = config.metadata
    payload["split_sizes"] = {"train": train_size, **eval_sizes}
    payload["steps_per_epoch"] = steps_per_epoch
    payload["parameter_count"] = parameter_count(model) + (parameter_count(value_head) if value_head is not None else 0)
    return payload
