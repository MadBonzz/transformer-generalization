from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from grokking_transformer.experiment_utils import (  # noqa: E402
    RunConfig,
    parameter_count,
    transformer_architecture_name,
    transformer_run_prefix,
)
from grokking_transformer.logging_utils import append_csv, append_csv_stable, append_jsonl, ensure_dir, write_json  # noqa: E402
from grokking_transformer.model import TransformerBlock, TransformerConfig  # noqa: E402
from grokking_transformer.tasks import DatasetInfo, TaskDataset  # noqa: E402


DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "mixbox_base_case"
DEFAULT_NUM_BASE_COLORS = 2000


@dataclass(frozen=True)
class ColourRunConfig:
    study_name: str
    seed: int
    lr: float
    weight_decay: float
    batch_size: int
    max_steps: int
    eval_every: int
    log_every: int
    device: str
    output_dir: str
    transformer_n_layers: int
    checkpoint_every_steps: int | None
    checkpoint_steps: tuple[int, ...] | None
    metadata: dict[str, object]


class ColourSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs.long()
        self.targets = targets.long()

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class RGBOutputTransformer(nn.Module):
    def __init__(self, config: TransformerConfig, target_vocab_size: int, target_seq_len: int) -> None:
        super().__init__()
        self.config = config
        self.target_vocab_size = target_vocab_size
        self.target_seq_len = target_seq_len
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        if config.positional_embedding_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(config.seq_len, config.d_model))
        else:
            self.register_buffer("pos_embed", self._sinusoidal_position_encoding(config.seq_len, config.d_model), persistent=False)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_ln = nn.LayerNorm(config.d_model) if config.final_norm else nn.Identity()
        self.output_head = nn.Linear(config.d_model, target_seq_len * target_vocab_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)

    @staticmethod
    def _sinusoidal_position_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pos_embed = torch.zeros(seq_len, d_model, dtype=torch.float32)
        pos_embed[:, 0::2] = torch.sin(positions * div_term)
        pos_embed[:, 1::2] = torch.cos(positions * div_term)
        return pos_embed

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError("tokens must have shape [batch, seq_len]")
        if tokens.size(1) != self.config.seq_len:
            raise ValueError(f"expected seq_len={self.config.seq_len}, got {tokens.size(1)}")
        x = self.token_embed(tokens) + self.pos_embed.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        logits = self.output_head(x[:, -1, :])
        return logits.view(tokens.size(0), self.target_seq_len, self.target_vocab_size)


def dataset_name(num_base_colors: int) -> str:
    return f"colour_mixing_mixbox_100k_{num_base_colors}base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 12-run color-combination baseline: 4 transformer depths x 3 seeds."
    )
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-schedule",
        type=str,
        default="fixed",
        choices=["staged", "fixed", "none"],
        help="staged: every 1k through 25k, then every --checkpoint-every-steps; fixed: use only --checkpoint-every-steps.",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=10000)
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
    parser.add_argument("--single-layer", type=int, choices=[1, 2, 3, 4], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-seed", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def staged_checkpoint_steps(max_steps: int, interval_after_early: int) -> tuple[int, ...]:
    steps = set(range(1000, min(max_steps, 25000) + 1, 1000))
    if max_steps > 25000 and interval_after_early > 0:
        steps.update(range(50000, max_steps + 1, interval_after_early))
    return tuple(sorted(steps))


def checkpoint_steps_for_args(args: argparse.Namespace) -> tuple[int, ...] | None:
    if args.checkpoint_schedule == "none":
        return None
    if args.checkpoint_schedule == "staged":
        return staged_checkpoint_steps(args.max_steps, args.checkpoint_every_steps)
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


def load_split_dataset(dataset_dir: Path, split: str) -> ColourSequenceDataset:
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    with (dataset_dir / "tokenized_examples.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] != split:
                continue
            inputs.append([int(token_id) for token_id in row["input_ids"].split()])
            targets.append([int(token_id) for token_id in row["target_ids"].split()])
    if not inputs:
        raise ValueError(f"dataset split {split!r} is empty in {dataset_dir}")
    return ColourSequenceDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))


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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(*, layers: int, vocab_size: int, target_vocab_size: int, seq_len: int, target_seq_len: int) -> nn.Module:
    if layers == 1:
        config = TransformerConfig.neel_nanda(vocab_size=vocab_size, seq_len=seq_len)
    elif layers >= 2:
        base_config = TransformerConfig.power_grokking(vocab_size=vocab_size, seq_len=seq_len)
        config = TransformerConfig(**{**asdict(base_config), "n_layers": layers})
    else:
        raise ValueError("layers must be positive")
    return RGBOutputTransformer(config, target_vocab_size, target_seq_len)


def make_loader(dataset: ColourSequenceDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, drop_last=False)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def evaluate(model: nn.Module, dataset: ColourSequenceDataset, *, device: torch.device, batch_size: int) -> dict[str, float]:
    loader = make_loader(dataset, batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_rows = 0
    token_correct = 0
    exact_correct = 0
    channel_correct = [0, 0, 0]
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = compute_loss(logits, targets)
            predictions = logits.argmax(dim=-1)
            rows = inputs.size(0)
            total_loss += loss.item() * rows
            total_rows += rows
            token_correct += (predictions == targets).sum().item()
            exact_correct += (predictions == targets).all(dim=-1).sum().item()
            for index in range(3):
                channel_correct[index] += (predictions[:, index] == targets[:, index]).sum().item()
    return {
        "loss": total_loss / max(total_rows, 1),
        "token_accuracy": token_correct / max(3 * total_rows, 1),
        "exact_match_accuracy": exact_correct / max(total_rows, 1),
        "red_accuracy": channel_correct[0] / max(total_rows, 1),
        "green_accuracy": channel_correct[1] / max(total_rows, 1),
        "blue_accuracy": channel_correct[2] / max(total_rows, 1),
    }


def cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "cuda_allocated_mb": 0.0,
            "cuda_reserved_mb": 0.0,
            "cuda_peak_allocated_mb": 0.0,
            "cuda_peak_reserved_mb": 0.0,
        }
    return {
        "cuda_allocated_mb": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
        "cuda_reserved_mb": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
    }


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_environment_payload(device: torch.device) -> dict[str, object]:
    return {
        "created_at_utc": timestamp_utc(),
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
    }


def metrics_fieldnames(eval_split_names: list[str]) -> list[str]:
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
    for split_name in eval_split_names:
        for suffix in ["loss", "token_accuracy", "exact_match_accuracy", "red_accuracy", "green_accuracy", "blue_accuracy"]:
            fieldnames.append(f"{split_name}_{suffix}")
    return fieldnames


def write_progress(path: Path, *, config: ColourRunConfig, status: str, step: int, train_size: int, steps_per_epoch: int, train_loss: float | None, eval_metrics: dict[str, dict[str, float]] | None) -> None:
    payload: dict[str, object] = {
        "updated_at_utc": timestamp_utc(),
        "status": status,
        "study_name": config.study_name,
        "run_name": Path(config.output_dir).name,
        "seed": config.seed,
        "step": step,
        "max_steps": config.max_steps,
        "progress_fraction": 1.0 if config.max_steps <= 0 else min(max(step / config.max_steps, 0.0), 1.0),
        "train_size": train_size,
        "steps_per_epoch": steps_per_epoch,
        "output_dir": config.output_dir,
    }
    if train_loss is not None:
        payload["train_update_loss"] = train_loss
    if eval_metrics is not None:
        for split_name, split_metrics in eval_metrics.items():
            for key, value in split_metrics.items():
                payload[f"{split_name}_{key}"] = value
    write_json(path, payload)


def should_checkpoint(config: ColourRunConfig, step: int) -> bool:
    if config.checkpoint_steps is not None:
        return step in config.checkpoint_steps
    return config.checkpoint_every_steps is not None and step % config.checkpoint_every_steps == 0


def save_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "result": result,
        },
        path,
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
    config = ColourRunConfig(
        study_name="colour_combination_base",
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
            "num_value_tokens": int(metadata["num_value_tokens"]),
            "num_t_tokens": int(metadata["num_t_tokens"]),
            "target_sequence_length": int(metadata["target_sequence_length"]),
            "split_fractions": metadata["split_fractions"],
        },
    )
    set_seed(seed)
    output_dir = ensure_dir(config.output_dir)
    device = torch.device(config.device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_dataset = load_split_dataset(args.dataset_dir, "train")
    val_dataset = load_split_dataset(args.dataset_dir, "val")
    test_dataset = load_split_dataset(args.dataset_dir, "test")
    eval_datasets = {"val": val_dataset, "test": test_dataset}
    eval_datasets_with_train = {"train": train_dataset, **eval_datasets}
    vocab_size = int(metadata["vocab_size"])
    target_vocab_size = int(metadata["target_vocab_size"])
    seq_len = int(metadata["sequence_length"])
    target_seq_len = int(metadata["target_sequence_length"])
    model = build_model(
        layers=layers,
        vocab_size=vocab_size,
        target_vocab_size=target_vocab_size,
        seq_len=seq_len,
        target_seq_len=target_seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
    train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
    train_iterator = iter(train_loader)
    steps_per_epoch = len(train_loader)

    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "dataset_info": {
                "vocab_size": vocab_size,
                "target_vocab_size": target_vocab_size,
                "seq_len": seq_len,
                "target_seq_len": target_seq_len,
            },
            "split_sizes": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
            "steps_per_epoch": steps_per_epoch,
            "parameter_count": parameter_count(model),
        },
    )
    write_json(output_dir / "runtime_environment.json", runtime_environment_payload(device))
    torch.save(
        {
            "train": {"inputs": train_dataset.inputs.cpu(), "targets": train_dataset.targets.cpu()},
            "val": {"inputs": val_dataset.inputs.cpu(), "targets": val_dataset.targets.cpu()},
            "test": {"inputs": test_dataset.inputs.cpu(), "targets": test_dataset.targets.cpu()},
        },
        output_dir / "dataset_snapshot.pt",
    )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_csv_path = output_dir / "metrics.csv"
    progress_path = output_dir / "progress.json"
    metric_fields = metrics_fieldnames(["train", *sorted(eval_datasets.keys())])
    write_progress(
        progress_path,
        config=config,
        status="starting",
        step=0,
        train_size=len(train_dataset),
        steps_per_epoch=steps_per_epoch,
        train_loss=None,
        eval_metrics=None,
    )

    final_eval: dict[str, dict[str, float]] = {}
    train_loss = float("nan")
    step = 0
    progress = tqdm(total=config.max_steps, desc=output_dir.name, leave=False, dynamic_ncols=True)
    try:
        for step in range(1, config.max_steps + 1):
            try:
                inputs, targets = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                inputs, targets = next(train_iterator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = compute_loss(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss = float(loss.item())

            should_log = step == 1 or step % config.log_every == 0 or step % config.eval_every == 0 or step == config.max_steps
            if should_log:
                record: dict[str, object] = {
                    "step": step,
                    "train_update_loss": train_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "param_norm": float(torch.sqrt(sum(parameter.detach().float().pow(2).sum() for parameter in model.parameters())).item()),
                }
                record.update(cuda_memory_stats(device))
                if step == 1 or step % config.eval_every == 0 or step == config.max_steps:
                    final_eval = {}
                    for split_name, dataset in eval_datasets_with_train.items():
                        split_metrics = evaluate(model, dataset, device=device, batch_size=config.batch_size)
                        final_eval[split_name] = split_metrics
                        for key, value in split_metrics.items():
                            record[f"{split_name}_{key}"] = value
                append_jsonl(metrics_path, record)
                append_csv_stable(metrics_csv_path, metric_fields, record)
                postfix = {"loss": f"{train_loss:.4f}"}
                if "test" in final_eval:
                    postfix["test_exact"] = f"{final_eval['test']['exact_match_accuracy']:.3f}"
                progress.set_postfix(postfix, refresh=False)

            if step == 1 or step % config.log_every == 0 or step % config.eval_every == 0 or step == config.max_steps:
                write_progress(
                    progress_path,
                    config=config,
                    status="running",
                    step=step,
                    train_size=len(train_dataset),
                    steps_per_epoch=steps_per_epoch,
                    train_loss=train_loss,
                    eval_metrics=final_eval or None,
                )

            if should_checkpoint(config, step):
                save_checkpoint(
                    output_dir / "checkpoints" / f"step_{step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    result={"step": step, "output_dir": str(output_dir), **config.metadata},
                )
            progress.update(1)
    except Exception:
        write_progress(
            progress_path,
            config=config,
            status="failed",
            step=step,
            train_size=len(train_dataset),
            steps_per_epoch=steps_per_epoch,
            train_loss=None if math.isnan(train_loss) else train_loss,
            eval_metrics=final_eval or None,
        )
        progress.close()
        raise

    progress.close()
    if not final_eval:
        final_eval = {
            split_name: evaluate(model, dataset, device=device, batch_size=config.batch_size)
            for split_name, dataset in eval_datasets_with_train.items()
        }
    final_checkpoint_path = (
        output_dir / "checkpoints" / f"step_{step:06d}.pt"
        if should_checkpoint(config, step)
        else output_dir / "final_checkpoint.pt"
    )
    result = {
        "study_name": config.study_name,
        "seed": config.seed,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "completed_steps": step,
        "output_dir": str(output_dir),
        "checkpoint_path": str(final_checkpoint_path),
        "train_size": len(train_dataset),
        "steps_per_epoch": steps_per_epoch,
        "parameter_count": parameter_count(model),
        **cuda_memory_stats(device),
        **config.metadata,
    }
    for split_name, split_metrics in final_eval.items():
        for key, value in split_metrics.items():
            result[f"{split_name}_{key}"] = value
    append_csv(output_dir / "summary_row.csv", result)
    write_json(output_dir / "result.json", result)
    write_progress(
        progress_path,
        config=config,
        status="completed",
        step=step,
        train_size=len(train_dataset),
        steps_per_epoch=steps_per_epoch,
        train_loss=train_loss,
        eval_metrics=final_eval,
    )
    save_checkpoint(final_checkpoint_path, model=model, optimizer=optimizer, step=step, result=result)
    return result


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
        "full_train_eval_logged": True,
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
        "checkpoint_steps": list(staged_checkpoint_steps(args.max_steps, args.checkpoint_every_steps))
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


def read_log_tail(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return f"<missing log: {path}>"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read log {path}: {exc}>"
    return "\n".join(lines[-max_lines:])


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
                log_path = item["log_path"]
                assert isinstance(log_path, Path)
                tqdm.write(f"[FAILED] layers={item['layers']} seed={item['seed']} return_code={return_code}")
                tqdm.write(f"[FAILED LOG TAIL] {log_path}\n{read_log_tail(log_path)}")
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
                    "log_path": log_path,
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
