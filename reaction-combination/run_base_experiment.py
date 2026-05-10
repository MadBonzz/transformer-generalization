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
    parameter_count,
    transformer_architecture_name,
    transformer_run_prefix,
)
from grokking_transformer.logging_utils import append_csv, append_csv_stable, append_jsonl, ensure_dir, write_json  # noqa: E402
from grokking_transformer.model import TransformerBlock, TransformerConfig  # noqa: E402

sys.path.insert(0, str(THIS_DIR))
from generate_reaction_dataset import NULL_TOKEN, parse_formula  # noqa: E402


DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "reaction_base_case"


@dataclass(frozen=True)
class ReactionRunConfig:
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


class ReactionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs.long()
        self.targets = targets.long()

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class TwoOutputGrokkingTransformer(nn.Module):
    """Same transformer presets as src/model.py, with a two-token product head."""

    def __init__(self, config: TransformerConfig, target_vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.target_vocab_size = target_vocab_size
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        if config.positional_embedding_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(config.seq_len, config.d_model))
        else:
            self.register_buffer("pos_embed", self._sinusoidal_position_encoding(config.seq_len, config.d_model), persistent=False)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_ln = nn.LayerNorm(config.d_model) if config.final_norm else nn.Identity()
        self.output_head = nn.Linear(config.d_model, 2 * target_vocab_size, bias=False)
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
        return logits.view(tokens.size(0), 2, self.target_vocab_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 6-run reaction-combination baseline: 2 transformer depths x 3 seeds."
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
        default="staged",
        choices=["staged", "fixed", "none"],
        help="staged: every 1k through 25k, then every --checkpoint-every-steps; fixed: use only --checkpoint-every-steps.",
    )
    parser.add_argument("--checkpoint-every-steps", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--num-rows", type=int, default=100000)
    parser.add_argument("--max-scale", type=int, default=12)
    parser.add_argument("--element-max-scale", type=int, default=1)
    parser.add_argument("--element-synthesis-fraction", type=float, default=0.50)
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


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_name(num_rows: int) -> str:
    return f"reaction_combination_{num_rows}"


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return args.dataset_dir
    return args.output_root / "dataset" / dataset_name(args.num_rows)


def run_dir_for(output_root: Path, *, layers: int, seed: int, learning_rate: float, weight_decay: float, batch_size: int) -> Path:
    prefix = transformer_run_prefix(layers)
    run_name = f"{prefix}_reaction_mix_seed{seed}_lr{learning_rate}_wd{weight_decay}_bs{batch_size}"
    return output_root / "runs" / run_name


def generate_dataset(args: argparse.Namespace, dataset_dir: Path) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "dataset_generation.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(THIS_DIR / "generate_reaction_dataset.py"),
            "--output-dir",
            str(dataset_dir),
            "--seed",
            str(args.dataset_seed),
            "--num-rows",
            str(args.num_rows),
            "--max-scale",
            str(args.max_scale),
            "--element-max-scale",
            str(args.element_max_scale),
            "--element-synthesis-fraction",
            str(args.element_synthesis_fraction),
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


def load_metadata(dataset_dir: Path) -> dict[str, object]:
    with (dataset_dir / "metadata.json").open(encoding="utf-8") as file:
        return json.load(file)


def load_vocab(dataset_dir: Path) -> tuple[dict[str, int], dict[int, int]]:
    species_to_id: dict[str, int] = {}
    amount_to_id: dict[int, int] = {}
    with (dataset_dir / "vocab.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            token_id = int(row["token_id"])
            if row["kind"] in {"species", "null"}:
                species_to_id[row["value"]] = token_id
            elif row["kind"] == "amount":
                amount_to_id[int(row["value"])] = token_id
    return species_to_id, amount_to_id


def load_split_dataset(dataset_dir: Path, split: str) -> ReactionDataset:
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    with (dataset_dir / "tokenized_examples.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] != split:
                continue
            inputs.append(
                [
                    int(row["input_0_reactant_1_id"]),
                    int(row["input_1_amount_1_id"]),
                    int(row["input_2_reactant_2_id"]),
                    int(row["input_3_amount_2_id"]),
                ]
            )
            targets.append([int(row["target_0_output_1_id"]), int(row["target_1_output_2_id"])])
    if not inputs:
        raise ValueError(f"dataset split {split!r} is empty in {dataset_dir}")
    return ReactionDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long))


def parse_equation_side(side: str) -> tuple[list[str], list[int]]:
    formulas: list[str] = []
    coeffs: list[int] = []
    for part in side.split("+"):
        item = part.strip()
        match = re_match_formula(item)
        coeffs.append(match[0])
        formulas.append(match[1])
    return formulas, coeffs


def re_match_formula(item: str) -> tuple[int, str]:
    import re

    match = re.match(r"^(\d+)?(.+)$", item)
    if match is None:
        raise ValueError(f"invalid equation item: {item!r}")
    return int(match.group(1) or "1"), match.group(2)


def atom_counts(formulas: list[str], coeffs: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for formula, coeff in zip(formulas, coeffs, strict=True):
        if formula == NULL_TOKEN:
            continue
        parsed = parse_formula(formula)
        for element, count in parsed.items():
            counts[element] = counts.get(element, 0) + count * coeff
    return counts


def validate_dataset(dataset_dir: Path) -> dict[str, object]:
    species_to_id, amount_to_id = load_vocab(dataset_dir)
    metadata = load_metadata(dataset_dir)
    split_counts: dict[str, int] = {}
    null_rows = 0
    two_output_rows = 0
    family_counts: dict[str, int] = {}
    element_synthesis_rows = 0
    element_reactant_species: set[str] = set()
    row_count = 0
    used_species: set[str] = set()
    used_amounts: set[int] = set()

    with (dataset_dir / "reaction_combination.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row_count += 1
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            family = row.get("family", "")
            family_counts[family] = family_counts.get(family, 0) + 1
            if family == "element_synthesis":
                element_synthesis_rows += 1
                element_reactant_species.add(row["reactant_1"])
                element_reactant_species.add(row["reactant_2"])
            species_values = [row["reactant_1"], row["reactant_2"], row["output_1"], row["output_2"]]
            amount_values = [int(row["amount_1"]), int(row["amount_2"])]
            for species in species_values:
                if species not in species_to_id:
                    raise ValueError(f"missing species token for {species!r}")
                used_species.add(species)
            for amount in amount_values:
                if amount not in amount_to_id:
                    raise ValueError(f"missing amount token for {amount}")
                used_amounts.add(amount)

            if row["output_2"] == NULL_TOKEN:
                null_rows += 1
            else:
                two_output_rows += 1

            lhs, rhs = row["equation"].split("->")
            reactants, reactant_coeffs = parse_equation_side(lhs)
            products, product_coeffs = parse_equation_side(rhs)
            if atom_counts(reactants, reactant_coeffs) != atom_counts(products, product_coeffs):
                raise ValueError(f"unbalanced row: {row['equation']}")

    expected_rows = int(metadata["num_examples"])
    if row_count != expected_rows:
        raise ValueError(f"metadata num_examples={expected_rows}, actual rows={row_count}")
    if NULL_TOKEN not in species_to_id:
        raise ValueError("NULL token is missing from vocab")

    validation = {
        "validated_at_utc": timestamp_utc(),
        "row_count": row_count,
        "split_counts": split_counts,
        "species_token_count": len(species_to_id),
        "amount_token_count": len(amount_to_id),
        "used_species_count": len(used_species),
        "used_amount_count": len(used_amounts),
        "two_output_rows": two_output_rows,
        "single_output_rows_with_NULL": null_rows,
        "family_counts": family_counts,
        "element_synthesis_rows": element_synthesis_rows,
        "unique_element_reactant_tokens_in_element_synthesis": len(element_reactant_species),
        "null_token_id": species_to_id[NULL_TOKEN],
        "tokenization": "Each full element/molecule formula is one species token ID; formulas are not split into atom/subformula tokens.",
        "chemistry_grounding": (
            "Every CSV row is generated from curated element-element synthesis, acid-base neutralization, "
            "or solubility-rule double-displacement templates, and every written equation is atom-balance validated."
        ),
    }
    write_json(dataset_dir / "dataset_validation.json", validation)
    return validation


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(*, layers: int, vocab_size: int, target_vocab_size: int, seq_len: int) -> nn.Module:
    if layers == 1:
        config = TransformerConfig.neel_nanda(vocab_size=vocab_size, seq_len=seq_len)
    elif layers == 2:
        config = TransformerConfig.power_grokking(vocab_size=vocab_size, seq_len=seq_len)
    else:
        raise ValueError("layers must be 1 or 2")
    return TwoOutputGrokkingTransformer(config, target_vocab_size)


def make_loader(dataset: ReactionDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, drop_last=False)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def evaluate(model: nn.Module, dataset: ReactionDataset, *, device: torch.device, batch_size: int) -> dict[str, float]:
    loader = make_loader(dataset, batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_rows = 0
    token_correct = 0
    exact_correct = 0
    output_1_correct = 0
    output_2_correct = 0
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
            output_1_correct += (predictions[:, 0] == targets[:, 0]).sum().item()
            output_2_correct += (predictions[:, 1] == targets[:, 1]).sum().item()
    return {
        "loss": total_loss / max(total_rows, 1),
        "token_accuracy": token_correct / max(2 * total_rows, 1),
        "exact_match_accuracy": exact_correct / max(total_rows, 1),
        "output_1_accuracy": output_1_correct / max(total_rows, 1),
        "output_2_accuracy": output_2_correct / max(total_rows, 1),
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


def git_value(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def runtime_environment_payload(device: torch.device) -> dict[str, object]:
    git_status = git_value(["status", "--short"])
    payload: dict[str, object] = {
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
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_is_dirty": bool(git_status),
        "git_status_short": git_status or "",
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": torch.cuda.get_device_capability(device),
            }
        )
    return payload


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


def export_dataset_snapshot(path: Path, train: ReactionDataset, eval_datasets: dict[str, ReactionDataset]) -> None:
    payload: dict[str, dict[str, torch.Tensor]] = {
        "train": {"inputs": train.inputs.cpu(), "targets": train.targets.cpu()}
    }
    for split_name, dataset in eval_datasets.items():
        payload[split_name] = {"inputs": dataset.inputs.cpu(), "targets": dataset.targets.cpu()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def export_prediction_table(path: Path, model: nn.Module, dataset: ReactionDataset, *, device: torch.device, batch_size: int) -> None:
    rows: list[dict[str, object]] = []
    loader = make_loader(dataset, batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        offset = 0
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            probabilities = torch.softmax(logits, dim=-1)
            confidences, predictions = probabilities.max(dim=-1)
            batch_size_now = inputs.size(0)
            for i in range(batch_size_now):
                row_index = offset + i
                target_pair = dataset.targets[row_index].tolist()
                prediction_pair = predictions[i].cpu().tolist()
                rows.append(
                    {
                        "index": row_index,
                        "input_tokens": " ".join(str(int(x)) for x in dataset.inputs[row_index].tolist()),
                        "target_0": int(target_pair[0]),
                        "target_1": int(target_pair[1]),
                        "prediction_0": int(prediction_pair[0]),
                        "prediction_1": int(prediction_pair[1]),
                        "is_exact_match": int(prediction_pair == target_pair),
                        "is_output_0_correct": int(prediction_pair[0] == target_pair[0]),
                        "is_output_1_correct": int(prediction_pair[1] == target_pair[1]),
                        "confidence_0": float(confidences[i, 0].item()),
                        "confidence_1": float(confidences[i, 1].item()),
                    }
                )
            offset += batch_size_now
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_progress(path: Path, *, config: ReactionRunConfig, status: str, step: int, train_size: int, steps_per_epoch: int, train_loss: float | None, eval_metrics: dict[str, dict[str, float]] | None) -> None:
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
        for suffix in ["loss", "token_accuracy", "exact_match_accuracy", "output_1_accuracy", "output_2_accuracy"]:
            fieldnames.append(f"{split_name}_{suffix}")
    return fieldnames


def should_checkpoint(config: ReactionRunConfig, step: int) -> bool:
    if config.checkpoint_steps is not None:
        return step in config.checkpoint_steps
    return config.checkpoint_every_steps is not None and step % config.checkpoint_every_steps == 0


def run_one(args: argparse.Namespace, *, layers: int, seed: int) -> dict[str, object]:
    metadata = load_metadata(args.dataset_dir)
    output_dir = run_dir_for(
        args.output_root,
        layers=layers,
        seed=seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )
    config = ReactionRunConfig(
        study_name="reaction_combination_base",
        seed=seed,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        device=resolve_device(args.device),
        output_dir=str(output_dir),
        transformer_n_layers=layers,
        checkpoint_every_steps=args.checkpoint_every_steps if args.checkpoint_schedule == "fixed" else None,
        checkpoint_steps=checkpoint_steps_for_args(args),
        metadata={
            "dataset": str(metadata["name"]),
            "dataset_dir": str(args.dataset_dir),
            "dataset_seed": args.dataset_seed,
            "architecture": transformer_architecture_name(layers),
            "input_format": metadata["input_format"],
            "target_format": metadata["target_format"],
            "null_token": metadata["null_token"],
            "split_fractions": metadata["split_fractions"],
            "num_species_tokens_including_null": metadata["num_species_tokens_including_null"],
            "num_amount_tokens": metadata["num_amount_tokens"],
            "target_design": metadata["target_design"],
            "element_synthesis_fraction": metadata.get("element_synthesis_fraction"),
            "element_synthesis_rows": metadata.get("element_synthesis_rows"),
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
    model = build_model(layers=layers, vocab_size=vocab_size, target_vocab_size=target_vocab_size, seq_len=seq_len).to(device)
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
            },
            "split_sizes": {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
            "steps_per_epoch": steps_per_epoch,
            "parameter_count": parameter_count(model),
        },
    )
    write_json(output_dir / "runtime_environment.json", runtime_environment_payload(device))
    export_dataset_snapshot(output_dir / "dataset_snapshot.pt", train_dataset, eval_datasets)

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
    result = {
        "study_name": config.study_name,
        "seed": config.seed,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "max_steps": config.max_steps,
        "completed_steps": step,
        "output_dir": str(output_dir),
        "checkpoint_path": str(output_dir / "final_checkpoint.pt"),
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
    export_prediction_table(output_dir / "train_predictions.csv", model, train_dataset, device=device, batch_size=config.batch_size)
    for split_name, dataset in eval_datasets.items():
        export_prediction_table(output_dir / f"{split_name}_predictions.csv", model, dataset, device=device, batch_size=config.batch_size)
    save_checkpoint(output_dir / "final_checkpoint.pt", model=model, optimizer=optimizer, step=step, result=result)
    return result


def build_jobs(args: argparse.Namespace) -> list[tuple[int, int]]:
    return [(layers, seed) for layers in args.layers for seed in args.seeds]


def write_manifest(args: argparse.Namespace, jobs: list[tuple[int, int]], dataset_dir: Path, validation: dict[str, object]) -> None:
    manifest = {
        "dataset_dir": str(dataset_dir),
        "output_root": str(args.output_root),
        "dataset_kind": "chemistry_reaction_binary_two_output",
        "dataset_seed": args.dataset_seed,
        "num_rows": args.num_rows,
        "split_fractions": {"train": 0.5, "val": 0.25, "test": 0.25},
        "full_train_eval_logged": True,
        "element_synthesis_fraction": args.element_synthesis_fraction,
        "element_max_scale": args.element_max_scale,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "log_every": args.log_every,
        "checkpoint_schedule": args.checkpoint_schedule,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "checkpoint_steps": list(staged_checkpoint_steps(args.max_steps, args.checkpoint_every_steps))
        if args.checkpoint_schedule == "staged"
        else None,
        "dataset_generation_log": str(args.output_root / "dataset_generation.log"),
        "dataset_validation": validation,
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
    write_json(args.output_root / "experiment_manifest.json", manifest)


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
    return [
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
        "--num-rows",
        str(args.num_rows),
        "--max-scale",
        str(args.max_scale),
        "--element-max-scale",
        str(args.element_max_scale),
        "--element-synthesis-fraction",
        str(args.element_synthesis_fraction),
    ]


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

    progress = tqdm(total=len(jobs), desc="reaction-combination runs", unit="run", dynamic_ncols=True)
    tqdm.write(f"[SCHEDULER START] total_runs={len(jobs)} | max_parallel={max_parallel} | output={args.output_root}")
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
        if pending or running:
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

    print(f"Generating deterministic reaction dataset in {args.dataset_dir}")
    generate_dataset(args, args.dataset_dir)
    validation = validate_dataset(args.dataset_dir)
    print(f"Validated {validation['row_count']} rows; species tokens={validation['species_token_count']}; amount tokens={validation['amount_token_count']}")
    jobs = build_jobs(args)
    write_manifest(args, jobs, args.dataset_dir, validation)

    if args.parallel_workers == 1:
        for layers, seed in jobs:
            print(f"Running reaction-combination baseline: layers={layers}, seed={seed}")
            run_one(args, layers=layers, seed=seed)
    else:
        run_parallel(args, jobs, args.dataset_dir)

    aggregate_summary(args.output_root, jobs, args)
    print(f"Completed {len(jobs)} runs. Bundle: {args.output_root}")
    print(f"Summary: {args.output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
