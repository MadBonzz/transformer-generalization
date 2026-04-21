from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _plot_series(
    *,
    output_path: Path,
    title: str,
    series_specs: list[tuple[str, list[int], list[float]]],
    ylabel: str,
) -> None:
    available = [(label, xs, values) for label, xs, values in series_specs if values]
    if not available:
        return

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, xs, values in available:
        axis.plot(xs, values, label=label, linewidth=1.75)
    axis.set_title(title)
    axis.set_xlabel("Step")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def generate_run_plots(output_dir: str | Path) -> list[str]:
    run_dir = Path(output_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []

    records = _read_jsonl(metrics_path)
    if not records:
        return []

    written: list[str] = []

    loss_keys = (
        ("train_update_loss", "train_update_loss"),
        ("train_loss", "train_eval_loss"),
        ("test_loss", "test_loss"),
        ("cross_add_loss", "cross_add_loss"),
        ("cross_mul_loss", "cross_mul_loss"),
    )
    loss_series = [
        (
            label,
            [int(record["step"]) for record in records if key in record],
            [float(record[key]) for record in records if key in record],
        )
        for key, label in loss_keys
    ]
    loss_path = run_dir / "loss_curves.png"
    _plot_series(
        output_path=loss_path,
        title=f"{run_dir.name} Loss Curves",
        series_specs=loss_series,
        ylabel="Loss",
    )
    if loss_path.exists():
        written.append(str(loss_path))

    accuracy_keys = (
        ("train_true_accuracy", "train_true_accuracy"),
        ("test_true_accuracy", "test_true_accuracy"),
        ("train_label_accuracy", "train_label_accuracy"),
        ("test_label_accuracy", "test_label_accuracy"),
        ("cross_add_true_accuracy", "cross_add_true_accuracy"),
        ("cross_mul_true_accuracy", "cross_mul_true_accuracy"),
    )
    accuracy_series = [
        (
            label,
            [int(record["step"]) for record in records if key in record],
            [float(record[key]) for record in records if key in record],
        )
        for key, label in accuracy_keys
    ]
    accuracy_path = run_dir / "accuracy_curves.png"
    _plot_series(
        output_path=accuracy_path,
        title=f"{run_dir.name} Accuracy Curves",
        series_specs=accuracy_series,
        ylabel="Accuracy",
    )
    if accuracy_path.exists():
        written.append(str(accuracy_path))

    rl_keys = (
        ("train_reward_mean", "train_reward_mean"),
        ("train_entropy", "train_entropy"),
        ("train_kl", "train_kl"),
        ("train_value_loss", "train_value_loss"),
        ("train_clip_fraction", "train_clip_fraction"),
    )
    rl_series = [
        (
            label,
            [int(record["step"]) for record in records if key in record],
            [float(record[key]) for record in records if key in record],
        )
        for key, label in rl_keys
    ]
    rl_path = run_dir / "rl_diagnostics.png"
    _plot_series(
        output_path=rl_path,
        title=f"{run_dir.name} RL Diagnostics",
        series_specs=rl_series,
        ylabel="Value",
    )
    if rl_path.exists():
        written.append(str(rl_path))

    return written
