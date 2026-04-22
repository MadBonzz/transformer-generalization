from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Study 3 token PCA plots by train fraction.")
    parser.add_argument("--root", type=str, default="outputs/study3_range_transfer")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-fractions", type=str, default="0.1,0.3,0.5")
    parser.add_argument("--annotate-every", type=int, default=10)
    return parser.parse_args()


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ v[:, :2]
    variances = coords.var(dim=0, unbiased=False)
    variance_ratio = variances / centered.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    return coords, variance_ratio


def save_coords_csv(path: Path, coords: torch.Tensor, add_offset: int, mul_offset: int, modulus: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token_id", "group", "display_number", "pc1", "pc2"])
        writer.writeheader()
        for token_id in range(add_offset, add_offset + modulus):
            writer.writerow(
                {
                    "token_id": token_id,
                    "group": "add_range",
                    "display_number": token_id - add_offset,
                    "pc1": float(coords[token_id, 0].item()),
                    "pc2": float(coords[token_id, 1].item()),
                }
            )
        for token_id in range(mul_offset, mul_offset + modulus):
            writer.writerow(
                {
                    "token_id": token_id,
                    "group": "mul_range",
                    "display_number": token_id - mul_offset,
                    "pc1": float(coords[token_id, 0].item()),
                    "pc2": float(coords[token_id, 1].item()),
                }
            )


def save_plot(
    path: Path,
    *,
    coords: torch.Tensor,
    add_offset: int,
    mul_offset: int,
    modulus: int,
    title: str,
    variance_ratio: torch.Tensor,
    annotate_every: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    add_slice = coords[add_offset : add_offset + modulus]
    mul_slice = coords[mul_offset : mul_offset + modulus]

    ax.scatter(add_slice[:, 0].numpy(), add_slice[:, 1].numpy(), s=28, c="#1f77b4", label="tokens 0-99", alpha=0.85)
    ax.scatter(mul_slice[:, 0].numpy(), mul_slice[:, 1].numpy(), s=28, c="#d62728", label="tokens 100-199", alpha=0.85)

    if annotate_every > 0:
        for idx in range(0, modulus, annotate_every):
            ax.annotate(str(idx), (float(add_slice[idx, 0]), float(add_slice[idx, 1])), fontsize=6, color="#1f77b4")
            ax.annotate(str(idx), (float(mul_slice[idx, 0]), float(mul_slice[idx, 1])), fontsize=6, color="#d62728")

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({float(variance_ratio[0].item()) * 100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({float(variance_ratio[1].item()) * 100:.2f}% var)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    train_fractions = parse_csv_list(args.train_fractions)
    summary_rows: list[dict[str, object]] = []

    for tf in train_fractions:
        run_dirs = sorted(root.glob(f"transformer_tf{tf}_ce_seed*_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}"))
        run_dirs = [path for path in run_dirs if (path / "final_checkpoint.pt").exists() and (path / "config.json").exists()]
        if not run_dirs:
            print(f"skip tf={tf}: no matching runs")
            continue

        config = load_json(run_dirs[0] / "config.json")
        metadata = dict(config.get("metadata", {}))
        modulus = int(metadata["input_modulus"])
        add_offset = int(metadata["add_offset"])
        mul_offset = int(metadata["mul_offset"])

        embeds = []
        for run_dir in run_dirs:
            checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu")
            embeds.append(checkpoint["model_state_dict"]["token_embed.weight"].detach().cpu())
        mean_embed = torch.stack(embeds, dim=0).mean(dim=0)

        coords, variance_ratio = pca_2d(mean_embed)
        stem = f"token_embedding_pca_tf{tf}_lr{args.lr}_wd{args.weight_decay}_bs{args.batch_size}"
        coords_csv_path = root / f"{stem}.csv"
        plot_path = root / f"{stem}.png"
        save_coords_csv(coords_csv_path, coords, add_offset, mul_offset, modulus)
        save_plot(
            plot_path,
            coords=coords,
            add_offset=add_offset,
            mul_offset=mul_offset,
            modulus=modulus,
            title=f"Study 3 Token PCA | train_fraction={tf} | mean over {len(run_dirs)} seeds",
            variance_ratio=variance_ratio,
            annotate_every=args.annotate_every,
        )
        summary_rows.append(
            {
                "train_fraction": tf,
                "num_runs": len(run_dirs),
                "pc1_variance_ratio": float(variance_ratio[0].item()),
                "pc2_variance_ratio": float(variance_ratio[1].item()),
                "coords_csv": str(coords_csv_path),
                "plot_path": str(plot_path),
            }
        )

    summary_path = root / "token_embedding_pca_by_train_fraction_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["train_fraction", "num_runs", "pc1_variance_ratio", "pc2_variance_ratio", "coords_csv", "plot_path"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
