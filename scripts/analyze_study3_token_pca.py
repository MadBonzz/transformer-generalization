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
    parser = argparse.ArgumentParser(description="Run PCA on Study 3 token embeddings and save plots.")
    parser.add_argument("--root", type=str, default="outputs/study3_range_transfer")
    parser.add_argument("--annotate-every", type=int, default=10)
    parser.add_argument("--include-summary", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ v[:, :2]
    variances = coords.var(dim=0, unbiased=False)
    variance_ratio = variances / centered.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    return coords, variance_ratio


def _write_coords_csv(
    path: Path,
    *,
    coords: torch.Tensor,
    add_offset: int,
    mul_offset: int,
    modulus: int,
) -> None:
    fieldnames = ["token_id", "group", "display_number", "pc1", "pc2"]
    rows: list[dict[str, object]] = []
    for token_id in range(add_offset, add_offset + modulus):
        display_number = token_id - add_offset
        rows.append(
            {
                "token_id": token_id,
                "group": "add_range",
                "display_number": display_number,
                "pc1": float(coords[token_id, 0].item()),
                "pc2": float(coords[token_id, 1].item()),
            }
        )
    for token_id in range(mul_offset, mul_offset + modulus):
        display_number = token_id - mul_offset
        rows.append(
            {
                "token_id": token_id,
                "group": "mul_range",
                "display_number": display_number,
                "pc1": float(coords[token_id, 0].item()),
                "pc2": float(coords[token_id, 1].item()),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(
    path: Path,
    *,
    coords: torch.Tensor,
    add_offset: int,
    mul_offset: int,
    modulus: int,
    run_name: str,
    variance_ratio: torch.Tensor,
    annotate_every: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    add_slice = coords[add_offset : add_offset + modulus]
    mul_slice = coords[mul_offset : mul_offset + modulus]

    ax.scatter(
        add_slice[:, 0].numpy(),
        add_slice[:, 1].numpy(),
        s=28,
        c="#1f77b4",
        label="tokens 0-99",
        alpha=0.8,
    )
    ax.scatter(
        mul_slice[:, 0].numpy(),
        mul_slice[:, 1].numpy(),
        s=28,
        c="#d62728",
        label="tokens 100-199",
        alpha=0.8,
    )

    if annotate_every > 0:
        for idx in range(0, modulus, annotate_every):
            ax.annotate(str(idx), (float(add_slice[idx, 0]), float(add_slice[idx, 1])), fontsize=6, color="#1f77b4")
            ax.annotate(str(idx), (float(mul_slice[idx, 0]), float(mul_slice[idx, 1])), fontsize=6, color="#d62728")

    ax.set_title(run_name)
    ax.set_xlabel(f"PC1 ({float(variance_ratio[0].item()) * 100:.2f}% var)")
    ax.set_ylabel(f"PC2 ({float(variance_ratio[1].item()) * 100:.2f}% var)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "final_checkpoint.pt").exists() and (path / "config.json").exists()
    )

    summary_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        config = _load_json(run_dir / "config.json")
        metadata = dict(config.get("metadata", {}))
        modulus = int(metadata["input_modulus"])
        add_offset = int(metadata["add_offset"])
        mul_offset = int(metadata["mul_offset"])

        checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu")
        token_embed = checkpoint["model_state_dict"]["token_embed.weight"].detach().cpu()
        coords, variance_ratio = _pca_2d(token_embed)

        coords_csv_path = run_dir / "token_embedding_pca.csv"
        plot_path = run_dir / "token_embedding_pca.png"
        _write_coords_csv(
            coords_csv_path,
            coords=coords,
            add_offset=add_offset,
            mul_offset=mul_offset,
            modulus=modulus,
        )
        _save_plot(
            plot_path,
            coords=coords,
            add_offset=add_offset,
            mul_offset=mul_offset,
            modulus=modulus,
            run_name=run_dir.name,
            variance_ratio=variance_ratio,
            annotate_every=args.annotate_every,
        )

        add_centroid = coords[add_offset : add_offset + modulus].mean(dim=0)
        mul_centroid = coords[mul_offset : mul_offset + modulus].mean(dim=0)
        centroid_distance = torch.norm(add_centroid - mul_centroid).item()
        summary_rows.append(
            {
                "run": run_dir.name,
                "pc1_variance_ratio": float(variance_ratio[0].item()),
                "pc2_variance_ratio": float(variance_ratio[1].item()),
                "centroid_distance": centroid_distance,
                "coords_csv": str(coords_csv_path),
                "plot_path": str(plot_path),
            }
        )

    if args.include_summary:
        summary_path = root / "token_embedding_pca_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["run", "pc1_variance_ratio", "pc2_variance_ratio", "centroid_distance", "coords_csv", "plot_path"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"wrote summary to {summary_path}")

    print(f"processed {len(run_dirs)} runs under {root}")


if __name__ == "__main__":
    main()
