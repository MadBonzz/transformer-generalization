from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.tasks import _build_contiguous_sets, _build_interleaved_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA and t-SNE on Study 3 token embeddings for the new scenario format.")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--annotate-every", type=int, default=1)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--include-summary", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ v[:, :2]
    variances = coords.var(dim=0, unbiased=False)
    variance_ratio = variances / centered.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    return coords, variance_ratio


def tsne_2d(matrix: torch.Tensor, *, perplexity: float, seed: int) -> torch.Tensor:
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(matrix.numpy())
    return torch.tensor(embedding, dtype=torch.float32)


def study3_sets(metadata: dict[str, object]) -> tuple[list[int], list[int]]:
    modulus = int(metadata["input_modulus"])
    task_scenario = str(metadata["task_scenario"])
    interleave_chunk_size = metadata.get("interleave_chunk_size")
    add_set_size = metadata.get("add_set_size")
    if task_scenario == "interleaved_partitioned_ops":
        if interleave_chunk_size is None:
            raise ValueError("interleave_chunk_size is required for interleaved_partitioned_ops")
        return _build_interleaved_sets(modulus, int(interleave_chunk_size))
    return _build_contiguous_sets(modulus, int(add_set_size) if add_set_size is not None else None)


def write_coords_csv(path: Path, coords: torch.Tensor, set_a: list[int], set_b: list[int], x_name: str, y_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token_id", "group", "display_number", x_name, y_name])
        writer.writeheader()
        for token_id in set_a:
            writer.writerow(
                {
                    "token_id": token_id,
                    "group": "set_a",
                    "display_number": token_id,
                    x_name: float(coords[token_id, 0].item()),
                    y_name: float(coords[token_id, 1].item()),
                }
            )
        for token_id in set_b:
            writer.writerow(
                {
                    "token_id": token_id,
                    "group": "set_b",
                    "display_number": token_id,
                    x_name: float(coords[token_id, 0].item()),
                    y_name: float(coords[token_id, 1].item()),
                }
            )


def save_plot(
    path: Path,
    *,
    coords: torch.Tensor,
    set_a: list[int],
    set_b: list[int],
    title: str,
    x_label: str,
    y_label: str,
    annotate_every: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)

    a_coords = coords[set_a]
    b_coords = coords[set_b]

    ax.scatter(a_coords[:, 0].numpy(), a_coords[:, 1].numpy(), s=28, c="#1f77b4", label="set A", alpha=0.85)
    ax.scatter(b_coords[:, 0].numpy(), b_coords[:, 1].numpy(), s=28, c="#d62728", label="set B", alpha=0.85)

    if annotate_every > 0:
        for idx, token_id in enumerate(set_a):
            if idx % annotate_every == 0:
                ax.annotate(str(token_id), (float(coords[token_id, 0]), float(coords[token_id, 1])), fontsize=5, color="#1f77b4")
        for idx, token_id in enumerate(set_b):
            if idx % annotate_every == 0:
                ax.annotate(str(token_id), (float(coords[token_id, 0]), float(coords[token_id, 1])), fontsize=5, color="#d62728")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    run_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "final_checkpoint.pt").exists() and (path / "config.json").exists()
    )

    summary_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        config = load_json(run_dir / "config.json")
        metadata = dict(config.get("metadata", {}))
        set_a, set_b = study3_sets(metadata)

        checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu")
        token_embed = checkpoint["model_state_dict"]["token_embed.weight"].detach().cpu()

        pca_coords, variance_ratio = pca_2d(token_embed)
        tsne_coords = tsne_2d(token_embed, perplexity=args.tsne_perplexity, seed=args.tsne_seed)

        pca_csv = run_dir / "token_embedding_pca_variant.csv"
        pca_png = run_dir / "token_embedding_pca_variant.png"
        tsne_csv = run_dir / "token_embedding_tsne_variant.csv"
        tsne_png = run_dir / "token_embedding_tsne_variant.png"

        write_coords_csv(pca_csv, pca_coords, set_a, set_b, "pc1", "pc2")
        save_plot(
            pca_png,
            coords=pca_coords,
            set_a=set_a,
            set_b=set_b,
            title=f"{run_dir.name} | PCA",
            x_label=f"PC1 ({float(variance_ratio[0].item()) * 100:.2f}% var)",
            y_label=f"PC2 ({float(variance_ratio[1].item()) * 100:.2f}% var)",
            annotate_every=args.annotate_every,
        )

        write_coords_csv(tsne_csv, tsne_coords, set_a, set_b, "tsne1", "tsne2")
        save_plot(
            tsne_png,
            coords=tsne_coords,
            set_a=set_a,
            set_b=set_b,
            title=f"{run_dir.name} | t-SNE perplexity={args.tsne_perplexity:g}",
            x_label="t-SNE 1",
            y_label="t-SNE 2",
            annotate_every=args.annotate_every,
        )

        a_centroid = pca_coords[set_a].mean(dim=0)
        b_centroid = pca_coords[set_b].mean(dim=0)
        summary_rows.append(
            {
                "run": run_dir.name,
                "pca_pc1_variance_ratio": float(variance_ratio[0].item()),
                "pca_pc2_variance_ratio": float(variance_ratio[1].item()),
                "pca_centroid_distance": float(torch.norm(a_centroid - b_centroid).item()),
                "pca_csv": str(pca_csv),
                "pca_plot": str(pca_png),
                "tsne_csv": str(tsne_csv),
                "tsne_plot": str(tsne_png),
            }
        )

    if args.include_summary:
        summary_path = root / "token_embedding_variant_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run",
                    "pca_pc1_variance_ratio",
                    "pca_pc2_variance_ratio",
                    "pca_centroid_distance",
                    "pca_csv",
                    "pca_plot",
                    "tsne_csv",
                    "tsne_plot",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"wrote summary to {summary_path}")

    print(f"processed {len(run_dirs)} runs under {root}")


if __name__ == "__main__":
    main()
