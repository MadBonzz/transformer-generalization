from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.tasks import _build_study3_groups


COLOR_CYCLE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA and/or t-SNE on Study 3 token embeddings across retained result layouts."
    )
    parser.add_argument("--root", required=True, type=str, help="Run directory or parent directory containing runs.")
    parser.add_argument("--methods", type=str, default="pca,tsne", help="Comma-separated subset of pca,tsne.")
    parser.add_argument("--annotate-every", type=int, default=1)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--include-summary", action="store_true")
    return parser.parse_args()


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("at least one method must be selected")
    invalid = [method for method in methods if method not in {"pca", "tsne"}]
    if invalid:
        raise ValueError(f"unsupported methods: {', '.join(invalid)}")
    return methods


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_run_dirs(root: Path) -> list[Path]:
    if (root / "final_checkpoint.pt").exists() and (root / "config.json").exists():
        return [root]
    run_dirs = {
        checkpoint.parent
        for checkpoint in root.rglob("final_checkpoint.pt")
        if (checkpoint.parent / "config.json").exists()
    }
    return sorted(run_dirs)


def _pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2, center=False)
    coords = centered @ v[:, :2]
    variances = coords.var(dim=0, unbiased=False)
    variance_ratio = variances / centered.var(dim=0, unbiased=False).sum().clamp_min(1e-12)
    return coords, variance_ratio


def _tsne_2d(matrix: torch.Tensor, *, perplexity: float, seed: int) -> torch.Tensor:
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(matrix.numpy())
    return torch.tensor(embedding, dtype=torch.float32)


def _offset_groups(metadata: dict[str, object]) -> list[tuple[str, list[int]]]:
    modulus = int(metadata["input_modulus"])
    add_offset = int(metadata["add_offset"])
    mul_offset = int(metadata["mul_offset"])
    add_group = list(range(add_offset, add_offset + modulus))
    mul_group = list(range(mul_offset, mul_offset + modulus))
    return [("add_range", add_group), ("mul_range", mul_group)]


def _study3_groups(metadata: dict[str, object]) -> list[tuple[str, list[int]]]:
    groups = _build_study3_groups(
        modulus=int(metadata["input_modulus"]),
        scenario=str(metadata["task_scenario"]),
        add_set_size=(None if metadata.get("add_set_size") is None else int(metadata["add_set_size"])),
        interleave_chunk_size=(
            None if metadata.get("interleave_chunk_size") is None else int(metadata["interleave_chunk_size"])
        ),
    )
    return [(f"group_{index}", group) for index, group in enumerate(groups)]


def _token_groups(config: dict[str, object]) -> list[tuple[str, list[int]]]:
    metadata = dict(config.get("metadata", {}))
    if "task_scenario" in metadata:
        return _study3_groups(metadata)
    if "add_offset" in metadata and "mul_offset" in metadata:
        return _offset_groups(metadata)
    raise ValueError("unsupported config metadata for embedding analysis")


def _centroid_distance_stats(coords: torch.Tensor, groups: list[tuple[str, list[int]]]) -> dict[str, float]:
    centroids = [coords[token_ids].mean(dim=0) for _, token_ids in groups]
    if len(centroids) < 2:
        return {
            "centroid_distance_min": 0.0,
            "centroid_distance_max": 0.0,
            "centroid_distance_mean": 0.0,
        }

    distances: list[float] = []
    for source_index in range(len(centroids)):
        for target_index in range(source_index + 1, len(centroids)):
            distances.append(float(torch.norm(centroids[source_index] - centroids[target_index]).item()))
    return {
        "centroid_distance_min": min(distances),
        "centroid_distance_max": max(distances),
        "centroid_distance_mean": sum(distances) / len(distances),
    }


def _write_coords_csv(
    path: Path,
    *,
    coords: torch.Tensor,
    groups: list[tuple[str, list[int]]],
    x_name: str,
    y_name: str,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token_id", "group", "display_number", x_name, y_name])
        writer.writeheader()
        for group_name, token_ids in groups:
            for display_index, token_id in enumerate(token_ids):
                writer.writerow(
                    {
                        "token_id": token_id,
                        "group": group_name,
                        "display_number": display_index,
                        x_name: float(coords[token_id, 0].item()),
                        y_name: float(coords[token_id, 1].item()),
                    }
                )


def _save_plot(
    path: Path,
    *,
    coords: torch.Tensor,
    groups: list[tuple[str, list[int]]],
    title: str,
    x_label: str,
    y_label: str,
    annotate_every: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7), dpi=180)
    for index, (group_name, token_ids) in enumerate(groups):
        color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
        group_coords = coords[token_ids]
        axis.scatter(
            group_coords[:, 0].numpy(),
            group_coords[:, 1].numpy(),
            s=28,
            c=color,
            label=group_name,
            alpha=0.85,
        )
        if annotate_every > 0:
            for display_index, token_id in enumerate(token_ids):
                if display_index % annotate_every != 0:
                    continue
                axis.annotate(
                    str(token_id),
                    (float(coords[token_id, 0]), float(coords[token_id, 1])),
                    fontsize=5,
                    color=color,
                )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _analyze_run(
    run_dir: Path,
    *,
    methods: list[str],
    annotate_every: int,
    tsne_perplexity: float,
    tsne_seed: int,
) -> list[dict[str, object]]:
    config = _load_json(run_dir / "config.json")
    groups = _token_groups(config)
    checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu")
    token_embed = checkpoint["model_state_dict"]["token_embed.weight"].detach().cpu()

    summary_rows: list[dict[str, object]] = []

    if "pca" in methods:
        pca_coords, variance_ratio = _pca_2d(token_embed)
        pca_csv = run_dir / "token_embedding_pca.csv"
        pca_png = run_dir / "token_embedding_pca.png"
        _write_coords_csv(pca_csv, coords=pca_coords, groups=groups, x_name="pc1", y_name="pc2")
        _save_plot(
            pca_png,
            coords=pca_coords,
            groups=groups,
            title=f"{run_dir.name} | PCA",
            x_label=f"PC1 ({float(variance_ratio[0].item()) * 100:.2f}% var)",
            y_label=f"PC2 ({float(variance_ratio[1].item()) * 100:.2f}% var)",
            annotate_every=annotate_every,
        )
        summary_rows.append(
            {
                "run": run_dir.name,
                "method": "pca",
                "coord_csv": str(pca_csv),
                "plot_path": str(pca_png),
                "component_1_variance_ratio": float(variance_ratio[0].item()),
                "component_2_variance_ratio": float(variance_ratio[1].item()),
                **_centroid_distance_stats(pca_coords, groups),
            }
        )

    if "tsne" in methods:
        tsne_coords = _tsne_2d(token_embed, perplexity=tsne_perplexity, seed=tsne_seed)
        tsne_csv = run_dir / "token_embedding_tsne.csv"
        tsne_png = run_dir / "token_embedding_tsne.png"
        _write_coords_csv(tsne_csv, coords=tsne_coords, groups=groups, x_name="tsne1", y_name="tsne2")
        _save_plot(
            tsne_png,
            coords=tsne_coords,
            groups=groups,
            title=f"{run_dir.name} | t-SNE perplexity={tsne_perplexity:g}",
            x_label="t-SNE 1",
            y_label="t-SNE 2",
            annotate_every=annotate_every,
        )
        summary_rows.append(
            {
                "run": run_dir.name,
                "method": "tsne",
                "coord_csv": str(tsne_csv),
                "plot_path": str(tsne_png),
                "component_1_variance_ratio": math.nan,
                "component_2_variance_ratio": math.nan,
                **_centroid_distance_stats(tsne_coords, groups),
            }
        )

    return summary_rows


def main() -> None:
    args = parse_args()
    methods = _parse_methods(args.methods)
    root = Path(args.root)
    run_dirs = _iter_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"no run directories found under {root}")

    summary_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        summary_rows.extend(
            _analyze_run(
                run_dir,
                methods=methods,
                annotate_every=args.annotate_every,
                tsne_perplexity=args.tsne_perplexity,
                tsne_seed=args.tsne_seed,
            )
        )

    if args.include_summary:
        summary_path = root / "token_embedding_analysis_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run",
                    "method",
                    "coord_csv",
                    "plot_path",
                    "component_1_variance_ratio",
                    "component_2_variance_ratio",
                    "centroid_distance_min",
                    "centroid_distance_max",
                    "centroid_distance_mean",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"wrote summary to {summary_path}")

    print(f"processed {len(run_dirs)} runs under {root}")


if __name__ == "__main__":
    main()
