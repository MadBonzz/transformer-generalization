from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


DEFAULT_BUNDLE = (
    Path(__file__).resolve().parent
    / "outputs"
    / "mixbox_base_case"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE over learned colour-token representations.")
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["token_embed", "unembed"],
        default=["token_embed", "unembed"],
    )
    return parser.parse_args()


def resolve_dataset_dir(bundle_dir: Path) -> Path:
    dataset_root = bundle_dir / "dataset"
    if not dataset_root.exists():
        raise FileNotFoundError(f"missing dataset directory: {dataset_root}")
    candidates = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one dataset directory under {dataset_root}, got {len(candidates)}")
    return candidates[0]


def load_color_vocab(bundle_dir: Path) -> list[dict[str, str]]:
    vocab_path = resolve_dataset_dir(bundle_dir) / "vocab.csv"
    with vocab_path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["kind"] == "hex"]
    if not rows:
        raise ValueError(f"no hex-token rows found in {vocab_path}")
    return rows


def representation_matrix(checkpoint_path: Path, representation: str, num_color_tokens: int) -> torch.Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    if representation == "token_embed":
        matrix = state["token_embed.weight"]
    elif representation == "unembed":
        matrix = state["unembed.weight"]
    else:
        raise ValueError(f"unsupported representation={representation}")
    return matrix[:num_color_tokens].detach().float()


def run_tsne(matrix: torch.Tensor, *, perplexity: float, seed: int) -> list[tuple[float, float]]:
    embedded = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="cosine",
    ).fit_transform(matrix.numpy())
    return [(float(x), float(y)) for x, y in embedded]


def write_coordinates(
    *,
    path: Path,
    color_vocab: list[dict[str, str]],
    coords: list[tuple[float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["token_id", "token", "hex_code", "tsne_x", "tsne_y"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, (x_value, y_value) in zip(color_vocab, coords):
            writer.writerow(
                {
                    "token_id": row["token_id"],
                    "token": row["token"],
                    "hex_code": row["value"],
                    "tsne_x": x_value,
                    "tsne_y": y_value,
                }
            )


def draw_single_plot(
    *,
    path: Path,
    coords: list[tuple[float, float]],
    colors: list[str],
    title: str,
) -> None:
    x_values = [item[0] for item in coords]
    y_values = [item[1] for item in coords]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.scatter(x_values, y_values, c=colors, s=18, alpha=0.95, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_grid_plot(
    *,
    path: Path,
    run_coords: list[tuple[str, list[tuple[float, float]]]],
    colors: list[str],
    representation: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=180)
    for ax, (run_name, coords) in zip(axes.flatten(), run_coords):
        x_values = [item[0] for item in coords]
        y_values = [item[1] for item in coords]
        ax.scatter(x_values, y_values, c=colors, s=10, alpha=0.95, linewidths=0)
        ax.set_title(run_name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    fig.suptitle(f"Colour-token t-SNE: {representation}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.bundle_dir / "analysis" / "token_tsne"
    color_vocab = load_color_vocab(args.bundle_dir)
    colors = [row["value"] for row in color_vocab]
    run_dirs = sorted((args.bundle_dir / "runs").iterdir())

    for representation in args.representations:
        run_coords: list[tuple[str, list[tuple[float, float]]]] = []
        for run_dir in run_dirs:
            checkpoint_path = run_dir / "final_checkpoint.pt"
            matrix = representation_matrix(checkpoint_path, representation, len(color_vocab))
            coords = run_tsne(matrix, perplexity=args.perplexity, seed=args.seed)
            run_coords.append((run_dir.name, coords))
            write_coordinates(
                path=output_dir / representation / f"{run_dir.name}_coords.csv",
                color_vocab=color_vocab,
                coords=coords,
            )
            draw_single_plot(
                path=output_dir / representation / f"{run_dir.name}.png",
                coords=coords,
                colors=colors,
                title=f"{run_dir.name}\n{representation}",
            )
        draw_grid_plot(
            path=output_dir / f"{representation}_all_runs.png",
            run_coords=run_coords,
            colors=colors,
            representation=representation,
        )
        print(f"Wrote {representation} plots to {output_dir / representation}")
        print(f"Wrote grid plot to {output_dir / f'{representation}_all_runs.png'}")


if __name__ == "__main__":
    main()
