from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from color_mixing import (
    EQ_TOKEN,
    PALETTE_LEVELS,
    RATIO_VALUES,
    build_mix_examples,
    build_palette,
    build_vocab_rows,
    iter_example_token_rows,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "color_mixing_linear_srgb_1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic linear-light sRGB color-mixing dataset."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-examples", type=int, default=60000)
    parser.add_argument("--chain-examples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict] | list[object]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    if isinstance(first, dict):
        fieldnames = list(first.keys())
        dict_rows = rows
    else:
        fieldnames = list(asdict(first).keys())
        dict_rows = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    palette = build_palette()
    vocab = build_vocab_rows()
    examples = build_mix_examples(
        random_examples=args.random_examples,
        chain_examples=args.chain_examples,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    example_rows = list(iter_example_token_rows(examples))

    write_csv(output_dir / "palette.csv", palette)
    write_csv(output_dir / "vocab.csv", vocab)
    write_csv(output_dir / "examples.csv", example_rows)

    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for example in examples:
        split_counts[example.split] = split_counts.get(example.split, 0) + 1
        source_counts[example.source_type] = source_counts.get(example.source_type, 0) + 1

    metadata = {
        "name": "color_mixing_linear_srgb_1000",
        "mixing_rule": "linear-light sRGB weighted average, converted back to sRGB, then channel-wise quantized to the fixed 10x10x10 sRGB palette",
        "row_format": [
            "color1_token",
            "ratio1_token",
            "color2_token",
            "ratio2_token",
            EQ_TOKEN,
            "target_color_token",
        ],
        "sequence_length": 5,
        "target_vocab_size": 1000,
        "vocab_size": 1012,
        "color_token_ids": [0, 999],
        "ratio_token_ids": [1000, 1010],
        "eq_token_id": 1011,
        "palette_levels_per_channel": list(PALETTE_LEVELS),
        "ratio_values_percent": list(RATIO_VALUES),
        "seed": args.seed,
        "num_examples": len(examples),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "notes": [
            "Color names in palette.csv are nearest CSS4 names in OKLab space and are metadata for plotting, not training labels.",
            "Chain rows use a carried left color that was produced by mixing two source colors; the carried ratio is the sum of those source ratios.",
            "This dataset models additive light/display color mixing, not subtractive paint mixing.",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Wrote {len(examples)} examples to {output_dir / 'examples.csv'}")
    print(f"Wrote {len(palette)} color tokens and {len(vocab)} total tokens")
    print(f"Split counts: {split_counts}")
    print(f"Source counts: {source_counts}")


if __name__ == "__main__":
    main()
