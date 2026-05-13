from __future__ import annotations

import argparse
import colorsys
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
NUM_BASE_COLORS = 2000
NUM_ROWS = 100_000
MIN_PAIR_RGB_DISTANCE = 25.0
HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "mixbox_100k_2000base"
MIXING_MODEL = "mixbox_lerp_nearest_base_palette"
TRAIN_FRACTION = 0.50
VAL_FRACTION = 0.25

PIGMENT_ANCHORS: tuple[tuple[str, str], ...] = (
    ("Cadmium Yellow", "#FEEC00"),
    ("Hansa Yellow", "#FCD300"),
    ("Yellow Ochre", "#C99700"),
    ("Cadmium Orange", "#FF6900"),
    ("Vermilion", "#E34234"),
    ("Cadmium Red", "#FF2702"),
    ("Alizarin Crimson", "#8A0303"),
    ("Quinacridone Magenta", "#80022E"),
    ("Cobalt Violet", "#4E0042"),
    ("Dioxazine Purple", "#2E0854"),
    ("Ultramarine Blue", "#190059"),
    ("Cobalt Blue", "#002185"),
    ("Phthalo Blue", "#0D1B44"),
    ("Cerulean Blue", "#007BA7"),
    ("Turquoise", "#00A6A6"),
    ("Phthalo Green", "#003C32"),
    ("Permanent Green", "#076D16"),
    ("Sap Green", "#507D2A"),
    ("Viridian", "#40826D"),
    ("Raw Sienna", "#D68A00"),
    ("Burnt Sienna", "#8A360F"),
    ("Raw Umber", "#635147"),
    ("Burnt Umber", "#5C4033"),
    ("Titanium White", "#F8F8F2"),
    ("Zinc White", "#F4F4F4"),
    ("Payne's Gray", "#536878"),
    ("Ivory Black", "#1C1C1C"),
)

RATIO_POOL: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (1, 6),
    (6, 1),
    (1, 8),
    (8, 1),
)

RATIO_CATEGORIES: dict[str, tuple[tuple[int, int], ...]] = {
    "equal": ((1, 1),),
    "mild": ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)),
    "strong": ((1, 4), (4, 1), (1, 5), (5, 1), (1, 6), (6, 1)),
    "extreme": ((1, 8), (8, 1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 100k synthetic pigment-like two-colour mixing dataset using Mixbox."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-base-colors", type=int, default=NUM_BASE_COLORS)
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    return parser.parse_args()


def import_mixbox():
    try:
        import mixbox  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Mixbox is required. Install it with `pip install pymixbox`; "
            "the imported module name is `mixbox`."
        ) from exc
    if not hasattr(mixbox, "lerp"):
        raise ImportError("The installed `mixbox` module does not expose `lerp(rgb1, rgb2, t)`.")
    return mixbox


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    if not HEX_PATTERN.match(hex_code):
        raise ValueError(f"invalid hex colour: {hex_code!r}")
    stripped = hex_code[1:]
    return int(stripped[0:2], 16), int(stripped[2:4], 16), int(stripped[4:6], 16)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def clamp_channel(value: float) -> int:
    return int(round(min(255.0, max(0.0, value))))


def normalize_rgb(rgb: tuple[float, float, float] | tuple[int, int, int] | list[float] | list[int]) -> tuple[int, int, int]:
    if len(rgb) != 3:
        raise ValueError(f"expected RGB triplet, got {rgb!r}")
    return tuple(clamp_channel(float(channel)) for channel in rgb)  # type: ignore[return-value]


def perturb_rgb_hsv(
    rgb: tuple[int, int, int],
    *,
    rng: np.random.Generator,
    hue_sigma: float,
    sat_sigma: float,
    value_sigma: float,
) -> tuple[int, int, int]:
    red, green, blue = (channel / 255.0 for channel in rgb)
    hue, sat, value = colorsys.rgb_to_hsv(red, green, blue)
    hue = (hue + float(rng.normal(0.0, hue_sigma))) % 1.0
    sat = min(1.0, max(0.0, sat * float(rng.normal(1.0, sat_sigma))))
    value = min(1.0, max(0.0, value * float(rng.normal(1.0, value_sigma))))
    out = colorsys.hsv_to_rgb(hue, sat, value)
    return normalize_rgb(tuple(channel * 255.0 for channel in out))


def mix_rgb_linear(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int], amount2: float) -> tuple[int, int, int]:
    amount1 = 1.0 - amount2
    return normalize_rgb(
        (
            (amount1 * rgb1[0]) + (amount2 * rgb2[0]),
            (amount1 * rgb1[1]) + (amount2 * rgb2[1]),
            (amount1 * rgb1[2]) + (amount2 * rgb2[2]),
        )
    )


def build_candidate_palette(rng: np.random.Generator) -> list[str]:
    candidates: list[str] = []
    anchor_rgbs = [hex_to_rgb(hex_code) for _, hex_code in PIGMENT_ANCHORS]

    for rgb in anchor_rgbs:
        candidates.append(rgb_to_hex(rgb))

    for rgb in anchor_rgbs:
        for _ in range(120):
            candidates.append(
                rgb_to_hex(
                    perturb_rgb_hsv(
                        rgb,
                        rng=rng,
                        hue_sigma=0.018,
                        sat_sigma=0.16,
                        value_sigma=0.13,
                    )
                )
            )
        for amount in (0.10, 0.18, 0.26, 0.34, 0.45):
            candidates.append(rgb_to_hex(mix_rgb_linear(rgb, (248, 248, 242), amount)))
            candidates.append(rgb_to_hex(mix_rgb_linear(rgb, (28, 28, 28), amount)))

    earth_rgbs = [
        hex_to_rgb("#D68A00"),
        hex_to_rgb("#8A360F"),
        hex_to_rgb("#635147"),
        hex_to_rgb("#5C4033"),
        hex_to_rgb("#7B3F00"),
        hex_to_rgb("#A0522D"),
    ]
    for rgb in earth_rgbs:
        for _ in range(90):
            candidates.append(
                rgb_to_hex(
                    perturb_rgb_hsv(
                        rgb,
                        rng=rng,
                        hue_sigma=0.012,
                        sat_sigma=0.20,
                        value_sigma=0.18,
                    )
                )
            )

    for value in np.linspace(12, 246, 42):
        gray = int(round(value))
        warm = normalize_rgb((gray + 4, gray + 2, gray - 2))
        cool = normalize_rgb((gray - 2, gray + 1, gray + 5))
        candidates.extend([rgb_to_hex((gray, gray, gray)), rgb_to_hex(warm), rgb_to_hex(cool)])

    deduped = list(dict.fromkeys(hex_code.upper() for hex_code in candidates))
    return deduped


def select_diverse_palette(candidates: list[str], num_colors: int) -> list[str]:
    if num_colors < len(PIGMENT_ANCHORS):
        raise ValueError("num_colors must be at least the number of pigment anchors")

    anchor_hexes = [hex_code.upper() for _, hex_code in PIGMENT_ANCHORS]
    selected = list(dict.fromkeys(anchor_hexes))
    remaining = [hex_code for hex_code in candidates if hex_code not in set(selected)]

    remaining_rgb = np.array([hex_to_rgb(hex_code) for hex_code in remaining], dtype=np.float64)
    selected_rgb = np.array([hex_to_rgb(hex_code) for hex_code in selected], dtype=np.float64)
    min_dist_sq = np.min(np.sum((remaining_rgb[:, None, :] - selected_rgb[None, :, :]) ** 2, axis=2), axis=1)

    while len(selected) < num_colors:
        best_index = int(np.argmax(min_dist_sq))
        best_hex = remaining[best_index]
        selected.append(best_hex)

        new_rgb = np.array(hex_to_rgb(best_hex), dtype=np.float64)
        new_dist_sq = np.sum((remaining_rgb - new_rgb) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, new_dist_sq)
        min_dist_sq[best_index] = -1.0

    return selected


def rgb_distance(hex1: str, hex2: str) -> float:
    rgb1 = hex_to_rgb(hex1)
    rgb2 = hex_to_rgb(hex2)
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(rgb1, rgb2)))


def build_qualifying_pairs(base_hexes: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for i, hex1 in enumerate(base_hexes):
        for hex2 in base_hexes[i + 1 :]:
            if rgb_distance(hex1, hex2) >= MIN_PAIR_RGB_DISTANCE:
                pairs.append((hex1, hex2))
    return pairs


def nearest_palette_hex(
    rgb: tuple[int, int, int],
    *,
    palette_hexes: list[str],
    palette_rgbs: np.ndarray,
) -> tuple[str, float]:
    target = np.array(rgb, dtype=np.float64)
    distances_sq = np.sum((palette_rgbs - target) ** 2, axis=1)
    best_index = int(np.argmin(distances_sq))
    return palette_hexes[best_index], float(math.sqrt(float(distances_sq[best_index])))


def ratio_category(ratio: tuple[int, int]) -> str:
    for category, ratios in RATIO_CATEGORIES.items():
        if ratio in ratios:
            return category
    raise ValueError(f"unknown ratio: {ratio}")


def weighted_ratio_choice(rng: random.Random) -> tuple[int, int]:
    category = rng.choices(
        population=("mild", "strong", "extreme"),
        weights=(0.65, 0.27, 0.08),
        k=1,
    )[0]
    return rng.choice(RATIO_CATEGORIES[category])


def choose_ratios_for_pair(rng: random.Random) -> list[tuple[int, int]]:
    target_count = rng.randint(2, 4)
    ratios: list[tuple[int, int]] = []
    if rng.random() < 0.75:
        ratios.append((1, 1))

    attempts = 0
    while len(ratios) < target_count and attempts < 100:
        attempts += 1
        ratio = weighted_ratio_choice(rng)
        reverse = (ratio[1], ratio[0])
        if ratio in ratios:
            continue
        if reverse in ratios and rng.random() > 0.25:
            continue
        ratios.append(ratio)

    while len(ratios) < target_count:
        ratio = rng.choice(RATIO_POOL)
        if ratio not in ratios:
            ratios.append(ratio)

    return ratios


def generate_rows(
    *,
    base_hexes: list[str],
    num_rows: int,
    rng: random.Random,
    mixbox_module,
) -> tuple[pd.DataFrame, pd.Series, int]:
    pairs = build_qualifying_pairs(base_hexes)
    if not pairs:
        raise ValueError("no qualifying colour pairs found")
    rng.shuffle(pairs)
    palette_rgbs = np.array([hex_to_rgb(hex_code) for hex_code in base_hexes], dtype=np.float64)

    internal_rows: list[dict[str, object]] = []
    pair_index = 0
    while len(internal_rows) < num_rows:
        if pair_index >= len(pairs):
            raise RuntimeError(
                f"ran out of unique qualifying pairs after {len(internal_rows)} rows; "
                "increase the palette size or allow repeated pairs"
            )
        hex1, hex2 = pairs[pair_index]
        pair_index += 1

        for ratio1_parts, ratio2_parts in choose_ratios_for_pair(rng):
            total_parts = ratio1_parts + ratio2_parts
            t = ratio2_parts / total_parts
            mixbox_rgb = normalize_rgb(mixbox_module.lerp(hex_to_rgb(hex1), hex_to_rgb(hex2), t))
            output_hex, quantization_distance = nearest_palette_hex(
                mixbox_rgb,
                palette_hexes=base_hexes,
                palette_rgbs=palette_rgbs,
            )
            internal_rows.append(
                {
                    "hex_1": hex1,
                    "hex_2": hex2,
                    "ratio_1_parts": ratio1_parts,
                    "ratio_2_parts": ratio2_parts,
                    "t": round(t, 6),
                    "output_hex": output_hex,
                    "_ratio": f"{ratio1_parts}:{ratio2_parts}",
                    "_ratio_category": ratio_category((ratio1_parts, ratio2_parts)),
                    "_mixbox_hex": rgb_to_hex(mixbox_rgb),
                    "_output_r": hex_to_rgb(output_hex)[0],
                    "_output_g": hex_to_rgb(output_hex)[1],
                    "_output_b": hex_to_rgb(output_hex)[2],
                    "_quantization_distance": quantization_distance,
                }
            )
            if len(internal_rows) >= num_rows:
                break

    rng.shuffle(internal_rows)
    internal_rows = internal_rows[:num_rows]
    internal_df = pd.DataFrame(internal_rows)
    compact_df = internal_df[["hex_1", "hex_2", "ratio_1_parts", "ratio_2_parts", "t", "output_hex"]].copy()
    ratio_distribution = internal_df["_ratio_category"].value_counts(normalize=True).sort_index() * 100.0
    unique_pairs_used = int(internal_df[["hex_1", "hex_2"]].drop_duplicates().shape[0])

    validate_dataset(compact_df, internal_df, base_hexes)
    return compact_df, ratio_distribution, unique_pairs_used


def validate_dataset(compact_df: pd.DataFrame, internal_df: pd.DataFrame, base_hexes: list[str]) -> None:
    expected_rows = len(compact_df)
    if len(set(base_hexes)) != len(base_hexes):
        raise AssertionError("base palette contains duplicates")
    if expected_rows != NUM_ROWS:
        raise AssertionError(f"expected {NUM_ROWS} rows, got {len(compact_df)}")
    if (compact_df["hex_1"] == compact_df["hex_2"]).any():
        raise AssertionError("found row with hex_1 == hex_2")
    if not compact_df["t"].between(0.0, 1.0, inclusive="neither").all():
        raise AssertionError("found t outside the open interval (0, 1)")
    if not compact_df["output_hex"].map(lambda value: bool(HEX_PATTERN.match(str(value)))).all():
        raise AssertionError("found invalid output_hex value")
    base_set = set(base_hexes)
    if not compact_df["hex_1"].map(lambda value: value in base_set).all():
        raise AssertionError("found hex_1 outside base palette")
    if not compact_df["hex_2"].map(lambda value: value in base_set).all():
        raise AssertionError("found hex_2 outside base palette")
    if not compact_df["output_hex"].map(lambda value: value in base_set).all():
        raise AssertionError("found output_hex outside base palette")
    for column in ("_output_r", "_output_g", "_output_b"):
        if not internal_df[column].between(0, 255, inclusive="both").all():
            raise AssertionError(f"found {column} outside [0, 255]")
        if not np.issubdtype(internal_df[column].dtype, np.integer):
            raise AssertionError(f"{column} is not integer typed")


def write_outputs(output_dir: Path, compact_df: pd.DataFrame, base_hexes: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    palette_df = pd.DataFrame({"hex": base_hexes})
    compact_df.to_csv(output_dir / "colour_mixing_100k.csv", index=False)
    palette_df.to_csv(output_dir / "base_palette.csv", index=False)
    palette_df.to_csv(output_dir / f"base_palette_{len(base_hexes)}.csv", index=False)


def add_deterministic_splits(compact_df: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    split_df = compact_df.copy()
    permutation = rng.permutation(len(split_df))
    splits = np.empty(len(split_df), dtype=object)
    train_end = int(len(split_df) * TRAIN_FRACTION)
    val_end = train_end + int(len(split_df) * VAL_FRACTION)
    splits[permutation[:train_end]] = "train"
    splits[permutation[train_end:val_end]] = "val"
    splits[permutation[val_end:]] = "test"
    split_df.insert(0, "id", np.arange(len(split_df), dtype=np.int64))
    split_df["split"] = splits
    return split_df


def write_training_files(output_dir: Path, compact_df: pd.DataFrame, base_hexes: list[str], *, seed: int) -> None:
    split_df = add_deterministic_splits(compact_df, seed=seed)
    channel_to_id = {value: value for value in range(256)}
    amount_values = sorted(
        set(int(value) for value in split_df["ratio_1_parts"].unique())
        | set(int(value) for value in split_df["ratio_2_parts"].unique())
    )
    amount_token_start = 256
    amount_to_id = {amount: amount_token_start + index for index, amount in enumerate(amount_values)}
    plus_token_id = amount_token_start + len(amount_values)
    equals_token_id = plus_token_id + 1

    tokenized_rows = []
    for row in split_df.itertuples(index=False):
        rgb_1 = hex_to_rgb(row.hex_1)
        rgb_2 = hex_to_rgb(row.hex_2)
        output_rgb = hex_to_rgb(row.output_hex)
        amount_1_id = amount_to_id[int(row.ratio_1_parts)]
        amount_2_id = amount_to_id[int(row.ratio_2_parts)]
        input_ids = (
            amount_1_id,
            *rgb_1,
            plus_token_id,
            amount_2_id,
            *rgb_2,
            equals_token_id,
        )
        tokenized_rows.append(
            {
                "id": int(row.id),
                "split": row.split,
                "input_ids": " ".join(str(token) for token in input_ids),
                "target_ids": " ".join(str(token) for token in output_rgb),
                "input_0_amount_1_id": amount_1_id,
                "input_1_rgb_1_r_id": rgb_1[0],
                "input_2_rgb_1_g_id": rgb_1[1],
                "input_3_rgb_1_b_id": rgb_1[2],
                "input_4_plus_id": plus_token_id,
                "input_5_amount_2_id": amount_2_id,
                "input_6_rgb_2_r_id": rgb_2[0],
                "input_7_rgb_2_g_id": rgb_2[1],
                "input_8_rgb_2_b_id": rgb_2[2],
                "input_9_equals_id": equals_token_id,
                "target_0_output_r_id": output_rgb[0],
                "target_1_output_g_id": output_rgb[1],
                "target_2_output_b_id": output_rgb[2],
                "hex_1": row.hex_1,
                "hex_2": row.hex_2,
                "ratio_1_parts": int(row.ratio_1_parts),
                "ratio_2_parts": int(row.ratio_2_parts),
                "t": float(row.t),
                "output_hex": row.output_hex,
            }
        )

    vocab_rows = []
    for value, token_id in channel_to_id.items():
        vocab_rows.append(
            {
                "token_id": token_id,
                "token": f"VALUE_{value:03d}",
                "kind": "value",
                "value": value,
            }
        )
    for amount, token_id in amount_to_id.items():
        vocab_rows.append(
            {
                "token_id": token_id,
                "token": f"AMOUNT_{amount}",
                "kind": "amount",
                "value": amount,
            }
        )
    vocab_rows.extend(
        [
            {"token_id": plus_token_id, "token": "PLUS", "kind": "special", "value": "+"},
            {"token_id": equals_token_id, "token": "EQUALS", "kind": "special", "value": "="},
        ]
    )

    split_counts = split_df["split"].value_counts().to_dict()
    metadata = {
        "name": f"colour_mixing_mixbox_100k_{len(base_hexes)}base",
        "dataset_kind": "mixbox_pigment_like",
        "mixing_model": MIXING_MODEL,
        "mixing_rule": "Mixbox pigment-like interpolation followed by nearest-neighbour quantization to the fixed base palette",
        "raw_mixing_rule": "raw_rgb = mixbox.lerp(rgb_1, rgb_2, t)",
        "target_rule": "output_hex = nearest base-palette hex to raw_rgb by Euclidean RGB distance",
        "sequence_length": 10,
        "target_sequence_length": 3,
        "input_format": ["amount_1_token", "rgb_1_r", "rgb_1_g", "rgb_1_b", "PLUS", "amount_2_token", "rgb_2_r", "rgb_2_g", "rgb_2_b", "EQUALS"],
        "target_format": ["output_r", "output_g", "output_b"],
        "num_rows": len(split_df),
        "num_base_colours": len(base_hexes),
        "num_hex_tokens": 0,
        "num_t_tokens": 0,
        "num_value_tokens": 256,
        "num_amount_tokens": len(amount_values),
        "amount_values": amount_values,
        "plus_token_id": plus_token_id,
        "equals_token_id": equals_token_id,
        "vocab_size": len(vocab_rows),
        "target_vocab_size": 256,
        "value_token_ids": [0, 255],
        "split_fractions": {"train": TRAIN_FRACTION, "val": VAL_FRACTION, "test": 1.0 - TRAIN_FRACTION - VAL_FRACTION},
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "seed": seed,
        "notes": [
            "RGB channel/output tokens are integer values in [0, 255].",
            "Amount tokens are separate from RGB value tokens to disambiguate coefficients from colour channels.",
            "Input is 10 tokens: amount_1, RGB colour 1, PLUS, amount_2, RGB colour 2, EQUALS.",
            "Target is 3 tokens: output RGB.",
            "The compact CSV stores hex_1, hex_2, ratio_1_parts, ratio_2_parts, t, output_hex.",
        ],
    }

    pd.DataFrame(tokenized_rows).to_csv(output_dir / "tokenized_examples.csv", index=False)
    pd.DataFrame(vocab_rows).to_csv(output_dir / "vocab.csv", index=False)
    split_df.to_csv(output_dir / "colour_mixing_100k_with_splits.csv", index=False)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        import json

        json.dump(metadata, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        print(f"Using non-default seed {args.seed}; requested reproducibility seed is {SEED}.")
    if args.num_rows != NUM_ROWS:
        raise ValueError(f"this script is configured to generate exactly {NUM_ROWS} rows")

    np_rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    mixbox_module = import_mixbox()

    candidates = build_candidate_palette(np_rng)
    base_hexes = select_diverse_palette(candidates, args.num_base_colors)
    compact_df, ratio_distribution, unique_pairs_used = generate_rows(
        base_hexes=base_hexes,
        num_rows=args.num_rows,
        rng=py_rng,
        mixbox_module=mixbox_module,
    )
    write_outputs(args.output_dir, compact_df, base_hexes)
    write_training_files(args.output_dir, compact_df, base_hexes, seed=args.seed)

    print(f"Base colours: {len(base_hexes)}")
    print(f"Mixture rows: {len(compact_df)}")
    print("Ratio distribution by category (%):")
    for category in ("equal", "mild", "strong", "extreme"):
        print(f"  {category}: {ratio_distribution.get(category, 0.0):.2f}")
    print(f"Unique input pairs used: {unique_pairs_used}")
    print(f"Unique output_hex values: {compact_df['output_hex'].nunique()}")
    amount_values = sorted(
        set(int(value) for value in compact_df["ratio_1_parts"].unique())
        | set(int(value) for value in compact_df["ratio_2_parts"].unique())
    )
    print(f"Model vocabulary size: {256 + len(amount_values) + 2} tokens (256 RGB values + {len(amount_values)} amount tokens + PLUS/EQUALS)")
    print("Model input sequence length: 10")
    print("Model target sequence length: 3")
    print(f"Amount token values: {amount_values}")
    print(f"Train/val/test rows: {int(NUM_ROWS * TRAIN_FRACTION)}/{int(NUM_ROWS * VAL_FRACTION)}/{NUM_ROWS - int(NUM_ROWS * TRAIN_FRACTION) - int(NUM_ROWS * VAL_FRACTION)}")
    print("Example rows:")
    print(compact_df.head(10).to_string(index=False))
    print(f"Wrote {args.output_dir / 'colour_mixing_100k.csv'}")
    print(f"Wrote {args.output_dir / 'base_palette.csv'}")


if __name__ == "__main__":
    main()
