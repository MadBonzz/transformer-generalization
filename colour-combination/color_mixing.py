from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable


PALETTE_LEVELS = (0, 28, 57, 85, 113, 142, 170, 198, 227, 255)
RATIO_VALUES = tuple(range(0, 101, 10))
EQ_TOKEN = "="


@dataclass(frozen=True)
class ColorToken:
    token_id: int
    token: str
    hex_code: str
    r: int
    g: int
    b: int
    nearest_css_name: str
    nearest_css_hex: str
    oklab_l: float
    oklab_a: float
    oklab_b: float


@dataclass(frozen=True)
class MixExample:
    example_id: int
    split: str
    source_type: str
    color1_id: int
    ratio1_id: int
    color2_id: int
    ratio2_id: int
    eq_id: int
    target_id: int
    color1_hex: str
    ratio1_percent: int
    color2_hex: str
    ratio2_percent: int
    target_hex: str
    chain_left_color1_id: int | None = None
    chain_left_ratio1_percent: int | None = None
    chain_left_color2_id: int | None = None
    chain_left_ratio2_percent: int | None = None


def srgb_channel_to_linear(channel: float) -> float:
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def linear_channel_to_srgb(channel: float) -> int:
    channel = min(1.0, max(0.0, channel))
    if channel <= 0.0031308:
        value = 12.92 * channel
    else:
        value = 1.055 * (channel ** (1.0 / 2.4)) - 0.055
    return int(round(value * 255.0))


def rgb_to_linear(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(srgb_channel_to_linear(channel) for channel in rgb)  # type: ignore[return-value]


def linear_to_rgb(linear_rgb: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(linear_channel_to_srgb(channel) for channel in linear_rgb)  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    stripped = hex_code.strip().lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"expected 6-digit hex code, got {hex_code!r}")
    return int(stripped[0:2], 16), int(stripped[2:4], 16), int(stripped[4:6], 16)


def quantize_rgb_to_palette(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(PALETTE_LEVELS, key=lambda level: abs(level - channel)) for channel in rgb)  # type: ignore[return-value]


def color_id_from_rgb(rgb: tuple[int, int, int]) -> int:
    try:
        r_index = PALETTE_LEVELS.index(rgb[0])
        g_index = PALETTE_LEVELS.index(rgb[1])
        b_index = PALETTE_LEVELS.index(rgb[2])
    except ValueError as exc:
        raise ValueError(f"RGB value is not on the palette grid: {rgb}") from exc
    return (r_index * 100) + (g_index * 10) + b_index


def rgb_from_color_id(color_id: int) -> tuple[int, int, int]:
    if not 0 <= color_id < 1000:
        raise ValueError(f"color_id must be in [0, 999], got {color_id}")
    r_index = color_id // 100
    g_index = (color_id // 10) % 10
    b_index = color_id % 10
    return PALETTE_LEVELS[r_index], PALETTE_LEVELS[g_index], PALETTE_LEVELS[b_index]


def ratio_token_id(percent: int) -> int:
    if percent not in RATIO_VALUES:
        raise ValueError(f"ratio must be one of {RATIO_VALUES}, got {percent}")
    return 1000 + (percent // 10)


def mix_rgb_linear_light(
    rgb1: tuple[int, int, int],
    ratio1_percent: int,
    rgb2: tuple[int, int, int],
    ratio2_percent: int,
) -> tuple[int, int, int]:
    if ratio1_percent not in RATIO_VALUES or ratio2_percent not in RATIO_VALUES:
        raise ValueError("ratios must be percentages in increments of 10")
    total = ratio1_percent + ratio2_percent
    if total <= 0:
        raise ValueError("at least one ratio must be non-zero")

    weight1 = ratio1_percent / total
    weight2 = ratio2_percent / total
    linear1 = rgb_to_linear(rgb1)
    linear2 = rgb_to_linear(rgb2)
    mixed_linear = tuple((weight1 * left) + (weight2 * right) for left, right in zip(linear1, linear2))
    return quantize_rgb_to_palette(linear_to_rgb(mixed_linear))  # type: ignore[arg-type]


def mix_color_ids(color1_id: int, ratio1_percent: int, color2_id: int, ratio2_percent: int) -> int:
    target_rgb = mix_rgb_linear_light(
        rgb_from_color_id(color1_id),
        ratio1_percent,
        rgb_from_color_id(color2_id),
        ratio2_percent,
    )
    return color_id_from_rgb(target_rgb)


def rgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    red, green, blue = rgb_to_linear(rgb)
    l_value = (0.4122214708 * red) + (0.5363325363 * green) + (0.0514459929 * blue)
    m_value = (0.2119034982 * red) + (0.6806995451 * green) + (0.1073969566 * blue)
    s_value = (0.0883024619 * red) + (0.2817188376 * green) + (0.6299787005 * blue)

    l_root = math.copysign(abs(l_value) ** (1.0 / 3.0), l_value)
    m_root = math.copysign(abs(m_value) ** (1.0 / 3.0), m_value)
    s_root = math.copysign(abs(s_value) ** (1.0 / 3.0), s_value)

    return (
        (0.2104542553 * l_root) + (0.7936177850 * m_root) - (0.0040720468 * s_root),
        (1.9779984951 * l_root) - (2.4285922050 * m_root) + (0.4505937099 * s_root),
        (0.0259040371 * l_root) + (0.7827717662 * m_root) - (0.8086757660 * s_root),
    )


def _css4_colors() -> dict[str, str]:
    try:
        from matplotlib.colors import CSS4_COLORS
    except ImportError:
        return {
            "black": "#000000",
            "white": "#ffffff",
            "red": "#ff0000",
            "lime": "#00ff00",
            "blue": "#0000ff",
            "yellow": "#ffff00",
            "cyan": "#00ffff",
            "magenta": "#ff00ff",
            "gray": "#808080",
            "pink": "#ffc0cb",
            "orange": "#ffa500",
            "purple": "#800080",
            "brown": "#a52a2a",
            "green": "#008000",
        }
    return {name: value.lower() for name, value in CSS4_COLORS.items()}


def nearest_css_color_name(rgb: tuple[int, int, int]) -> tuple[str, str]:
    target = rgb_to_oklab(rgb)
    best_name = ""
    best_hex = ""
    best_distance = float("inf")
    for name, hex_code in _css4_colors().items():
        candidate = rgb_to_oklab(hex_to_rgb(hex_code))
        distance = sum((left - right) ** 2 for left, right in zip(target, candidate))
        if distance < best_distance:
            best_name = name
            best_hex = hex_code
            best_distance = distance
    return best_name, best_hex


def build_palette() -> list[ColorToken]:
    palette: list[ColorToken] = []
    for color_id in range(1000):
        rgb = rgb_from_color_id(color_id)
        hex_code = rgb_to_hex(rgb)
        css_name, css_hex = nearest_css_color_name(rgb)
        oklab_l, oklab_a, oklab_b = rgb_to_oklab(rgb)
        palette.append(
            ColorToken(
                token_id=color_id,
                token=f"COLOR_{color_id:03d}",
                hex_code=hex_code,
                r=rgb[0],
                g=rgb[1],
                b=rgb[2],
                nearest_css_name=css_name,
                nearest_css_hex=css_hex,
                oklab_l=oklab_l,
                oklab_a=oklab_a,
                oklab_b=oklab_b,
            )
        )
    return palette


def build_vocab_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for color in build_palette():
        rows.append(
            {
                "token_id": color.token_id,
                "token": color.token,
                "kind": "color",
                "value": color.hex_code,
            }
        )
    for ratio in RATIO_VALUES:
        rows.append(
            {
                "token_id": ratio_token_id(ratio),
                "token": f"RATIO_{ratio}",
                "kind": "ratio",
                "value": ratio,
            }
        )
    rows.append({"token_id": 1011, "token": EQ_TOKEN, "kind": "symbol", "value": EQ_TOKEN})
    return rows


def valid_ratio_pairs() -> list[tuple[int, int]]:
    return [(left, right) for left in RATIO_VALUES for right in RATIO_VALUES if left + right > 0]


def _split_for_index(index: int, total: int, train_fraction: float, val_fraction: float) -> str:
    train_cutoff = int(total * train_fraction)
    val_cutoff = train_cutoff + int(total * val_fraction)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def build_mix_examples(
    *,
    random_examples: int,
    chain_examples: int,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> list[MixExample]:
    if random_examples < 0 or chain_examples < 0:
        raise ValueError("example counts must be non-negative")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if val_fraction < 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    rng = random.Random(seed)
    ratio_pairs = valid_ratio_pairs()
    rows: list[dict[str, int | str | None]] = []
    seen_inputs: set[tuple[int, int, int, int]] = set()

    def append_row(
        *,
        source_type: str,
        color1_id: int,
        ratio1_percent: int,
        color2_id: int,
        ratio2_percent: int,
        chain_left_color1_id: int | None = None,
        chain_left_ratio1_percent: int | None = None,
        chain_left_color2_id: int | None = None,
        chain_left_ratio2_percent: int | None = None,
    ) -> bool:
        key = (color1_id, ratio1_percent, color2_id, ratio2_percent)
        if key in seen_inputs:
            return False
        seen_inputs.add(key)
        target_id = mix_color_ids(color1_id, ratio1_percent, color2_id, ratio2_percent)
        rows.append(
            {
                "source_type": source_type,
                "color1_id": color1_id,
                "ratio1_percent": ratio1_percent,
                "color2_id": color2_id,
                "ratio2_percent": ratio2_percent,
                "target_id": target_id,
                "chain_left_color1_id": chain_left_color1_id,
                "chain_left_ratio1_percent": chain_left_ratio1_percent,
                "chain_left_color2_id": chain_left_color2_id,
                "chain_left_ratio2_percent": chain_left_ratio2_percent,
            }
        )
        return True

    anchors = [0, 9, 90, 99, 900, 909, 990, 999, 555, 500, 50, 5]
    for color_id in range(1000):
        for anchor_id in anchors:
            ratio1, ratio2 = rng.choice(ratio_pairs)
            append_row(
                source_type="anchor",
                color1_id=color_id,
                ratio1_percent=ratio1,
                color2_id=anchor_id,
                ratio2_percent=ratio2,
            )
            if len(rows) >= random_examples:
                break
        if len(rows) >= random_examples:
            break

    while len(rows) < random_examples:
        color1_id = rng.randrange(1000)
        color2_id = rng.randrange(1000)
        ratio1, ratio2 = rng.choice(ratio_pairs)
        append_row(
            source_type="random",
            color1_id=color1_id,
            ratio1_percent=ratio1,
            color2_id=color2_id,
            ratio2_percent=ratio2,
        )

    chain_count = 0
    while chain_count < chain_examples:
        left1_id = rng.randrange(1000)
        left2_id = rng.randrange(1000)
        left_ratio1, left_ratio2 = rng.choice(
            [(left, right) for left, right in ratio_pairs if left + right <= 100]
        )
        carried_ratio = left_ratio1 + left_ratio2
        if carried_ratio == 0:
            continue
        carried_id = mix_color_ids(left1_id, left_ratio1, left2_id, left_ratio2)
        right_id = rng.randrange(1000)
        if append_row(
            source_type="chain",
            color1_id=carried_id,
            ratio1_percent=carried_ratio,
            color2_id=right_id,
            ratio2_percent=rng.choice(RATIO_VALUES[1:]),
            chain_left_color1_id=left1_id,
            chain_left_ratio1_percent=left_ratio1,
            chain_left_color2_id=left2_id,
            chain_left_ratio2_percent=left_ratio2,
        ):
            chain_count += 1

    rng.shuffle(rows)
    total = len(rows)
    examples: list[MixExample] = []
    for index, row in enumerate(rows):
        color1_id = int(row["color1_id"])
        color2_id = int(row["color2_id"])
        target_id = int(row["target_id"])
        ratio1_percent = int(row["ratio1_percent"])
        ratio2_percent = int(row["ratio2_percent"])
        examples.append(
            MixExample(
                example_id=index,
                split=_split_for_index(index, total, train_fraction, val_fraction),
                source_type=str(row["source_type"]),
                color1_id=color1_id,
                ratio1_id=ratio_token_id(ratio1_percent),
                color2_id=color2_id,
                ratio2_id=ratio_token_id(ratio2_percent),
                eq_id=1011,
                target_id=target_id,
                color1_hex=rgb_to_hex(rgb_from_color_id(color1_id)),
                ratio1_percent=ratio1_percent,
                color2_hex=rgb_to_hex(rgb_from_color_id(color2_id)),
                ratio2_percent=ratio2_percent,
                target_hex=rgb_to_hex(rgb_from_color_id(target_id)),
                chain_left_color1_id=(
                    None if row["chain_left_color1_id"] is None else int(row["chain_left_color1_id"])
                ),
                chain_left_ratio1_percent=(
                    None
                    if row["chain_left_ratio1_percent"] is None
                    else int(row["chain_left_ratio1_percent"])
                ),
                chain_left_color2_id=(
                    None if row["chain_left_color2_id"] is None else int(row["chain_left_color2_id"])
                ),
                chain_left_ratio2_percent=(
                    None
                    if row["chain_left_ratio2_percent"] is None
                    else int(row["chain_left_ratio2_percent"])
                ),
            )
        )
    return examples


def iter_example_token_rows(examples: Iterable[MixExample]) -> Iterable[dict[str, int | str | None]]:
    for example in examples:
        yield {
            "example_id": example.example_id,
            "split": example.split,
            "source_type": example.source_type,
            "input_0_color1_id": example.color1_id,
            "input_1_ratio1_id": example.ratio1_id,
            "input_2_color2_id": example.color2_id,
            "input_3_ratio2_id": example.ratio2_id,
            "input_4_eq_id": example.eq_id,
            "target_color_id": example.target_id,
            "color1_hex": example.color1_hex,
            "ratio1_percent": example.ratio1_percent,
            "color2_hex": example.color2_hex,
            "ratio2_percent": example.ratio2_percent,
            "target_hex": example.target_hex,
            "chain_left_color1_id": example.chain_left_color1_id,
            "chain_left_ratio1_percent": example.chain_left_ratio1_percent,
            "chain_left_color2_id": example.chain_left_color2_id,
            "chain_left_ratio2_percent": example.chain_left_ratio2_percent,
        }
