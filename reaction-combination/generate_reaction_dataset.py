from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


SEED = 42
NUM_ROWS = 100_000
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "reaction_combination_100k"
TRAIN_FRACTION = 0.50
VAL_FRACTION = 0.25
NULL_TOKEN = "NULL"


@dataclass(frozen=True)
class Ion:
    formula: str
    charge: int
    family: str


@dataclass(frozen=True)
class CandidateReaction:
    reactant_1: str
    reactant_2: str
    products: tuple[str, ...]
    family: str
    source_rule: str
    note: str = ""


CATIONS: tuple[Ion, ...] = (
    Ion("Li", 1, "group1"),
    Ion("Na", 1, "group1"),
    Ion("K", 1, "group1"),
    Ion("Rb", 1, "group1"),
    Ion("Cs", 1, "group1"),
    Ion("NH4", 1, "ammonium"),
    Ion("Ag", 1, "transition"),
    Ion("Mg", 2, "alkaline_earth"),
    Ion("Ca", 2, "alkaline_earth"),
    Ion("Sr", 2, "alkaline_earth"),
    Ion("Ba", 2, "alkaline_earth"),
    Ion("Zn", 2, "transition"),
    Ion("Fe", 2, "transition"),
    Ion("Fe", 3, "transition"),
    Ion("Cu", 2, "transition"),
    Ion("Ni", 2, "transition"),
    Ion("Co", 2, "transition"),
    Ion("Mn", 2, "transition"),
    Ion("Pb", 2, "post_transition"),
    Ion("Sn", 2, "post_transition"),
    Ion("Al", 3, "post_transition"),
    Ion("Cr", 3, "transition"),
)

ANIONS: tuple[Ion, ...] = (
    Ion("F", -1, "halide"),
    Ion("Cl", -1, "halide"),
    Ion("Br", -1, "halide"),
    Ion("I", -1, "halide"),
    Ion("NO3", -1, "nitrate"),
    Ion("NO2", -1, "nitrite"),
    Ion("C2H3O2", -1, "acetate"),
    Ion("ClO3", -1, "chlorate"),
    Ion("ClO4", -1, "perchlorate"),
    Ion("OH", -1, "hydroxide"),
    Ion("S", -2, "sulfide"),
    Ion("SO4", -2, "sulfate"),
    Ion("SO3", -2, "sulfite"),
    Ion("CO3", -2, "carbonate"),
    Ion("C2O4", -2, "oxalate"),
    Ion("CrO4", -2, "chromate"),
    Ion("PO4", -3, "phosphate"),
)

ACID_ANIONS: tuple[Ion, ...] = (
    Ion("F", -1, "acid"),
    Ion("Cl", -1, "acid"),
    Ion("Br", -1, "acid"),
    Ion("I", -1, "acid"),
    Ion("NO3", -1, "acid"),
    Ion("NO2", -1, "acid"),
    Ion("ClO3", -1, "acid"),
    Ion("ClO4", -1, "acid"),
    Ion("C2H3O2", -1, "acid"),
    Ion("S", -2, "acid"),
    Ion("SO4", -2, "acid"),
    Ion("SO3", -2, "acid"),
    Ion("CO3", -2, "acid"),
    Ion("C2O4", -2, "acid"),
    Ion("CrO4", -2, "acid"),
    Ion("PO4", -3, "acid"),
)

BASE_CATIONS: tuple[Ion, ...] = (
    Ion("Li", 1, "base"),
    Ion("Na", 1, "base"),
    Ion("K", 1, "base"),
    Ion("Rb", 1, "base"),
    Ion("Cs", 1, "base"),
    Ion("Mg", 2, "base"),
    Ion("Ca", 2, "base"),
    Ion("Sr", 2, "base"),
    Ion("Ba", 2, "base"),
)

SYNTHESIS_REACTIONS: tuple[CandidateReaction, ...] = (
    CandidateReaction("H2", "F2", ("HF",), "synthesis", "curated synthesis"),
    CandidateReaction("H2", "Cl2", ("HCl",), "synthesis", "curated synthesis"),
    CandidateReaction("H2", "Br2", ("HBr",), "synthesis", "curated synthesis"),
    CandidateReaction("H2", "I2", ("HI",), "synthesis", "curated synthesis"),
    CandidateReaction("N2", "H2", ("NH3",), "synthesis", "curated synthesis"),
    CandidateReaction("H2", "O2", ("H2O",), "synthesis", "curated synthesis"),
    CandidateReaction("C", "O2", ("CO2",), "synthesis", "curated synthesis"),
    CandidateReaction("C", "O2", ("CO",), "synthesis", "curated synthesis limited oxygen"),
    CandidateReaction("S", "O2", ("SO2",), "synthesis", "curated synthesis"),
    CandidateReaction("SO2", "O2", ("SO3",), "synthesis", "curated synthesis"),
    CandidateReaction("NO", "O2", ("NO2",), "synthesis", "curated synthesis"),
    CandidateReaction("CO", "O2", ("CO2",), "synthesis", "curated synthesis"),
    CandidateReaction("P4", "O2", ("P4O10",), "synthesis", "curated synthesis"),
    CandidateReaction("Mg", "O2", ("MgO",), "synthesis", "curated metal oxide"),
    CandidateReaction("Ca", "O2", ("CaO",), "synthesis", "curated metal oxide"),
    CandidateReaction("Zn", "O2", ("ZnO",), "synthesis", "curated metal oxide"),
    CandidateReaction("Cu", "O2", ("CuO",), "synthesis", "curated metal oxide"),
    CandidateReaction("Cu", "O2", ("Cu2O",), "synthesis", "curated metal oxide"),
    CandidateReaction("Fe", "O2", ("Fe2O3",), "synthesis", "curated metal oxide"),
    CandidateReaction("Fe", "O2", ("Fe3O4",), "synthesis", "curated metal oxide"),
    CandidateReaction("Al", "O2", ("Al2O3",), "synthesis", "curated metal oxide"),
    CandidateReaction("Li", "N2", ("Li3N",), "synthesis", "curated metal nitride"),
    CandidateReaction("Mg", "N2", ("Mg3N2",), "synthesis", "curated metal nitride"),
    CandidateReaction("Ca", "N2", ("Ca3N2",), "synthesis", "curated metal nitride"),
    CandidateReaction("Al", "N2", ("AlN",), "synthesis", "curated metal nitride"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 100k balance-checked binary reaction dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    parser.add_argument("--max-scale", type=int, default=12)
    parser.add_argument("--include-reversed-order", action="store_true", default=True)
    parser.add_argument("--no-reversed-order", dest="include_reversed_order", action="store_false")
    return parser.parse_args()


def parse_formula(formula: str) -> Counter[str]:
    index = 0

    def parse_number() -> int:
        nonlocal index
        start = index
        while index < len(formula) and formula[index].isdigit():
            index += 1
        return int(formula[start:index]) if index > start else 1

    def parse_group() -> Counter[str]:
        nonlocal index
        counts: Counter[str] = Counter()
        while index < len(formula):
            char = formula[index]
            if char == "(":
                index += 1
                inner = parse_group()
                multiplier = parse_number()
                for element, count in inner.items():
                    counts[element] += count * multiplier
            elif char == ")":
                index += 1
                return counts
            else:
                match = re.match(r"[A-Z][a-z]?", formula[index:])
                if match is None:
                    raise ValueError(f"invalid formula {formula!r} at index {index}")
                element = match.group(0)
                index += len(element)
                counts[element] += parse_number()
        return counts

    parsed = parse_group()
    if index != len(formula):
        raise ValueError(f"failed to parse full formula: {formula!r}")
    return parsed


def species_counts(formula: str) -> Counter[str]:
    if formula == NULL_TOKEN:
        return Counter()
    return parse_formula(formula)


def is_polyatomic(formula: str) -> bool:
    return len(re.findall(r"[A-Z][a-z]?", formula)) > 1


def format_group(formula: str, count: int) -> str:
    if count == 1:
        return formula
    if is_polyatomic(formula):
        return f"({formula}){count}"
    return f"{formula}{count}"


def neutral_compound(cation: Ion, anion: Ion) -> str:
    if cation.charge <= 0 or anion.charge >= 0:
        raise ValueError(f"expected cation and anion, got {cation}, {anion}")
    cation_charge = cation.charge
    anion_charge = abs(anion.charge)
    divisor = math.gcd(cation_charge, anion_charge)
    cation_count = anion_charge // divisor
    anion_count = cation_charge // divisor
    return f"{format_group(cation.formula, cation_count)}{format_group(anion.formula, anion_count)}"


def acid_formula(anion: Ion) -> str:
    proton_count = abs(anion.charge)
    hydrogen = "H" if proton_count == 1 else f"H{proton_count}"
    return f"{hydrogen}{anion.formula}"


def is_group1_or_ammonium(cation: Ion) -> bool:
    return cation.formula in {"Li", "Na", "K", "Rb", "Cs", "NH4"}


def is_soluble(cation: Ion, anion: Ion) -> bool:
    if is_group1_or_ammonium(cation):
        return True
    if anion.formula in {"NO3", "NO2", "C2H3O2", "ClO3", "ClO4"}:
        return True
    if anion.formula in {"Cl", "Br", "I"}:
        return cation.formula not in {"Ag", "Pb"}
    if anion.formula == "F":
        return is_group1_or_ammonium(cation)
    if anion.formula == "SO4":
        return cation.formula not in {"Ba", "Sr", "Pb", "Ca", "Ag"}
    if anion.formula == "OH":
        return cation.formula in {"Li", "Na", "K", "Rb", "Cs", "Ba", "Sr", "Ca"}
    if anion.formula == "S":
        return is_group1_or_ammonium(cation) or cation.formula in {"Ca", "Sr", "Ba"}
    if anion.formula in {"CO3", "SO3", "C2O4", "CrO4", "PO4"}:
        return is_group1_or_ammonium(cation)
    return False


def build_count_matrix(reactants: tuple[str, ...], products: tuple[str, ...]) -> tuple[list[str], list[list[Fraction]]]:
    elements = sorted(set().union(*(species_counts(species).keys() for species in (*reactants, *products))))
    species = (*reactants, *products)
    matrix: list[list[Fraction]] = []
    for element in elements:
        row = []
        for index, formula in enumerate(species):
            sign = 1 if index < len(reactants) else -1
            row.append(Fraction(sign * species_counts(formula).get(element, 0), 1))
        matrix.append(row)
    return elements, matrix


def rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    if not matrix:
        return matrix, []
    rows = [row[:] for row in matrix]
    row_count = len(rows)
    col_count = len(rows[0])
    pivot_cols: list[int] = []
    pivot_row = 0
    for col in range(col_count):
        selected = None
        for row in range(pivot_row, row_count):
            if rows[row][col] != 0:
                selected = row
                break
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        divisor = rows[pivot_row][col]
        rows[pivot_row] = [value / divisor for value in rows[pivot_row]]
        for row in range(row_count):
            if row == pivot_row:
                continue
            factor = rows[row][col]
            if factor != 0:
                rows[row] = [left - factor * right for left, right in zip(rows[row], rows[pivot_row])]
        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == row_count:
            break
    return rows, pivot_cols


def balance_coefficients(reactants: tuple[str, ...], products: tuple[str, ...]) -> tuple[int, ...] | None:
    _, matrix = build_count_matrix(reactants, products)
    reduced, pivot_cols = rref(matrix)
    col_count = len(reactants) + len(products)
    free_cols = [col for col in range(col_count) if col not in pivot_cols]
    if not free_cols:
        return None

    solution = [Fraction(0, 1) for _ in range(col_count)]
    free_col = free_cols[-1]
    solution[free_col] = Fraction(1, 1)
    for row_index, pivot_col in enumerate(pivot_cols):
        solution[pivot_col] = -reduced[row_index][free_col]

    non_zero = [value for value in solution if value != 0]
    if not non_zero:
        return None
    if all(value < 0 for value in non_zero):
        solution = [-value for value in solution]
    if any(value <= 0 for value in solution):
        return None

    lcm = 1
    for value in solution:
        lcm = math.lcm(lcm, value.denominator)
    integer_values = [int(value * lcm) for value in solution]
    divisor = 0
    for value in integer_values:
        divisor = math.gcd(divisor, abs(value))
    integer_values = [value // divisor for value in integer_values]
    return tuple(integer_values)


def validate_balanced(reactants: tuple[str, ...], reactant_coeffs: tuple[int, ...], products: tuple[str, ...], product_coeffs: tuple[int, ...]) -> None:
    left: Counter[str] = Counter()
    right: Counter[str] = Counter()
    for species, coeff in zip(reactants, reactant_coeffs):
        for element, count in species_counts(species).items():
            left[element] += count * coeff
    for species, coeff in zip(products, product_coeffs):
        for element, count in species_counts(species).items():
            right[element] += count * coeff
    if left != right:
        raise ValueError(f"unbalanced reaction: {reactants} {reactant_coeffs} -> {products} {product_coeffs}; {left} != {right}")


def build_candidates() -> list[CandidateReaction]:
    candidates: list[CandidateReaction] = []
    candidates.extend(SYNTHESIS_REACTIONS)

    for acid_anion in ACID_ANIONS:
        acid = acid_formula(acid_anion)
        for base_cation in BASE_CATIONS:
            base = neutral_compound(base_cation, Ion("OH", -1, "hydroxide"))
            salt = neutral_compound(base_cation, acid_anion)
            candidates.append(
                CandidateReaction(
                    acid,
                    base,
                    (salt, "H2O"),
                    "acid_base_neutralization",
                    "acid + metal hydroxide -> salt + water",
                )
            )

    soluble_salts = []
    for cation in CATIONS:
        for anion in ANIONS:
            if is_soluble(cation, anion):
                soluble_salts.append((neutral_compound(cation, anion), cation, anion))

    for salt_1, cation_1, anion_1 in soluble_salts:
        for salt_2, cation_2, anion_2 in soluble_salts:
            if cation_1.formula == cation_2.formula or anion_1.formula == anion_2.formula:
                continue
            product_1 = neutral_compound(cation_1, anion_2)
            product_2 = neutral_compound(cation_2, anion_1)
            product_1_soluble = is_soluble(cation_1, anion_2)
            product_2_soluble = is_soluble(cation_2, anion_1)
            if product_1_soluble and product_2_soluble:
                continue
            products = (product_1, product_2)
            if product_1_soluble and not product_2_soluble:
                products = (product_2, product_1)
            elif not product_1_soluble and not product_2_soluble and product_2 < product_1:
                products = (product_2, product_1)
            candidates.append(
                CandidateReaction(
                    salt_1,
                    salt_2,
                    products,
                    "aqueous_double_displacement",
                    "soluble salts exchange ions; retained only when at least one product is insoluble by standard solubility rules",
                )
            )

    return list(dict.fromkeys(candidates))


def format_term(amount: int, species: str) -> str:
    return species if amount == 1 else f"{amount}{species}"


def format_equation(reactants: tuple[str, str], reactant_coeffs: tuple[int, int], products: tuple[str, ...], product_coeffs: tuple[int, ...]) -> str:
    left = " + ".join(format_term(coeff, species) for species, coeff in zip(reactants, reactant_coeffs))
    right = " + ".join(format_term(coeff, species) for species, coeff in zip(products, product_coeffs))
    return f"{left} -> {right}"


def build_pool(*, max_scale: int, include_reversed_order: bool) -> list[dict[str, object]]:
    if max_scale < 1:
        raise ValueError("max_scale must be >= 1")
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for candidate in build_candidates():
        reactants = (candidate.reactant_1, candidate.reactant_2)
        products = candidate.products
        coeffs = balance_coefficients(reactants, products)
        if coeffs is None:
            continue
        reactant_coeffs = tuple(coeffs[:2])
        product_coeffs = tuple(coeffs[2:])
        validate_balanced(reactants, reactant_coeffs, products, product_coeffs)

        input_orders = [(reactants, reactant_coeffs, "canonical")]
        if include_reversed_order and reactants[0] != reactants[1]:
            input_orders.append(((reactants[1], reactants[0]), (reactant_coeffs[1], reactant_coeffs[0]), "reversed"))

        for scale in range(1, max_scale + 1):
            scaled_product_coeffs = tuple(coeff * scale for coeff in product_coeffs)
            output_1 = products[0]
            output_2 = products[1] if len(products) > 1 else NULL_TOKEN
            output_1_amount = scaled_product_coeffs[0]
            output_2_amount = scaled_product_coeffs[1] if len(products) > 1 else 0
            for ordered_reactants, ordered_coeffs, order in input_orders:
                scaled_reactant_coeffs = tuple(coeff * scale for coeff in ordered_coeffs)
                key = (
                    ordered_reactants[0],
                    scaled_reactant_coeffs[0],
                    ordered_reactants[1],
                    scaled_reactant_coeffs[1],
                    output_1,
                    output_2,
                )
                if key in seen:
                    continue
                seen.add(key)
                validate_balanced(
                    ordered_reactants,
                    scaled_reactant_coeffs,
                    tuple(product for product in products if product != NULL_TOKEN),
                    scaled_product_coeffs,
                )
                rows.append(
                    {
                        "reactant_1": ordered_reactants[0],
                        "amount_1": scaled_reactant_coeffs[0],
                        "reactant_2": ordered_reactants[1],
                        "amount_2": scaled_reactant_coeffs[1],
                        "output_1": output_1,
                        "output_1_amount": output_1_amount,
                        "output_2": output_2,
                        "output_2_amount": output_2_amount,
                        "equation": format_equation(ordered_reactants, scaled_reactant_coeffs, products, scaled_product_coeffs),
                        "family": candidate.family,
                        "source_rule": candidate.source_rule,
                        "order": order,
                        "scale": scale,
                        "note": candidate.note,
                    }
                )
    return rows


def build_rows(*, num_rows: int, max_scale: int, include_reversed_order: bool, seed: int) -> list[dict[str, object]]:
    pool = build_pool(max_scale=max_scale, include_reversed_order=include_reversed_order)
    if len(pool) < num_rows:
        raise ValueError(
            f"only generated {len(pool)} valid rows, fewer than requested {num_rows}; increase --max-scale"
        )
    rng = random.Random(seed)
    rows = rng.sample(pool, num_rows)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["id"] = index
        row["split"] = split_for_index(index, len(rows))
    return rows


def split_for_index(index: int, total: int) -> str:
    train_end = int(total * TRAIN_FRACTION)
    val_end = train_end + int(total * VAL_FRACTION)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def build_vocab(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int], dict[int, int]]:
    species = sorted(
        {
            *(str(row["reactant_1"]) for row in rows),
            *(str(row["reactant_2"]) for row in rows),
            *(str(row["output_1"]) for row in rows),
            *(str(row["output_2"]) for row in rows),
        }
    )
    if NULL_TOKEN not in species:
        species.append(NULL_TOKEN)
        species = sorted(species)
    amounts = sorted({int(row["amount_1"]) for row in rows} | {int(row["amount_2"]) for row in rows})
    species_to_id = {formula: index for index, formula in enumerate(species)}
    amount_token_start = len(species)
    amount_to_id = {amount: amount_token_start + index for index, amount in enumerate(amounts)}

    vocab_rows: list[dict[str, object]] = []
    for formula, token_id in species_to_id.items():
        kind = "null" if formula == NULL_TOKEN else "species"
        vocab_rows.append({"token_id": token_id, "token": f"SPECIES_{token_id:04d}", "kind": kind, "value": formula})
    for amount, token_id in amount_to_id.items():
        vocab_rows.append({"token_id": token_id, "token": f"AMOUNT_{amount}", "kind": "amount", "value": amount})
    return vocab_rows, species_to_id, amount_to_id


def build_tokenized_rows(
    rows: list[dict[str, object]],
    species_to_id: dict[str, int],
    amount_to_id: dict[int, int],
) -> list[dict[str, object]]:
    tokenized = []
    for row in rows:
        tokenized.append(
            {
                "id": row["id"],
                "split": row["split"],
                "input_0_reactant_1_id": species_to_id[str(row["reactant_1"])],
                "input_1_amount_1_id": amount_to_id[int(row["amount_1"])],
                "input_2_reactant_2_id": species_to_id[str(row["reactant_2"])],
                "input_3_amount_2_id": amount_to_id[int(row["amount_2"])],
                "target_0_output_1_id": species_to_id[str(row["output_1"])],
                "target_1_output_2_id": species_to_id[str(row["output_2"])],
                "reactant_1": row["reactant_1"],
                "amount_1": row["amount_1"],
                "reactant_2": row["reactant_2"],
                "amount_2": row["amount_2"],
                "output_1": row["output_1"],
                "output_1_amount": row["output_1_amount"],
                "output_2": row["output_2"],
                "output_2_amount": row["output_2_amount"],
                "equation": row["equation"],
                "family": row["family"],
            }
        )
    return tokenized


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(
        num_rows=args.num_rows,
        max_scale=args.max_scale,
        include_reversed_order=args.include_reversed_order,
        seed=args.seed,
    )
    vocab_rows, species_to_id, amount_to_id = build_vocab(rows)
    tokenized_rows = build_tokenized_rows(rows, species_to_id, amount_to_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "reaction_combination.csv", rows)
    write_csv(args.output_dir / "tokenized_examples.csv", tokenized_rows)
    write_csv(args.output_dir / "vocab.csv", vocab_rows)

    split_counts = Counter(str(row["split"]) for row in rows)
    family_counts = Counter(str(row["family"]) for row in rows)
    two_output_count = sum(1 for row in rows if row["output_2"] != NULL_TOKEN)
    metadata = {
        "name": "reaction_combination_binary_two_output_100k",
        "scientific_scope": (
            "Atom-balanced binary reactions generated from curated synthesis reactions, "
            "acid-base neutralization templates, and standard aqueous solubility-rule precipitation templates."
        ),
        "target_design": "two_product_species_tokens",
        "target_note": "Product stoichiometric coefficients are stored as output_1_amount/output_2_amount but are not model targets.",
        "null_token": NULL_TOKEN,
        "sequence_length": 4,
        "input_format": ["reactant_1_species_token", "amount_1_token", "reactant_2_species_token", "amount_2_token"],
        "target_format": ["output_1_species_token", "output_2_species_or_NULL_token"],
        "num_examples": len(rows),
        "num_species_tokens_including_null": len(species_to_id),
        "num_amount_tokens": len(amount_to_id),
        "vocab_size": len(vocab_rows),
        "target_vocab_size": len(species_to_id),
        "split_fractions": {"train": TRAIN_FRACTION, "val": VAL_FRACTION, "test": 1.0 - TRAIN_FRACTION - VAL_FRACTION},
        "split_counts": dict(split_counts),
        "family_counts": dict(family_counts),
        "two_output_rows": two_output_count,
        "single_output_rows_with_NULL": len(rows) - two_output_count,
        "seed": args.seed,
        "max_scale": args.max_scale,
        "include_reversed_order": args.include_reversed_order,
        "validation": "Every accepted row is atom-balanced by formula parsing before it is written.",
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print(f"Wrote {len(rows)} reaction examples to {args.output_dir / 'reaction_combination.csv'}")
    print(f"Species tokens including NULL: {len(species_to_id)}")
    print(f"Amount tokens: {len(amount_to_id)}")
    print(f"Vocab size: {len(vocab_rows)}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Family counts: {dict(family_counts)}")
    print(f"Two-output rows: {two_output_count}")
    print(f"Single-output rows with NULL: {len(rows) - two_output_count}")
    print("Example rows:")
    for row in rows[:10]:
        print(
            "  "
            f"{row['reactant_1']} {row['amount_1']} {row['reactant_2']} {row['amount_2']} "
            f"-> {row['output_1']} {row['output_2']} | {row['equation']}"
        )


if __name__ == "__main__":
    main()
