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
DEFAULT_ELEMENT_SYNTHESIS_FRACTION = 0.50
DEFAULT_ELEMENT_MAX_SCALE = 1
MAX_DOUBLE_DISPLACEMENT_CANDIDATES = 140_000
MAX_NO_REACTION_CANDIDATES = 80_000
DEFAULT_NO_REACTION_FRACTION = 0.25


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
    Ion("Be", 2, "alkaline_earth"),
    Ion("Cd", 2, "transition"),
    Ion("Hg", 2, "transition"),
    Ion("Cu", 1, "transition"),
    Ion("Co", 3, "transition"),
    Ion("Ni", 3, "transition"),
    Ion("Mn", 3, "transition"),
    Ion("Mn", 4, "transition"),
    Ion("Cr", 2, "transition"),
    Ion("Ti", 2, "transition"),
    Ion("Ti", 3, "transition"),
    Ion("Ti", 4, "transition"),
    Ion("V", 2, "transition"),
    Ion("V", 3, "transition"),
    Ion("V", 5, "transition"),
    Ion("Zr", 4, "transition"),
    Ion("Hf", 4, "transition"),
    Ion("Sc", 3, "transition"),
    Ion("Y", 3, "transition"),
    Ion("La", 3, "lanthanide"),
    Ion("Ce", 3, "lanthanide"),
    Ion("Ce", 4, "lanthanide"),
    Ion("Ga", 3, "post_transition"),
    Ion("In", 3, "post_transition"),
    Ion("Bi", 3, "post_transition"),
    Ion("Sb", 3, "metalloid"),
    Ion("Au", 1, "transition"),
    Ion("Au", 3, "transition"),
    Ion("Pt", 2, "transition"),
    Ion("Pt", 4, "transition"),
    Ion("Pd", 2, "transition"),
    Ion("Mo", 6, "transition"),
    Ion("W", 6, "transition"),
    Ion("Nb", 5, "transition"),
    Ion("Ta", 5, "transition"),
)

ANIONS: tuple[Ion, ...] = (
    Ion("H", -1, "hydride"),
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
    Ion("O", -2, "oxide"),
    Ion("N", -3, "nitride"),
    Ion("P", -3, "phosphide"),
    Ion("Se", -2, "selenide"),
    Ion("Te", -2, "telluride"),
    Ion("CN", -1, "cyanide"),
    Ion("SCN", -1, "thiocyanate"),
    Ion("OCN", -1, "cyanate"),
    Ion("MnO4", -1, "permanganate"),
    Ion("HCO3", -1, "bicarbonate"),
    Ion("HSO4", -1, "bisulfate"),
    Ion("HSO3", -1, "bisulfite"),
    Ion("H2PO4", -1, "dihydrogen_phosphate"),
    Ion("HPO4", -2, "hydrogen_phosphate"),
    Ion("BO3", -3, "borate"),
    Ion("SiO3", -2, "silicate"),
    Ion("BrO3", -1, "bromate"),
    Ion("IO3", -1, "iodate"),
    Ion("ClO", -1, "hypochlorite"),
    Ion("IO4", -1, "periodate"),
    Ion("SeO4", -2, "selenate"),
    Ion("S2O3", -2, "thiosulfate"),
    Ion("MoO4", -2, "molybdate"),
    Ion("WO4", -2, "tungstate"),
    Ion("AsO4", -3, "arsenate"),
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
    Ion("HCO3", -1, "acid"),
    Ion("HSO4", -1, "acid"),
    Ion("HSO3", -1, "acid"),
    Ion("H2PO4", -1, "acid"),
    Ion("HPO4", -2, "acid"),
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
    Ion("Zn", 2, "base"),
    Ion("Fe", 2, "base"),
    Ion("Cu", 2, "base"),
    Ion("Al", 3, "base"),
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


def element_synthesis_reactions() -> tuple[CandidateReaction, ...]:
    """Conservative binary element synthesis set for common inorganic products.

    These rows intentionally use elemental reactants only. The balancing step later
    rejects any malformed formula, and all accepted equations are atom-balanced.
    """

    candidates: list[CandidateReaction] = []
    metal_cations = tuple(cation for cation in CATIONS if cation.formula != "NH4")
    unique_metal_cations = list(dict.fromkeys(metal_cations))

    halogens = (
        ("F2", Ion("F", -1, "halide")),
        ("Cl2", Ion("Cl", -1, "halide")),
        ("Br2", Ion("Br", -1, "halide")),
        ("I2", Ion("I", -1, "halide")),
    )
    for cation in unique_metal_cations:
        for halogen_species, anion in halogens:
            product = neutral_compound(cation, anion)
            candidates.append(
                CandidateReaction(
                    cation.formula,
                    halogen_species,
                    (product,),
                    "element_synthesis",
                    "metal + elemental halogen -> binary metal halide",
                    note="binary element synthesis",
                )
            )

    oxide_anion = Ion("O", -2, "oxide")
    sulfide_anion = Ion("S", -2, "sulfide")
    for cation in unique_metal_cations:
        candidates.append(
            CandidateReaction(
                cation.formula,
                "O2",
                (neutral_compound(cation, oxide_anion),),
                "element_synthesis",
                "metal + oxygen -> binary metal oxide",
                note="binary element synthesis",
            )
        )
        candidates.append(
            CandidateReaction(
                cation.formula,
                "S",
                (neutral_compound(cation, sulfide_anion),),
                "element_synthesis",
                "metal + sulfur -> binary metal sulfide",
                note="binary element synthesis",
            )
        )

    for reactant_species, anion, rule in (
        ("N2", Ion("N", -3, "nitride"), "metal + nitrogen -> binary metal nitride"),
        ("P4", Ion("P", -3, "phosphide"), "metal + phosphorus -> binary metal phosphide"),
        ("Se", Ion("Se", -2, "selenide"), "metal + selenium -> binary metal selenide"),
        ("Te", Ion("Te", -2, "telluride"), "metal + tellurium -> binary metal telluride"),
    ):
        for cation in unique_metal_cations:
            candidates.append(
                CandidateReaction(
                    cation.formula,
                    reactant_species,
                    (neutral_compound(cation, anion),),
                    "element_synthesis",
                    rule,
                    note="binary element synthesis",
                )
            )

    hydride_anion = Ion("H", -1, "hydride")
    hydride_metals = tuple(
        cation
        for cation in unique_metal_cations
        if (cation.formula, cation.charge) in {
            ("Li", 1),
            ("Na", 1),
            ("K", 1),
            ("Rb", 1),
            ("Cs", 1),
            ("Mg", 2),
            ("Ca", 2),
            ("Sr", 2),
            ("Ba", 2),
        }
    )
    for cation in hydride_metals:
        candidates.append(
            CandidateReaction(
                cation.formula,
                "H2",
                (neutral_compound(cation, hydride_anion),),
                "element_synthesis",
                "metal + hydrogen -> binary metal hydride",
                note="binary element synthesis",
            )
        )

    nonmetal_binary_reactions = (
        CandidateReaction("H2", "F2", ("HF",), "element_synthesis", "hydrogen + halogen -> hydrogen halide"),
        CandidateReaction("H2", "Cl2", ("HCl",), "element_synthesis", "hydrogen + halogen -> hydrogen halide"),
        CandidateReaction("H2", "Br2", ("HBr",), "element_synthesis", "hydrogen + halogen -> hydrogen halide"),
        CandidateReaction("H2", "I2", ("HI",), "element_synthesis", "hydrogen + halogen -> hydrogen halide"),
        CandidateReaction("H2", "S", ("H2S",), "element_synthesis", "hydrogen + sulfur -> hydrogen sulfide"),
        CandidateReaction("H2", "Se", ("H2Se",), "element_synthesis", "hydrogen + selenium -> hydrogen selenide"),
        CandidateReaction("H2", "Te", ("H2Te",), "element_synthesis", "hydrogen + tellurium -> hydrogen telluride"),
        CandidateReaction("N2", "H2", ("NH3",), "element_synthesis", "nitrogen + hydrogen -> ammonia"),
        CandidateReaction("H2", "O2", ("H2O",), "element_synthesis", "hydrogen + oxygen -> water"),
        CandidateReaction("C", "O2", ("CO2",), "element_synthesis", "carbon + oxygen -> carbon dioxide"),
        CandidateReaction("C", "O2", ("CO",), "element_synthesis", "carbon + oxygen -> carbon monoxide"),
        CandidateReaction("S", "O2", ("SO2",), "element_synthesis", "sulfur + oxygen -> sulfur dioxide"),
        CandidateReaction("Se", "O2", ("SeO2",), "element_synthesis", "selenium + oxygen -> selenium dioxide"),
        CandidateReaction("Te", "O2", ("TeO2",), "element_synthesis", "tellurium + oxygen -> tellurium dioxide"),
        CandidateReaction("B", "O2", ("B2O3",), "element_synthesis", "boron + oxygen -> boron oxide"),
        CandidateReaction("Si", "O2", ("SiO2",), "element_synthesis", "silicon + oxygen -> silicon dioxide"),
        CandidateReaction("P4", "O2", ("P4O10",), "element_synthesis", "phosphorus + oxygen -> tetraphosphorus decoxide"),
        CandidateReaction("P4", "Cl2", ("PCl3",), "element_synthesis", "phosphorus + chlorine -> phosphorus trichloride"),
        CandidateReaction("P4", "Cl2", ("PCl5",), "element_synthesis", "phosphorus + chlorine -> phosphorus pentachloride"),
        CandidateReaction("P4", "F2", ("PF3",), "element_synthesis", "phosphorus + fluorine -> phosphorus trifluoride"),
        CandidateReaction("P4", "F2", ("PF5",), "element_synthesis", "phosphorus + fluorine -> phosphorus pentafluoride"),
        CandidateReaction("S", "F2", ("SF6",), "element_synthesis", "sulfur + fluorine -> sulfur hexafluoride"),
        CandidateReaction("S", "Cl2", ("SCl2",), "element_synthesis", "sulfur + chlorine -> sulfur dichloride"),
        CandidateReaction("Si", "Cl2", ("SiCl4",), "element_synthesis", "silicon + chlorine -> silicon tetrachloride"),
        CandidateReaction("Si", "F2", ("SiF4",), "element_synthesis", "silicon + fluorine -> silicon tetrafluoride"),
        CandidateReaction("B", "Cl2", ("BCl3",), "element_synthesis", "boron + chlorine -> boron trichloride"),
        CandidateReaction("B", "F2", ("BF3",), "element_synthesis", "boron + fluorine -> boron trifluoride"),
    )
    candidates.extend(nonmetal_binary_reactions)
    return tuple(dict.fromkeys(candidates))


COMBUSTION_REACTIONS: tuple[CandidateReaction, ...] = (
    CandidateReaction("CH4", "O2", ("CO2", "H2O"), "combustion", "hydrocarbon + oxygen -> carbon dioxide + water"),
    CandidateReaction("C2H6", "O2", ("CO2", "H2O"), "combustion", "hydrocarbon + oxygen -> carbon dioxide + water"),
    CandidateReaction("C2H4", "O2", ("CO2", "H2O"), "combustion", "hydrocarbon + oxygen -> carbon dioxide + water"),
    CandidateReaction("C2H2", "O2", ("CO2", "H2O"), "combustion", "hydrocarbon + oxygen -> carbon dioxide + water"),
    CandidateReaction("C3H8", "O2", ("CO2", "H2O"), "combustion", "hydrocarbon + oxygen -> carbon dioxide + water"),
    CandidateReaction("C4H10", "O2", ("CO2", "H2O"), "combustion", "hydrocarbon + oxygen -> carbon dioxide + water"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 100k balance-checked binary reaction dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-rows", type=int, default=NUM_ROWS)
    parser.add_argument("--max-scale", type=int, default=12)
    parser.add_argument("--element-max-scale", type=int, default=DEFAULT_ELEMENT_MAX_SCALE)
    parser.add_argument("--element-synthesis-fraction", type=float, default=DEFAULT_ELEMENT_SYNTHESIS_FRACTION)
    parser.add_argument("--no-reaction-fraction", type=float, default=DEFAULT_NO_REACTION_FRACTION)
    parser.add_argument("--split-strategy", choices=["generalization", "random"], default="generalization")
    parser.add_argument("--allow-stoichiometric-variations", action="store_true")
    parser.add_argument("--include-reversed-order", action="store_true", default=False)
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
    if anion.formula in {
        "NO3",
        "NO2",
        "C2H3O2",
        "ClO3",
        "ClO4",
        "MnO4",
        "HSO4",
        "H2PO4",
        "BrO3",
        "IO3",
        "IO4",
        "SCN",
        "OCN",
    }:
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
    if anion.formula in {"O", "N", "P", "Se", "Te", "H", "HPO4", "BO3", "SiO3", "SeO4", "S2O3", "MoO4", "WO4", "AsO4"}:
        return is_group1_or_ammonium(cation)
    if anion.formula in {"CN", "HCO3", "HSO3", "ClO"}:
        return is_group1_or_ammonium(cation)
    return False


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def common_cation_by_formula() -> dict[str, Ion]:
    preferred_charges = {
        "Li": 1,
        "Na": 1,
        "K": 1,
        "Rb": 1,
        "Cs": 1,
        "Mg": 2,
        "Ca": 2,
        "Sr": 2,
        "Ba": 2,
        "Cr": 3,
        "Mn": 2,
        "Co": 2,
        "Al": 3,
        "Zn": 2,
        "Fe": 2,
        "Ni": 2,
        "Sn": 2,
        "Pb": 2,
        "Cu": 2,
        "Hg": 2,
        "Cd": 2,
        "Ag": 1,
    }
    by_formula: dict[str, Ion] = {}
    for cation in CATIONS:
        if cation.formula in preferred_charges and cation.charge == preferred_charges[cation.formula]:
            by_formula[cation.formula] = cation
    return by_formula


def soluble_salt_pool() -> list[tuple[str, Ion, Ion]]:
    salts = []
    for cation in CATIONS:
        for anion in ANIONS:
            if is_soluble(cation, anion):
                salts.append((neutral_compound(cation, anion), cation, anion))
    return salts


def single_displacement_reactions() -> list[CandidateReaction]:
    """Metal activity-series single-displacement reactions.

    A more reactive free metal displaces a less reactive metal cation from a salt.
    The set is intentionally conservative and uses common aqueous lab examples.
    """

    activity = ["Mg", "Al", "Mn", "Zn", "Cr", "Fe", "Co", "Ni", "Sn", "Pb", "Cu", "Hg", "Ag"]
    rank = {metal: index for index, metal in enumerate(activity)}
    cations = common_cation_by_formula()
    salt_anions = (
        Ion("NO3", -1, "nitrate"),
        Ion("SO4", -2, "sulfate"),
        Ion("Cl", -1, "halide"),
        Ion("Br", -1, "halide"),
        Ion("I", -1, "halide"),
        Ion("C2H3O2", -1, "acetate"),
        Ion("ClO3", -1, "chlorate"),
        Ion("ClO4", -1, "perchlorate"),
    )
    candidates: list[CandidateReaction] = []
    for incoming in activity:
        incoming_cation = cations[incoming]
        for displaced in activity:
            if rank[incoming] >= rank[displaced]:
                continue
            displaced_cation = cations[displaced]
            for anion in salt_anions:
                salt = neutral_compound(displaced_cation, anion)
                product_salt = neutral_compound(incoming_cation, anion)
                candidates.append(
                    CandidateReaction(
                        incoming,
                        salt,
                        (product_salt, displaced),
                        "single_displacement_metal",
                        "more reactive metal + metal salt -> new salt + displaced metal",
                    )
                )
    return candidates


def halogen_displacement_reactions() -> list[CandidateReaction]:
    """Halogen activity-series reactions: F2 > Cl2 > Br2 > I2."""

    halogens = [
        ("F2", Ion("F", -1, "halide")),
        ("Cl2", Ion("Cl", -1, "halide")),
        ("Br2", Ion("Br", -1, "halide")),
        ("I2", Ion("I", -1, "halide")),
    ]
    halogen_salt_metals = {
        "Li",
        "Na",
        "K",
        "Rb",
        "Cs",
        "Mg",
        "Ca",
        "Sr",
        "Ba",
        "Zn",
        "Fe",
        "Cu",
        "Ni",
        "Co",
        "Mn",
        "Sn",
        "Pb",
        "Cd",
        "Hg",
        "Ag",
    }
    cations = tuple(cation for cation in CATIONS if cation.formula in halogen_salt_metals)
    candidates: list[CandidateReaction] = []
    for incoming_index, (incoming_species, incoming_anion) in enumerate(halogens):
        for displaced_species, displaced_anion in halogens[incoming_index + 1 :]:
            for cation in cations:
                salt = neutral_compound(cation, displaced_anion)
                product_salt = neutral_compound(cation, incoming_anion)
                candidates.append(
                    CandidateReaction(
                        incoming_species,
                        salt,
                        (product_salt, displaced_species),
                        "halogen_displacement",
                        "more reactive halogen + halide salt -> new halide salt + displaced halogen",
                    )
                )
    return candidates


def acid_metal_reactions() -> list[CandidateReaction]:
    """Active metal + non-oxidizing acid -> salt + hydrogen gas."""

    cations = common_cation_by_formula()
    active_metals = ("Mg", "Al", "Mn", "Zn", "Cr", "Fe", "Co", "Ni", "Sn")
    acid_anions = (
        Ion("Cl", -1, "acid"),
        Ion("Br", -1, "acid"),
        Ion("I", -1, "acid"),
        Ion("SO4", -2, "acid"),
        Ion("C2H3O2", -1, "acid"),
    )
    candidates: list[CandidateReaction] = []
    for metal in active_metals:
        if metal not in cations:
            continue
        for anion in acid_anions:
            candidates.append(
                CandidateReaction(
                    metal,
                    acid_formula(anion),
                    (neutral_compound(cations[metal], anion), "H2"),
                    "acid_metal",
                    "active metal + non-oxidizing acid -> salt + hydrogen",
                )
            )
    return candidates


def no_reaction_candidates(positive_pairs: set[tuple[str, str]]) -> list[CandidateReaction]:
    """Conservative no-net-reaction examples under the generator's chemistry rules."""

    candidates: list[CandidateReaction] = []

    # Soluble aqueous salt pairs where double displacement leaves all ions soluble.
    salts = soluble_salt_pool()
    count = 0
    seen_no_reaction_pairs: set[tuple[str, str]] = set()
    for salt_1, cation_1, anion_1 in salts:
        for salt_2, cation_2, anion_2 in salts:
            if count >= MAX_NO_REACTION_CANDIDATES:
                break
            if cation_1.formula == cation_2.formula or anion_1.formula == anion_2.formula:
                continue
            pair = canonical_pair(salt_1, salt_2)
            if pair in positive_pairs or pair in seen_no_reaction_pairs:
                continue
            seen_no_reaction_pairs.add(pair)
            product_1_soluble = is_soluble(cation_1, anion_2)
            product_2_soluble = is_soluble(cation_2, anion_1)
            if product_1_soluble and product_2_soluble:
                candidates.append(
                    CandidateReaction(
                        salt_1,
                        salt_2,
                        (),
                        "no_reaction_aqueous_spectator",
                        "soluble salt pair; no precipitate, gas, or weak electrolyte forms",
                    )
                )
                count += 1
        if count >= MAX_NO_REACTION_CANDIDATES:
            break

    # Element pairs with no ordinary binary reaction in this curated rule scope.
    element_pairs = (
        ("F2", "Cl2"),
        ("F2", "Br2"),
        ("F2", "I2"),
        ("Cl2", "Br2"),
        ("Cl2", "I2"),
        ("Br2", "I2"),
        ("N2", "O2"),
        ("N2", "Cl2"),
        ("N2", "Br2"),
        ("O2", "Cl2"),
        ("O2", "Br2"),
        ("H2", "N2"),
    )
    for left, right in element_pairs:
        if canonical_pair(left, right) not in positive_pairs:
            candidates.append(
                CandidateReaction(
                    left,
                    right,
                    (),
                    "no_reaction_element_pair",
                    "no ordinary binary reaction under the dataset's standard-condition rule scope",
                )
            )

    # Metals that are not reactive enough to displace the salt cation.
    activity = ["Mg", "Al", "Mn", "Zn", "Cr", "Fe", "Co", "Ni", "Sn", "Pb", "Cu", "Hg", "Ag"]
    rank = {metal: index for index, metal in enumerate(activity)}
    cations = common_cation_by_formula()
    anions = (Ion("NO3", -1, "nitrate"), Ion("SO4", -2, "sulfate"), Ion("Cl", -1, "halide"))
    for incoming in activity:
        for protected in activity:
            if rank[incoming] < rank[protected]:
                continue
            if incoming == protected:
                continue
            for anion in anions:
                salt = neutral_compound(cations[protected], anion)
                if canonical_pair(incoming, salt) in positive_pairs:
                    continue
                candidates.append(
                    CandidateReaction(
                        incoming,
                        salt,
                        (),
                        "no_reaction_single_displacement",
                        "metal is not reactive enough to displace the salt cation",
                    )
                )
    return candidates


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
    positive_candidates: list[CandidateReaction] = []
    positive_candidates.extend(element_synthesis_reactions())
    positive_candidates.extend(SYNTHESIS_REACTIONS)
    positive_candidates.extend(COMBUSTION_REACTIONS)
    positive_candidates.extend(single_displacement_reactions())
    positive_candidates.extend(halogen_displacement_reactions())
    positive_candidates.extend(acid_metal_reactions())

    for acid_anion in ACID_ANIONS:
        acid = acid_formula(acid_anion)
        for base_cation in BASE_CATIONS:
            base = neutral_compound(base_cation, Ion("OH", -1, "hydroxide"))
            salt = neutral_compound(base_cation, acid_anion)
            positive_candidates.append(
                CandidateReaction(
                    acid,
                    base,
                    (salt, "H2O"),
                    "acid_base_neutralization",
                    "acid + metal hydroxide -> salt + water",
                )
            )

    soluble_salts = soluble_salt_pool()

    seen_salt_pairs: set[tuple[str, str]] = set()
    double_displacement_count = 0
    for salt_1, cation_1, anion_1 in soluble_salts:
        for salt_2, cation_2, anion_2 in soluble_salts:
            if double_displacement_count >= MAX_DOUBLE_DISPLACEMENT_CANDIDATES:
                break
            if cation_1.formula == cation_2.formula or anion_1.formula == anion_2.formula:
                continue
            salt_pair_key = tuple(sorted((salt_1, salt_2)))
            if salt_pair_key in seen_salt_pairs:
                continue
            seen_salt_pairs.add(salt_pair_key)
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
            positive_candidates.append(
                CandidateReaction(
                    salt_1,
                    salt_2,
                    products,
                    "aqueous_double_displacement",
                    "soluble salts exchange ions; retained only when at least one product is insoluble by standard solubility rules",
                )
            )
            double_displacement_count += 1
        if double_displacement_count >= MAX_DOUBLE_DISPLACEMENT_CANDIDATES:
            break

    positive_candidates = list(dict.fromkeys(positive_candidates))
    positive_pairs = {canonical_pair(candidate.reactant_1, candidate.reactant_2) for candidate in positive_candidates}
    candidates = positive_candidates + no_reaction_candidates(positive_pairs)
    return list(dict.fromkeys(candidates))


def format_term(amount: int, species: str) -> str:
    return species if amount == 1 else f"{amount}{species}"


def format_equation(reactants: tuple[str, str], reactant_coeffs: tuple[int, int], products: tuple[str, ...], product_coeffs: tuple[int, ...]) -> str:
    left = " + ".join(format_term(coeff, species) for species, coeff in zip(reactants, reactant_coeffs))
    right = " + ".join(format_term(coeff, species) for species, coeff in zip(products, product_coeffs)) if products else f"{NULL_TOKEN} + {NULL_TOKEN}"
    return f"{left} -> {right}"


def build_pool(
    *,
    max_scale: int,
    element_max_scale: int,
    include_reversed_order: bool,
    allow_stoichiometric_variations: bool,
) -> list[dict[str, object]]:
    if max_scale < 1:
        raise ValueError("max_scale must be >= 1")
    if element_max_scale < 1:
        raise ValueError("element_max_scale must be >= 1")
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    for candidate in build_candidates():
        reactants = (candidate.reactant_1, candidate.reactant_2)
        products = candidate.products
        if not products:
            input_orders = [(reactants, (1, 1), "canonical")]
            if include_reversed_order and reactants[0] != reactants[1]:
                input_orders.append(((reactants[1], reactants[0]), (1, 1), "reversed"))
            for ordered_reactants, ordered_coeffs, order in input_orders:
                key = (ordered_reactants[0], 1, ordered_reactants[1], 1, NULL_TOKEN, NULL_TOKEN)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "reactant_1": ordered_reactants[0],
                        "amount_1": ordered_coeffs[0],
                        "reactant_2": ordered_reactants[1],
                        "amount_2": ordered_coeffs[1],
                        "output_1": NULL_TOKEN,
                        "output_1_amount": 0,
                        "output_2": NULL_TOKEN,
                        "output_2_amount": 0,
                        "equation": format_equation(ordered_reactants, ordered_coeffs, (), ()),
                        "family": candidate.family,
                        "source_rule": candidate.source_rule,
                        "order": order,
                        "scale": 1,
                        "note": candidate.note,
                        "split_group": f"{candidate.family}:{'|'.join(canonical_pair(*ordered_reactants))}",
                    }
                )
            continue

        coeffs = balance_coefficients(reactants, products)
        if coeffs is None:
            continue
        reactant_coeffs = tuple(coeffs[:2])
        product_coeffs = tuple(coeffs[2:])
        validate_balanced(reactants, reactant_coeffs, products, product_coeffs)

        input_orders = [(reactants, reactant_coeffs, "canonical")]
        if include_reversed_order and reactants[0] != reactants[1]:
            input_orders.append(((reactants[1], reactants[0]), (reactant_coeffs[1], reactant_coeffs[0]), "reversed"))

        if allow_stoichiometric_variations:
            scale_limit = element_max_scale if candidate.family == "element_synthesis" else max_scale
        else:
            scale_limit = 1
        for scale in range(1, scale_limit + 1):
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
                        "split_group": f"{candidate.family}:{'|'.join(canonical_pair(*ordered_reactants))}->{'|'.join(products)}",
                    }
                )
    return rows


def build_rows(
    *,
    num_rows: int,
    max_scale: int,
    element_max_scale: int,
    element_synthesis_fraction: float,
    no_reaction_fraction: float,
    split_strategy: str,
    include_reversed_order: bool,
    allow_stoichiometric_variations: bool,
    seed: int,
) -> list[dict[str, object]]:
    if not 0.0 <= element_synthesis_fraction <= 1.0:
        raise ValueError("element_synthesis_fraction must be in [0, 1]")
    if not 0.0 <= no_reaction_fraction <= 1.0:
        raise ValueError("no_reaction_fraction must be in [0, 1]")
    pool = build_pool(
        max_scale=max_scale,
        element_max_scale=element_max_scale,
        include_reversed_order=include_reversed_order,
        allow_stoichiometric_variations=allow_stoichiometric_variations,
    )
    if len(pool) < num_rows:
        raise ValueError(
            f"only generated {len(pool)} valid rows, fewer than requested {num_rows}; increase --max-scale"
        )
    rng = random.Random(seed)
    rows = sample_rows_by_family(
        pool,
        num_rows=num_rows,
        element_synthesis_fraction=element_synthesis_fraction,
        no_reaction_fraction=no_reaction_fraction,
        rng=rng,
    )
    if split_strategy == "generalization":
        assign_generalization_splits(rows, rng=random.Random(seed + 10_000))
    elif split_strategy == "random":
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row["split"] = split_for_index(index, len(rows))
    else:
        raise ValueError(f"unknown split strategy: {split_strategy}")
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["id"] = index
    return rows


def family_is_no_reaction(family: object) -> bool:
    return str(family).startswith("no_reaction")


def sample_rows_by_family(
    pool: list[dict[str, object]],
    *,
    num_rows: int,
    element_synthesis_fraction: float,
    no_reaction_fraction: float,
    rng: random.Random,
) -> list[dict[str, object]]:
    by_family: dict[str, list[dict[str, object]]] = {}
    for row in pool:
        by_family.setdefault(str(row["family"]), []).append(row)

    selected: list[dict[str, object]] = []
    used_keys: set[int] = set()

    def draw(families: list[str], count: int) -> int:
        available = [row for family in families for row in by_family.get(family, []) if id(row) not in used_keys]
        take = min(count, len(available))
        if take <= 0:
            return 0
        sampled = rng.sample(available, take)
        for row in sampled:
            used_keys.add(id(row))
        selected.extend(sampled)
        return take

    def draw_balanced(families: list[str], count: int) -> int:
        taken = 0
        remaining = count
        active = [family for family in families if any(id(row) not in used_keys for row in by_family.get(family, []))]
        while remaining > 0 and active:
            per_family = max(1, math.ceil(remaining / len(active)))
            progress = 0
            for family in list(active):
                if remaining <= 0:
                    break
                progress += draw([family], min(per_family, remaining))
                remaining = count - progress - taken
            taken += progress
            active = [family for family in families if any(id(row) not in used_keys for row in by_family.get(family, []))]
            if progress == 0:
                break
        return taken

    no_reaction_families = sorted(family for family in by_family if family_is_no_reaction(family))
    element_families = ["element_synthesis"] if "element_synthesis" in by_family else []

    no_reaction_target = round(num_rows * no_reaction_fraction)
    element_target = round(num_rows * element_synthesis_fraction)
    no_reaction_taken = draw_balanced(no_reaction_families, no_reaction_target)
    element_target = min(element_target, num_rows - len(selected))
    element_taken = draw(element_families, element_target)

    priority_positive_families = [
        "acid_base_neutralization",
        "single_displacement_metal",
        "halogen_displacement",
        "acid_metal",
        "combustion",
        "synthesis",
    ]
    priority_positive_taken = draw(
        [family for family in priority_positive_families if family in by_family],
        num_rows - len(selected),
    )

    remaining = num_rows - len(selected)
    positive_other_families = sorted(
        family
        for family in by_family
        if family not in element_families
        and family not in priority_positive_families
        and not family_is_no_reaction(family)
    )
    if not positive_other_families:
        raise ValueError("no non-element positive reaction families were generated")

    while remaining > 0:
        progress = 0
        active = [
            family
            for family in positive_other_families
            if any(id(row) not in used_keys for row in by_family.get(family, []))
        ]
        if not active:
            fallback = [row for row in pool if id(row) not in used_keys]
            if len(fallback) < remaining:
                raise ValueError(f"only {len(selected) + len(fallback)} unique rows available, fewer than requested {num_rows}")
            for row in rng.sample(fallback, remaining):
                used_keys.add(id(row))
                selected.append(row)
            break
        per_family = max(1, math.ceil(remaining / len(active)))
        for family in active:
            if remaining <= 0:
                break
            progress += draw([family], min(per_family, remaining))
            remaining = num_rows - len(selected)
        if progress == 0:
            raise ValueError(f"could not sample enough rows; selected {len(selected)} of {num_rows}")

    rng.shuffle(selected)
    print(f"Requested no-reaction rows: {no_reaction_target}; sampled {no_reaction_taken}")
    print(f"Requested element-synthesis rows: {element_target}; sampled {element_taken}")
    print(f"Prioritized non-double-displacement positive rows: {priority_positive_taken}")
    return selected


def assign_generalization_splits(rows: list[dict[str, object]], *, rng: random.Random) -> None:
    """Assign whole chemistry groups to splits so group keys do not cross splits."""

    targets = {
        "train": round(len(rows) * TRAIN_FRACTION),
        "val": round(len(rows) * VAL_FRACTION),
    }
    targets["test"] = len(rows) - targets["train"] - targets["val"]
    split_counts = {"train": 0, "val": 0, "test": 0}

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("split_group", row["equation"])), []).append(row)
    groups = list(grouped.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    for group in groups:
        def score(split: str) -> tuple[float, int]:
            target = max(targets[split], 1)
            return (split_counts[split] / target, split_counts[split])

        split = min(("train", "val", "test"), key=score)
        for row in group:
            row["split"] = split
        split_counts[split] += len(group)


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
        element_max_scale=args.element_max_scale,
        element_synthesis_fraction=args.element_synthesis_fraction,
        no_reaction_fraction=args.no_reaction_fraction,
        split_strategy=args.split_strategy,
        include_reversed_order=args.include_reversed_order,
        allow_stoichiometric_variations=args.allow_stoichiometric_variations,
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
    no_reaction_count = sum(1 for row in rows if str(row["family"]).startswith("no_reaction"))
    metadata = {
        "name": "reaction_combination_binary_two_output_100k",
        "scientific_scope": (
            "Atom-balanced binary reactions generated from curated element-element synthesis reactions, "
            "single-displacement reactions, halogen-displacement reactions, acid-metal reactions, combustion, "
            "acid-base neutralization templates, standard aqueous solubility-rule precipitation templates, "
            "and explicit no-net-reaction controls."
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
        "no_reaction_rows_with_NULL_NULL": no_reaction_count,
        "seed": args.seed,
        "max_scale": args.max_scale,
        "element_max_scale": args.element_max_scale,
        "element_synthesis_fraction": args.element_synthesis_fraction,
        "no_reaction_fraction": args.no_reaction_fraction,
        "element_synthesis_rows": family_counts.get("element_synthesis", 0),
        "split_strategy": args.split_strategy,
        "include_reversed_order": args.include_reversed_order,
        "allow_stoichiometric_variations": args.allow_stoichiometric_variations,
        "unique_reaction_policy": (
            "Default generation writes each canonical reaction once: no reversed-order duplicates and no scaled "
            "stoichiometric variants. Balanced base coefficients are still used as input amounts."
        ),
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
    print(f"Element-synthesis rows: {family_counts.get('element_synthesis', 0)}")
    print(f"No-reaction rows with NULL NULL: {no_reaction_count}")
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
