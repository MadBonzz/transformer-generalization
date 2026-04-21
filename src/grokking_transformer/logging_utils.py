from __future__ import annotations

import csv
import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, object]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def append_csv(path: str | Path, row: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_csv_stable(path: str | Path, fieldnames: list[str], row: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    normalized_row = {field: row.get(field, "") for field in fieldnames}
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(normalized_row)
