from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three grokking studies sequentially.")
    parser.add_argument("--profile", choices=("pilot", "full10"), default="pilot")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--transformer-layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--studies", type=str, default="", help="Comma-separated subset of studies: study1,study2,study3")
    parser.add_argument("--seeds", type=str, default="", help="Optional comma-separated seed list passed through to each study.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max-steps override applied to all selected studies.")
    parser.add_argument("--study1-max-steps", type=int, default=10_000_000)
    parser.add_argument("--study2-max-steps", type=int, default=10_000_000)
    parser.add_argument("--study3-max-steps", type=int, default=10_000_000)
    parser.add_argument("--study1-modulus", type=int, default=None)
    parser.add_argument("--study2-modulus", type=int, default=None)
    parser.add_argument("--study3-modulus", type=int, default=None, help="Input numeric range size for Study 3.")
    parser.add_argument("--study3-output-modulus", type=int, default=None, help="Optional output modulus override for Study 3.")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-vram-mb", type=float, default=500.0)
    parser.add_argument("--safety-margin-mb", type=float, default=500.0)
    parser.add_argument("--per-process-overhead-mb", type=float, default=256.0)
    parser.add_argument("--min-free-system-ram-mb", type=float, default=0.0)
    parser.add_argument("--system-ram-safety-margin-mb", type=float, default=0.0)
    parser.add_argument("--per-process-ram-mb", type=float, default=0.0)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--launch-settle-sec", type=float, default=1.0)
    return parser.parse_args()


def _parse_selected_studies(raw: str) -> list[str]:
    if not raw:
        return ["study1", "study2", "study3"]

    selected = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"study1", "study2", "study3"}
    invalid = [item for item in selected if item not in valid]
    if invalid:
        raise ValueError(f"unsupported study selection: {', '.join(invalid)}")
    if not selected:
        raise ValueError("at least one study must be selected")
    return selected


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    selected_studies = _parse_selected_studies(args.studies)
    common_max_steps = args.max_steps
    studies: list[dict[str, object]] = []
    if "study1" in selected_studies:
        studies.append(
            {
                "script": root / "run_loss_vs_rl.py",
                "output_root": Path(args.output_root) / "study1_loss_vs_rl",
                "max_steps": common_max_steps if common_max_steps is not None else args.study1_max_steps,
                "extra_args": (["--modulus", str(args.study1_modulus)] if args.study1_modulus is not None else []),
            }
        )
    if "study2" in selected_studies:
        studies.append(
            {
                "script": root / "run_fake_labels.py",
                "output_root": Path(args.output_root) / "study2_fake_labels",
                "max_steps": common_max_steps if common_max_steps is not None else args.study2_max_steps,
                "extra_args": (["--modulus", str(args.study2_modulus)] if args.study2_modulus is not None else []),
            }
        )
    if "study3" in selected_studies:
        study3_extra_args: list[str] = []
        if args.study3_modulus is not None:
            study3_extra_args.extend(["--modulus", str(args.study3_modulus)])
        if args.study3_output_modulus is not None:
            study3_extra_args.extend(["--output-modulus", str(args.study3_output_modulus)])
        studies.append(
            {
                "script": root / "run_range_transfer.py",
                "output_root": Path(args.output_root) / "study3_range_transfer",
                "max_steps": common_max_steps if common_max_steps is not None else args.study3_max_steps,
                "extra_args": study3_extra_args,
            }
        )

    study_progress = tqdm(total=len(studies), desc="studies", unit="study", dynamic_ncols=True)
    for study_index, study in enumerate(studies, start=1):
        study_name = study["output_root"].name
        study_extra_args = list(study["extra_args"])
        tqdm.write(
            f"[STUDY {study_index}/{len(studies)} START] {study_name} | "
            f"max_steps={study['max_steps']} | output={study['output_root']}"
        )
        if args.parallel_workers == 1:
            command = [
                sys.executable,
                str(study["script"]),
                "--profile",
                args.profile,
                "--output-root",
                str(study["output_root"]),
                "--device",
                args.device,
                "--transformer-layers",
                str(args.transformer_layers),
                "--max-steps",
                str(study["max_steps"]),
            ]
            if args.seeds:
                command.extend(["--seeds", args.seeds])
            command.extend(study_extra_args)
            subprocess.run(command, check=True)
            study_progress.update(1)
            tqdm.write(f"[STUDY {study_index}/{len(studies)} DONE] {study_name}")
            continue

        manifest_path = Path(study["output_root"]) / "manifest.jsonl"
        manifest_command = [
            sys.executable,
            str(study["script"]),
            "--profile",
            args.profile,
            "--output-root",
            str(study["output_root"]),
            "--device",
            args.device,
            "--transformer-layers",
            str(args.transformer_layers),
            "--max-steps",
            str(study["max_steps"]),
            "--manifest-out",
            str(manifest_path),
            "--manifest-only",
        ]
        if args.seeds:
            manifest_command.extend(["--seeds", args.seeds])
        manifest_command.extend(study_extra_args)
        subprocess.run(manifest_command, check=True)
        job_count = sum(1 for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())
        tqdm.write(f"[STUDY {study_index}/{len(studies)} MANIFEST] {study_name}: {job_count} runs")
        subprocess.run(
            [
                sys.executable,
                str(root / "launch_manifest_parallel.py"),
                "--manifest",
                str(manifest_path),
                "--max-parallel",
                str(args.parallel_workers),
                "--gpu-index",
                str(args.gpu_index),
                "--min-free-vram-mb",
                str(args.min_free_vram_mb),
                "--safety-margin-mb",
                str(args.safety_margin_mb),
                "--per-process-overhead-mb",
                str(args.per_process_overhead_mb),
                "--min-free-system-ram-mb",
                str(args.min_free_system_ram_mb),
                "--system-ram-safety-margin-mb",
                str(args.system_ram_safety_margin_mb),
                "--per-process-ram-mb",
                str(args.per_process_ram_mb),
                "--poll-interval-sec",
                str(args.poll_interval_sec),
                "--launch-settle-sec",
                str(args.launch_settle_sec),
                "--summary-out",
                str(Path(study["output_root"]) / "summary.csv"),
            ],
            check=True,
        )
        study_progress.update(1)
        tqdm.write(
            f"[STUDY {study_index}/{len(studies)} DONE] {study_name} | "
            f"summary={Path(study['output_root']) / 'summary.csv'}"
        )
    study_progress.close()


if __name__ == "__main__":
    main()
