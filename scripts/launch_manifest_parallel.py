from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.job_runner import aggregate_results, estimate_vram_mb, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch manifest jobs in parallel using live GPU free-VRAM checks.")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--min-free-vram-mb", type=float, default=500.0)
    parser.add_argument("--safety-margin-mb", type=float, default=500.0)
    parser.add_argument("--per-process-overhead-mb", type=float, default=256.0)
    parser.add_argument("--min-free-system-ram-mb", type=float, default=0.0)
    parser.add_argument("--system-ram-safety-margin-mb", type=float, default=0.0)
    parser.add_argument("--per-process-ram-mb", type=float, default=0.0)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--launch-settle-sec", type=float, default=1.0)
    parser.add_argument("--summary-out", required=True, type=str)
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts_utc": _timestamp(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _query_free_vram_mb(gpu_index: int) -> tuple[float, float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is required for live VRAM scheduling but was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to query GPU memory with nvidia-smi: {exc.output.strip()}") from exc

    if not output:
        raise RuntimeError(f"nvidia-smi returned no GPU memory data for gpu_index={gpu_index}")

    first_line = output.splitlines()[0]
    free_raw, total_raw = [item.strip() for item in first_line.split(",", maxsplit=1)]
    return float(free_raw), float(total_raw)


def _query_free_system_ram_mb() -> tuple[float, float] | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None

    free_mb = (page_size * available_pages) / (1024 ** 2)
    total_mb = (page_size * total_pages) / (1024 ** 2)
    return float(free_mb), float(total_mb)


def _pick_job_index(pending: list[dict[str, object]], capacity_mb: float) -> int | None:
    eligible = [i for i, job in enumerate(pending) if float(job["estimated_vram_mb"]) <= capacity_mb]
    if not eligible:
        return None
    return max(eligible, key=lambda index: float(pending[index]["estimated_vram_mb"]))


def _launch_job(
    *,
    manifest_path: Path,
    job_record: dict[str, object],
    root: Path,
    event_log_path: Path,
) -> dict[str, object]:
    job = job_record["job"]
    index = int(job_record["index"])
    estimated_vram_mb = float(job_record["estimated_vram_mb"])
    output_dir = Path(job["run_config"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "launcher.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(root / "run_experiment_job.py"), "--manifest", str(manifest_path), "--job-index", str(index)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _append_event(
        event_log_path,
        {
            "event": "launch",
            "job_index": index,
            "estimated_vram_mb": estimated_vram_mb,
            "output_dir": str(output_dir),
            "pid": process.pid,
        },
    )
    return {
        "process": process,
        "log_handle": log_handle,
        "job_index": index,
        "estimated_vram_mb": estimated_vram_mb,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    jobs = read_manifest(manifest_path)
    root = Path(__file__).resolve().parent
    event_log_path = Path(args.summary_out).parent / "scheduler_events.jsonl"
    parallel_limit = sys.maxsize if args.max_parallel <= 0 else args.max_parallel
    system_ram_supported = _query_free_system_ram_mb() is not None

    pending: list[dict[str, object]] = []
    for index, job in enumerate(jobs):
        pending.append(
            {
                "index": index,
                "job": job,
                "estimated_vram_mb": estimate_vram_mb(job, per_process_overhead_mb=args.per_process_overhead_mb),
            }
        )

    pending.sort(key=lambda item: float(item["estimated_vram_mb"]), reverse=True)
    running: list[dict[str, object]] = []
    _, total_vram_mb = _query_free_vram_mb(args.gpu_index)
    _append_event(
        event_log_path,
        {
            "event": "scheduler_start",
            "gpu_index": args.gpu_index,
            "total_vram_mb": total_vram_mb,
            "pending_jobs": len(pending),
            "max_parallel": None if args.max_parallel <= 0 else args.max_parallel,
            "min_free_vram_mb": args.min_free_vram_mb,
            "safety_margin_mb": args.safety_margin_mb,
            "per_process_overhead_mb": args.per_process_overhead_mb,
            "min_free_system_ram_mb": args.min_free_system_ram_mb,
            "system_ram_safety_margin_mb": args.system_ram_safety_margin_mb,
            "per_process_ram_mb": args.per_process_ram_mb,
            "system_ram_supported": system_ram_supported,
        },
    )

    while pending or running:
        still_running: list[dict[str, object]] = []
        for item in running:
            return_code = item["process"].poll()
            if return_code is None:
                still_running.append(item)
                continue
            item["log_handle"].close()
            _append_event(
                event_log_path,
                {
                    "event": "finish",
                    "job_index": item["job_index"],
                    "return_code": return_code,
                    "output_dir": item["output_dir"],
                    "pid": item["process"].pid,
                },
            )
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, item["process"].args)
        running = still_running

        launched_any = False
        while pending and len(running) < parallel_limit:
            free_vram_mb, total_vram_mb = _query_free_vram_mb(args.gpu_index)
            capacity_mb = free_vram_mb - args.safety_margin_mb
            if capacity_mb < args.min_free_vram_mb:
                _append_event(
                    event_log_path,
                    {
                        "event": "wait",
                        "reason": "below_min_free_threshold",
                        "free_vram_mb": free_vram_mb,
                        "total_vram_mb": total_vram_mb,
                        "capacity_mb": capacity_mb,
                        "running_jobs": len(running),
                        "pending_jobs": len(pending),
                    },
                )
                break

            system_ram = _query_free_system_ram_mb()
            if system_ram is not None and args.per_process_ram_mb > 0.0:
                free_system_ram_mb, total_system_ram_mb = system_ram
                system_capacity_mb = free_system_ram_mb - args.system_ram_safety_margin_mb
                if system_capacity_mb < max(args.min_free_system_ram_mb, args.per_process_ram_mb):
                    _append_event(
                        event_log_path,
                        {
                            "event": "wait",
                            "reason": "below_system_ram_threshold",
                            "free_system_ram_mb": free_system_ram_mb,
                            "total_system_ram_mb": total_system_ram_mb,
                            "system_capacity_mb": system_capacity_mb,
                            "running_jobs": len(running),
                            "pending_jobs": len(pending),
                        },
                    )
                    break

            selected_index = _pick_job_index(pending, capacity_mb)
            if selected_index is None and not running:
                smallest_index = min(range(len(pending)), key=lambda index: float(pending[index]["estimated_vram_mb"]))
                if float(pending[smallest_index]["estimated_vram_mb"]) <= free_vram_mb:
                    selected_index = smallest_index

            if selected_index is None:
                _append_event(
                    event_log_path,
                    {
                        "event": "wait",
                        "reason": "no_job_fits_current_free_vram",
                        "free_vram_mb": free_vram_mb,
                        "total_vram_mb": total_vram_mb,
                        "capacity_mb": capacity_mb,
                        "running_jobs": len(running),
                        "pending_jobs": len(pending),
                        "smallest_pending_estimated_vram_mb": min(
                            float(job["estimated_vram_mb"]) for job in pending
                        ),
                    },
                )
                break

            job_record = pending.pop(selected_index)
            running.append(
                _launch_job(
                    manifest_path=manifest_path,
                    job_record=job_record,
                    root=root,
                    event_log_path=event_log_path,
                )
            )
            launched_any = True
            time.sleep(args.launch_settle_sec)

        if pending or running:
            if not launched_any:
                time.sleep(args.poll_interval_sec)

    _append_event(event_log_path, {"event": "scheduler_end"})
    aggregate_results(manifest_path, args.summary_out)


if __name__ == "__main__":
    main()
