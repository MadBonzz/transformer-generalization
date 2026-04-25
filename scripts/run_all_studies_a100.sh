#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
PROFILE="${PROFILE:-full10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/full10_run}"
GPU_INDEX="${GPU_INDEX:-0}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-1}"
STUDIES="${STUDIES:-}"
SEEDS="${SEEDS:-}"
MAX_STEPS="${MAX_STEPS:-}"
STUDY1_MODULUS="${STUDY1_MODULUS:-}"
STUDY2_MODULUS="${STUDY2_MODULUS:-}"
STUDY3_MODULUS="${STUDY3_MODULUS:-}"
STUDY3_OUTPUT_MODULUS="${STUDY3_OUTPUT_MODULUS:-}"

# For 1x A100 40 GB, 16 vCPU, 112 GB RAM:
# - cap workers at 16 because CPU is the first hard limit
# - keep a 2 GB VRAM reserve and a 16 GB RAM reserve
# - assume each child process can cost roughly 3 GB system RAM
# - use a 768 MB GPU overhead estimate per child for safer packing
PARALLEL_WORKERS="${PARALLEL_WORKERS:-16}"
MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-2000}"
SAFETY_MARGIN_MB="${SAFETY_MARGIN_MB:-2000}"
PER_PROCESS_OVERHEAD_MB="${PER_PROCESS_OVERHEAD_MB:-768}"
MIN_FREE_SYSTEM_RAM_MB="${MIN_FREE_SYSTEM_RAM_MB:-12288}"
SYSTEM_RAM_SAFETY_MARGIN_MB="${SYSTEM_RAM_SAFETY_MARGIN_MB:-16384}"
PER_PROCESS_RAM_MB="${PER_PROCESS_RAM_MB:-3072}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-2}"
LAUNCH_SETTLE_SEC="${LAUNCH_SETTLE_SEC:-1}"

cmd=(
  "$PYTHON_BIN" -u "$ROOT_DIR/scripts/run_all_studies.py"
  --profile "$PROFILE"
  --output-root "$OUTPUT_ROOT"
  --device cuda
  --transformer-layers "$TRANSFORMER_LAYERS"
  --parallel-workers "$PARALLEL_WORKERS"
  --gpu-index "$GPU_INDEX"
  --min-free-vram-mb "$MIN_FREE_VRAM_MB"
  --safety-margin-mb "$SAFETY_MARGIN_MB"
  --per-process-overhead-mb "$PER_PROCESS_OVERHEAD_MB"
  --min-free-system-ram-mb "$MIN_FREE_SYSTEM_RAM_MB"
  --system-ram-safety-margin-mb "$SYSTEM_RAM_SAFETY_MARGIN_MB"
  --per-process-ram-mb "$PER_PROCESS_RAM_MB"
  --poll-interval-sec "$POLL_INTERVAL_SEC"
  --launch-settle-sec "$LAUNCH_SETTLE_SEC"
)

if [[ -n "$STUDIES" ]]; then
  cmd+=(--studies "$STUDIES")
fi
if [[ -n "$SEEDS" ]]; then
  cmd+=(--seeds "$SEEDS")
fi
if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$STUDY1_MODULUS" ]]; then
  cmd+=(--study1-modulus "$STUDY1_MODULUS")
fi
if [[ -n "$STUDY2_MODULUS" ]]; then
  cmd+=(--study2-modulus "$STUDY2_MODULUS")
fi
if [[ -n "$STUDY3_MODULUS" ]]; then
  cmd+=(--study3-modulus "$STUDY3_MODULUS")
fi
if [[ -n "$STUDY3_OUTPUT_MODULUS" ]]; then
  cmd+=(--study3-output-modulus "$STUDY3_OUTPUT_MODULUS")
fi

exec "${cmd[@]}"
