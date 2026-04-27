# Transformer Generalization

Small PyTorch setup for modular-addition grokking experiments inspired by:

- Power et al. (2022), "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
- Nanda et al. (2023), "Progress measures for grokking via mechanistic interpretability"

This project includes:

- a modular-addition dataset with the fixed-operation, 3-token context used in Neel Nanda's setup
- a simple 1-layer decoder-only transformer
- a 2-layer MLP matching the corrupted-label paper's one-hot + quadratic-activation setup
- a small training script
- an experiment suite for loss-vs-RL, fake-label, and range-transfer studies
- smoke tests that verify forward/backward/training behavior

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe train.py --steps 200 --device cuda
.\.venv\Scripts\python.exe train.py --model mlp --steps 200 --full-batch --device cuda
```

## Experiment scripts

Pilot runs:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_loss_vs_rl.py --profile pilot
.\.venv\Scripts\python.exe .\scripts\run_fake_labels.py --profile pilot
.\.venv\Scripts\python.exe .\scripts\run_range_transfer.py --profile pilot
```

Sequential execution:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_all_studies.py --profile pilot
```

Heavier 10-seed presets:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_all_studies.py --profile full10
```

Cloud A100 run on Linux:

```bash
chmod +x scripts/run_all_studies_a100.sh
scripts/run_all_studies_a100.sh
```

The repo is configured for a Python 3.13 environment with dependencies installed from `requirements.txt`.

This wrapper is tuned for a `1x A100 40 GB / 16 vCPU / 112 GB RAM` box:

- `PARALLEL_WORKERS=16`
- VRAM reserve of `2 GB`
- system RAM reserve of `16 GB`
- per-process RAM estimate of `3 GB`

You can override any of these by exporting environment variables before running the script.

Outputs are written under `outputs/` with one directory per run, a `metrics.jsonl` trace, a `config.json`, a final `result.json`, and a per-study `summary.csv`.

## Notes

- Inputs are `[a, b, =]`.
- The model predicts `(a + b) mod p` from the final position.
- Default architecture is intentionally close to the small 1-layer model used in the mechanistic grokking paper: `d_model=128`, `n_heads=4`, `d_head=32`, `d_mlp=512`.
- The MLP path uses the corrupted-label paper's setup: concatenate one-hot encodings of `a` and `b`, apply one hidden layer with quadratic activation, and train against one-hot targets.
- The transformer RL path uses a lightweight GRPO-style grouped policy optimization adapted to single-token classification.
- The MLP RL path uses a lightweight PPO-style actor-critic objective for the same single-step classification setting.
- RL reward sweeps include binary reward and an absolute-difference partial reward.
- The range-transfer study uses explicit operator tokens and disjoint numeric token ranges.
