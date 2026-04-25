# AGENTS.md

## Project Purpose
This repository studies grokking and compositional generalization in small transformers and MLPs on modular arithmetic tasks. The core focus is whether a model can learn arithmetic rules that transfer beyond the biased or partial training support it sees during optimization.

There are three active study families in the codebase:
1. `study1_loss_vs_rl`: compares cross-entropy training against RL-style objectives on single-operator modular arithmetic.
2. `study2_fake_labels`: studies robustness to label corruption and whether the model can recover the true arithmetic rule under noisy supervision.
3. `study3_range_transfer`: studies distribution shift and transfer across ranges, subsets, operators, and pairing structures. All retained archived results in this repository are from Study 3.

## Repository Layout
- `src/grokking_transformer/`: reusable implementation code.
- `scripts/`: experiment entrypoints, orchestration, and manifest generation.
- `tests/`: smoke tests and task-builder/manifest tests.
- `outputs/old_results/`: retained archived result batches.
- `train.py`: small top-level training entrypoint from the earlier project layout.
- `README.md`: general project description.
- `grokking_notes_compiled.md`: working notes.

## Source Code Structure
### `src/grokking_transformer/tasks.py`
Defines the arithmetic tasks, tokenization, dataset splits, and Study 3 scenario builders.

Important responsibilities:
- Implements operators: `add`, `sub`, `mul`, `div`, and `poly`.
- Builds single-operator datasets for Study 1 and Study 2.
- Builds the earlier range-transfer task with token offsets.
- Builds Study 3 variants including:
  - 2-subset partitioned operator setups
  - 2-subset both-ops benchmarks
  - interleaved subset setups
  - whole-set pair/operator split setups
  - 4-subset / 4-operator missing-operator setups
- Defines `DatasetInfo`, `TaskDataset`, `TaskSplit`, `RangeTransferSplit`, and `Study3Split`.

### `src/grokking_transformer/model.py`
Implements the transformer models used across studies.

Current architecture presets:
- `TransformerConfig.neel_nanda(...)`
  - 1 layer
  - `d_model=128`
  - `4` heads with `d_head=32`
  - `d_mlp=512`
  - `ReLU`
  - no `LayerNorm`
  - learned positional embeddings
- `TransformerConfig.power_grokking(...)`
  - 2 layers
  - `d_model=128`
  - `4` heads with `d_head=32`
  - `d_mlp=512`
  - `ReLU`
  - post-residual `LayerNorm`
  - sinusoidal positional embeddings
  - zero dropout
  - bias-free MLP layers

Current parameter counts for the Study 3 tokenized setting (`vocab_size=134`, `seq_len=4`):
- 1-layer Neel preset: `232064` trainable parameters
- 2-layer grokking preset: `428544` trainable parameters

The model code exposes:
- `MultiHeadSelfAttention`
- `TransformerBlock`
- `GrokkingTransformer`

### `src/grokking_transformer/mlp.py`
Defines the modular arithmetic MLP baseline used in Study 1 and Study 2.

### `src/grokking_transformer/experiment_utils.py`
Shared training/runtime utilities.

Important responsibilities:
- `RunConfig`: canonical run specification.
- `build_model(...)`: chooses transformer vs MLP and selects the correct transformer preset from `transformer_n_layers`.
- `run_training(...)`: the main train/eval loop.
- `evaluate_dataset(...)`: standardized evaluation.
- `transformer_run_prefix(...)`: naming convention for `transformer` vs `transformer2`.
- `transformer_architecture_name(...)`: canonical architecture metadata string.
- `run_config_payload(...)`: canonical manifest serialization.

### `src/grokking_transformer/job_runner.py`
Manifest execution and scheduler-facing helpers.

Important responsibilities:
- Rehydrates jobs from manifest rows.
- Dispatches `single_operator`, `range_transfer`, and `study3_variant` tasks.
- Estimates VRAM based on model type, batch size, sequence length, and transformer depth.
- Aggregates manifest outputs.

### `src/grokking_transformer/train_utils.py`
Core minibatch train/eval steps.

### `src/grokking_transformer/rl.py`
GRPO/PPO-related code used by Study 1 and Study 2.

### `src/grokking_transformer/logging_utils.py`
Writers for JSON, JSONL, and CSV metrics/progress outputs.

### `src/grokking_transformer/data.py`
Earlier modular-addition dataset utilities used by smoke tests and the original simplified setup.

## Script Entry Points
### `scripts/run_loss_vs_rl.py`
Study 1 entrypoint.
- Generates Study 1 jobs.
- Supports `--transformer-layers 1|2`.
- Can write a manifest or execute runs directly.

### `scripts/run_fake_labels.py`
Study 2 entrypoint.
- Generates fake-label corruption and inversion jobs.
- Supports `--transformer-layers 1|2`.

### `scripts/run_range_transfer.py`
Study 3 entrypoint.
- Generates the Study 3 scenario suite.
- Supports `--transformer-layers 1|2`.
- Current default suite is the retained 4-subset suite plus the two whole-set additions.

### `scripts/run_all_studies.py`
Top-level multi-study orchestrator. Builds manifests and optionally launches the parallel manifest scheduler.

### `scripts/run_all_studies_a100.sh`
Linux/A100 helper wrapper around `run_all_studies.py`.

### `scripts/launch_manifest_parallel.py`
Parallel manifest scheduler used for A100 runs.

### `scripts/aggregate_manifest_results.py`
Collects completed manifest rows into summary CSVs.

### `scripts/analyze_study3_embeddings.py`
Reusable Study 3 embedding analysis script kept after the repo cleanup.

## Tokenization and Vocabulary
The code distinguishes between:
- input vocabulary size: all tokens the model can read
- target vocabulary size: valid prediction classes

For task-token Study 3 runs, the input generally looks like:
- `[a, operator_token, b, eq_token]`

For no-task-token variants, the input is:
- `[a, b, eq_token]`

The operator token is an ordinary learned embedding. It changes the input sequence but is not itself a valid prediction target.

## Model Switching Contract
The data pipeline is independent of transformer depth. Switching architectures should not require task changes.

Use:
- `--transformer-layers 1` for the Neel-style 1-layer transformer
- `--transformer-layers 2` for the original grokking-style 2-layer transformer

The generated run naming convention is:
- `transformer_...` for the 1-layer preset
- `transformer2_...` for the 2-layer preset

The same convention is also stored in run metadata as:
- `neel_nanda_1layer`
- `power_etal_2layer`

## Testing
The retained tests are intentionally small and targeted.

Main test files:
- `tests/test_experiments.py`
  - task-builder correctness
  - scenario wiring
  - manifest counts
  - reward logic
  - progress tracking
- `tests/test_smoke.py`
  - transformer forward passes
  - Neel preset sanity checks
  - 2-layer preset sanity checks
  - small overfitting smoke tests

Typical verification command:
```bash
python -m unittest tests.test_experiments tests.test_smoke -v
```

## Retained Results
Only three Study 3 result archives are intentionally kept in the repository:
1. `outputs/old_results/study3_18runs_mod100_tasktoken_20260422`
2. `outputs/old_results/study3_42runs_mod131_a100_20260423`
3. `outputs/old_results/study3_39runs_mod131_fourset_a100_20260424`

These are the canonical retained result batches after cleanup. Other older/incomplete result trees were removed.

## Result Archive Conventions
Each retained result archive contains:
- per-run folders
- a manifest
- a `summary.csv`
- one or more compact result tables
- a human-readable summary file
- sometimes additional analysis outputs such as milestone CSVs or embedding analyses

## Study 3 Retained Batches
### 1. `study3_18runs_mod100_tasktoken_20260422`
Earlier local 18-run Study 3 batch.
- modulus/output modulus: `100`
- two disjoint numeric ranges:
  - addition on `0..99`
  - multiplication on `100..199`
- explicit task token enabled
- `6` configs x `3` seeds = `18` runs

### 2. `study3_42runs_mod131_a100_20260423`
Main expanded 2-subset A100 batch.
- modulus/output modulus: `131`
- contiguous two-set partition over `0..130`
- `14` configs x `3` seeds = `42` runs
- includes partitioned, no-task-token, within-set, all-pairs, and interleaved scenarios

### 3. `study3_39runs_mod131_fourset_a100_20260424`
Follow-up 4-subset A100 batch.
- modulus/output modulus: `131`
- four contiguous subsets over `0..130`
- operators: `+`, `-`, `*`, `/`
- `13` configs x `3` seeds = `39` runs
- includes the four-set missing-operator ablations plus whole-set baselines

## Practical Guidance For New Code Agents
If starting from this file alone:
1. Use `scripts/run_range_transfer.py` for any Study 3 rerun or modification.
2. Treat `tasks.py` as the source of truth for scenario semantics.
3. Treat `experiment_utils.py` as the source of truth for architecture selection and run serialization.
4. Treat `outputs/old_results/` as immutable archived results.
5. If changing transformer depth, do it through `transformer_n_layers` / `--transformer-layers`, not by editing task code.
6. Run `python -m unittest tests.test_experiments tests.test_smoke -v` after any nontrivial code changes.
