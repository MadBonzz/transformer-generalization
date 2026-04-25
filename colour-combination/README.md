# Colour Combination Dataset and Baseline

This folder contains the self-contained color-mixing dataset generator and the
baseline runner for the grokking-style colour-combination experiment.

## Dataset

The dataset models additive light/display mixing with linear-light sRGB:

```text
color1 ratio1 color2 ratio2 = target_color
```

Token layout:

- Color tokens: `0..999`
- Ratio tokens: `1000..1010` for `0%, 10%, ..., 100%`
- Equals token: `1011`
- Input sequence length: `5`
- Target classes: `0..999`

Generate the default dataset:

```bash
python colour-combination/create_dataset.py
```

Default output:

```text
colour-combination/datasets/color_mixing_linear_srgb_1000/
```

The experiment runner does not require this pre-generated folder. It generates
the same dataset on the fly, using `--dataset-seed`, inside the selected output
bundle.

## Baseline Runs

Run the base experiment:

```bash
python colour-combination/run_base_experiment.py
```

This runs six jobs:

- 1-layer Neel/Nanda-style transformer, seeds `0, 1, 2`
- 2-layer Power-style transformer, seeds `0, 1, 2`
- `100000` training steps per run
- weight decay `0.5`
- staged checkpoints under each run's `checkpoints/` folder:
  `1000, 2000, ..., 10000`, then `15000, 20000, ..., 50000`,
  then `60000, 70000, ..., 100000`

Default outputs:

```text
colour-combination/outputs/base_case/
```

This folder is the complete run bundle. It contains:

- `dataset/color_mixing_linear_srgb_1000/`
- `runs/<run_name>/`
- `launcher_logs/` when `--parallel-workers` is not `1`
- `experiment_manifest.json`
- `summary.csv`

Parallel run example:

```bash
python colour-combination/run_base_experiment.py --parallel-workers 2
```

Deterministic dataset controls:

```bash
python colour-combination/run_base_experiment.py --dataset-seed 20260425 --random-examples 60000 --chain-examples 15000
```

Useful shorter smoke run:

```bash
python colour-combination/run_base_experiment.py --layers 1 --seeds 0 --max-steps 1 --eval-every 1 --output-dir colour-combination/outputs/smoke
```
