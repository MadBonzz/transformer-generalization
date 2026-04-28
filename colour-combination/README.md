# Colour Combination Dataset and Baseline

This folder contains the self-contained color-mixing dataset generator and the
baseline runner for the grokking-style colour-combination experiment.

## Dataset

The current baseline experiment uses the Mixbox pigment-like generator in
`generate_mixbox_dataset.py`. The saved compact dataset has this row format:

```text
hex_1,hex_2,t,output_hex
```

`t` is the percentage of colour 2 in Mixbox interpolation terms:

```text
t = ratio_2_parts / (ratio_1_parts + ratio_2_parts)
```

The runner generates the dataset on the fly with seed `42` by default. It also
writes training files used by the model:

- `tokenized_examples.csv`
- `vocab.csv`
- `metadata.json`

Token layout:

- Each base-palette hex code is one shared color token.
- Mixbox produces the raw mixture, then the target is quantized to the nearest base-palette color.
- Each unique `t` value is a separate input-only token.
- Input sequence length: `3`, as `[hex_1_token, hex_2_token, t_token]`.
- Output target: one hex token.
- Default color vocabulary: `2000` base hex tokens.
- Default total input vocabulary: `2015` tokens, from `2000` base colors plus `15` `t` tokens.
- Split: `50%` train, `25%` validation, `25%` test.

## Baseline Runs

Run the base experiment:

```bash
python colour-combination/run_base_experiment.py
```

This runs six jobs:

- 1-layer Neel/Nanda-style transformer, seeds `0, 1, 2`
- 2-layer Power-style transformer, seeds `0, 1, 2`
- `500000` training steps per run
- weight decay `0.5`
- uniform checkpoints every `25000` steps under each run's `checkpoints/` folder

Default outputs:

```text
colour-combination/outputs/mixbox_base_case/
```

This folder is the complete run bundle. It contains:

- `dataset/colour_mixing_mixbox_100k_2000base/`
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
python colour-combination/run_base_experiment.py --dataset-seed 42
```

Palette-size control:

```bash
python colour-combination/run_base_experiment.py --num-base-colors 1000
```

Useful shorter smoke run:

```bash
python colour-combination/run_base_experiment.py --layers 1 --seeds 0 --max-steps 1 --eval-every 1 --output-dir colour-combination/outputs/smoke
```

## Mixbox Pigment-Like Dataset

For a larger synthetic pigment-like dataset, use the Mixbox generator:

```bash
python colour-combination/generate_mixbox_dataset.py
```

Default output:

```text
colour-combination/outputs/mixbox_100k_2000base/
```

It writes:

- `colour_mixing_100k.csv`
- `base_palette.csv`
- `base_palette_2000.csv`
- `colour_mixing_100k_with_splits.csv`
- `tokenized_examples.csv`
- `vocab.csv`
- `metadata.json`

The saved mixture columns are compact:

```text
hex_1,hex_2,t,output_hex
```

`t` is the percentage of colour 2 in Mixbox interpolation terms, so `t=0.5`
means a `1:1` mix and `t=0.666667` means `1:2`. RGB values are used only
internally for Mixbox, nearest-palette quantization, and validation. Mixbox is
installed from the `pymixbox` package and imported as `mixbox`.
