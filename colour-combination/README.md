# Colour Combination Dataset and Baseline

This folder contains the self-contained color-mixing dataset generator and the
baseline runner for the grokking-style colour-combination experiment.

## Dataset

The current baseline experiment uses the Mixbox pigment-like generator in
`generate_mixbox_dataset.py`. The saved compact dataset has this row format:

```text
hex_1,hex_2,ratio_1_parts,ratio_2_parts,t,output_hex
```

`t` is the percentage of colour 2 in Mixbox interpolation terms:

```text
t = ratio_2_parts / (ratio_1_parts + ratio_2_parts)
```

So `t=0.333333` means colour 2 is one third of the mixture, e.g. a
`2:1` mix of colour 1 to colour 2. A `1:2` mix gives `t=0.666667`.

The runner generates the dataset on the fly with seed `42` by default. It also
writes training files used by the model:

- `tokenized_examples.csv`
- `vocab.csv`
- `metadata.json`

Token layout:

- Each token is an integer value in `[0, 255]`.
- Each colour is represented as three RGB channel tokens.
- Mixbox produces the raw mixture, then the target is quantized to the nearest base-palette color.
- Amounts use separate `AMOUNT_n` tokens, so amount `2` is not confused with RGB value `2`.
- `PLUS` separates the two colours and `EQUALS` marks the end of the input.
- Input sequence length: `10`, as `[amount_1, r1, g1, b1, PLUS, amount_2, r2, g2, b2, EQUALS]`.
- Output target length: `3`, as `[out_r, out_g, out_b]`.
- Model vocabulary size: `265` for the default ratio pool: `256` RGB values, `7` amount tokens, `PLUS`, and `EQUALS`.
- Target vocabulary size: `256`.
- Split: `50%` train, `25%` validation, `25%` test.

## Baseline Runs

Run the base experiment:

```bash
python colour-combination/run_base_experiment.py
```

This runs twelve jobs:

- 1-layer Neel/Nanda-style transformer, seeds `0, 1, 2`
- 2-, 3-, and 4-layer Power-style transformers, seeds `0, 1, 2`
- `100000` training steps per run
- weight decay `0.5`
- full train/val/test metrics every `100` steps
- fixed checkpoints every `10000` steps under each run's `checkpoints/` folder

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
python scripts/run_colour_only.py --parallel-workers 12 --device cuda
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
hex_1,hex_2,ratio_1_parts,ratio_2_parts,t,output_hex
```

`t` is the percentage of colour 2 in Mixbox interpolation terms, so `t=0.5`
means a `1:1` mix and `t=0.666667` means `1:2`. RGB values are used only
internally for Mixbox, nearest-palette quantization, validation, and the
model-facing RGB-token dataset. Mixbox is installed from the `pymixbox` package
and imported as `mixbox`.
