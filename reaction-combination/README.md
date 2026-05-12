# Reaction Combination Dataset

This folder contains a deterministic binary chemical-reaction dataset generator.

The model-facing format is:

```text
reactant_1 amount_1 reactant_2 amount_2 -> output_1 output_2
```

Examples:

```text
HCl 1 NaOH 1 -> NaCl H2O
H2 1 Cl2 1 -> HCl NULL
F2 1 Cl2 1 -> NULL NULL
```

`NULL` is an explicit species token used when the reaction has only one product
species, and `NULL NULL` is used for explicit no-net-reaction examples. Product
stoichiometric coefficients are stored in the CSV as
`output_1_amount` and `output_2_amount`, but the model target is only the two
product species tokens. This keeps the input context fixed at 4 tokens and the
target fixed at 2 tokens.

The dataset is generated from:

- curated binary element-element synthesis reactions,
- curated single-product synthesis reactions,
- metal single-displacement reactions from a conservative activity series,
- halogen-displacement reactions,
- active metal + non-oxidizing acid reactions,
- simple hydrocarbon combustion reactions,
- acid-base neutralization templates,
- aqueous double-displacement reactions filtered by standard solubility rules,
- explicit no-net-reaction controls, including soluble spectator salt pairs,
  non-displacing metal/salt pairs, and selected element pairs.

Every positive accepted row is atom-balanced by parsing the formulas and
validating atom counts before it is written. No-reaction rows are validated by
requiring both outputs to be `NULL`. By default, each canonical reaction is
written once: there are no scaled stoichiometric variants and no reversed-order
duplicate inputs. The dataset is deterministic with seed `42`.

Generate the dataset:

```bash
python reaction-combination/generate_reaction_dataset.py
```

Default output:

```text
reaction-combination/outputs/reaction_combination_100k/
```

Files written:

- `reaction_combination.csv`: human-readable reaction rows.
- `tokenized_examples.csv`: four-token inputs and two-token targets.
- `vocab.csv`: species tokens, amount tokens, and the `NULL` token.
- `metadata.json`: dataset shape, split counts, and validation metadata.

Default dataset shape:

- `100,000` examples.
- `50%` train, `25%` validation, `25%` test.
- the default split strategy holds out whole chemistry groups so a `split_group`
  does not appear in more than one split.
- `4` input tokens: reactant species, amount, reactant species, amount.
- `2` output tokens: product species, product species-or-`NULL`.
- about `2.5k` species tokens including `NULL` in the current default generation.
- about `10` amount tokens in the current default generation.
- about `600` unique element-element synthesis rows in the current default generation.
- `25%` no-reaction rows by default.

To change size while keeping deterministic sampling:

```bash
python reaction-combination/generate_reaction_dataset.py --num-rows 100000 --seed 42
```

Useful controls:

```bash
python reaction-combination/generate_reaction_dataset.py \
  --no-reaction-fraction 0.25 \
  --split-strategy generalization
```

Run the baseline experiment:

```bash
python reaction-combination/run_base_experiment.py --parallel-workers 1
```

Default experiment settings:

- `6` runs: transformer layers `1, 2` crossed with seeds `0, 1, 2`.
- `500,000` training steps per run.
- full train/val/test metrics every `500` steps.
- staged checkpoints every `1,000` steps through `25,000`, then every `25,000` steps.
- `batch_size=256`, `learning_rate=1e-3`, `weight_decay=0.5`.
- one output bundle under `reaction-combination/outputs/reaction_base_case/`.

The runner generates the dataset on the fly, validates tokenization and atom
balance, writes `dataset_generation.log`, `dataset_validation.json`,
`experiment_manifest.json`, `summary.csv`, and per-run `config.json`,
`metrics.jsonl`, `metrics.csv`, `progress.json`, `result.json`,
`dataset_snapshot.pt`, checkpoints, final checkpoint, and val/test prediction
CSVs.
