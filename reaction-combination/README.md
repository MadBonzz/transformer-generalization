# Reaction Combination Dataset

This folder contains a deterministic binary chemical-reaction dataset generator.

The model-facing format is:

```text
amount_1 expanded_reactant_1 + amount_2 expanded_reactant_2 -> amount_3 expanded_product_1 + amount_4 expanded_product_2
```

Examples:

```text
1 H Cl + 1 Na OH -> 1 Na Cl + 1 H H O
1 H2 + 1 Cl2 -> 2 H Cl + NULL NULL
1 F2 + 1 Cl2 -> NULL NULL + NULL NULL
```

Formulas are expanded into repeated element/polyatomic-unit tokens. For example,
`Ca(ClO4)2` becomes `Ca ClO4 ClO4`, and `2H2SO4` is represented as
`2 H H SO4`: the leading stoichiometric coefficient is a single amount token,
while within-formula counts are expanded. Standalone elemental molecules such as
`Cl2`, `O2`, `N2`, and `P4` remain distinct unit tokens, so `Cl2` is not confused
with chloride `Cl`. `NULL NULL` is used as one missing product slot;
no-reaction rows use `NULL NULL + NULL NULL`.

Inputs and targets are padded to fixed lengths with `PAD` for batching. The
current default 100k dataset filters out hydrocarbons and any formula whose
expanded representation has 8 or more unit tokens.

The dataset is generated from:

- curated binary element-element synthesis reactions,
- curated single-product synthesis reactions,
- metal single-displacement reactions from a conservative activity series,
- halogen-displacement reactions,
- active metal + non-oxidizing acid reactions,
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
- `tokenized_examples.csv`: padded expanded-token inputs and padded expanded-token targets.
- `vocab.csv`: unit tokens, amount tokens, `+`, `->`, `NULL`, and `PAD`.
- `metadata.json`: dataset shape, split counts, and validation metadata.

Default dataset shape:

- `100,000` examples.
- `50%` train, `25%` validation, `25%` test.
- the default split strategy holds out whole chemistry groups so a `split_group`
  does not appear in more than one split.
- variable expanded inputs padded to a fixed sequence length.
- variable expanded product-side targets padded to a fixed target sequence length.
- about `90` unit tokens in the current default generation.
- about `8` amount tokens in the current default generation.
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

- `12` runs: transformer layers `1, 2, 3, 4` crossed with seeds `0, 1, 2`.
- `100,000` training steps per run.
- full train/val/test metrics every `100` steps.
- fixed checkpoints every `10,000` steps.
- `batch_size=256`, `learning_rate=1e-3`, `weight_decay=0.5`.
- one output bundle under `reaction-combination/outputs/reaction_base_case/`.

The runner generates the dataset on the fly, validates tokenization and atom
balance, writes `dataset_generation.log`, `dataset_validation.json`,
`experiment_manifest.json`, `summary.csv`, and per-run `config.json`,
`metrics.jsonl`, `metrics.csv`, `progress.json`, `result.json`,
`dataset_snapshot.pt`, checkpoints, `checkpoint_path` in `result.json`, and val/test prediction
CSVs.
