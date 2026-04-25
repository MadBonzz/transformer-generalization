# RESULTS.md

## Scope
This file summarizes the three retained experiment batches currently preserved in the repository. All three are Study 3 experiment batches, each probing a different form of arithmetic transfer failure or success under structured distribution shift.

Retained result archives:
1. `outputs/old_results/study3_18runs_mod100_tasktoken_20260422`
2. `outputs/old_results/study3_42runs_mod131_a100_20260423`
3. `outputs/old_results/study3_39runs_mod131_fourset_a100_20260424`

## Common Training Setup
Across the retained Study 3 batches:
- objective: cross-entropy
- optimizer: AdamW
- model for archived runs: 1-layer transformer
- default transformer scale: `d_model=128`, `4` heads, `d_head=32`, `d_mlp=512`
- later code now distinguishes:
  - 1-layer Neel preset
  - 2-layer grokking preset
- archived runs themselves were all produced with the 1-layer transformer

## Experiment 1: 18-run local mod-100 task-token batch
Archive: `outputs/old_results/study3_18runs_mod100_tasktoken_20260422`

### What it tested
This was the earlier Study 3 range-transfer setup before the later contiguous-subset scenario expansion. It used a confounded token-offset formulation:
- addition examples occupied one numeric token range
- multiplication examples occupied a different numeric token range
- a task token was also present

The hypothesis was whether the model could learn arithmetic that transfers when the operation-number association is structurally biased.

### Batch structure
- `6` configs x `3` seeds = `18` runs
- varied:
  - `train_fraction`: `0.1`, `0.3`, `0.5`
  - `lr`: `1e-3`, `3e-4`
  - `weight_decay`: `1.0`, `0.5`
  - `batch_size`: `256`, `128`

### Averaged results by config
| lr | weight_decay | batch_size | train_fraction | train_accuracy | test_accuracy | cross_add_accuracy | cross_mul_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0010 | 1 | 256 | 0.1000 | 1 | 0.0794 | 0.0043 | 0.0076 |
| 0.0010 | 1 | 256 | 0.3000 | 0.9999 | 0.9999 | 0.0040 | 0.0040 |
| 0.0003 | 1 | 256 | 0.5000 | 1 | 1 | 0.0040 | 0.0040 |
| 0.0010 | 0.5000 | 256 | 0.5000 | 1.0000 | 0.9999 | 0.0047 | 0.0047 |
| 0.0010 | 1 | 128 | 0.5000 | 0.9984 | 0.9981 | 0.0045 | 0.0057 |
| 0.0010 | 1 | 256 | 0.5000 | 0.9997 | 0.9997 | 0.0041 | 0.0044 |

### Main takeaways
- `train_fraction=0.1` did not generalize on the normal split.
- `train_fraction=0.3` and `0.5` generalized quickly on the normal split.
- cross-task / reverse-style transfer remained near chance.
- because the batch used both disjoint token ranges and an explicit task token, it is best treated as an earlier, more confounded precursor to the later Study 3 design.

### Detailed notes
See the experiment-level file in the archive folder for the complete per-run table.

## Experiment 2: 42-run A100 mod-131 two-subset batch
Archive: `outputs/old_results/study3_42runs_mod131_a100_20260423`

### What it tested
This was the main two-subset Study 3 batch. The core question was whether a transformer trained on biased subset-operator assignments learns:
- a genuine operator abstraction
- a genuine number abstraction
- or merely subset-specific heuristics that fail under reversal or unseen cross-subset pairings

### Batch structure
- `14` configs x `3` seeds = `42` runs
- scenario families:
  - `contiguous_partitioned_ops`
  - `contiguous_partitioned_ops_no_task_token`
  - `contiguous_both_ops_within_set`
  - `contiguous_both_ops_all_pairs`
  - `interleaved10_partitioned_ops`
  - `interleaved20_partitioned_ops`

### Averaged results by config
| scenario_name | use_task_token | lr | weight_decay | batch_size | train_fraction | avg_train_acc | avg_test_acc | avg_reverse_add_acc | avg_reverse_mul_acc | avg_cross_pair_add_acc | avg_cross_pair_mul_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contiguous_both_ops_all_pairs | True | 0.0010 | 1 | 256 | 0.5000 | 0.8651 | 0.8531 |  |  | 0.7232 | 0.9950 |
| contiguous_both_ops_within_set | True | 0.0010 | 1 | 256 | 0.5000 | 0.9953 | 0.9906 |  |  | 0.0000 | 0.0152 |
| contiguous_partitioned_ops | True | 0.0010 | 1 | 256 | 0.1000 | 1 | 0.0281 | 0.0077 | 0.0073 | 0.0033 | 0.0045 |
| contiguous_partitioned_ops | True | 0.0010 | 1 | 256 | 0.1500 | 0.9902 | 0.0359 | 0.0073 | 0.0072 | 0.0026 | 0.0034 |
| contiguous_partitioned_ops | True | 0.0010 | 1 | 256 | 0.2000 | 1 | 0.0763 | 0.0071 | 0.0084 | 0.0024 | 0.0039 |
| contiguous_partitioned_ops | True | 0.0010 | 1 | 256 | 0.2500 | 1 | 0.5062 | 0.0068 | 0.0070 | 0.0026 | 0.0052 |
| contiguous_partitioned_ops | True | 0.0010 | 1 | 256 | 0.3000 | 1 | 0.9637 | 0.0077 | 0.0071 | 0.0022 | 0.0053 |
| contiguous_partitioned_ops | True | 0.0003 | 1 | 256 | 0.5000 | 1 | 0.9937 | 0.0076 | 0.0075 | 0.0018 | 0.0046 |
| contiguous_partitioned_ops | True | 0.0010 | 0.5000 | 256 | 0.5000 | 1 | 0.9974 | 0.0076 | 0.0078 | 0.0014 | 0.0055 |
| contiguous_partitioned_ops | True | 0.0010 | 1 | 128 | 0.5000 | 0.9998 | 0.9936 | 0.0076 | 0.0073 | 0.0032 | 0.0047 |
| contiguous_partitioned_ops | True | 0.0010 | 1 | 256 | 0.5000 | 0.9989 | 0.9897 | 0.0074 | 0.0074 | 0.0016 | 0.0044 |
| contiguous_partitioned_ops_no_task_token | False | 0.0010 | 1 | 256 | 0.5000 | 0.9973 | 0.9879 | 0.0076 | 0.0073 | 0.0031 | 0.0044 |
| interleaved10_partitioned_ops | True | 0.0010 | 1 | 256 | 0.5000 | 0.9985 | 0.9914 | 0.0077 | 0.0073 | 0.0026 | 0.0045 |
| interleaved20_partitioned_ops | True | 0.0010 | 1 | 256 | 0.5000 | 1 | 0.9995 | 0.0078 | 0.0075 | 0.0020 | 0.0063 |

### Scenario-level summary
| scenario_name | runs | avg_train_acc | avg_test_acc | avg_reverse_add_acc | avg_reverse_mul_acc | avg_cross_pair_add_acc | avg_cross_pair_mul_acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contiguous_both_ops_all_pairs | 3 | 0.8651 | 0.8531 |  |  | 0.7232 | 0.9950 |
| contiguous_both_ops_within_set | 3 | 0.9953 | 0.9906 |  |  | 0.0000 | 0.0152 |
| contiguous_partitioned_ops | 27 | 0.9988 | 0.6205 | 0.0074 | 0.0074 | 0.0024 | 0.0046 |
| contiguous_partitioned_ops_no_task_token | 3 | 0.9973 | 0.9879 | 0.0076 | 0.0073 | 0.0031 | 0.0044 |
| interleaved10_partitioned_ops | 3 | 0.9985 | 0.9914 | 0.0077 | 0.0073 | 0.0026 | 0.0045 |
| interleaved20_partitioned_ops | 3 | 1 | 0.9995 | 0.0078 | 0.0075 | 0.0020 | 0.0063 |

### Main takeaways
- The clearest grokking-like delayed generalization on the normal split appeared in `contiguous_partitioned_ops` around `train_fraction=0.3`.
- Reverse-set and cross-pair transfer stayed near chance across all partitioned and interleaved variants.
- Removing the task token did not rescue reverse/cross transfer.
- Interleaving improved ordinary test accuracy but not compositional transfer.
- `contiguous_both_ops_within_set` showed high in-distribution accuracy but essentially zero cross-pair transfer.
- `contiguous_both_ops_all_pairs` was the only scenario with strong cross-pair transfer, especially for multiplication.

### Detailed notes
See the experiment-level file in the archive folder for the full per-seed run table.

## Experiment 3: 39-run A100 mod-131 four-subset batch
Archive: `outputs/old_results/study3_39runs_mod131_fourset_a100_20260424`

### What it tested
This batch replaced the earlier default 2-subset suite with a harder 4-subset / 4-operator setup. The main question was whether the model could generalize to the missing operator for each subset and whether scaling the structural complexity exposed deeper failures of abstraction.

### Batch structure
- `13` configs x `3` seeds = `39` runs
- scenario families:
  - `four_set_missing_ops_within_set`
  - `four_set_all_ops_within_set`
  - `four_set_all_ops_all_pairs`
  - `whole_set_operator_complement`
  - `whole_set_pair_split_both_ops`

### Averaged results by config
| scenario_name | lr | weight_decay | batch_size | train_fraction | use_task_token | train_acc | test_acc | reverse_add_acc | reverse_sub_acc | reverse_mul_acc | reverse_div_acc | cross_add_acc | cross_sub_acc | cross_mul_acc | cross_div_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| four_set_all_ops_all_pairs | 0.0010 | 1 | 256 | 0.5000 | True | 0.3735 | 0.3461 |  |  |  |  | 0.4692 | 0.4634 | 0.2567 | 0.2472 |
| four_set_all_ops_within_set | 0.0010 | 1 | 256 | 0.5000 | True | 0.9916 | 0.9631 |  |  |  |  | 0.0018 | 0.0027 | 0.0157 | 0.0080 |
| four_set_missing_ops_within_set | 0.0003 | 1 | 256 | 0.5000 | True | 0.9994 | 0.9858 | 0.0373 | 0.1243 | 0.4775 | 0.4189 | 0.0239 | 0.0068 | 0.0152 | 0.0075 |
| four_set_missing_ops_within_set | 0.0010 | 0.5000 | 256 | 0.5000 | True | 0.9950 | 0.9812 | 0.0254 | 0.2632 | 0.8733 | 0.8467 | 0.0028 | 0.0009 | 0.0155 | 0.0090 |
| four_set_missing_ops_within_set | 0.0010 | 1 | 128 | 0.5000 | True | 0.9460 | 0.7535 | 0.0217 | 0.0306 | 0.0673 | 0.1624 | 0.0083 | 0.0080 | 0.0164 | 0.0089 |
| four_set_missing_ops_within_set | 0.0010 | 1 | 256 | 0.1000 | True | 0.9894 | 0.0218 | 0.0070 | 0.0144 | 0.0092 | 0.0065 | 0.0040 | 0.0028 | 0.0188 | 0.0123 |
| four_set_missing_ops_within_set | 0.0010 | 1 | 256 | 0.1500 | True | 1 | 0.0181 | 0.0064 | 0.0181 | 0.0067 | 0.0088 | 0.0036 | 0.0020 | 0.0191 | 0.0114 |
| four_set_missing_ops_within_set | 0.0010 | 1 | 256 | 0.2000 | True | 0.9869 | 0.0268 | 0.0049 | 0.0230 | 0.0073 | 0.0124 | 0.0033 | 0.0011 | 0.0185 | 0.0113 |
| four_set_missing_ops_within_set | 0.0010 | 1 | 256 | 0.2500 | True | 0.9847 | 0.0343 | 0.0077 | 0.0171 | 0.0040 | 0.0124 | 0.0030 | 0.0013 | 0.0190 | 0.0116 |
| four_set_missing_ops_within_set | 0.0010 | 1 | 256 | 0.3000 | True | 0.9979 | 0.1255 | 0.0067 | 0.0196 | 0.0101 | 0.0107 | 0.0024 | 0.0008 | 0.0187 | 0.0108 |
| four_set_missing_ops_within_set | 0.0010 | 1 | 256 | 0.5000 | True | 0.9994 | 0.9826 | 0.0193 | 0.0511 | 0.7190 | 0.5072 | 0.0023 | 0.0020 | 0.0159 | 0.0093 |
| whole_set_operator_complement | 0.0010 | 1 | 256 | 0.5000 | True | 0.3290 | 0.3078 |  |  |  |  |  |  |  |  |
| whole_set_pair_split_both_ops | 0.0010 | 1 | 256 | 0.5000 | True | 0.7618 | 0.7430 |  |  |  |  |  |  |  |  |

### Scenario-level summary
| scenario_name | runs | train_acc | test_acc | reverse_add_acc | reverse_sub_acc | reverse_mul_acc | reverse_div_acc | cross_add_acc | cross_sub_acc | cross_mul_acc | cross_div_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| four_set_all_ops_all_pairs | 3 | 0.3735 | 0.3461 |  |  |  |  | 0.4692 | 0.4634 | 0.2567 | 0.2472 |
| four_set_all_ops_within_set | 3 | 0.9916 | 0.9631 |  |  |  |  | 0.0018 | 0.0027 | 0.0157 | 0.0080 |
| four_set_missing_ops_within_set | 27 | 0.9887 | 0.4366 | 0.0152 | 0.0624 | 0.2416 | 0.2207 | 0.0060 | 0.0028 | 0.0175 | 0.0102 |
| whole_set_operator_complement | 3 | 0.3290 | 0.3078 |  |  |  |  |  |  |  |  |
| whole_set_pair_split_both_ops | 3 | 0.7618 | 0.7430 |  |  |  |  |  |  |  |  |

### Main takeaways
- The 4-subset missing-operator setting was much harder than the earlier 2-subset version.
- For the main ablation, normal generalization remained weak through `train_fraction <= 0.3` and only became strong at `train_fraction=0.5`.
- Reverse transfer was highly operator-dependent:
  - multiplication and division transferred much better than addition and subtraction.
- `four_set_all_ops_within_set` reproduced the within-set entanglement result from the 2-subset batch: strong in-distribution performance, near-zero cross-pair transfer.
- `four_set_all_ops_all_pairs` underfit and was unstable across seeds at `500k` steps.
- `whole_set_operator_complement` performed poorly.
- `whole_set_pair_split_both_ops` was unstable across seeds.

### Grokking-style milestone summary
| seed | lr | weight_decay | batch_size | train_fraction | train_hits_99_step | test_hits_50_step | test_hits_90_step | test_hits_95_step | final_test_acc | final_reverse_add_acc | final_reverse_sub_acc | final_reverse_mul_acc | final_reverse_div_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0010 | 1 | 256 | 0.1000 | 1500 |  |  |  | 0.0200 | 0.0101 | 0.0110 | 0.0110 | 0.0068 |
| 1 | 0.0010 | 1 | 256 | 0.1000 | 4000 |  |  |  | 0.0264 | 0.0037 | 0.0202 | 0.0083 | 0.0059 |
| 2 | 0.0010 | 1 | 256 | 0.1000 | 3000 |  |  |  | 0.0190 | 0.0073 | 0.0119 | 0.0083 | 0.0068 |
| 0.0000 | 0.0010 | 1 | 256 | 0.1500 | 500 |  |  |  | 0.0168 | 0.0055 | 0.0193 | 0.0055 | 0.0107 |
| 1 | 0.0010 | 1 | 256 | 0.1500 | 500 |  |  |  | 0.0224 | 0.0073 | 0.0193 | 0.0064 | 0.0059 |
| 2 | 0.0010 | 1 | 256 | 0.1500 | 500 |  |  |  | 0.0150 | 0.0064 | 0.0156 | 0.0083 | 0.0098 |
| 0.0000 | 0.0010 | 1 | 256 | 0.2000 | 5500 |  |  |  | 0.0317 | 0.0046 | 0.0211 | 0.0083 | 0.0137 |
| 1 | 0.0010 | 1 | 256 | 0.2000 | 1000 |  |  |  | 0.0246 | 0.0064 | 0.0202 | 0.0037 | 0.0117 |
| 2 | 0.0010 | 1 | 256 | 0.2000 | 3000 |  |  |  | 0.0240 | 0.0037 | 0.0275 | 0.0101 | 0.0117 |

### Detailed notes
See the experiment-level file in the archive folder for the full per-seed run table.

## Cross-Experiment Comparison
### How the experiments relate
- The 18-run batch is the earliest, more confounded version of the Study 3 hypothesis.
- The 42-run batch is the main two-subset study that established the gap between ordinary test generalization and true transfer.
- The 39-run batch scales the structural difficulty from 2 subsets / 2 operators to 4 subsets / 4 operators and shows the abstraction problem becomes much harder.

### High-level conclusion
Across the retained Study 3 batches, the strongest recurring result is:
- ordinary held-out test accuracy can become very high
- but reverse-task and cross-support transfer often remain at or near chance

This indicates the models frequently learn training-support-specific solutions rather than a single clean arithmetic abstraction that is invariant to subset assignment, operator assignment, or pairing structure.
