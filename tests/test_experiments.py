from __future__ import annotations

import json
import shutil
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from uuid import uuid4

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from grokking_transformer.experiment_utils import RunConfig, run_training
from grokking_transformer.mlp import MLPConfig, ModularMLP
from grokking_transformer.model import GrokkingTransformer, TransformerConfig
from grokking_transformer.rl import GRPOConfig, PPOConfig, grpo_update, ppo_update
from grokking_transformer.rewards import compute_reward
from grokking_transformer.tasks import build_range_transfer_task, build_single_operator_task, build_study3_task
from run_range_transfer import build_job_specs


class TaskBuilderTest(unittest.TestCase):
    def test_single_operator_corruption_respects_fraction(self) -> None:
        task = build_single_operator_task(
            modulus=13,
            operator="add",
            train_fraction=0.25,
            seed=0,
            corruption_fraction=0.5,
            corruption_seed=1,
        )
        corruption_rate = task.train.corrupted_mask.float().mean().item()
        self.assertGreater(corruption_rate, 0.4)
        self.assertLess(corruption_rate, 0.6)

    def test_range_transfer_seq_len_and_targets(self) -> None:
        task = build_range_transfer_task(modulus=17, train_fraction=0.3, seed=0, add_offset=0, mul_offset=100)
        self.assertEqual(task.info.seq_len, 3)
        self.assertEqual(task.info.operator_token_ids, {})
        self.assertTrue(task.cross_add.targets.max().item() < task.info.target_vocab_size)
        self.assertTrue(task.cross_mul.targets.max().item() < task.info.target_vocab_size)

    def test_range_transfer_can_keep_input_range_and_change_output_modulus(self) -> None:
        task = build_range_transfer_task(
            modulus=251,
            output_modulus=97,
            train_fraction=0.3,
            seed=0,
            add_offset=0,
            mul_offset=1000,
        )
        self.assertEqual(task.info.seq_len, 3)
        self.assertEqual(task.info.target_vocab_size, 97)
        self.assertGreater(task.info.vocab_size, task.info.target_vocab_size)
        self.assertLess(task.train.targets.max().item(), task.info.target_vocab_size)
        self.assertLess(task.cross_add.targets.max().item(), task.info.target_vocab_size)
        self.assertLess(task.cross_mul.targets.max().item(), task.info.target_vocab_size)

    def test_study3_partitioned_ops_without_task_token_has_seq_len_3(self) -> None:
        task = build_study3_task(
            modulus=17,
            output_modulus=17,
            train_fraction=0.3,
            seed=0,
            scenario="partitioned_ops",
            use_task_token=False,
            add_set_size=9,
        )
        self.assertEqual(task.info.seq_len, 3)
        self.assertEqual(task.info.operator_token_ids, {})
        self.assertIn("reverse_add", task.final_evals)
        self.assertIn("cross_pair_mul", task.final_evals)

    def test_study3_interleaved_partitioned_ops_respects_chunking(self) -> None:
        task = build_study3_task(
            modulus=30,
            output_modulus=30,
            train_fraction=0.3,
            seed=0,
            scenario="interleaved_partitioned_ops",
            use_task_token=True,
            interleave_chunk_size=10,
        )
        eq_token = task.info.eq_token_id
        add_token = task.info.operator_token_ids["add"]
        mul_token = task.info.operator_token_ids["mul"]
        reverse_mul_inputs = {tuple(row.tolist()) for row in task.final_evals["reverse_mul"].inputs}
        cross_pair_add_inputs = {tuple(row.tolist()) for row in task.final_evals["cross_pair_add"].inputs}
        self.assertIn((0, mul_token, 8, eq_token), reverse_mul_inputs)
        self.assertIn((0, mul_token, 23, eq_token), reverse_mul_inputs)
        self.assertNotIn((0, mul_token, 15, eq_token), reverse_mul_inputs)
        self.assertIn((0, add_token, 15, eq_token), cross_pair_add_inputs)


class GRPOSmokeTest(unittest.TestCase):
    def test_grpo_update_runs(self) -> None:
        task = build_single_operator_task(modulus=13, operator="add", train_fraction=0.3, seed=0)
        model = GrokkingTransformer(TransformerConfig(vocab_size=task.info.vocab_size))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = (task.train.inputs[:8], task.train.targets[:8])
        metrics = grpo_update(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=torch.device("cpu"),
            target_vocab_size=task.info.target_vocab_size,
            config=GRPOConfig(n_samples=4, policy_epochs=2, reward_mode="binary"),
        )
        self.assertIn("reward_mean", metrics)


class PPOSmokeTest(unittest.TestCase):
    def test_ppo_update_runs(self) -> None:
        task = build_single_operator_task(modulus=13, operator="add", train_fraction=0.3, seed=0)
        model = ModularMLP(MLPConfig(prime=13, hidden_dim=64))
        value_head = torch.nn.Linear(13, 1)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(value_head.parameters()), lr=1e-3)
        batch = (task.train.inputs[:8], task.train.targets[:8])
        metrics = ppo_update(
            model=model,
            value_head=value_head,
            batch=batch,
            optimizer=optimizer,
            device=torch.device("cpu"),
            target_vocab_size=task.info.target_vocab_size,
            config=PPOConfig(n_samples=4, policy_epochs=2, reward_mode="partial_absolute"),
        )
        self.assertIn("value_loss", metrics)


class RewardTest(unittest.TestCase):
    def test_partial_absolute_reward(self) -> None:
        rewards = compute_reward(
            torch.tensor([0, 4, 9]),
            torch.tensor([0, 7, 9]),
            target_vocab_size=10,
            reward_mode="partial_absolute",
        )
        self.assertAlmostEqual(float(rewards[0].item()), 1.0)
        self.assertAlmostEqual(float(rewards[1].item()), 1.0 - 3.0 / 9.0)
        self.assertAlmostEqual(float(rewards[2].item()), 1.0)

    def test_binary_plus_partial_absolute_reward(self) -> None:
        rewards = compute_reward(
            torch.tensor([0, 4, 9]),
            torch.tensor([0, 7, 9]),
            target_vocab_size=10,
            reward_mode="binary_plus_partial_absolute",
        )
        self.assertAlmostEqual(float(rewards[0].item()), 1.0)
        self.assertAlmostEqual(float(rewards[1].item()), 1.0 - 3.0 / 9.0)
        self.assertAlmostEqual(float(rewards[2].item()), 1.0)


class ProgressTrackingTest(unittest.TestCase):
    def test_run_training_writes_completed_progress_state(self) -> None:
        task = build_single_operator_task(modulus=13, operator="add", train_fraction=0.3, seed=0)
        tmp_dir = Path("outputs") / "_test_runs" / f"progress_{uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
        run_dir = tmp_dir / "progress_run"
        summary_csv_path = tmp_dir / "summary.csv"
        config = RunConfig(
            study_name="progress_test",
            model_type="transformer",
            objective="ce",
            seed=0,
            lr=1e-3,
            weight_decay=0.0,
            batch_size=16,
            max_steps=2,
            eval_every=1,
            log_every=1,
            device="cpu",
            output_dir=str(run_dir),
        )

        result = run_training(
            config=config,
            info=task.info,
            train_dataset=task.train,
            eval_datasets={"train": task.train, "test": task.test},
            summary_csv_path=summary_csv_path,
        )

        progress_path = run_dir / "progress.json"
        self.assertTrue(progress_path.exists())
        progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(progress_payload["status"], "completed")
        self.assertEqual(progress_payload["study_name"], "progress_test")
        self.assertEqual(progress_payload["run_name"], run_dir.name)
        self.assertEqual(progress_payload["step"], 2)
        self.assertEqual(progress_payload["max_steps"], 2)
        self.assertAlmostEqual(float(progress_payload["progress_fraction"]), 1.0)
        self.assertEqual(progress_payload["output_dir"], str(run_dir))
        self.assertEqual(result["completed_steps"], 2)


class Study3ManifestBuilderTest(unittest.TestCase):
    def test_study3_manifest_count_matches_expected_reduced_sweep(self) -> None:
        args = Namespace(
            profile="full10",
            output_root="outputs/test_study3",
            sweep_mode="baseline_ablation",
            scenarios="contiguous_partitioned_ops,contiguous_both_ops_within_set,contiguous_both_ops_all_pairs,contiguous_partitioned_ops_no_task_token,interleaved10_partitioned_ops,interleaved20_partitioned_ops",
            modulus=131,
            output_modulus=None,
            add_set_size=66,
            train_fraction=None,
            train_fractions="",
            lrs="",
            weight_decays="",
            batch_sizes="",
            baseline_lr=1e-3,
            baseline_weight_decay=1.0,
            baseline_batch_size=256,
            baseline_train_fraction=0.5,
            max_steps=100_000,
            eval_every=500,
            log_every=100,
            device="cpu",
            seeds="",
            manifest_out="",
            manifest_only=False,
        )
        jobs = build_job_specs(args)
        self.assertEqual(len(jobs), 42)

    def test_only_no_task_token_ablation_disables_task_token(self) -> None:
        args = Namespace(
            profile="full10",
            output_root="outputs/test_study3",
            sweep_mode="baseline_ablation",
            scenarios="contiguous_partitioned_ops,contiguous_partitioned_ops_no_task_token",
            modulus=131,
            output_modulus=None,
            add_set_size=66,
            train_fraction=None,
            train_fractions="",
            lrs="",
            weight_decays="",
            batch_sizes="",
            baseline_lr=1e-3,
            baseline_weight_decay=1.0,
            baseline_batch_size=256,
            baseline_train_fraction=0.5,
            max_steps=100_000,
            eval_every=500,
            log_every=100,
            device="cpu",
            seeds="0",
            manifest_out="",
            manifest_only=False,
        )
        jobs = build_job_specs(args)
        task_token_flags = {
            job["run_config"]["metadata"]["scenario_name"]: job["task"]["use_task_token"]
            for job in jobs
        }
        self.assertTrue(task_token_flags["contiguous_partitioned_ops"])
        self.assertFalse(task_token_flags["contiguous_partitioned_ops_no_task_token"])


if __name__ == "__main__":
    unittest.main()
