from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.mlp import MLPConfig, ModularMLP
from grokking_transformer.model import GrokkingTransformer, TransformerConfig
from grokking_transformer.rl import GRPOConfig, PPOConfig, grpo_update, ppo_update
from grokking_transformer.rewards import compute_reward
from grokking_transformer.tasks import build_range_transfer_task, build_single_operator_task


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
        self.assertEqual(task.info.seq_len, 4)
        self.assertTrue(task.cross_add.targets.max().item() < task.info.target_vocab_size)
        self.assertTrue(task.cross_mul.targets.max().item() < task.info.target_vocab_size)


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


if __name__ == "__main__":
    unittest.main()
