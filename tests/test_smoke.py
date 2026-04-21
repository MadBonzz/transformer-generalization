from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grokking_transformer.data import ModularAdditionDataset, create_data_splits
from grokking_transformer.mlp import MLPConfig, ModularMLP
from grokking_transformer.model import GrokkingTransformer, TransformerConfig
from grokking_transformer.train_utils import evaluate, train_step


class GrokkingTransformerSmokeTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        config = TransformerConfig(vocab_size=114)
        model = GrokkingTransformer(config)
        tokens = torch.tensor([[1, 2, 113], [5, 7, 113]], dtype=torch.long)
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 114))

    def test_small_batch_can_overfit(self) -> None:
        torch.manual_seed(0)
        train_split, _ = create_data_splits(prime=13, train_fraction=0.25, seed=0)
        subset_inputs = train_split.inputs[:32]
        subset_targets = train_split.targets[:32]
        dataset = ModularAdditionDataset(subset_inputs, subset_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        config = TransformerConfig(vocab_size=14, d_model=64, n_heads=4, d_head=16, d_mlp=128)
        model = GrokkingTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        device = torch.device("cpu")

        for _ in range(200):
            for batch in loader:
                train_step(model, batch, optimizer, device, target_vocab_size=13)

        metrics = evaluate(model, loader, device, target_vocab_size=13)
        self.assertGreaterEqual(metrics.accuracy, 0.95)


class ModularMLPSmokeTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        config = MLPConfig(prime=13, hidden_dim=64)
        model = ModularMLP(config)
        tokens = torch.tensor([[1, 2, 13], [5, 7, 13]], dtype=torch.long)
        logits = model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 13))

    def test_small_batch_can_overfit(self) -> None:
        torch.manual_seed(0)
        train_split, _ = create_data_splits(prime=13, train_fraction=0.25, seed=0)
        subset_inputs = train_split.inputs[:32]
        subset_targets = train_split.targets[:32]
        dataset = ModularAdditionDataset(subset_inputs, subset_targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = ModularMLP(MLPConfig(prime=13, hidden_dim=128))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        device = torch.device("cpu")

        for _ in range(300):
            for batch in loader:
                train_step(
                    model,
                    batch,
                    optimizer,
                    device,
                    target_vocab_size=13,
                    loss_type="mse_one_hot",
                )

        metrics = evaluate(model, loader, device, target_vocab_size=13, loss_type="mse_one_hot")
        self.assertGreaterEqual(metrics.accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()
