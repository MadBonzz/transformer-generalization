from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from grokking_transformer.data import ModularAdditionDataset, create_data_splits
from grokking_transformer.mlp import MLPConfig, ModularMLP
from grokking_transformer.model import GrokkingTransformer, TransformerConfig
from grokking_transformer.train_utils import evaluate, train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modular-addition model.")
    parser.add_argument("--model", choices=("transformer", "mlp"), default="transformer")
    parser.add_argument("--prime", type=int, default=113)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_split, test_split = create_data_splits(
        prime=args.prime,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    train_dataset = ModularAdditionDataset(train_split.inputs, train_split.targets)
    test_dataset = ModularAdditionDataset(test_split.inputs, test_split.targets)

    train_batch_size = len(train_dataset) if args.full_batch else args.batch_size
    test_batch_size = len(test_dataset) if args.full_batch else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    if args.model == "transformer":
        config = TransformerConfig(vocab_size=args.prime + 1)
        model = GrokkingTransformer(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98),
        )
        loss_type = "cross_entropy"
        print(
            f"config: d_model={config.d_model} n_heads={config.n_heads} "
            f"d_head={config.d_head} d_mlp={config.d_mlp} n_layers={config.n_layers}"
        )
    else:
        config = MLPConfig(prime=args.prime, hidden_dim=args.mlp_hidden_dim)
        model = ModularMLP(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_type = "mse_one_hot"
        print(f"config: hidden_dim={config.hidden_dim} activation=quadratic n_layers=2")

    print(f"device={device} train_examples={len(train_dataset)} test_examples={len(test_dataset)}")
    print(f"model={args.model} batch_size={train_batch_size} loss_type={loss_type}")

    step = 0
    while step < args.steps:
        for batch in train_loader:
            step += 1
            train_loss = train_step(
                model,
                batch,
                optimizer,
                device,
                target_vocab_size=args.prime,
                loss_type=loss_type,
            )

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                train_metrics = evaluate(
                    model,
                    train_loader,
                    device,
                    target_vocab_size=args.prime,
                    loss_type=loss_type,
                )
                test_metrics = evaluate(
                    model,
                    test_loader,
                    device,
                    target_vocab_size=args.prime,
                    loss_type=loss_type,
                )
                print(
                    f"step={step:04d} train_loss={train_loss:.4f} "
                    f"train_acc={train_metrics.accuracy:.3f} test_acc={test_metrics.accuracy:.3f}"
                )

            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
