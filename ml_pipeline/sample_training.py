"""
MLCopilot AI - Sample Training Pipeline
A simple PyTorch training example that INTENTIONALLY causes problems
so the AI system can detect and diagnose them.

Scenarios:
  1. exploding_gradients  — high LR, no clipping
  2. overfitting          — large model, tiny dataset, many epochs
  3. vanishing_gradients  — deep network with sigmoid activation
  4. healthy              — well-configured baseline for comparison

Usage:
    python sample_training.py --scenario exploding_gradients
    python sample_training.py --scenario overfitting
    python sample_training.py --scenario vanishing_gradients
    python sample_training.py --scenario healthy
"""

import argparse
import math
import time
import sys
import os

# Ensure project root is on the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ml_pipeline.metrics_logger import MetricsLogger


# ============================
# Model Definitions
# ============================

class SimpleNet(nn.Module):
    """A basic feedforward network."""

    def __init__(self, input_dim=20, hidden_dim=64, output_dim=2, depth=3, activation="relu"):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * depth + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "relu":
                    layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LargeOverfitNet(nn.Module):
    """An excessively large model designed to overfit on small data."""

    def __init__(self, input_dim=20, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepSigmoidNet(nn.Module):
    """A deep network with sigmoid activations — causes vanishing gradients."""

    def __init__(self, input_dim=20, output_dim=2, depth=15):
        super().__init__()
        layers = []
        dims = [input_dim] + [32] * depth + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        # Initialize with small weights to encourage vanishing
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ============================
# Data Generation
# ============================

def generate_data(n_samples=1000, input_dim=20, noise=0.1):
    """Generate synthetic binary classification data."""
    X = torch.randn(n_samples, input_dim)
    weights = torch.randn(input_dim)
    logits = X @ weights + noise * torch.randn(n_samples)
    y = (logits > 0).long()
    return X, y


# ============================
# Training Function
# ============================

def compute_grad_norm(model):
    """Compute the total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def train(scenario: str, api_url: str = "http://localhost:8000", offline: bool = False):
    """Run a training loop with the specified scenario."""

    print(f"\n{'='*60}")
    print(f"  MLCopilot AI - Sample Training Pipeline")
    print(f"  Scenario: {scenario}")
    print(f"{'='*60}\n")

    # ---- Scenario configs ----
    configs = {
        "exploding_gradients": {
            "model": SimpleNet(hidden_dim=128, depth=4),
            "lr": 0.5,          # Way too high
            "epochs": 30,
            "n_train": 500,
            "batch_size": 32,
            "description": "High LR with no gradient clipping -> exploding gradients",
        },
        "overfitting": {
            "model": LargeOverfitNet(),
            "lr": 0.001,
            "epochs": 60,
            "n_train": 50,       # Very small dataset
            "batch_size": 16,
            "description": "Large model + tiny dataset -> overfitting",
        },
        "vanishing_gradients": {
            "model": DeepSigmoidNet(depth=15),
            "lr": 0.01,
            "epochs": 30,
            "n_train": 500,
            "batch_size": 32,
            "description": "Deep sigmoid network -> vanishing gradients",
        },
        "healthy": {
            "model": SimpleNet(hidden_dim=64, depth=2),
            "lr": 0.001,
            "epochs": 30,
            "n_train": 1000,
            "batch_size": 64,
            "description": "Well-configured training -> healthy baseline",
        },
    }

    if scenario not in configs:
        print(f"Unknown scenario: {scenario}")
        print(f"Available scenarios: {list(configs.keys())}")
        return

    cfg = configs[scenario]
    model = cfg["model"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    n_train = cfg["n_train"]
    batch_size = cfg["batch_size"]

    print(f"Description: {cfg['description']}")
    print(f"LR: {lr}, Epochs: {epochs}, Train samples: {n_train}\n")

    # ---- Generate data ----
    X_train, y_train = generate_data(n_samples=n_train)
    X_val, y_val = generate_data(n_samples=200)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- MLCopilot Logger ----
    logger = MetricsLogger(
        experiment_name=f"sample_{scenario}",
        api_url=api_url,
        config={
            "scenario": scenario,
            "lr": lr,
            "epochs": epochs,
            "n_train": n_train,
            "batch_size": batch_size,
            "model": model.__class__.__name__,
        },
        offline=offline,
    )

    # ---- Training loop ----
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                grad_norm = float("nan")
                logger.log(
                    epoch=epoch,
                    loss=float("nan"),
                    val_loss=float("nan"),
                    grad_norm=grad_norm,
                    lr=lr,
                    batch_size=batch_size,
                )
                print(f"\n[FAIL] Training diverged at epoch {epoch}! Loss is NaN/Inf.")
                logger.analyze()
                logger.complete()
                return

            loss.backward()

            # NO gradient clipping in problem scenarios
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total
        grad_norm = compute_grad_norm(model)

        # ---- Validation ----
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_total += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        # ---- Log to MLCopilot ----
        logger.log(
            epoch=epoch,
            loss=train_loss,
            val_loss=val_loss,
            accuracy=train_acc,
            val_accuracy=val_acc,
            grad_norm=grad_norm,
            lr=lr,
            batch_size=batch_size,
        )

        # Small delay so it feels like real training
        time.sleep(0.05)

    # ---- Final analysis ----
    print(f"\n{'='*60}")
    print("  Training Complete — Running Final Analysis...")
    print(f"{'='*60}\n")
    result = logger.analyze()
    logger.complete()

    summary = logger.get_summary()
    print(f"\n--- Summary ---")
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Epochs: {summary['total_epochs']}")
    print(f"Issues detected: {summary['issues_found']}")


# ============================
# CLI entry point
# ============================

def main():
    parser = argparse.ArgumentParser(description="MLCopilot Sample Training Pipeline")
    parser.add_argument(
        "--scenario",
        type=str,
        default="exploding_gradients",
        choices=["exploding_gradients", "overfitting", "vanishing_gradients", "healthy"],
        help="Training scenario to simulate",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="MLCopilot backend URL",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run without the backend server",
    )
    args = parser.parse_args()
    train(args.scenario, args.api_url, args.offline)


if __name__ == "__main__":
    main()
