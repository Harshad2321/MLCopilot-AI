"""
MLCopilot AI — Example Training Script  (training_example/train_model.py)
=========================================================================
Demonstrates how to integrate the MLCopilot SDK into any PyTorch training loop.

Four scenarios are provided — each intentionally triggers a different ML issue
so you can watch MLCopilot detect and explain it in real time.

Usage
-----
    # From the project root:
    python training_example/train_model.py --scenario exploding_gradients
    python training_example/train_model.py --scenario overfitting
    python training_example/train_model.py --scenario vanishing_gradients
    python training_example/train_model.py --scenario healthy

    # Point at a remote backend:
    python training_example/train_model.py --scenario overfitting --api-url https://your-server.com
"""

import argparse
import math
import sys
import os

# Allow imports from the project root when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sdk.mlcopilot_logger import MLCopilotLogger


# ── Model definitions ─────────────────────────────────────────────────────────

class SimpleNet(nn.Module):
    """Configurable fully-connected network for classification."""

    def __init__(self, input_size=20, hidden_size=64, num_layers=3,
                 num_classes=2, activation="relu", dropout=0.0):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "leaky": nn.LeakyReLU}[activation]

        layers = [nn.Linear(input_size, hidden_size), act_fn()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), act_fn()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Scenario presets ──────────────────────────────────────────────────────────

SCENARIOS = {
    "exploding_gradients": dict(
        lr=0.5, epochs=30, samples=500, batch_size=32,
        hidden_size=64, num_layers=2, activation="relu",
        clip_grad=False,
        description="High LR (0.5) with no gradient clipping → gradient norms explode.",
    ),
    "overfitting": dict(
        lr=0.001, epochs=60, samples=60, batch_size=16,
        hidden_size=512, num_layers=5, activation="relu",
        clip_grad=False,
        description="Large model + tiny dataset → model memorises training data.",
    ),
    "vanishing_gradients": dict(
        lr=0.001, epochs=40, samples=500, batch_size=32,
        hidden_size=64, num_layers=8, activation="sigmoid",
        clip_grad=False,
        description="Deep network with sigmoid activations → gradients shrink to zero.",
    ),
    "healthy": dict(
        lr=0.001, epochs=30, samples=1000, batch_size=64,
        hidden_size=128, num_layers=3, activation="relu",
        clip_grad=True,
        description="Well-configured run — no issues expected.",
    ),
}


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(scenario_name: str, api_url: str):
    cfg = SCENARIOS[scenario_name]
    print(f"\n{'='*60}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Purpose  : {cfg['description']}")
    print(f"  Epochs   : {cfg['epochs']}   LR: {cfg['lr']}")
    print(f"{'='*60}\n")

    # ── Synthetic dataset ─────────────────────────────────────────────────────
    torch.manual_seed(42)
    n_train = cfg["samples"]
    n_val   = max(50, n_train // 5)
    n_feat  = 20

    X_train = torch.randn(n_train, n_feat)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).long()
    X_val   = torch.randn(n_val, n_feat)
    y_val   = (X_val[:, 0] + X_val[:, 1] > 0).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),   batch_size=128)

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = SimpleNet(
        input_size=n_feat,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        activation=cfg["activation"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    # ── MLCopilot SDK integration ─────────────────────────────────────────────
    logger = MLCopilotLogger(
        run_id=f"{scenario_name}_{torch.randint(1000,9999,(1,)).item()}",
        api_url=api_url,
        experiment_name=scenario_name,
    )

    # ── Per-epoch training ────────────────────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if math.isnan(loss.item()):
                print("[MLCopilot] NaN loss detected — stopping training.")
                logger.finish()
                return

            loss.backward()

            # Compute gradient norm before optional clipping
            grad_norm = _compute_grad_norm(model)

            if cfg.get("clip_grad"):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y_batch).sum().item()
            total      += X_batch.size(0)

        train_loss = total_loss / total
        train_acc  = correct    / total

        # --- Validate ---
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits     = model(X_batch)
                val_loss_total += criterion(logits, y_batch).item() * X_batch.size(0)
                val_correct    += (logits.argmax(1) == y_batch).sum().item()
                val_total      += X_batch.size(0)

        val_loss = val_loss_total / val_total
        val_acc  = val_correct    / val_total

        # --- Log to MLCopilot ---
        logger.log(
            epoch         = epoch,
            train_loss    = round(train_loss, 6),
            val_loss      = round(val_loss, 6),
            accuracy      = round(train_acc, 4),
            learning_rate = cfg["lr"],
            gradient_norm = round(grad_norm, 6),
        )

    logger.finish()


def _compute_grad_norm(model: nn.Module) -> float:
    """Return the global L2 gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLCopilot AI — Example Training Script")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="healthy",
        help="Training scenario to run (default: healthy)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="MLCopilot backend URL (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    run_training(args.scenario, args.api_url)


if __name__ == "__main__":
    main()
