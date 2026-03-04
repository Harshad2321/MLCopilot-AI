"""
MLCopilot AI - Hyperparameter Optimizer
Uses Optuna to suggest optimal hyperparameters for training.
"""

import sys
import os
import math

# Ensure project root is on the import path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import optuna
from optuna.samplers import TPESampler

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
import torch.optim as optim_module
from torch.utils.data import DataLoader, TensorDataset


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimizer.
    Runs a search over learning rate, hidden size, depth, dropout, etc.
    """

    def __init__(
        self,
        input_dim: int = 20,
        output_dim: int = 2,
        X_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        max_epochs: int = 20,
        n_trials: int = 30,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.max_epochs = max_epochs
        self.n_trials = n_trials

    def _build_model(self, trial):
        """Build a model based on trial hyperparameters."""
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
        depth = trial.suggest_int("depth", 1, 5)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "GELU"])

        act_map = {
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "GELU": nn.GELU,
        }

        layers = []
        dims = [self.input_dim] + [hidden_dim] * depth + [self.output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_map[activation]())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def _objective(self, trial):
        """Optuna objective: minimize validation loss."""
        model = self._build_model(trial)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

        if optimizer_name == "Adam":
            optimizer = optim_module.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim_module.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim_module.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        # Train
        model.train()
        for epoch in range(self.max_epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    return float("inf")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Pruning: report intermediate value
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_val)
                val_loss = criterion(val_outputs, self.y_val).item()
            model.train()

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Final validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(self.X_val)
            val_loss = criterion(val_outputs, self.y_val).item()
            _, preds = torch.max(val_outputs, 1)
            val_acc = (preds == self.y_val).float().mean().item()

        return val_loss

    def optimize(self) -> dict:
        """
        Run the hyperparameter optimization.
        Returns the best hyperparameters and trial info.
        """
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )

        print(f"\n[MLCopilot Optimizer] Starting hyperparameter search ({self.n_trials} trials)...\n")

        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\n{'='*50}")
        print(f"  Optimization Complete!")
        print(f"  Best val_loss: {best.value:.4f}")
        print(f"  Best params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        print(f"{'='*50}\n")

        return {
            "best_params": best.params,
            "best_val_loss": best.value,
            "n_trials": self.n_trials,
            "all_trials": [
                {
                    "number": t.number,
                    "value": t.value if t.value is not None else None,
                    "params": t.params,
                    "state": str(t.state),
                }
                for t in study.trials
            ],
        }


def run_optimization():
    """Standalone function to run hyperparameter optimization."""
    # Generate data
    X = torch.randn(1000, 20)
    weights = torch.randn(20)
    y = (X @ weights > 0).long()

    X_train, y_train = X[:800], y[:800]
    X_val, y_val = X[800:], y[800:]

    optimizer = HyperparameterOptimizer(
        input_dim=20,
        output_dim=2,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=15,
        n_trials=25,
    )

    result = optimizer.optimize()
    return result


if __name__ == "__main__":
    run_optimization()
