"""
MLCopilot AI - Suggestion Engine
Generates fix suggestions, parameter changes, code snippets, and explanations.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Suggestion knowledge base: maps (problem, cause) → actionable fixes
# ---------------------------------------------------------------------------
SUGGESTION_DB = {
    "Exploding Gradients": {
        "general": {
            "fixes": [
                "Reduce learning rate (try 0.0001 or 0.0003)",
                "Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
                "Add Batch Normalization layers",
                "Use a learning rate scheduler with warmup",
            ],
            "code_suggestion": """# Add gradient clipping to your training loop:
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()""",
            "param_changes": {"learning_rate": 0.0003, "max_grad_norm": 1.0},
            "explanation": (
                "Large gradients are destabilizing the training process. "
                "Reducing the learning rate lowers the magnitude of parameter updates. "
                "Gradient clipping caps the norm of gradients to prevent explosive growth. "
                "Batch Normalization normalizes layer inputs, smoothing the loss landscape."
            ),
        },
    },
    "Vanishing Gradients": {
        "general": {
            "fixes": [
                "Replace sigmoid/tanh with ReLU or LeakyReLU activations",
                "Add residual/skip connections",
                "Use proper weight initialization (Xavier or Kaiming)",
                "Reduce network depth or add Batch Normalization",
            ],
            "code_suggestion": """# Replace activation function:
# Before: self.activation = nn.Sigmoid()
self.activation = nn.LeakyReLU(0.01)

# Add skip connection in forward():
def forward(self, x):
    residual = x
    out = self.layer(x)
    out = out + residual  # skip connection
    return self.activation(out)""",
            "param_changes": {"activation": "LeakyReLU", "init": "kaiming_normal"},
            "explanation": (
                "Gradients are becoming too small to effectively update weights. "
                "ReLU-family activations don't saturate for positive inputs, preserving gradients. "
                "Skip connections allow gradients to flow directly through the network. "
                "Proper initialization keeps activations and gradients in a stable range."
            ),
        },
    },
    "Overfitting": {
        "general": {
            "fixes": [
                "Add Dropout (p=0.3 to 0.5) between layers",
                "Add weight decay / L2 regularization (e.g., 1e-4)",
                "Use data augmentation to increase effective dataset size",
                "Reduce model complexity (fewer layers or smaller hidden sizes)",
                "Implement early stopping based on validation loss",
            ],
            "code_suggestion": """# Add dropout:
self.dropout = nn.Dropout(p=0.3)

# In forward():
x = self.dropout(self.fc1(x))

# Add weight decay to optimizer:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Early stopping:
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience:
        print("Early stopping triggered")
        break""",
            "param_changes": {"dropout": 0.3, "weight_decay": 1e-4},
            "explanation": (
                "The model is memorizing training data rather than learning general patterns. "
                "Dropout randomly deactivates neurons during training, preventing co-adaptation. "
                "Weight decay penalizes large weights, encouraging simpler solutions. "
                "Early stopping halts training before the model overfits."
            ),
        },
    },
    "Underfitting": {
        "general": {
            "fixes": [
                "Increase model capacity (more layers or larger hidden sizes)",
                "Increase learning rate (try 0.001 or 0.003)",
                "Train for more epochs",
                "Improve feature engineering or input preprocessing",
                "Reduce regularization if it's too strong",
            ],
            "code_suggestion": """# Increase model capacity:
self.fc1 = nn.Linear(input_size, 256)  # was 64
self.fc2 = nn.Linear(256, 128)         # was 32
self.fc3 = nn.Linear(128, num_classes)

# Increase learning rate:
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)""",
            "param_changes": {"learning_rate": 0.003, "hidden_size": 256},
            "explanation": (
                "The model doesn't have enough capacity or training to learn the data distribution. "
                "Increasing model size gives it more representational power. "
                "A higher learning rate helps the model converge faster. "
                "More training epochs give the optimizer more time to find a good solution."
            ),
        },
    },
    "Loss Stagnation": {
        "general": {
            "fixes": [
                "Increase learning rate or use a learning rate scheduler",
                "Try a different optimizer (Adam, AdamW, SGD with momentum)",
                "Re-initialize weights and restart training",
                "Add learning rate warmup followed by cosine annealing",
            ],
            "code_suggestion": """# Use cosine annealing scheduler:
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# In training loop:
optimizer.step()
scheduler.step()

# Or try AdamW:
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)""",
            "param_changes": {"scheduler": "CosineAnnealingLR", "optimizer": "AdamW"},
            "explanation": (
                "Training progress has stalled - the optimizer may be stuck in a local minimum "
                "or saddle point. A learning rate scheduler can help escape by varying the step size. "
                "Different optimizers explore the loss landscape differently."
            ),
        },
    },
    "Learning Rate Too High": {
        "general": {
            "fixes": [
                "Reduce learning rate by 3–10x (try 0.0003 or 0.0001)",
                "Add learning rate warmup for the first few epochs",
                "Use a scheduler to reduce LR on plateau",
            ],
            "code_suggestion": """# Reduce LR and add scheduler:
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# After validation:
scheduler.step(val_loss)""",
            "param_changes": {"learning_rate": 0.0003},
            "explanation": (
                "The current learning rate causes the optimizer to overshoot good minima. "
                "A lower learning rate produces more stable, gradual updates. "
                "ReduceLROnPlateau automatically decreases LR when progress stalls."
            ),
        },
    },
    "Loss Divergence": {
        "general": {
            "fixes": [
                "Reduce learning rate significantly (try 1e-4)",
                "Add gradient clipping",
                "Check for data preprocessing errors (NaN, very large values)",
                "Use mixed-precision training with loss scaling",
            ],
            "code_suggestion": """# Emergency fix - reduce LR and clip gradients:
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)""",
            "param_changes": {"learning_rate": 0.0001, "max_grad_norm": 0.5},
            "explanation": (
                "Training has diverged, meaning the loss has grown out of control. "
                "This is typically caused by an overly aggressive learning rate or uncontrolled gradients. "
                "Immediately reducing the LR and adding gradient clipping should stabilize training."
            ),
        },
    },
    "NaN/Inf Detected": {
        "general": {
            "fixes": [
                "Check input data for NaN or extremely large values",
                "Add gradient clipping with a conservative norm (0.5)",
                "Reduce learning rate significantly",
                "Add epsilon to division and log operations",
                "Use torch.nan_to_num() to sanitize outputs",
            ],
            "code_suggestion": """# Sanitize inputs:
X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

# Add epsilon to prevent log(0):
loss = -torch.log(predictions + 1e-8)

# Clip gradients conservatively:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)""",
            "param_changes": {"learning_rate": 0.0001, "max_grad_norm": 0.5},
            "explanation": (
                "NaN or Inf values in training metrics indicate numerical instability. "
                "This often stems from division by zero, log(0), or extremely large gradient updates. "
                "Sanitizing inputs, adding epsilon for numerical stability, and aggressive gradient "
                "clipping can fix this issue."
            ),
        },
    },
}


class SuggestionEngine:
    """
    Generates fix suggestions and human-readable explanations
    for detected issues and their root causes.
    """

    def __init__(self):
        self.knowledge_base = SUGGESTION_DB

    def generate(
        self, issues: list[dict], root_causes: list[dict]
    ) -> list[dict]:
        """
        Generate suggestions for each detected issue.
        Incorporates root cause information for more targeted advice.
        """
        suggestions = []

        # Group root causes by problem
        causes_by_problem = {}
        for rc in root_causes:
            prob = rc.get("problem", "")
            causes_by_problem.setdefault(prob, []).append(rc)

        seen_problems = set()
        for issue in issues:
            name = issue.get("name", "")
            if name in seen_problems:
                continue
            seen_problems.add(name)

            problem_db = self.knowledge_base.get(name, {})
            general = problem_db.get("general", {})

            if not general:
                # Fallback for unknown issues
                suggestions.append({
                    "problem": name,
                    "severity": issue.get("severity", "medium"),
                    "fixes": ["Review training configuration and data pipeline"],
                    "code_suggestion": "",
                    "param_changes": {},
                    "explanation": issue.get("description", "An issue was detected."),
                    "root_causes": causes_by_problem.get(name, []),
                })
                continue

            prob_causes = causes_by_problem.get(name, [])
            top_cause = prob_causes[0] if prob_causes else None

            suggestion_entry = {
                "problem": name,
                "severity": issue.get("severity", "medium"),
                "fixes": general.get("fixes", []),
                "code_suggestion": general.get("code_suggestion", ""),
                "param_changes": general.get("param_changes", {}),
                "explanation": general.get("explanation", ""),
                "root_causes": [
                    {"cause": rc["cause"], "confidence": rc["confidence"]}
                    for rc in prob_causes
                ],
            }

            if top_cause:
                suggestion_entry["primary_cause"] = top_cause["cause"]
                suggestion_entry["cause_confidence"] = top_cause["confidence"]

            suggestions.append(suggestion_entry)

        return suggestions

    def format_report(self, suggestions: list[dict]) -> str:
        """
        Format suggestions into a human-readable report.
        """
        if not suggestions:
            return "[OK] MLCopilot Analysis: No issues detected. Training looks healthy!"

        lines = []
        lines.append("=" * 60)
        lines.append("  MLCopilot AI - Training Analysis Report")
        lines.append("=" * 60)

        for i, s in enumerate(suggestions, 1):
            severity_icon = {
                "critical": "[CRITICAL]",
                "high": "[HIGH]",
                "medium": "[MEDIUM]",
                "low": "[LOW]",
            }.get(s.get("severity", "medium"), "[?]")

            lines.append(f"\n{severity_icon} Issue #{i}: {s['problem']}")
            lines.append("-" * 40)

            if s.get("root_causes"):
                lines.append("\n  Root Cause Analysis:")
                for rc in s["root_causes"]:
                    lines.append(f"    * {rc['cause']} (confidence: {rc['confidence']:.0%})")

            lines.append("\n  Suggested Fixes:")
            for fix in s.get("fixes", []):
                lines.append(f"    -> {fix}")

            if s.get("param_changes"):
                lines.append("\n  Recommended Parameter Changes:")
                for param, value in s["param_changes"].items():
                    lines.append(f"    {param}: {value}")

            if s.get("code_suggestion"):
                lines.append("\n  Code Suggestion:")
                for code_line in s["code_suggestion"].split("\n"):
                    lines.append(f"    {code_line}")

            lines.append(f"\n  Explanation:")
            # Wrap explanation text
            explanation = s.get("explanation", "")
            words = explanation.split()
            current_line = "    "
            for word in words:
                if len(current_line) + len(word) + 1 > 70:
                    lines.append(current_line)
                    current_line = "    " + word
                else:
                    current_line += " " + word if current_line.strip() else "    " + word
            if current_line.strip():
                lines.append(current_line)

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
