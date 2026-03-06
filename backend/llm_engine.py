"""
MLCopilot AI — LLM Root-Cause Explanation Engine  (backend/llm_engine.py)
=========================================================================
Generates human-readable explanations for detected ML issues using an LLM.

Supported providers (tried in priority order)
----------------------------------------------
1. Amazon Bedrock  — set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
                     Uses your AWS credits — recommended for AWS deployment.
                     Set BEDROCK_MODEL_ID (default: amazon.titan-text-express-v1)
2. OpenAI          — set OPENAI_API_KEY
3. Anthropic       — set ANTHROPIC_API_KEY
4. Fallback        — built-in rule-based explanation (no key needed)

The caller (main.py) always gets a plain string back regardless of provider.
"""

from __future__ import annotations
import os
import json

# ── Optional imports (graceful fallback if not installed) ────────────────────
try:
    import boto3 as _boto3
except ImportError:
    _boto3 = None

try:
    import openai as _openai
except ImportError:
    _openai = None

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

# ── Built-in fallback explanations ────────────────────────────────────────────
_FALLBACK_EXPLANATIONS: dict[str, str] = {
    "Exploding Gradients": (
        "Exploding gradients occur when the gradient norm grows very large during "
        "back-propagation. This is typically caused by a learning rate that is too "
        "high, missing gradient clipping, or improper weight initialisation. "
        "The fix is to clip gradients and reduce the learning rate."
    ),
    "Vanishing Gradients": (
        "Vanishing gradients occur when gradients become close to zero, preventing "
        "weight updates in early layers. Common causes include saturating activations "
        "(sigmoid/tanh) and very deep networks without residual connections. "
        "Switch to ReLU-family activations and add skip connections."
    ),
    "Overfitting": (
        "Overfitting means the model is memorising training data instead of learning "
        "general patterns. Signs: train loss keeps falling while val loss rises. "
        "Add Dropout, use weight decay, apply data augmentation, or reduce model size."
    ),
    "Underfitting": (
        "Underfitting means the model lacks capacity or training time to fit the data. "
        "Both train and val accuracy remain low. Increase model size, learning rate, "
        "or the number of training epochs."
    ),
    "Loss Stagnation": (
        "Loss stagnation means the model is no longer making meaningful progress. "
        "It may be stuck in a local minimum or saddle point. Try a learning-rate "
        "scheduler, a different optimiser (AdamW, SGD+momentum), or check for "
        "data-quality issues."
    ),
}


# ── Public interface ──────────────────────────────────────────────────────────

def generate_explanation(issue: dict, latest_metrics: dict) -> str:
    """
    Return a human-readable explanation for a detected issue.

    Tries (in order):
      1. Amazon Bedrock (if AWS credentials are set — uses your AWS credits)
      2. OpenAI  (if OPENAI_API_KEY is set)
      3. Anthropic (if ANTHROPIC_API_KEY is set)
      4. Built-in fallback explanation

    Parameters
    ----------
    issue : dict
        Issue dict produced by analyzer.py  (keys: issue, severity, reason, suggestions)
    latest_metrics : dict
        The most recent epoch's metric snapshot (for context in the prompt)
    """
    # 1. Try Amazon Bedrock first (AWS credits)
    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    if aws_key and _boto3:
        try:
            return _call_bedrock(issue, latest_metrics)
        except Exception:
            pass

    # 2. Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and _openai:
        try:
            return _call_openai(issue, latest_metrics, openai_key)
        except Exception:
            pass

    # 3. Try Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key and _anthropic:
        try:
            return _call_anthropic(issue, latest_metrics, anthropic_key)
        except Exception:
            pass

    # 4. Fallback
    return _fallback_explanation(issue)


# ── Provider implementations ──────────────────────────────────────────────────

def _build_prompt(issue: dict, metrics: dict) -> str:
    return (
        "You are an expert ML engineer. A training run has produced this issue:\n\n"
        f"Issue: {issue.get('issue')}\n"
        f"Severity: {issue.get('severity')}\n"
        f"Reason: {issue.get('reason')}\n"
        f"Latest metrics: epoch={metrics.get('epoch')}, "
        f"train_loss={metrics.get('train_loss')}, "
        f"val_loss={metrics.get('val_loss')}, "
        f"accuracy={metrics.get('accuracy')}, "
        f"gradient_norm={metrics.get('gradient_norm')}\n\n"
        "In 3-4 sentences, explain:\n"
        "1. Why this issue happens.\n"
        "2. What impact it has on the model.\n"
        "3. The single most important fix.\n"
        "Be concise, practical, and avoid jargon."
    )


def _call_bedrock(issue: dict, metrics: dict) -> str:
    """Call Amazon Bedrock — uses AWS credits, no extra cost beyond what you pay AWS."""
    region     = os.getenv("AWS_REGION", "us-east-1")
    model_id   = os.getenv("BEDROCK_MODEL_ID", "amazon.titan-text-express-v1")
    prompt     = _build_prompt(issue, metrics)

    client = _boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token     = os.getenv("AWS_SESSION_TOKEN"),  # optional, for temp creds
    )

    # Titan text model body format
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature":   0.3,
            "topP":          0.9,
        },
    })

    response = client.invoke_model(
        modelId     = model_id,
        body        = body,
        contentType = "application/json",
        accept      = "application/json",
    )
    result = json.loads(response["body"].read())
    return result["results"][0]["outputText"].strip()


def _call_openai(issue: dict, metrics: dict, api_key: str) -> str:
    client = _openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert ML debugging assistant."},
            {"role": "user",   "content": _build_prompt(issue, metrics)},
        ],
        max_tokens=250,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(issue: dict, metrics: dict, api_key: str) -> str:
    client = _anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=250,
        messages=[
            {"role": "user", "content": _build_prompt(issue, metrics)},
        ],
    )
    return message.content[0].text.strip()


def _fallback_explanation(issue: dict) -> str:
    issue_name = issue.get("issue", "")
    return _FALLBACK_EXPLANATIONS.get(
        issue_name,
        (
            f"{issue_name} was detected. Check the reason field for evidence and "
            "follow the suggested fixes to resolve the problem."
        ),
    )
