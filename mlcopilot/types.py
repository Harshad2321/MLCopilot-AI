"""
MLCopilot Type Definitions
Core data structures, enums, and constants for the monitoring system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============================================================================
# ENUMS
# ============================================================================

class AnomalyType(Enum):
    """Types of training anomalies that can be detected."""
    EXPLODING_GRADIENTS = "exploding_gradients"
    VANISHING_GRADIENTS = "vanishing_gradients"
    LOSS_DIVERGENCE = "loss_divergence"
    LOSS_PLATEAU = "loss_plateau"
    NAN_LOSS = "nan_loss"
    OVERFITTING = "overfitting"
    LR_INSTABILITY = "lr_instability"


class Severity(Enum):
    """Severity level of detected issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CauseCategory(Enum):
    """Root cause categories for training failures."""
    DATA_ISSUE = "data_issue"
    MODEL_ARCHITECTURE = "model_architecture"
    HYPERPARAMETER = "hyperparameter"
    OPTIMIZATION = "optimization"
    NUMERICAL_INSTABILITY = "numerical_instability"
    UNKNOWN = "unknown"


class Priority(Enum):
    """Priority level for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MetricSnapshot:
    """Single point-in-time snapshot of training metrics."""
    epoch: int
    batch: int
    loss: float
    grad_norm: float
    learning_rate: float
    param_mean: float
    param_std: float
    param_max: float
    timestamp: float
    
    # Optional validation metrics
    val_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'epoch': self.epoch,
            'batch': self.batch,
            'loss': self.loss,
            'grad_norm': self.grad_norm,
            'learning_rate': self.learning_rate,
            'param_mean': self.param_mean,
            'param_std': self.param_std,
            'param_max': self.param_max,
            'timestamp': self.timestamp,
            'val_loss': self.val_loss
        }


@dataclass
class DetectionResult:
    """Result from anomaly detection."""
    anomaly_type: AnomalyType
    confidence: float  # 0.0 to 1.0
    severity: Severity
    detected_at_epoch: int
    detected_at_batch: int
    description: str
    metric_snapshot: MetricSnapshot
    raw_values: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'anomaly_type': self.anomaly_type.value,
            'confidence': self.confidence,
            'severity': self.severity.value,
            'detected_at_epoch': self.detected_at_epoch,
            'detected_at_batch': self.detected_at_batch,
            'description': self.description,
            'metric_snapshot': self.metric_snapshot.to_dict(),
            'raw_values': self.raw_values
        }


@dataclass
class Diagnosis:
    """Root cause analysis result."""
    detection: DetectionResult
    cause_category: CauseCategory
    primary_cause: str
    contributing_factors: List[str]
    reasoning: str
    model_context: Dict[str, Any]
    optimizer_context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'detection': self.detection.to_dict(),
            'cause_category': self.cause_category.value,
            'primary_cause': self.primary_cause,
            'contributing_factors': self.contributing_factors,
            'reasoning': self.reasoning,
            'model_context': self.model_context,
            'optimizer_context': self.optimizer_context
        }


@dataclass
class Recommendation:
    """Actionable recommendation to fix training issue."""
    priority: Priority
    category: str
    action: str
    current_value: Optional[str]
    suggested_value: Optional[str]
    reasoning: str
    code_example: Optional[str] = None
    expected_impact: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'priority': self.priority.value,
            'category': self.category,
            'action': self.action,
            'current_value': self.current_value,
            'suggested_value': self.suggested_value,
            'reasoning': self.reasoning,
            'code_example': self.code_example,
            'expected_impact': self.expected_impact
        }


# ============================================================================
# DETECTION CONSTANTS
# ============================================================================

class DetectionThresholds:
    """Thresholds for anomaly detection."""
    
    # Gradient thresholds
    EXPLODING_GRAD_THRESHOLD = 10.0
    VANISHING_GRAD_THRESHOLD = 1e-7
    GRAD_MULTIPLIER_THRESHOLD = 3.0  # For comparing to moving average
    
    # Loss thresholds
    LOSS_DIVERGENCE_MULTIPLIER = 2.0  # 2x initial loss
    LOSS_PLATEAU_THRESHOLD = 0.001  # Minimum change
    LOSS_PLATEAU_WINDOW = 50  # Number of batches
    
    # Learning rate thresholds
    HIGH_LR_THRESHOLD = 0.1
    LOW_LR_THRESHOLD = 1e-6
    
    # Overfitting thresholds
    OVERFITTING_GAP_THRESHOLD = 0.5  # train_loss + 0.5 < val_loss
    OVERFITTING_RATIO_THRESHOLD = 1.5  # val_loss / train_loss > 1.5
    
    # General
    CONFIDENCE_HIGH = 0.8
    CONFIDENCE_MEDIUM = 0.5
    CONFIDENCE_LOW = 0.3


class MonitoringConfig:
    """Configuration for monitoring behavior."""
    
    # Buffer settings
    MAX_METRICS_BUFFER = 1000  # Keep last N metric snapshots
    
    # Check frequency
    CHECK_INTERVAL_BATCHES = 10  # Check for anomalies every N batches
    
    # Moving average window
    MOVING_AVG_WINDOW = 20
    
    # Minimum batches before detection
    WARMUP_BATCHES = 5  # Don't detect issues in first N batches


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_confidence(value: float, threshold: float, 
                        inverse: bool = False) -> float:
    """
    Calculate confidence score based on how far value exceeds threshold.
    
    Confidence ranges from 0.0 (at threshold) to 1.0 (far beyond threshold).
    
    Formula (non-inverse): confidence = min(1.0, (value/threshold - 1) / 2)
    - At threshold: confidence = 0.0
    - At 2x threshold: confidence = 0.5
    - At 3x threshold: confidence = 1.0
    
    Args:
        value: The measured value
        threshold: The threshold value
        inverse: If True, confidence increases as value < threshold
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if inverse:
        if value >= threshold:
            return 0.0
        ratio = value / threshold
        # Confidence = 1.0 when value = 0, 0.0 when value = threshold
        confidence = 1.0 - ratio
    else:
        if value <= threshold:
            return 0.0
        ratio = value / threshold
        # Maps ratio=1.0 → 0.0, ratio=3.0 → 1.0
        confidence = min(1.0, (ratio - 1.0) / 2.0)
    
    return max(0.0, min(1.0, confidence))


def determine_severity(confidence: float) -> Severity:
    """Determine severity level based on confidence score."""
    if confidence >= 0.9:
        return Severity.CRITICAL
    elif confidence >= 0.7:
        return Severity.HIGH
    elif confidence >= 0.5:
        return Severity.MEDIUM
    else:
        return Severity.LOW
