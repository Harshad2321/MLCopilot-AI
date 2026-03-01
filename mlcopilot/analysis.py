"""
MLCopilot Root Cause Analysis
Infers root causes of detected training failures using expert rules.
"""

from typing import Dict, List, Any

from .types import (
    DetectionResult, Diagnosis, AnomalyType, CauseCategory,
    DetectionThresholds
)


class RootCauseAnalyzer:
    """
    Analyzes detected anomalies and infers root causes using rule-based reasoning.
    """
    
    def analyze(self, 
                detection: DetectionResult,
                model_info: Dict[str, Any],
                optimizer_info: Dict[str, Any]) -> Diagnosis:
        """
        Perform root cause analysis on a detected anomaly.
        
        Args:
            detection: DetectionResult from failure detector
            model_info: Model architecture metadata
            optimizer_info: Optimizer configuration
        
        Returns:
            Diagnosis with inferred root cause
        """
        # Route to appropriate analysis function
        analyzer_map = {
            AnomalyType.EXPLODING_GRADIENTS: self._analyze_exploding_gradients,
            AnomalyType.VANISHING_GRADIENTS: self._analyze_vanishing_gradients,
            AnomalyType.LOSS_DIVERGENCE: self._analyze_loss_divergence,
            AnomalyType.LOSS_PLATEAU: self._analyze_loss_plateau,
            AnomalyType.NAN_LOSS: self._analyze_nan_loss,
            AnomalyType.OVERFITTING: self._analyze_overfitting,
        }
        
        analyzer = analyzer_map.get(detection.anomaly_type)
        
        if analyzer:
            return analyzer(detection, model_info, optimizer_info)
        else:
            # Fallback for unknown anomaly types
            return self._analyze_unknown(detection, model_info, optimizer_info)
    
    def _analyze_exploding_gradients(self,
                                     detection: DetectionResult,
                                     model_info: Dict[str, Any],
                                     optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze exploding gradients."""
        lr = optimizer_info.get('learning_rate', 0.0)
        has_normalization = model_info.get('has_normalization', False)
        num_layers = model_info.get('num_layers', 0)
        
        contributing_factors = []
        
        # Check learning rate
        if lr > DetectionThresholds.HIGH_LR_THRESHOLD:
            cause_category = CauseCategory.HYPERPARAMETER
            primary_cause = "Learning rate too high"
            contributing_factors.append(f"Learning rate ({lr}) exceeds recommended threshold")
            reasoning = (
                f"The learning rate ({lr}) is very high, causing weight updates to overshoot. "
                "Large learning rates can cause gradients to explode as the optimizer takes "
                "steps that are too large in parameter space."
            )
        elif not has_normalization:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Missing normalization layers"
            contributing_factors.append("No BatchNorm or LayerNorm detected in model")
            reasoning = (
                "The model lacks normalization layers (BatchNorm/LayerNorm). "
                "Without normalization, activations can grow unbounded through the network, "
                "leading to exploding gradients during backpropagation."
            )
        elif num_layers > 10:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Very deep network without proper architecture"
            contributing_factors.append(f"Model has {num_layers} layers")
            reasoning = (
                f"The model is deep ({num_layers} layers) which can cause gradient instability. "
                "Consider using residual connections (ResNet-style) or proper initialization."
            )
        else:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Optimization instability"
            contributing_factors.append("Gradient clipping may be needed")
            reasoning = (
                "Gradients are exploding during backpropagation. "
                "This could be due to unstable optimization dynamics or numerical issues."
            )
        
        # Additional context
        grad_norm = detection.metric_snapshot.grad_norm
        contributing_factors.append(f"Gradient norm reached {grad_norm:.2f}")
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_vanishing_gradients(self,
                                     detection: DetectionResult,
                                     model_info: Dict[str, Any],
                                     optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze vanishing gradients."""
        lr = optimizer_info.get('learning_rate', 0.0)
        num_layers = model_info.get('num_layers', 0)
        
        contributing_factors = []
        
        if num_layers > 10:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Very deep network causing gradient decay"
            contributing_factors.append(f"Model has {num_layers} layers")
            reasoning = (
                f"The model is very deep ({num_layers} layers). "
                "In deep networks, gradients can diminish exponentially as they propagate "
                "backwards through layers, especially with saturating activation functions like sigmoid/tanh."
            )
        elif lr < DetectionThresholds.LOW_LR_THRESHOLD:
            cause_category = CauseCategory.HYPERPARAMETER
            primary_cause = "Learning rate too small"
            contributing_factors.append(f"Learning rate ({lr:.2e}) is extremely small")
            reasoning = (
                f"The learning rate ({lr:.2e}) is extremely small, resulting in nearly zero gradients. "
                "While this won't prevent learning entirely, it will make training extremely slow."
            )
        else:
            cause_category = CauseCategory.MODEL_ARCHITECTURE
            primary_cause = "Architecture causing gradient flow issues"
            contributing_factors.append("Possible saturating activations or initialization issues")
            reasoning = (
                "Gradients are vanishing during backpropagation. "
                "This often occurs with sigmoid/tanh activations, poor weight initialization, "
                "or missing residual connections in deep networks."
            )
        
        grad_norm = detection.metric_snapshot.grad_norm
        contributing_factors.append(f"Gradient norm dropped to {grad_norm:.2e}")
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_loss_divergence(self,
                                 detection: DetectionResult,
                                 model_info: Dict[str, Any],
                                 optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze loss divergence."""
        lr = optimizer_info.get('learning_rate', 0.0)
        
        contributing_factors = []
        
        if lr > DetectionThresholds.HIGH_LR_THRESHOLD:
            cause_category = CauseCategory.HYPERPARAMETER
            primary_cause = "Learning rate too high causing instability"
            contributing_factors.append(f"Learning rate ({lr}) is very high")
            reasoning = (
                f"The learning rate ({lr}) is too high for stable training. "
                "High learning rates cause the optimizer to overshoot minima, "
                "leading to diverging loss values."
            )
        else:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Training instability"
            contributing_factors.append("Loss diverged from initial value")
            reasoning = (
                "The loss is diverging instead of decreasing. "
                "This indicates fundamental training instability, possibly due to "
                "learning rate, batch size, or data issues."
            )
        
        current_loss = detection.metric_snapshot.loss
        initial_loss = detection.raw_values.get('initial_loss', 0)
        contributing_factors.append(f"Loss increased from {initial_loss:.4f} to {current_loss:.4f}")
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_loss_plateau(self,
                             detection: DetectionResult,
                             model_info: Dict[str, Any],
                             optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze loss plateau."""
        lr = optimizer_info.get('learning_rate', 0.0)
        
        contributing_factors = []
        
        if lr < 1e-5:
            cause_category = CauseCategory.HYPERPARAMETER
            primary_cause = "Learning rate too small"
            contributing_factors.append(f"Learning rate ({lr:.2e}) is very small")
            reasoning = (
                f"The learning rate ({lr:.2e}) is too small for meaningful progress. "
                "While the model is technically training, the updates are too small "
                "to escape local minima or make significant progress."
            )
        else:
            cause_category = CauseCategory.OPTIMIZATION
            primary_cause = "Stuck in local minimum or plateau"
            contributing_factors.append("Loss not changing over extended period")
            reasoning = (
                "The loss has plateaued, indicating the model may be stuck in a local minimum "
                "or a flat region of the loss landscape. Consider learning rate scheduling, "
                "momentum adjustments, or architectural changes."
            )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_nan_loss(self,
                         detection: DetectionResult,
                         model_info: Dict[str, Any],
                         optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze NaN loss."""
        lr = optimizer_info.get('learning_rate', 0.0)
        
        cause_category = CauseCategory.NUMERICAL_INSTABILITY
        primary_cause = "Numerical instability causing NaN"
        
        contributing_factors = [
            "Loss became NaN or Inf",
            "Training has completely diverged"
        ]
        
        if lr > DetectionThresholds.HIGH_LR_THRESHOLD:
            contributing_factors.append(f"Very high learning rate ({lr}) likely contributed")
        
        reasoning = (
            "The loss became NaN (Not a Number) or Inf, indicating severe numerical instability. "
            "This typically results from exploding gradients, division by zero, or log of negative values. "
            "Common causes: extremely high learning rate, missing gradient clipping, or data preprocessing issues."
        )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_overfitting(self,
                            detection: DetectionResult,
                            model_info: Dict[str, Any],
                            optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Analyze overfitting."""
        total_params = model_info.get('total_params', 0)
        
        cause_category = CauseCategory.MODEL_ARCHITECTURE
        primary_cause = "Model overfitting to training data"
        
        contributing_factors = [
            "Large gap between training and validation loss",
        ]
        
        if total_params > 1_000_000:
            contributing_factors.append(f"Large model ({total_params:,} parameters)")
        
        train_loss = detection.raw_values.get('train_loss', 0)
        val_loss = detection.raw_values.get('val_loss', 0)
        gap = detection.raw_values.get('gap', 0)
        
        reasoning = (
            f"Validation loss ({val_loss:.4f}) significantly exceeds training loss ({train_loss:.4f}), "
            f"with a gap of {gap:.4f}. This indicates the model is memorizing the training data "
            "rather than learning generalizable patterns. Consider regularization techniques, "
            "data augmentation, or reducing model capacity."
        )
        
        return Diagnosis(
            detection=detection,
            cause_category=cause_category,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            reasoning=reasoning,
            model_context=model_info,
            optimizer_context=optimizer_info
        )
    
    def _analyze_unknown(self,
                        detection: DetectionResult,
                        model_info: Dict[str, Any],
                        optimizer_info: Dict[str, Any]) -> Diagnosis:
        """Fallback analysis for unknown anomaly types."""
        return Diagnosis(
            detection=detection,
            cause_category=CauseCategory.UNKNOWN,
            primary_cause="Unknown training issue",
            contributing_factors=["Unable to determine specific cause"],
            reasoning="An anomaly was detected but root cause analysis is not available for this type.",
            model_context=model_info,
            optimizer_context=optimizer_info
        )
