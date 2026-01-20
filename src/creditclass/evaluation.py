"""
Model evaluation utilities for the CreditClass project.

This module provides functions for computing metrics, confusion matrices,
SHAP values, and model comparisons.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    average: str = "binary",
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a trained model.

    Args:
        model: Trained model with predict and predict_proba methods.
        X_test: Test features.
        y_test: True labels.
        average: Averaging method for multi-class ('binary', 'micro',
            'macro', 'weighted').

    Returns:
        Dictionary containing accuracy, precision, recall, F1, and AUC.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    # AUC requires probability predictions
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        n_classes = len(np.unique(y_test))

        if n_classes == 2:
            # Binary classification - use positive class probability
            metrics["auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            # Multi-class - use one-vs-rest
            metrics["auc"] = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
    else:
        metrics["auc"] = None

    return metrics


def get_confusion_matrix(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    normalise: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix for a trained model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        normalise: Normalisation mode - 'true', 'pred', 'all', or None.

    Returns:
        Confusion matrix as numpy array.
    """
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred, normalize=normalise)


def get_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve data.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: True labels (binary).

    Returns:
        Dictionary with 'fpr', 'tpr', and 'thresholds'.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def get_precision_recall_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute precision-recall curve data.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: True labels (binary).

    Returns:
        Dictionary with 'precision', 'recall', and 'thresholds'.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
    }


def get_calibration_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute calibration curve data.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: True labels (binary).
        n_bins: Number of bins for calibration.

    Returns:
        Dictionary with 'prob_true' and 'prob_pred'.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


def get_learning_curve_data(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "f1",
) -> Dict[str, np.ndarray]:
    """
    Compute learning curve data.

    Args:
        model: Unfitted model instance.
        X: Feature matrix.
        y: Labels.
        cv: Number of cross-validation folds.
        train_sizes: Training set sizes to evaluate.
        scoring: Scoring metric.

    Returns:
        Dictionary with 'train_sizes', 'train_scores', 'test_scores'.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=-1,
    )

    return {
        "train_sizes": train_sizes_abs,
        "train_scores_mean": train_scores.mean(axis=1),
        "train_scores_std": train_scores.std(axis=1),
        "test_scores_mean": test_scores.mean(axis=1),
        "test_scores_std": test_scores.std(axis=1),
    }


def compute_shap_values(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretability.

    Args:
        model: Trained model.
        X: Feature matrix for explanation.
        feature_names: Names of features.
        max_samples: Maximum samples to use for background data.

    Returns:
        Dictionary with 'shap_values', 'expected_value', and 'feature_names'.
    """
    # Sample background data if needed
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # Choose appropriate explainer based on model type
    model_type = type(model).__name__

    if model_type in ["RandomForestClassifier", "XGBClassifier"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # Use KernelExplainer for other models
        if len(X) > 50:
            background = shap.sample(X, 50)
        else:
            background = X

        def predict_fn(x):
            return model.predict_proba(x)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample)

    # Handle binary classification (return positive class SHAP values)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": feature_names,
        "X_sample": X_sample,
    }


def get_feature_importance(
    model: BaseEstimator,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ or coef_ attribute.
        feature_names: Names of features.

    Returns:
        DataFrame with feature names and importance scores, sorted by importance.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not have feature importance attributes.")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def compare_models(
    models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    average: str = "binary",
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.

    Args:
        models: Dictionary mapping model names to trained models.
        X_test: Test features.
        y_test: True labels.
        average: Averaging method for multi-class metrics.

    Returns:
        DataFrame with metrics for each model.
    """
    results = []

    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, average=average)
        metrics["model"] = name
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.set_index("model")

    # Reorder columns
    column_order = ["accuracy", "precision", "recall", "f1", "auc"]
    comparison_df = comparison_df[[c for c in column_order if c in comparison_df.columns]]

    return comparison_df.sort_values("f1", ascending=False)


def get_classification_report(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a text classification report.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        target_names: Names for each class.

    Returns:
        Formatted classification report string.
    """
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=target_names)
