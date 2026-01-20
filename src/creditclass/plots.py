"""
Visualisation utilities for the CreditClass project.

This module provides functions for creating publication-ready plots
using Matplotlib and Seaborn.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.base import BaseEstimator

from creditclass.evaluation import (
    get_calibration_curve,
    get_confusion_matrix,
    get_feature_importance,
    get_precision_recall_curve,
    get_roc_curve,
)


# Set default style
plt.style.use("seaborn-v0_8-whitegrid")

# Default colour palette
COLOURS = sns.color_palette("tab10")

# Default figure settings
DEFAULT_FIGSIZE = (8, 6)
DEFAULT_DPI = 300


def set_plot_style():
    """Set consistent plot style across all visualisations."""
    plt.rcParams.update({
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.dpi": 100,
        "savefig.dpi": DEFAULT_DPI,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Save a figure to disk.

    Args:
        fig: Matplotlib figure to save.
        filename: Name for the saved file (without extension).
        output_dir: Directory to save to. Defaults to outputs/figures/.
        dpi: Resolution for saved figure.

    Returns:
        Path to the saved figure.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "figures"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{filename}.png"
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight", facecolor="white")

    return file_path


def plot_confusion_matrix(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalise: bool = True,
    ax: Optional[plt.Axes] = None,
    cmap: str = "Blues",
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot confusion matrix for a trained model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        class_names: Names for each class.
        normalise: Whether to normalise values.
        ax: Matplotlib axes to plot on.
        cmap: Colour map for the heatmap.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    cm = get_confusion_matrix(
        model, X_test, y_test,
        normalise="true" if normalise else None,
    )

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalise else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title or "Confusion Matrix")

    return ax


def plot_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    colour: Optional[str] = None,
    show_auc: bool = True,
) -> plt.Axes:
    """
    Plot ROC curve for a trained model.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: True labels (binary).
        ax: Matplotlib axes to plot on.
        label: Label for the curve.
        colour: Colour for the curve.
        show_auc: Whether to include AUC in legend.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    roc_data = get_roc_curve(model, X_test, y_test)
    auc = np.trapezoid(roc_data["tpr"], roc_data["fpr"])

    curve_label = label or "Model"
    if show_auc:
        curve_label = f"{curve_label} (AUC = {auc:.3f})"

    ax.plot(
        roc_data["fpr"],
        roc_data["tpr"],
        label=curve_label,
        color=colour,
        linewidth=2,
    )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    return ax


def plot_roc_curves_comparison(
    models: Dict[str, BaseEstimator],
    X_test: np.ndarray,
    y_test: np.ndarray,
    ax: Optional[plt.Axes] = None,
    colours: Optional[List[str]] = None,
) -> plt.Axes:
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        models: Dictionary mapping model names to trained models.
        X_test: Test features.
        y_test: True labels (binary).
        ax: Matplotlib axes to plot on.
        colours: List of colours for each model.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if colours is None:
        colours = COLOURS

    for i, (name, model) in enumerate(models.items()):
        colour = colours[i % len(colours)]
        plot_roc_curve(model, X_test, y_test, ax=ax, label=name, colour=colour)

    ax.set_title("ROC Curves Comparison")

    return ax


def plot_precision_recall(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    colour: Optional[str] = None,
) -> plt.Axes:
    """
    Plot precision-recall curve for a trained model.

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: True labels (binary).
        ax: Matplotlib axes to plot on.
        label: Label for the curve.
        colour: Colour for the curve.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    pr_data = get_precision_recall_curve(model, X_test, y_test)

    ax.plot(
        pr_data["recall"],
        pr_data["precision"],
        label=label or "Model",
        color=colour,
        linewidth=2,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    return ax


def plot_learning_curve(
    learning_data: Dict[str, np.ndarray],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot learning curve showing train/validation scores vs training size.

    Args:
        learning_data: Dictionary from get_learning_curve_data().
        ax: Matplotlib axes to plot on.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    train_sizes = learning_data["train_sizes"]
    train_mean = learning_data["train_scores_mean"]
    train_std = learning_data["train_scores_std"]
    test_mean = learning_data["test_scores_mean"]
    test_std = learning_data["test_scores_std"]

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color=COLOURS[0],
    )
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.2,
        color=COLOURS[1],
    )

    ax.plot(train_sizes, train_mean, "o-", color=COLOURS[0], label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color=COLOURS[1], label="Validation score")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title or "Learning Curve")
    ax.legend(loc="lower right")

    return ax


def plot_calibration(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
) -> plt.Axes:
    """
    Plot calibration curve (reliability diagram).

    Args:
        model: Trained model with predict_proba method.
        X_test: Test features.
        y_test: True labels (binary).
        n_bins: Number of bins.
        ax: Matplotlib axes to plot on.
        label: Label for the curve.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    cal_data = get_calibration_curve(model, X_test, y_test, n_bins=n_bins)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")
    ax.plot(
        cal_data["prob_pred"],
        cal_data["prob_true"],
        "o-",
        label=label or "Model",
        linewidth=2,
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return ax


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
    ax: Optional[plt.Axes] = None,
    colour: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot horizontal bar chart of feature importances.

    Args:
        model: Trained model with feature importance.
        feature_names: Names of features.
        top_n: Number of top features to show.
        ax: Matplotlib axes to plot on.
        colour: Bar colour.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(6, top_n * 0.4)))

    importance_df = get_feature_importance(model, feature_names)
    top_features = importance_df.head(top_n)

    ax.barh(
        top_features["feature"],
        top_features["importance"],
        color=colour or COLOURS[0],
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title or f"Top {top_n} Feature Importances")
    ax.invert_yaxis()

    return ax


def plot_shap_summary(
    shap_data: Dict[str, Any],
    plot_type: str = "bar",
    max_display: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot SHAP summary visualisation.

    Args:
        shap_data: Dictionary from compute_shap_values().
        plot_type: 'bar' for bar plot, 'beeswarm' for beeswarm plot.
        max_display: Maximum features to display.
        ax: Matplotlib axes to plot on.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(6, max_display * 0.5)))
        plt.sca(ax)

    shap_values = shap_data["shap_values"]
    X_sample = shap_data["X_sample"]
    feature_names = shap_data["feature_names"]

    if plot_type == "bar":
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
    else:
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )

    return ax


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "coolwarm",
    annot: bool = True,
) -> plt.Axes:
    """
    Plot correlation heatmap for numerical features.

    Args:
        df: DataFrame containing features.
        columns: Columns to include. If None, uses all numerical columns.
        ax: Matplotlib axes to plot on.
        cmap: Colour map for the heatmap.
        annot: Whether to annotate cells with correlation values.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[columns].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        ax=ax,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
    )

    ax.set_title("Feature Correlation Matrix")

    return ax


def plot_class_distribution(
    y: pd.Series,
    class_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    colour: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot bar chart of class distribution.

    Args:
        y: Target series.
        class_names: Names for each class.
        ax: Matplotlib axes to plot on.
        colour: Bar colour.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    class_counts = y.value_counts().sort_index()

    if class_names is not None:
        class_counts.index = class_names

    ax.bar(
        class_counts.index.astype(str),
        class_counts.values,
        color=colour or COLOURS[0],
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title or "Class Distribution")

    # Add count labels on bars
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + max(class_counts.values) * 0.01, str(v), ha="center")

    return ax


def plot_distribution(
    data: pd.Series,
    ax: Optional[plt.Axes] = None,
    kind: str = "hist",
    bins: int = 30,
    colour: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot distribution of a numerical feature.

    Args:
        data: Series to plot.
        ax: Matplotlib axes to plot on.
        kind: 'hist' for histogram, 'kde' for density, 'both' for both.
        bins: Number of bins for histogram.
        colour: Plot colour.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    colour = colour or COLOURS[0]

    if kind == "hist":
        ax.hist(data, bins=bins, color=colour, alpha=0.7, edgecolor="white")
    elif kind == "kde":
        data.plot.kde(ax=ax, color=colour, linewidth=2)
    else:  # both
        ax.hist(data, bins=bins, color=colour, alpha=0.5, density=True, edgecolor="white")
        data.plot.kde(ax=ax, color=colour, linewidth=2)

    ax.set_xlabel(data.name or "Value")
    ax.set_ylabel("Frequency" if kind == "hist" else "Density")
    ax.set_title(title or f"Distribution of {data.name}")

    return ax


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "f1",
    ax: Optional[plt.Axes] = None,
    colours: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot bar chart comparing models on a metric.

    Args:
        comparison_df: DataFrame from compare_models().
        metric: Metric to plot.
        ax: Matplotlib axes to plot on.
        colours: List of colours for bars.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if colours is None:
        colours = COLOURS

    models = comparison_df.index.tolist()
    values = comparison_df[metric].values

    bars = ax.bar(models, values, color=colours[:len(models)])

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Model Comparison: {metric.upper()}")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")

    return ax


def plot_metrics_grouped_bar(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot grouped bar chart of multiple metrics across models.

    Args:
        comparison_df: DataFrame from compare_models().
        metrics: Metrics to plot. If None, uses all available.
        ax: Matplotlib axes to plot on.
        title: Plot title.

    Returns:
        Matplotlib axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    if metrics is None:
        metrics = comparison_df.columns.tolist()

    x = np.arange(len(comparison_df))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            comparison_df[metric],
            width,
            label=metric.upper(),
            color=COLOURS[i % len(COLOURS)],
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title(title or "Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha="right")
    ax.legend()

    return ax
