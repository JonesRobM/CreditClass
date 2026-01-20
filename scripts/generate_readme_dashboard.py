#!/usr/bin/env python
"""
Generate a polished dashboard figure for the README.

This script creates a composite visualisation combining:
- ROC curves for all models
- F1 score comparison bar chart
- Top feature importances
- Confusion matrix grid

The output is saved to outputs/figures/dashboard.png
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from creditclass.preprocessing import prepare_data
from creditclass.training import get_model, train_model, load_model, get_all_model_names
from creditclass.evaluation import (
    evaluate_model,
    get_roc_curve,
    get_confusion_matrix,
    get_feature_importance,
)


# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLOURS = sns.color_palette("tab10")
RANDOM_STATE = 42


def train_all_models():
    """Train all models and return them with test data."""
    print("Preparing data...")
    data = prepare_data(
        target_type="default",
        encoding_method="onehot",
        test_size=0.2,
        random_state=RANDOM_STATE,
        scale=True,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    # Model configurations (name, needs_scaling)
    model_configs = [
        ("logistic_regression", True),
        ("random_forest", False),
        ("xgboost", False),
        ("svm", True),
        ("knn", True),
    ]

    trained_models = {}

    for model_name, needs_scaling in model_configs:
        print(f"Training {model_name}...")

        X_tr = X_train_scaled if needs_scaling else X_train.values
        X_te = X_test_scaled if needs_scaling else X_test.values

        model = get_model(model_name)
        model = train_model(model, X_tr, y_train)

        trained_models[model_name] = {
            "model": model,
            "X_test": X_te,
        }

    return trained_models, y_test, feature_names


def create_dashboard(trained_models, y_test, feature_names, output_path):
    """Create and save the dashboard figure."""
    print("Creating dashboard...")

    # Create figure with custom grid
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. ROC Curves (top left, spans 2 columns)
    ax_roc = fig.add_subplot(gs[0, :2])

    for i, (model_name, model_data) in enumerate(trained_models.items()):
        model = model_data["model"]
        X_te = model_data["X_test"]

        roc_data = get_roc_curve(model, X_te, y_test)
        auc = np.trapezoid(roc_data["tpr"], roc_data["fpr"])

        display_name = model_name.replace("_", " ").title()
        ax_roc.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            label=f"{display_name} (AUC={auc:.3f})",
            color=COLOURS[i],
            linewidth=2,
        )

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.set_title("ROC Curves Comparison", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.02])

    # 2. F1 Score Bar Chart (top right)
    ax_f1 = fig.add_subplot(gs[0, 2])

    f1_scores = {}
    for model_name, model_data in trained_models.items():
        metrics = evaluate_model(model_data["model"], model_data["X_test"], y_test)
        f1_scores[model_name] = metrics["f1"]

    # Sort by F1 score
    sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
    model_names = [m[0].replace("_", " ").title() for m in sorted_models]
    scores = [m[1] for m in sorted_models]

    bars = ax_f1.barh(model_names, scores, color=COLOURS[:len(model_names)])
    ax_f1.set_xlabel("F1 Score", fontsize=11)
    ax_f1.set_title("F1 Score Comparison", fontsize=14, fontweight="bold")
    ax_f1.set_xlim([0, 1])
    ax_f1.invert_yaxis()

    # Add value labels
    for bar, score in zip(bars, scores):
        ax_f1.text(
            score + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=9,
        )

    # 3. Top 5 Feature Importance (bottom left)
    ax_feat = fig.add_subplot(gs[1, 0])

    # Use XGBoost for feature importance
    xgb_model = trained_models["xgboost"]["model"]
    importance_df = get_feature_importance(xgb_model, feature_names)
    top_features = importance_df.head(5)

    ax_feat.barh(
        top_features["feature"],
        top_features["importance"],
        color=COLOURS[2],
    )
    ax_feat.set_xlabel("Importance", fontsize=11)
    ax_feat.set_title("Top 5 Features (XGBoost)", fontsize=14, fontweight="bold")
    ax_feat.invert_yaxis()

    # 4. Confusion Matrix Grid (bottom middle and right)
    ax_cm1 = fig.add_subplot(gs[1, 1])
    ax_cm2 = fig.add_subplot(gs[1, 2])

    # Best model confusion matrix
    best_model_name = sorted_models[0][0]
    best_model = trained_models[best_model_name]["model"]
    best_X_test = trained_models[best_model_name]["X_test"]

    cm_best = get_confusion_matrix(best_model, best_X_test, y_test, normalise="true")

    sns.heatmap(
        cm_best,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Good", "Bad"],
        yticklabels=["Good", "Bad"],
        ax=ax_cm1,
        cbar_kws={"shrink": 0.8},
    )
    ax_cm1.set_xlabel("Predicted", fontsize=11)
    ax_cm1.set_ylabel("Actual", fontsize=11)
    ax_cm1.set_title(
        f"{best_model_name.replace('_', ' ').title()}\n(Best Model)",
        fontsize=14,
        fontweight="bold",
    )

    # Logistic Regression confusion matrix (for comparison)
    lr_model = trained_models["logistic_regression"]["model"]
    lr_X_test = trained_models["logistic_regression"]["X_test"]

    cm_lr = get_confusion_matrix(lr_model, lr_X_test, y_test, normalise="true")

    sns.heatmap(
        cm_lr,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=["Good", "Bad"],
        yticklabels=["Good", "Bad"],
        ax=ax_cm2,
        cbar_kws={"shrink": 0.8},
    )
    ax_cm2.set_xlabel("Predicted", fontsize=11)
    ax_cm2.set_ylabel("Actual", fontsize=11)
    ax_cm2.set_title("Logistic Regression\n(Baseline)", fontsize=14, fontweight="bold")

    # Add main title
    fig.suptitle(
        "CreditClass: Credit Default Prediction Dashboard",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Dashboard saved to: {output_path}")

    plt.close(fig)


def main():
    """Main entry point."""
    # Determine output path
    project_root = Path(__file__).parent.parent
    output_path = project_root / "outputs" / "figures" / "dashboard.png"

    # Train models
    trained_models, y_test, feature_names = train_all_models()

    # Create dashboard
    create_dashboard(trained_models, y_test, feature_names, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
