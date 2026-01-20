"""
Tests for the evaluation module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from creditclass.evaluation import (
    compare_models,
    evaluate_model,
    get_calibration_curve,
    get_classification_report,
    get_confusion_matrix,
    get_feature_importance,
    get_learning_curve_data,
    get_precision_recall_curve,
    get_roc_curve,
)
from creditclass.training import get_model, train_model


@pytest.fixture
def trained_model_and_data():
    """Create trained model with test data."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Split
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


@pytest.fixture
def feature_names():
    """Create feature names for testing."""
    return [f"feature_{i}" for i in range(20)]


class TestEvaluateModel:
    """Tests for model evaluation."""

    def test_returns_all_metrics(self, trained_model_and_data):
        """Test all expected metrics are returned."""
        model, _, X_test, _, y_test = trained_model_and_data

        metrics = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics

    def test_metrics_in_valid_range(self, trained_model_and_data):
        """Test metrics are in valid range [0, 1]."""
        model, _, X_test, _, y_test = trained_model_and_data

        metrics = evaluate_model(model, X_test, y_test)

        for name, value in metrics.items():
            if value is not None:
                assert 0 <= value <= 1, f"{name} out of range: {value}"

    def test_multiclass_average(self, trained_model_and_data):
        """Test multi-class averaging works."""
        model, _, X_test, _, y_test = trained_model_and_data

        # Test different averaging methods
        for average in ["binary", "macro", "weighted"]:
            metrics = evaluate_model(model, X_test, y_test, average=average)
            assert metrics["f1"] >= 0


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""

    def test_confusion_matrix_shape(self, trained_model_and_data):
        """Test confusion matrix has correct shape."""
        model, _, X_test, _, y_test = trained_model_and_data

        cm = get_confusion_matrix(model, X_test, y_test)

        assert cm.shape == (2, 2)

    def test_normalised_confusion_matrix(self, trained_model_and_data):
        """Test normalised confusion matrix sums to 1."""
        model, _, X_test, _, y_test = trained_model_and_data

        cm = get_confusion_matrix(model, X_test, y_test, normalise="true")

        # Each row should sum to 1
        row_sums = cm.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])


class TestCurves:
    """Tests for curve computations."""

    def test_roc_curve_data(self, trained_model_and_data):
        """Test ROC curve data structure."""
        model, _, X_test, _, y_test = trained_model_and_data

        roc_data = get_roc_curve(model, X_test, y_test)

        assert "fpr" in roc_data
        assert "tpr" in roc_data
        assert "thresholds" in roc_data
        assert len(roc_data["fpr"]) == len(roc_data["tpr"])

    def test_precision_recall_curve_data(self, trained_model_and_data):
        """Test precision-recall curve data structure."""
        model, _, X_test, _, y_test = trained_model_and_data

        pr_data = get_precision_recall_curve(model, X_test, y_test)

        assert "precision" in pr_data
        assert "recall" in pr_data
        assert "thresholds" in pr_data

    def test_calibration_curve_data(self, trained_model_and_data):
        """Test calibration curve data structure."""
        model, _, X_test, _, y_test = trained_model_and_data

        cal_data = get_calibration_curve(model, X_test, y_test, n_bins=5)

        assert "prob_true" in cal_data
        assert "prob_pred" in cal_data
        assert len(cal_data["prob_true"]) <= 5


class TestLearningCurve:
    """Tests for learning curve computation."""

    def test_learning_curve_data(self, trained_model_and_data):
        """Test learning curve data structure."""
        _, X_train, _, y_train, _ = trained_model_and_data

        model = LogisticRegression(random_state=42, max_iter=1000)
        lc_data = get_learning_curve_data(model, X_train, y_train, cv=3)

        assert "train_sizes" in lc_data
        assert "train_scores_mean" in lc_data
        assert "train_scores_std" in lc_data
        assert "test_scores_mean" in lc_data
        assert "test_scores_std" in lc_data


class TestFeatureImportance:
    """Tests for feature importance extraction."""

    def test_random_forest_importance(self, feature_names):
        """Test feature importance from Random Forest."""
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance_df = get_feature_importance(model, feature_names)

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == 20

    def test_logistic_regression_importance(self, feature_names):
        """Test feature importance from Logistic Regression."""
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)

        importance_df = get_feature_importance(model, feature_names)

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 20

    def test_importance_sorted_descending(self, feature_names):
        """Test importance is sorted in descending order."""
        X, y = make_classification(n_samples=100, n_features=20, random_state=42)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importance_df = get_feature_importance(model, feature_names)

        importances = importance_df["importance"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))


class TestCompareModels:
    """Tests for model comparison."""

    def test_compare_multiple_models(self, trained_model_and_data):
        """Test comparing multiple models."""
        _, X_train, X_test, y_train, y_test = trained_model_and_data

        models = {}
        for name in ["logistic_regression", "random_forest"]:
            model = get_model(name)
            models[name] = train_model(model, X_train, y_train)

        comparison = compare_models(models, X_test, y_test)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "logistic_regression" in comparison.index
        assert "random_forest" in comparison.index

    def test_comparison_sorted_by_f1(self, trained_model_and_data):
        """Test comparison is sorted by F1 score."""
        _, X_train, X_test, y_train, y_test = trained_model_and_data

        models = {}
        for name in ["logistic_regression", "random_forest"]:
            model = get_model(name)
            models[name] = train_model(model, X_train, y_train)

        comparison = compare_models(models, X_test, y_test)

        f1_values = comparison["f1"].values
        assert all(f1_values[i] >= f1_values[i + 1] for i in range(len(f1_values) - 1))


class TestClassificationReport:
    """Tests for classification report generation."""

    def test_report_string(self, trained_model_and_data):
        """Test classification report returns string."""
        model, _, X_test, _, y_test = trained_model_and_data

        report = get_classification_report(model, X_test, y_test)

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report
