"""
Tests for the training module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from creditclass.training import (
    DEFAULT_PARAMS,
    NeuralNetworkClassifier,
    get_all_model_names,
    get_model,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_classification_data():
    """Create multi-class classification dataset for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


class TestGetModel:
    """Tests for model factory function."""

    def test_all_model_names_valid(self):
        """Test all model names can be retrieved."""
        for model_name in get_all_model_names():
            model = get_model(model_name)
            assert model is not None

    def test_invalid_model_raises(self):
        """Test invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("invalid_model")

    def test_custom_params(self):
        """Test custom parameters are applied."""
        model = get_model("logistic_regression", params={"C": 0.5})
        assert model.C == 0.5

    def test_default_params_applied(self):
        """Test default parameters are applied."""
        model = get_model("random_forest")

        assert model.n_estimators == DEFAULT_PARAMS["random_forest"]["n_estimators"]
        assert model.max_depth == DEFAULT_PARAMS["random_forest"]["max_depth"]


class TestTrainModel:
    """Tests for model training."""

    @pytest.mark.parametrize("model_name", [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "svm",
        "knn",
    ])
    def test_sklearn_models_train(self, model_name, binary_classification_data):
        """Test sklearn-compatible models train without error."""
        X, y = binary_classification_data
        model = get_model(model_name)

        trained_model = train_model(model, X, y)

        assert hasattr(trained_model, "predict")
        predictions = trained_model.predict(X[:10])
        assert len(predictions) == 10

    def test_neural_network_trains(self, binary_classification_data):
        """Test neural network trains without error."""
        X, y = binary_classification_data
        model = get_model("neural_network", params={"epochs": 10})

        trained_model = train_model(model, X, y)

        assert hasattr(trained_model, "predict")
        predictions = trained_model.predict(X[:10])
        assert len(predictions) == 10

    @pytest.mark.parametrize("model_name", [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "svm",
    ])
    def test_models_have_predict_proba(self, model_name, binary_classification_data):
        """Test models have predict_proba method."""
        X, y = binary_classification_data
        model = get_model(model_name)
        trained_model = train_model(model, X, y)

        assert hasattr(trained_model, "predict_proba")
        probas = trained_model.predict_proba(X[:10])
        assert probas.shape == (10, 2)

    def test_multiclass_training(self, multiclass_classification_data):
        """Test models work with multi-class targets."""
        X, y = multiclass_classification_data
        model = get_model("random_forest")

        trained_model = train_model(model, X, y)
        predictions = trained_model.predict(X[:10])

        assert all(p in [0, 1, 2] for p in predictions)


class TestNeuralNetworkClassifier:
    """Tests for the neural network classifier wrapper."""

    def test_fit_predict(self, binary_classification_data):
        """Test neural network fit and predict."""
        X, y = binary_classification_data

        model = NeuralNetworkClassifier(
            hidden_sizes=[32, 16],
            epochs=10,
            random_state=42,
        )

        model.fit(X, y)
        predictions = model.predict(X[:10])

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, binary_classification_data):
        """Test neural network probability predictions."""
        X, y = binary_classification_data

        model = NeuralNetworkClassifier(epochs=10, random_state=42)
        model.fit(X, y)

        probas = model.predict_proba(X[:10])

        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_classes_attribute(self, binary_classification_data):
        """Test classes_ attribute is set."""
        X, y = binary_classification_data

        model = NeuralNetworkClassifier(epochs=5)
        model.fit(X, y)

        assert hasattr(model, "classes_")
        assert len(model.classes_) == 2


class TestSaveLoadModel:
    """Tests for model serialisation."""

    def test_save_load_roundtrip(self, binary_classification_data):
        """Test model can be saved and loaded."""
        X, y = binary_classification_data

        model = get_model("random_forest")
        trained_model = train_model(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save
            save_path = save_model(trained_model, "test_model", output_dir=tmpdir)
            assert save_path.exists()

            # Load
            loaded_model = load_model("test_model", model_dir=tmpdir)

            # Compare predictions
            original_preds = trained_model.predict(X[:10])
            loaded_preds = loaded_model.predict(X[:10])

            np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_save_creates_directory(self, binary_classification_data):
        """Test save creates output directory if needed."""
        X, y = binary_classification_data

        model = get_model("logistic_regression")
        trained_model = train_model(model, X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "models"

            save_path = save_model(trained_model, "test_model", output_dir=nested_dir)

            assert nested_dir.exists()
            assert save_path.exists()
