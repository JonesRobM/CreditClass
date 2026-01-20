"""
Model training utilities for the CreditClass project.

This module provides functions for creating, training, tuning, and
persisting classification models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# Default hyperparameters for each model
DEFAULT_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "random_state": 42,
    },
    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
        "n_jobs": -1,
    },
    "neural_network": {
        "hidden_sizes": [64, 32],
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 32,
    },
}

# Hyperparameter grids for tuning
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    },
    "xgboost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
}


class SimpleNeuralNetwork(nn.Module):
    """
    Simple feedforward neural network for classification.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout: float = 0.2,
    ):
        """
        Initialise the neural network.

        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for the neural network.
    """

    def __init__(
        self,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        """
        Initialise the classifier.

        Args:
            hidden_sizes: Hidden layer sizes.
            dropout: Dropout probability.
            learning_rate: Learning rate for optimiser.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            random_state: Random seed.
        """
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.model_ = None
        self.classes_ = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkClassifier":
        """
        Fit the neural network.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            Fitted classifier.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.model_ = SimpleNeuralNetwork(
            input_size=n_features,
            hidden_sizes=self.hidden_sizes,
            output_size=n_classes,
            dropout=self.dropout,
        ).to(self.device_)

        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
        )

        X_tensor = torch.FloatTensor(X).to(self.device_)
        y_tensor = torch.LongTensor(y).to(self.device_)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model_.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimiser.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimiser.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict.

        Returns:
            Predicted class labels.
        """
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device_)
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict.

        Returns:
            Class probabilities.
        """
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device_)
            outputs = self.model_(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()


def get_model(
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
    """
    Factory function to create a model by name.

    Args:
        model_name: Name of the model. One of: 'logistic_regression',
            'random_forest', 'xgboost', 'svm', 'knn', 'neural_network'.
        params: Optional custom parameters. If None, defaults are used.

    Returns:
        Configured model instance.

    Raises:
        ValueError: If model_name is not recognised.
    """
    model_classes = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "neural_network": NeuralNetworkClassifier,
    }

    if model_name not in model_classes:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(model_classes.keys())}"
        )

    model_params = DEFAULT_PARAMS.get(model_name, {}).copy()
    if params is not None:
        model_params.update(params)

    return model_classes[model_name](**model_params)


def train_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    """
    Train a model on the provided data.

    Args:
        model: Model instance to train.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted model.
    """
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    method: str = "grid",
    cv: int = 5,
    scoring: str = "f1",
    n_iter: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Tune hyperparameters using cross-validation.

    Args:
        model_name: Name of the model to tune.
        X_train: Training features.
        y_train: Training labels.
        param_grid: Custom parameter grid. If None, defaults are used.
        method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.
        cv: Number of cross-validation folds.
        scoring: Scoring metric.
        n_iter: Number of iterations for random search.
        random_state: Random seed.

    Returns:
        Dictionary with 'best_params', 'best_score', and 'best_model'.
    """
    if model_name == "neural_network":
        raise ValueError("Hyperparameter tuning not supported for neural network.")

    base_model = get_model(model_name)

    if param_grid is None:
        param_grid = PARAM_GRIDS.get(model_name, {})

    if method == "grid":
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
        )

    search.fit(X_train, y_train)

    return {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "best_model": search.best_estimator_,
    }


def save_model(
    model: BaseEstimator,
    model_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save a trained model to disk.

    Args:
        model: Trained model to save.
        model_name: Name for the saved file.
        output_dir: Directory to save to. Defaults to outputs/models/.

    Returns:
        Path to the saved model file.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "models"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, file_path)

    return file_path


def load_model(
    model_name: str,
    model_dir: Optional[Path] = None,
) -> BaseEstimator:
    """
    Load a trained model from disk.

    Args:
        model_name: Name of the saved model file (without extension).
        model_dir: Directory containing the model. Defaults to outputs/models/.

    Returns:
        Loaded model instance.
    """
    if model_dir is None:
        model_dir = Path(__file__).parent.parent.parent.parent / "outputs" / "models"

    file_path = Path(model_dir) / f"{model_name}.joblib"

    return joblib.load(file_path)


def get_all_model_names() -> List[str]:
    """
    Get list of all available model names.

    Returns:
        List of model name strings.
    """
    return [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "svm",
        "knn",
        "neural_network",
    ]
