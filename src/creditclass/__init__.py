"""
CreditClass: A lightweight classification demo on real-world credit data.

This package provides utilities for preprocessing, feature engineering,
model training, evaluation, and visualisation of credit classification tasks.
"""

__version__ = "0.1.0"

from creditclass.preprocessing import (
    download_data,
    load_data,
    encode_categoricals,
    create_default_target,
    create_tier_target,
    create_approval_target,
    split_data,
)
from creditclass.feature_engineering import (
    add_interaction_terms,
    bin_numerical_features,
    get_feature_pipeline,
)
from creditclass.training import (
    get_model,
    train_model,
    tune_hyperparameters,
    save_model,
    load_model,
)
from creditclass.evaluation import (
    evaluate_model,
    get_confusion_matrix,
    compute_shap_values,
    compare_models,
)

__all__ = [
    # Preprocessing
    "download_data",
    "load_data",
    "encode_categoricals",
    "create_default_target",
    "create_tier_target",
    "create_approval_target",
    "split_data",
    # Feature engineering
    "add_interaction_terms",
    "bin_numerical_features",
    "get_feature_pipeline",
    # Training
    "get_model",
    "train_model",
    "tune_hyperparameters",
    "save_model",
    "load_model",
    # Evaluation
    "evaluate_model",
    "get_confusion_matrix",
    "compute_shap_values",
    "compare_models",
]
