"""
Feature engineering utilities for the German Credit dataset.

This module provides functions for creating interaction terms, binning
numerical features, and building sklearn pipelines.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from creditclass.preprocessing import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


class InteractionTermsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for creating interaction terms between features.
    """

    def __init__(self, interactions: Optional[List[Tuple[str, str]]] = None):
        """
        Initialise the transformer.

        Args:
            interactions: List of (column1, column2) tuples for interactions.
                If None, default interactions are used.
        """
        self.interactions = interactions

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for this transformer)."""
        if self.interactions is None:
            self.interactions_ = [
                ("duration_months", "credit_amount"),
                ("age", "credit_amount"),
                ("duration_months", "installment_rate"),
            ]
        else:
            self.interactions_ = self.interactions
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction terms to the DataFrame.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with additional interaction columns.
        """
        X_out = X.copy()

        for col1, col2 in self.interactions_:
            if col1 in X_out.columns and col2 in X_out.columns:
                interaction_name = f"{col1}_x_{col2}"
                X_out[interaction_name] = X_out[col1] * X_out[col2]

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return feature names after transformation."""
        if input_features is None:
            return None

        output_features = list(input_features)
        for col1, col2 in self.interactions_:
            if col1 in input_features and col2 in input_features:
                output_features.append(f"{col1}_x_{col2}")

        return np.array(output_features)


def add_interaction_terms(
    df: pd.DataFrame,
    interactions: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Add interaction terms to a DataFrame.

    Creates multiplicative interactions between specified feature pairs.

    Args:
        df: Input DataFrame.
        interactions: List of (column1, column2) tuples. If None, uses defaults:
            - duration_months x credit_amount
            - age x credit_amount
            - duration_months x installment_rate

    Returns:
        DataFrame with additional interaction columns.
    """
    transformer = InteractionTermsTransformer(interactions=interactions)
    return transformer.fit_transform(df)


def bin_numerical_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_bins: int = 5,
    strategy: str = "quantile",
    encode: str = "ordinal",
) -> Tuple[pd.DataFrame, KBinsDiscretizer]:
    """
    Discretise numerical features into bins.

    Args:
        df: Input DataFrame.
        columns: Columns to bin. If None, bins all numerical columns.
        n_bins: Number of bins.
        strategy: Binning strategy - 'uniform', 'quantile', or 'kmeans'.
        encode: Encoding strategy - 'ordinal' or 'onehot'.

    Returns:
        Tuple of (binned DataFrame, fitted discretizer).
    """
    if columns is None:
        columns = [col for col in NUMERICAL_COLUMNS if col in df.columns]

    df_out = df.copy()

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        subsample=None,
    )

    binned_values = discretizer.fit_transform(df_out[columns])

    if encode == "ordinal":
        for i, col in enumerate(columns):
            df_out[f"{col}_binned"] = binned_values[:, i]
    else:
        # One-hot encoding produces sparse matrix
        binned_df = pd.DataFrame(
            binned_values.toarray() if hasattr(binned_values, "toarray") else binned_values,
            index=df_out.index,
        )
        # Rename columns
        binned_df.columns = [
            f"{col}_bin_{i}"
            for col in columns
            for i in range(n_bins)
        ]
        df_out = pd.concat([df_out, binned_df], axis=1)

    return df_out, discretizer


def create_amount_duration_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio of credit amount to duration.

    This represents the monthly payment burden.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with additional ratio column.
    """
    df_out = df.copy()

    if "credit_amount" in df_out.columns and "duration_months" in df_out.columns:
        df_out["amount_per_month"] = (
            df_out["credit_amount"] / df_out["duration_months"]
        )

    return df_out


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age group categories.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with additional age group column.
    """
    df_out = df.copy()

    if "age" in df_out.columns:
        df_out["age_group"] = pd.cut(
            df_out["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=["young", "young_adult", "middle", "mature", "senior"],
        )

    return df_out


def get_feature_pipeline(
    numerical_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    add_interactions: bool = True,
    scale_numerical: bool = True,
) -> Pipeline:
    """
    Create a sklearn pipeline for feature preprocessing.

    Args:
        numerical_columns: List of numerical column names.
        categorical_columns: List of categorical column names.
        add_interactions: Whether to add interaction terms.
        scale_numerical: Whether to scale numerical features.

    Returns:
        Configured sklearn Pipeline.
    """
    if numerical_columns is None:
        numerical_columns = NUMERICAL_COLUMNS

    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS

    # Numerical preprocessing
    numerical_steps = []
    if scale_numerical:
        numerical_steps.append(("scaler", StandardScaler()))

    numerical_pipeline = Pipeline(numerical_steps) if numerical_steps else "passthrough"

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="passthrough",
    )

    # Full pipeline
    pipeline_steps = [("preprocessor", preprocessor)]

    if add_interactions:
        # Note: interactions are added before preprocessing for simplicity
        # In production, you might want a different order
        pass

    return Pipeline(pipeline_steps)


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    This is a convenience function that applies:
    - Interaction terms
    - Amount/duration ratio
    - Age groups

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with all engineered features.
    """
    df_out = df.copy()

    # Add interaction terms
    df_out = add_interaction_terms(df_out)

    # Add amount per month ratio
    df_out = create_amount_duration_ratio(df_out)

    # Add age groups (if age is present and not encoded)
    if "age" in df_out.columns:
        df_out = create_age_groups(df_out)

    return df_out
