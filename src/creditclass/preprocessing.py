"""
Data preprocessing utilities for the UCI German Credit dataset.

This module handles downloading, loading, encoding, and preparing the data
for classification tasks.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# UCI German Credit dataset URL
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names for the German Credit dataset
COLUMN_NAMES = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "credit_risk",
]

# Categorical columns in the dataset
CATEGORICAL_COLUMNS = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

# Numerical columns in the dataset
NUMERICAL_COLUMNS = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]


def download_data(data_dir: Optional[Path] = None, force: bool = False) -> Path:
    """
    Download the UCI German Credit dataset.

    Args:
        data_dir: Directory to save the data. Defaults to data/raw/.
        force: If True, re-download even if file exists.

    Returns:
        Path to the downloaded data file.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "raw"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "german.data"

    if file_path.exists() and not force:
        return file_path

    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return file_path


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the German Credit dataset from file.

    Args:
        file_path: Path to the data file. If None, downloads the data first.

    Returns:
        DataFrame containing the raw dataset.
    """
    if file_path is None:
        file_path = download_data()

    df = pd.read_csv(
        file_path,
        sep=" ",
        header=None,
        names=COLUMN_NAMES,
    )

    return df


def encode_categoricals(
    df: pd.DataFrame,
    method: str = "onehot",
    encoders: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables in the dataset.

    Args:
        df: Input DataFrame.
        method: Encoding method - 'onehot' or 'label'.
        encoders: Pre-fitted encoders for consistent encoding. If None, new
            encoders are fitted.

    Returns:
        Tuple of (encoded DataFrame, dictionary of fitted encoders).
    """
    df_encoded = df.copy()
    fitted_encoders = encoders if encoders is not None else {}

    if method == "label":
        for col in CATEGORICAL_COLUMNS:
            if col in df_encoded.columns:
                if col not in fitted_encoders:
                    fitted_encoders[col] = LabelEncoder()
                    df_encoded[col] = fitted_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    df_encoded[col] = fitted_encoders[col].transform(
                        df_encoded[col].astype(str)
                    )

    elif method == "onehot":
        for col in CATEGORICAL_COLUMNS:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(
                    df_encoded[col],
                    prefix=col,
                    drop_first=True,
                    dtype=int,
                )
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])

    return df_encoded, fitted_encoders


def create_default_target(df: pd.DataFrame) -> pd.Series:
    """
    Create binary target for credit default prediction.

    The original target is 1 (good) or 2 (bad). We convert to:
    - 0: Good credit risk (no default)
    - 1: Bad credit risk (default)

    Args:
        df: DataFrame containing the credit_risk column.

    Returns:
        Binary target series.
    """
    return (df["credit_risk"] == 2).astype(int)


def create_tier_target(df: pd.DataFrame) -> pd.Series:
    """
    Create multi-class target for credit score tier classification.

    Derives risk tiers (low/medium/high) based on:
    - Original credit risk
    - Credit amount
    - Duration
    - Checking account status

    Args:
        df: DataFrame containing relevant columns.

    Returns:
        Multi-class target series with values 0 (low), 1 (medium), 2 (high).
    """
    # Create a risk score based on multiple factors
    risk_score = np.zeros(len(df))

    # Original credit risk contributes
    risk_score += (df["credit_risk"] == 2).astype(int) * 2

    # High credit amount increases risk
    amount_75th = df["credit_amount"].quantile(0.75)
    risk_score += (df["credit_amount"] > amount_75th).astype(int)

    # Long duration increases risk
    duration_75th = df["duration_months"].quantile(0.75)
    risk_score += (df["duration_months"] > duration_75th).astype(int)

    # No checking account (A14) increases risk
    if "checking_account_status" in df.columns:
        risk_score += (df["checking_account_status"] == "A14").astype(int)

    # Convert score to tiers
    tiers = pd.cut(
        risk_score,
        bins=[-np.inf, 1, 3, np.inf],
        labels=[0, 1, 2],
    ).astype(int)

    return tiers


def create_approval_target(df: pd.DataFrame) -> pd.Series:
    """
    Create binary target for loan approval prediction.

    Derives approval decision based on business rules:
    - Good credit risk
    - Credit amount below median
    - Stable employment (>1 year)

    Args:
        df: DataFrame containing relevant columns.

    Returns:
        Binary target series (1 = approved, 0 = denied).
    """
    # Start with good credit risk
    approved = (df["credit_risk"] == 1).astype(bool)

    # Credit amount should be reasonable (below 75th percentile)
    amount_75th = df["credit_amount"].quantile(0.75)
    approved &= df["credit_amount"] <= amount_75th

    # Employment stability (A73, A74, A75 indicate >= 1 year employment)
    if "employment_since" in df.columns:
        stable_employment = df["employment_since"].isin(["A73", "A74", "A75"])
        approved &= stable_employment

    return approved.astype(int)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.

    Args:
        X: Feature DataFrame.
        y: Target series.
        test_size: Proportion of data for test set.
        random_state: Random seed for reproducibility.
        stratify: Whether to stratify by target.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Args:
        X_train: Training features.
        X_test: Test features.
        scaler: Pre-fitted scaler. If None, a new one is fitted on X_train.

    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def prepare_data(
    target_type: str = "default",
    encoding_method: str = "onehot",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """
    Convenience function to load and prepare data for modelling.

    Args:
        target_type: One of 'default', 'tier', or 'approval'.
        encoding_method: 'onehot' or 'label'.
        test_size: Proportion for test set.
        random_state: Random seed.
        scale: Whether to scale features.

    Returns:
        Dictionary containing X_train, X_test, y_train, y_test, feature_names,
        and optionally scaler.
    """
    # Load data
    df = load_data()

    # Create target
    target_creators = {
        "default": create_default_target,
        "tier": create_tier_target,
        "approval": create_approval_target,
    }
    y = target_creators[target_type](df)

    # Remove target column from features
    feature_df = df.drop(columns=["credit_risk"])

    # Encode categoricals
    X_encoded, encoders = encode_categoricals(feature_df, method=encoding_method)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X_encoded, y, test_size=test_size, random_state=random_state
    )

    result = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X_encoded.columns.tolist(),
        "encoders": encoders,
    }

    if scale:
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        result["X_train_scaled"] = X_train_scaled
        result["X_test_scaled"] = X_test_scaled
        result["scaler"] = scaler

    return result
