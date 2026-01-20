"""
Tests for the preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from creditclass.preprocessing import (
    CATEGORICAL_COLUMNS,
    COLUMN_NAMES,
    NUMERICAL_COLUMNS,
    create_approval_target,
    create_default_target,
    create_tier_target,
    encode_categoricals,
    load_data,
    split_data,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "checking_account_status": np.random.choice(["A11", "A12", "A13", "A14"], n_samples),
        "duration_months": np.random.randint(6, 72, n_samples),
        "credit_history": np.random.choice(["A30", "A31", "A32", "A33", "A34"], n_samples),
        "purpose": np.random.choice(["A40", "A41", "A42", "A43"], n_samples),
        "credit_amount": np.random.randint(500, 20000, n_samples),
        "savings_account": np.random.choice(["A61", "A62", "A63", "A64", "A65"], n_samples),
        "employment_since": np.random.choice(["A71", "A72", "A73", "A74", "A75"], n_samples),
        "installment_rate": np.random.randint(1, 5, n_samples),
        "personal_status_sex": np.random.choice(["A91", "A92", "A93", "A94"], n_samples),
        "other_debtors": np.random.choice(["A101", "A102", "A103"], n_samples),
        "residence_since": np.random.randint(1, 5, n_samples),
        "property": np.random.choice(["A121", "A122", "A123", "A124"], n_samples),
        "age": np.random.randint(19, 75, n_samples),
        "other_installment_plans": np.random.choice(["A141", "A142", "A143"], n_samples),
        "housing": np.random.choice(["A151", "A152", "A153"], n_samples),
        "existing_credits": np.random.randint(1, 5, n_samples),
        "job": np.random.choice(["A171", "A172", "A173", "A174"], n_samples),
        "num_dependents": np.random.randint(1, 3, n_samples),
        "telephone": np.random.choice(["A191", "A192"], n_samples),
        "foreign_worker": np.random.choice(["A201", "A202"], n_samples),
        "credit_risk": np.random.choice([1, 2], n_samples),
    }

    return pd.DataFrame(data)


class TestLoadData:
    """Tests for data loading functions."""

    def test_column_names_count(self):
        """Verify column names list has correct count."""
        assert len(COLUMN_NAMES) == 21

    def test_categorical_columns_valid(self):
        """Verify categorical columns are in column names."""
        for col in CATEGORICAL_COLUMNS:
            assert col in COLUMN_NAMES

    def test_numerical_columns_valid(self):
        """Verify numerical columns are in column names."""
        for col in NUMERICAL_COLUMNS:
            assert col in COLUMN_NAMES


class TestEncodeCategoricals:
    """Tests for categorical encoding."""

    def test_label_encoding(self, sample_data):
        """Test label encoding produces integers."""
        encoded, encoders = encode_categoricals(sample_data, method="label")

        for col in CATEGORICAL_COLUMNS:
            if col in encoded.columns:
                assert encoded[col].dtype in [np.int32, np.int64]

    def test_onehot_encoding(self, sample_data):
        """Test one-hot encoding produces binary columns."""
        encoded, _ = encode_categoricals(sample_data, method="onehot")

        # Original categorical columns should be removed
        for col in CATEGORICAL_COLUMNS:
            assert col not in encoded.columns

        # New columns should be created
        assert len(encoded.columns) > len(NUMERICAL_COLUMNS)

    def test_encoding_preserves_rows(self, sample_data):
        """Test encoding preserves number of rows."""
        encoded_label, _ = encode_categoricals(sample_data, method="label")
        encoded_onehot, _ = encode_categoricals(sample_data, method="onehot")

        assert len(encoded_label) == len(sample_data)
        assert len(encoded_onehot) == len(sample_data)


class TestTargetCreation:
    """Tests for target variable creation."""

    def test_default_target_binary(self, sample_data):
        """Test default target is binary."""
        target = create_default_target(sample_data)

        assert set(target.unique()).issubset({0, 1})

    def test_default_target_conversion(self, sample_data):
        """Test correct conversion of credit_risk values."""
        target = create_default_target(sample_data)

        # Original value 2 (bad) should map to 1
        bad_mask = sample_data["credit_risk"] == 2
        assert (target[bad_mask] == 1).all()

        # Original value 1 (good) should map to 0
        good_mask = sample_data["credit_risk"] == 1
        assert (target[good_mask] == 0).all()

    def test_tier_target_multiclass(self, sample_data):
        """Test tier target has three classes."""
        target = create_tier_target(sample_data)

        assert set(target.unique()).issubset({0, 1, 2})

    def test_approval_target_binary(self, sample_data):
        """Test approval target is binary."""
        target = create_approval_target(sample_data)

        assert set(target.unique()).issubset({0, 1})

    def test_target_length_matches_data(self, sample_data):
        """Test all targets have same length as input data."""
        default_target = create_default_target(sample_data)
        tier_target = create_tier_target(sample_data)
        approval_target = create_approval_target(sample_data)

        assert len(default_target) == len(sample_data)
        assert len(tier_target) == len(sample_data)
        assert len(approval_target) == len(sample_data)


class TestSplitData:
    """Tests for data splitting."""

    def test_split_ratios(self, sample_data):
        """Test split produces correct proportions."""
        X = sample_data.drop(columns=["credit_risk"])
        y = sample_data["credit_risk"]

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        assert len(X_train) == pytest.approx(len(X) * 0.8, abs=2)
        assert len(X_test) == pytest.approx(len(X) * 0.2, abs=2)

    def test_split_preserves_columns(self, sample_data):
        """Test split preserves column structure."""
        X = sample_data.drop(columns=["credit_risk"])
        y = sample_data["credit_risk"]

        X_train, X_test, y_train, y_test = split_data(X, y)

        assert list(X_train.columns) == list(X.columns)
        assert list(X_test.columns) == list(X.columns)

    def test_stratified_split(self, sample_data):
        """Test stratified split maintains class proportions."""
        X = sample_data.drop(columns=["credit_risk"])
        y = create_default_target(sample_data)

        X_train, X_test, y_train, y_test = split_data(X, y, stratify=True)

        original_ratio = y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        assert train_ratio == pytest.approx(original_ratio, abs=0.1)
        assert test_ratio == pytest.approx(original_ratio, abs=0.1)
