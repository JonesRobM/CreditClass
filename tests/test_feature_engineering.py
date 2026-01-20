"""
Tests for the feature engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from creditclass.feature_engineering import (
    InteractionTermsTransformer,
    add_interaction_terms,
    bin_numerical_features,
    create_age_groups,
    create_amount_duration_ratio,
    engineer_all_features,
)


@pytest.fixture
def sample_features():
    """Create sample feature data for testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        "duration_months": np.random.randint(6, 72, n_samples),
        "credit_amount": np.random.randint(500, 20000, n_samples),
        "age": np.random.randint(19, 75, n_samples),
        "installment_rate": np.random.randint(1, 5, n_samples),
        "residence_since": np.random.randint(1, 5, n_samples),
    })


class TestInteractionTerms:
    """Tests for interaction term creation."""

    def test_default_interactions(self, sample_features):
        """Test default interaction terms are created."""
        result = add_interaction_terms(sample_features)

        assert "duration_months_x_credit_amount" in result.columns
        assert "age_x_credit_amount" in result.columns
        assert "duration_months_x_installment_rate" in result.columns

    def test_custom_interactions(self, sample_features):
        """Test custom interaction terms."""
        interactions = [("age", "duration_months")]
        result = add_interaction_terms(sample_features, interactions=interactions)

        assert "age_x_duration_months" in result.columns

    def test_interaction_values(self, sample_features):
        """Test interaction values are computed correctly."""
        result = add_interaction_terms(sample_features)

        expected = sample_features["duration_months"] * sample_features["credit_amount"]
        np.testing.assert_array_equal(
            result["duration_months_x_credit_amount"].values,
            expected.values,
        )

    def test_transformer_fit_transform(self, sample_features):
        """Test InteractionTermsTransformer works correctly."""
        transformer = InteractionTermsTransformer()
        result = transformer.fit_transform(sample_features)

        assert len(result) == len(sample_features)
        assert len(result.columns) > len(sample_features.columns)

    def test_preserves_original_columns(self, sample_features):
        """Test original columns are preserved."""
        result = add_interaction_terms(sample_features)

        for col in sample_features.columns:
            assert col in result.columns


class TestBinNumericalFeatures:
    """Tests for numerical feature binning."""

    def test_binning_creates_columns(self, sample_features):
        """Test binning creates new columns."""
        result, discretizer = bin_numerical_features(
            sample_features,
            columns=["age", "credit_amount"],
        )

        assert "age_binned" in result.columns
        assert "credit_amount_binned" in result.columns

    def test_binning_values_range(self, sample_features):
        """Test binned values are within expected range."""
        result, _ = bin_numerical_features(
            sample_features,
            columns=["age"],
            n_bins=5,
        )

        assert result["age_binned"].min() >= 0
        assert result["age_binned"].max() < 5

    def test_binning_preserves_rows(self, sample_features):
        """Test binning preserves number of rows."""
        result, _ = bin_numerical_features(sample_features)

        assert len(result) == len(sample_features)


class TestDerivedFeatures:
    """Tests for derived feature creation."""

    def test_amount_duration_ratio(self, sample_features):
        """Test amount per month ratio calculation."""
        result = create_amount_duration_ratio(sample_features)

        assert "amount_per_month" in result.columns

        expected = sample_features["credit_amount"] / sample_features["duration_months"]
        np.testing.assert_array_almost_equal(
            result["amount_per_month"].values,
            expected.values,
        )

    def test_age_groups(self, sample_features):
        """Test age group categorisation."""
        result = create_age_groups(sample_features)

        assert "age_group" in result.columns

        # Check categories exist
        valid_groups = {"young", "young_adult", "middle", "mature", "senior"}
        assert set(result["age_group"].dropna().unique()).issubset(valid_groups)

    def test_age_groups_boundaries(self):
        """Test age group boundaries are correct."""
        df = pd.DataFrame({"age": [20, 30, 40, 50, 60]})
        result = create_age_groups(df)

        assert result.loc[0, "age_group"] == "young"
        assert result.loc[1, "age_group"] == "young_adult"
        assert result.loc[2, "age_group"] == "middle"
        assert result.loc[3, "age_group"] == "mature"
        assert result.loc[4, "age_group"] == "senior"


class TestEngineerAllFeatures:
    """Tests for combined feature engineering."""

    def test_all_features_added(self, sample_features):
        """Test all feature engineering is applied."""
        result = engineer_all_features(sample_features)

        # Should have interaction terms
        assert "duration_months_x_credit_amount" in result.columns

        # Should have amount per month
        assert "amount_per_month" in result.columns

        # Should have age groups
        assert "age_group" in result.columns

    def test_preserves_original_data(self, sample_features):
        """Test original data is not modified."""
        original_cols = list(sample_features.columns)
        _ = engineer_all_features(sample_features)

        assert list(sample_features.columns) == original_cols
