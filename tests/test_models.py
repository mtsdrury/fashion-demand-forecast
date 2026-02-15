"""Tests for SeasonalNaive model (smoke tests, no external data needed)."""

import numpy as np
import pandas as pd
import pytest

from src.models import SeasonalNaive


@pytest.fixture
def monthly_series():
    """Two years of monthly data where each month has a distinct value."""
    idx = pd.period_range("2020-01", periods=24, freq="M")
    # Year 1: 100-111, Year 2: 200-211
    values = list(range(100, 112)) + list(range(200, 212))
    return pd.Series(values, index=idx, dtype=float)


class TestSeasonalNaive:
    def test_predict_repeats_last_year(self, monthly_series):
        model = SeasonalNaive()
        model.fit(monthly_series)
        predicted = model.predict(steps=12)
        # Should repeat the last 12 values (200-211)
        expected = np.array(list(range(200, 212)), dtype=float)
        np.testing.assert_array_equal(predicted, expected)

    def test_predict_shape(self, monthly_series):
        model = SeasonalNaive()
        model.fit(monthly_series)
        predicted = model.predict(steps=6)
        assert len(predicted) == 6

    def test_predict_wraps_for_long_horizon(self, monthly_series):
        model = SeasonalNaive()
        model.fit(monthly_series)
        predicted = model.predict(steps=18)
        assert len(predicted) == 18
        # Months 13-18 should wrap back to the start of last season
        np.testing.assert_array_equal(predicted[12:], predicted[:6])

    def test_predict_before_fit_raises(self):
        model = SeasonalNaive()
        with pytest.raises(RuntimeError, match="must be fit"):
            model.predict(steps=12)
