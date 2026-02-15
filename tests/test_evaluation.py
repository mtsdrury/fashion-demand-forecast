"""Tests for evaluation metrics on known values."""

import numpy as np
import pytest

from src.evaluation import compute_all_metrics, mae, mape, rmse


class TestRMSE:
    def test_perfect_predictions(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == 0.0

    def test_known_error(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        # errors are all 1.0, so RMSE = 1.0
        assert rmse(actual, predicted) == pytest.approx(1.0)

    def test_asymmetric(self):
        actual = np.array([0.0, 0.0])
        predicted = np.array([3.0, 4.0])
        # MSE = (9 + 16) / 2 = 12.5, RMSE = sqrt(12.5)
        assert rmse(actual, predicted) == pytest.approx(np.sqrt(12.5))


class TestMAE:
    def test_perfect_predictions(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert mae(actual, actual) == 0.0

    def test_known_error(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 4.0, 6.0])
        # |1| + |2| + |3| = 6, MAE = 2.0
        assert mae(actual, predicted) == pytest.approx(2.0)


class TestMAPE:
    def test_perfect_predictions(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert mape(actual, actual) == 0.0

    def test_known_error(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 220.0])
        # |10/100| + |20/200| = 0.1 + 0.1 = 0.2, mean = 0.1, * 100 = 10%
        assert mape(actual, predicted) == pytest.approx(10.0)

    def test_handles_zeros_in_actual(self):
        actual = np.array([0.0, 100.0])
        predicted = np.array([10.0, 110.0])
        # Zero actual is filtered out, only 100 vs 110 remains: 10%
        assert mape(actual, predicted) == pytest.approx(10.0)


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        metrics = compute_all_metrics(actual, predicted)
        assert set(metrics.keys()) == {"rmse", "mae", "mape"}

    def test_perfect_predictions(self):
        actual = np.array([1.0, 2.0, 3.0])
        metrics = compute_all_metrics(actual, actual)
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["mape"] == 0.0
