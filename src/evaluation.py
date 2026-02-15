"""Forecast evaluation metrics and rolling-window cross-validation."""

from typing import Protocol

import numpy as np
import pandas as pd

from src.config import CV_HORIZON_MONTHS, CV_INITIAL_TRAIN_YEARS, CV_STEP_MONTHS


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (as a percentage, e.g. 5.2 means 5.2%).

    Filters out periods where actual is zero to avoid division errors.
    """
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_all_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Compute RMSE, MAE, and MAPE for a forecast."""
    return {
        "rmse": rmse(actual, predicted),
        "mae": mae(actual, predicted),
        "mape": mape(actual, predicted),
    }


class ForecastModel(Protocol):
    """Protocol for forecast model classes used in cross-validation."""

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> None: ...
    def predict(
        self, steps: int, exog: pd.DataFrame | None = None
    ) -> np.ndarray: ...


def rolling_window_cv(
    model_factory,
    y: pd.Series,
    exog: pd.DataFrame | None = None,
    initial_train_years: int = CV_INITIAL_TRAIN_YEARS,
    horizon_months: int = CV_HORIZON_MONTHS,
    step_months: int = CV_STEP_MONTHS,
) -> list[dict]:
    """Rolling-window time series cross-validation.

    Parameters
    ----------
    model_factory : Callable that returns a new, unfitted model instance.
    y : Target time series with PeriodIndex.
    exog : Exogenous regressors aligned with y (optional).
    initial_train_years : Number of years in the initial training window.
    horizon_months : Forecast horizon per fold.
    step_months : Step size between fold start dates.

    Returns
    -------
    List of dicts, one per fold, each containing:
        - fold: fold number (0-indexed)
        - train_start, train_end: training window boundaries
        - test_start, test_end: test window boundaries
        - metrics: dict of RMSE, MAE, MAPE
        - actual: actual test values
        - predicted: predicted test values
    """
    results = []
    periods = y.index

    initial_train_end_idx = initial_train_years * 12
    fold = 0
    train_start_idx = 0

    while True:
        train_end_idx = initial_train_end_idx + fold * step_months
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + horizon_months

        if test_end_idx > len(periods):
            break

        train_y = y.iloc[train_start_idx:train_end_idx]
        test_y = y.iloc[test_start_idx:test_end_idx]

        train_exog = None
        test_exog = None
        if exog is not None:
            train_exog = exog.iloc[train_start_idx:train_end_idx]
            test_exog = exog.iloc[test_start_idx:test_end_idx]

        model = model_factory()
        model.fit(train_y, exog=train_exog)
        predicted = model.predict(steps=horizon_months, exog=test_exog)

        actual_values = test_y.values
        pred_values = np.asarray(predicted)

        results.append({
            "fold": fold,
            "train_start": str(periods[train_start_idx]),
            "train_end": str(periods[train_end_idx - 1]),
            "test_start": str(periods[test_start_idx]),
            "test_end": str(periods[test_end_idx - 1]),
            "metrics": compute_all_metrics(actual_values, pred_values),
            "actual": actual_values,
            "predicted": pred_values,
        })
        fold += 1

    return results
