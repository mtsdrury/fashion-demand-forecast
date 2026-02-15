"""Forecast models: Seasonal Naive, SARIMAX, and Prophet wrappers."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER


class SeasonalNaive:
    """Seasonal naive baseline: predict each month as its value from last year."""

    def __init__(self, season_length: int = 12):
        self.season_length = season_length
        self._last_season: np.ndarray | None = None

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> None:
        self._last_season = y.values[-self.season_length :]

    def predict(self, steps: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        if self._last_season is None:
            raise RuntimeError("Model must be fit before calling predict.")
        # Tile the last season to cover the requested number of steps
        repeats = (steps // self.season_length) + 1
        tiled = np.tile(self._last_season, repeats)
        return tiled[:steps]


class SARIMAXModel:
    """Wrapper around statsmodels SARIMAX with configurable order."""

    def __init__(
        self,
        order: tuple[int, int, int] = SARIMAX_ORDER,
        seasonal_order: tuple[int, int, int, int] = SARIMAX_SEASONAL_ORDER,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> None:
        # Convert PeriodIndex to timestamp for statsmodels compatibility
        y_ts = y.copy()
        y_ts.index = y_ts.index.to_timestamp()

        exog_ts = None
        if exog is not None:
            exog_ts = exog.copy()
            exog_ts.index = exog_ts.index.to_timestamp()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y_ts,
                exog=exog_ts,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fitted = model.fit(disp=False, maxiter=200)

    def predict(self, steps: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Model must be fit before calling predict.")

        exog_ts = None
        if exog is not None:
            exog_ts = exog.copy()
            exog_ts.index = exog_ts.index.to_timestamp()

        forecast = self._fitted.forecast(steps=steps, exog=exog_ts)
        return forecast.values


class ProphetModel:
    """Wrapper around Facebook Prophet with optional exogenous regressors."""

    def __init__(self, yearly_seasonality: bool = True, add_us_holidays: bool = True):
        self.yearly_seasonality = yearly_seasonality
        self.add_us_holidays = add_us_holidays
        self._model = None
        self._regressor_names: list[str] = []

    def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> None:
        from prophet import Prophet

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=False,
            daily_seasonality=False,
        )

        if self.add_us_holidays:
            self._model.add_country_holidays(country_name="US")

        # Build the Prophet DataFrame (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            "ds": y.index.to_timestamp(),
            "y": y.values,
        })

        # Add exogenous regressors if provided
        self._regressor_names = []
        if exog is not None:
            for col in exog.columns:
                self._model.add_regressor(col)
                df[col] = exog[col].values
                self._regressor_names.append(col)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(df)

    def predict(self, steps: int, exog: pd.DataFrame | None = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fit before calling predict.")

        future = self._model.make_future_dataframe(periods=steps, freq="MS")
        # Keep only the forecast period
        future = future.tail(steps).reset_index(drop=True)

        if exog is not None:
            for col in self._regressor_names:
                future[col] = exog[col].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model.predict(future)

        return forecast["yhat"].values
