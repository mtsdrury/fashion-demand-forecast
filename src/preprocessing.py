"""Merge data sources, deflate to real dollars, and run stationarity tests."""

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.config import ANALYSIS_START, CPI_BASE_YEAR


def deflate_sales(
    sales: pd.Series,
    cpi: pd.Series,
    base_year: int = CPI_BASE_YEAR,
) -> pd.Series:
    """Convert nominal sales to real (inflation-adjusted) sales.

    Uses the mean CPI value for the base year as the reference level.
    Formula: real_sales = nominal_sales * (base_cpi / current_cpi)
    """
    base_cpi = cpi[cpi.index.year == base_year].mean()
    real_sales = sales * (base_cpi / cpi)
    real_sales.name = "real_clothing_sales"
    return real_sales


def merge_datasets(
    sales: pd.Series,
    cpi: pd.Series,
    trends: pd.DataFrame | None = None,
    start: str = ANALYSIS_START,
) -> pd.DataFrame:
    """Merge all data sources on monthly index and filter to analysis period.

    Parameters
    ----------
    sales : Monthly clothing retail sales (nominal).
    cpi : Monthly CPI Apparel index.
    trends : Google Trends DataFrame (optional, can be None for models
             that don't use exogenous regressors).
    start : Start date for the analysis period.

    Returns
    -------
    DataFrame with columns: clothing_sales, cpi_apparel, real_clothing_sales,
    and optionally one column per Trends term.
    """
    df = pd.DataFrame({"clothing_sales": sales, "cpi_apparel": cpi})
    df["real_clothing_sales"] = deflate_sales(sales, cpi)

    if trends is not None:
        df = df.join(trends)

    # Filter to analysis period
    start_period = pd.Period(start, freq="M")
    df = df[df.index >= start_period]

    # Forward-fill small gaps (e.g., late FRED releases), then drop remaining NaNs
    df = df.ffill().dropna()

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features for modeling.

    Adds month (1-12) and year columns derived from the PeriodIndex.
    """
    df = df.copy()
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """Run the Augmented Dickey-Fuller test for stationarity.

    Returns a dict with the test statistic, p-value, critical values,
    and a boolean indicating whether the series is stationary at the
    given significance level.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
        "is_stationary": bool(result[1] < significance),
    }
