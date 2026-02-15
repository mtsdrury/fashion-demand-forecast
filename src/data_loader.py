"""Fetch FRED economic data and parse Google Trends CSV exports."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from src.config import (
    CLOTHING_SALES_SERIES,
    CPI_APPAREL_SERIES,
    DATA_RAW,
    FRED_START,
    GOOGLE_TRENDS_DIR,
    TRENDS_TERMS,
)

load_dotenv()


def _get_fred_client() -> Fred:
    """Create a FRED API client from the environment variable."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not found. Copy .env.example to .env and add your key."
        )
    return Fred(api_key=api_key)


def fetch_clothing_sales(start: str = FRED_START) -> pd.Series:
    """Fetch monthly US clothing retail sales from FRED.

    Returns a Series with a monthly PeriodIndex and values in millions of dollars.
    """
    fred = _get_fred_client()
    series = fred.get_series(CLOTHING_SALES_SERIES, observation_start=start)
    series.index = pd.to_datetime(series.index).to_period("M")
    series.name = "clothing_sales"
    return series


def fetch_cpi_apparel(start: str = FRED_START) -> pd.Series:
    """Fetch monthly CPI Apparel index from FRED.

    Returns a Series with a monthly PeriodIndex.
    """
    fred = _get_fred_client()
    series = fred.get_series(CPI_APPAREL_SERIES, observation_start=start)
    series.index = pd.to_datetime(series.index).to_period("M")
    series.name = "cpi_apparel"
    return series


def _term_to_filename(term: str) -> str:
    """Convert a Google Trends search term to its expected CSV filename."""
    return term.replace(" ", "_") + ".csv"


def load_single_trend(filepath: Path) -> pd.Series:
    """Parse a single Google Trends CSV export into a Series.

    Google Trends CSVs have a header section (first 2 lines with category info
    and a blank line) followed by Month,Value columns.
    """
    df = pd.read_csv(filepath, skiprows=2)

    # The first column is always "Month", second is "{term}: (United States)"
    df.columns = ["month", "value"]
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
    df = df.set_index("month")

    # Google Trends uses "<1" for very low values; replace with 0
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    return df["value"]


def load_google_trends(
    trends_dir: Path = GOOGLE_TRENDS_DIR,
    terms: list[str] = TRENDS_TERMS,
) -> pd.DataFrame:
    """Load and merge all Google Trends CSV exports into a single DataFrame.

    Returns a DataFrame with a monthly PeriodIndex and one column per search term.
    Column names are the search terms with spaces replaced by underscores.
    """
    frames = {}
    for term in terms:
        filename = _term_to_filename(term)
        filepath = trends_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Google Trends CSV not found: {filepath}\n"
                f"See data/raw/google_trends/README.md for download instructions."
            )
        col_name = term.replace(" ", "_")
        frames[col_name] = load_single_trend(filepath)

    df = pd.DataFrame(frames)
    df.index.name = "month"
    return df


def save_raw_data(clothing_sales: pd.Series, cpi_apparel: pd.Series) -> None:
    """Save fetched FRED data to CSV in data/raw/."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    clothing_sales.to_csv(DATA_RAW / "clothing_sales.csv", header=True)
    cpi_apparel.to_csv(DATA_RAW / "cpi_apparel.csv", header=True)
