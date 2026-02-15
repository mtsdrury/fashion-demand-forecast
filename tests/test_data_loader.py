"""Tests for Google Trends CSV parsing (no API key needed)."""

import pandas as pd
import pytest

from src.data_loader import _term_to_filename, load_single_trend, load_google_trends


def test_term_to_filename_single_word():
    assert _term_to_filename("athleisure") == "athleisure.csv"


def test_term_to_filename_multi_word():
    assert _term_to_filename("fast fashion") == "fast_fashion.csv"


@pytest.fixture
def sample_trends_csv(tmp_path):
    """Create a synthetic Google Trends CSV matching the real export format."""
    content = (
        "Category: All categories\n"
        "\n"
        "Month,athleisure: (United States)\n"
        "2004-01,0\n"
        "2004-02,5\n"
        "2004-03,<1\n"
        "2004-04,12\n"
    )
    filepath = tmp_path / "athleisure.csv"
    filepath.write_text(content)
    return filepath


def test_load_single_trend_shape(sample_trends_csv):
    series = load_single_trend(sample_trends_csv)
    assert len(series) == 4
    assert isinstance(series.index, pd.PeriodIndex)


def test_load_single_trend_values(sample_trends_csv):
    series = load_single_trend(sample_trends_csv)
    assert series.iloc[0] == 0
    assert series.iloc[1] == 5
    # "<1" should be coerced to 0
    assert series.iloc[2] == 0
    assert series.iloc[3] == 12


def test_load_single_trend_period_index(sample_trends_csv):
    series = load_single_trend(sample_trends_csv)
    assert series.index[0] == pd.Period("2004-01", freq="M")
    assert series.index[-1] == pd.Period("2004-04", freq="M")


@pytest.fixture
def sample_trends_dir(tmp_path):
    """Create a directory with two synthetic Google Trends CSVs."""
    for term, values in [("athleisure", "10,20,30"), ("streetwear", "5,15,25")]:
        content = (
            "Category: All categories\n"
            "\n"
            f"Month,{term}: (United States)\n"
            f"2004-01,{values.split(',')[0]}\n"
            f"2004-02,{values.split(',')[1]}\n"
            f"2004-03,{values.split(',')[2]}\n"
        )
        filepath = tmp_path / f"{term}.csv"
        filepath.write_text(content)
    return tmp_path


def test_load_google_trends_merge(sample_trends_dir):
    df = load_google_trends(
        trends_dir=sample_trends_dir, terms=["athleisure", "streetwear"]
    )
    assert list(df.columns) == ["athleisure", "streetwear"]
    assert len(df) == 3


def test_load_google_trends_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Google Trends CSV not found"):
        load_google_trends(trends_dir=tmp_path, terms=["nonexistent"])
