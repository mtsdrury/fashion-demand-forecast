"""Project constants and configuration."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
GOOGLE_TRENDS_DIR = DATA_RAW / "google_trends"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --- FRED Series ---
CLOTHING_SALES_SERIES = "MRTSSM448USN"  # US clothing retail sales, nominal, NSA
CPI_APPAREL_SERIES = "CPIAPPNS"  # CPI Apparel, not seasonally adjusted

# --- Google Trends ---
TRENDS_TERMS = [
    "athleisure",
    "fast fashion",
    "streetwear",
    "sustainable fashion",
    "luxury fashion",
]

# --- Date Ranges ---
# Google Trends data starts 2004-01, so that is the effective start for models
# using exogenous regressors. FRED data goes back further (1992).
FRED_START = "1992-01-01"
TRENDS_START = "2004-01-01"
ANALYSIS_START = "2004-01-01"  # common start for all models

# --- CPI Base Period ---
# Deflate nominal sales to real dollars using CPI Apparel.
# Base period: use the most recent full year available at analysis time.
CPI_BASE_YEAR = 2023

# --- Cross-Validation Parameters ---
CV_INITIAL_TRAIN_YEARS = 10  # initial training window (years)
CV_HORIZON_MONTHS = 12  # forecast horizon per fold (months)
CV_STEP_MONTHS = 12  # step between folds (months)

# --- SARIMAX Defaults ---
SARIMAX_ORDER = (1, 1, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, 12)  # monthly seasonality
