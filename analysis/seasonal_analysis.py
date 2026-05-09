"""
analysis/seasonal_analysis.py
Seasonal pattern analysis for the Bengaluru Climate Intelligence Platform.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DATA_PROC_DIR, DATA_SYN_DIR, SEASONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    for base in [DATA_PROC_DIR, DATA_SYN_DIR]:
        for fname in ["merged_dataset.csv", "meteorological.csv"]:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                df = pd.read_csv(p, parse_dates=["date"])
                logger.info(f"Loaded {p}")
                return df
    return pd.DataFrame()


def seasonal_stats(df: pd.DataFrame) -> dict:
    """Compute mean/std/min/max per season for key variables."""
    if df.empty or "season" not in df.columns:
        return {}
    numeric_cols = ["temperature_mean","temperature_max","rainfall","humidity",
                    "wind_speed","aqi","pm25"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    stats_dict = {}
    for season in SEASONS:
        subset = df[df["season"] == season]
        if subset.empty:
            continue
        stats_dict[season] = {}
        for col in numeric_cols:
            stats_dict[season][col] = {
                "mean":  round(float(subset[col].mean()), 2),
                "std":   round(float(subset[col].std()),  2),
                "min":   round(float(subset[col].min()),  2),
                "max":   round(float(subset[col].max()),  2),
                "median":round(float(subset[col].median()),2),
            }
    return stats_dict


def monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """12-month average of key variables (climatological mean)."""
    if df.empty or "month" not in df.columns:
        return pd.DataFrame()
    numeric_cols = ["temperature_mean","temperature_max","temperature_min",
                    "rainfall","humidity","wind_speed"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    return df.groupby("month")[numeric_cols].mean().round(2).reset_index()


def annual_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Annual averages showing long-term trends."""
    if df.empty or "year" not in df.columns:
        return pd.DataFrame()
    cols = ["temperature_mean","rainfall","humidity","aqi","pm25"]
    cols = [c for c in cols if c in df.columns]
    annual = df.groupby("year")[cols].mean().round(2).reset_index()
    # Attach OLS trend slope per variable
    for col in cols:
        x = annual["year"].values
        y = annual[col].values
        mask = ~np.isnan(y)
        if mask.sum() > 1:
            slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
            annual[f"{col}_trend_slope"] = round(slope, 4)
    return annual


def flood_risk_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Flood risk distribution by season."""
    if "flood_risk" not in df.columns or "season" not in df.columns:
        return pd.DataFrame()
    ct = pd.crosstab(df["season"], df["flood_risk"], normalize="index").round(3)
    return ct.reset_index()


def aqi_seasonal_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly AQI trend across years."""
    if "aqi" not in df.columns:
        return pd.DataFrame()
    return df.groupby(["year","month"])["aqi"].mean().round(1).reset_index()


def dengue_risk_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Dengue risk distribution by season."""
    if "dengue_risk" not in df.columns or "season" not in df.columns:
        return pd.DataFrame()
    ct = pd.crosstab(df["season"], df["dengue_risk"], normalize="index").round(3)
    return ct.reset_index()


def run_seasonal_analysis() -> dict:
    df = load_data()
    if df.empty:
        logger.error("No data available.")
        return {}
    results = {
        "seasonal_stats":       seasonal_stats(df),
        "monthly_climatology":  monthly_climatology(df),
        "annual_trend":         annual_trend(df),
        "flood_risk_seasonal":  flood_risk_seasonal(df),
        "aqi_seasonal_trend":   aqi_seasonal_trend(df),
        "dengue_risk_seasonal": dengue_risk_seasonal(df),
    }
    logger.info("Seasonal analysis complete.")
    return results


if __name__ == "__main__":
    results = run_seasonal_analysis()
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            print(f"\n── {k} ──\n{v.head()}")
        else:
            print(f"\n── {k} ──")
            for season, stats in v.items():
                print(f"  {season}: temp_mean={stats.get('temperature_mean',{}).get('mean','N/A')}")
    print("\n✅  Seasonal analysis complete.")
