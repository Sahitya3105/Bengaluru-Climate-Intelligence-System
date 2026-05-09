"""
pipeline/preprocessor.py
Data cleaning, normalization, temporal/spatial alignment,
and classification labeling for the Bengaluru Climate Platform.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    DATA_SYN_DIR, DATA_PROC_DIR,
    TEMP_CLASSES, RAINFALL_CLASSES, AQI_CLASSES,
    SEASONS, RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def load_csv(name: str, subdir: str = DATA_SYN_DIR) -> pd.DataFrame:
    path = os.path.join(subdir, f"{name}.csv")
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=1).columns else [])
    logger.info(f"Loaded {path}  ({len(df)} rows)")
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "interpolate") -> pd.DataFrame:
    """Fill NaNs using interpolation or forward-fill."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "interpolate":
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    else:
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def remove_outliers(df: pd.DataFrame, cols: list, z_threshold: float = 3.5) -> pd.DataFrame:
    """Replace z-score outliers with interpolated values."""
    for col in cols:
        if col not in df.columns:
            continue
        z = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-9))
        df.loc[z > z_threshold, col] = np.nan
    return handle_missing(df)


def normalize(df: pd.DataFrame, cols: list, method: str = "minmax") -> tuple:
    """Normalize selected columns; return (df, scaler)."""
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    existing = [c for c in cols if c in df.columns]
    df[existing] = scaler.fit_transform(df[existing])
    return df, scaler


# ─────────────────────────────────────────────
# Classification labelers
# ─────────────────────────────────────────────

def classify_temperature(temp: float) -> str:
    for label, (lo, hi) in TEMP_CLASSES.items():
        if lo <= temp < hi:
            return label
    return "Heat-Wave"


def classify_rainfall(rain: float) -> str:
    for label, (lo, hi) in RAINFALL_CLASSES.items():
        if lo <= rain < hi:
            return label
    return "Extreme"


def classify_aqi(aqi: float) -> str:
    for label, (lo, hi) in AQI_CLASSES.items():
        if lo <= aqi <= hi:
            return label
    return "Very Poor"


def classify_flood_risk(rainfall: float, humidity: float) -> str:
    score = rainfall * 0.7 + humidity * 0.3
    if score < 15:  return "Low"
    if score < 45:  return "Medium"
    return "High"


def classify_health_risk(temp: float, aqi: float, humidity: float) -> str:
    score = (temp / 45) * 40 + (aqi / 500) * 40 + (humidity / 100) * 20
    if score < 25:   return "Low"
    if score < 50:   return "Moderate"
    if score < 75:   return "High"
    return "Severe"


def classify_dengue_risk(temp: float, rainfall: float, humidity: float) -> str:
    # Mosquito breeding: warm+wet conditions
    score = 0
    if 25 <= temp <= 35: score += 2
    if rainfall > 10:    score += 2
    if humidity > 70:    score += 1
    if score <= 1: return "Low"
    if score <= 3: return "Medium"
    return "High"


# ─────────────────────────────────────────────
# Main preprocessing pipeline
# ─────────────────────────────────────────────

def preprocess_met(df_met: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing meteorological data ...")
    df = df_met.copy()
    outlier_cols = ["temperature_mean", "temperature_max", "temperature_min",
                    "rainfall", "humidity", "wind_speed", "pressure"]
    df = handle_missing(df)
    df = remove_outliers(df, outlier_cols)

    # Add class labels
    df["temp_class"]  = df["temperature_mean"].apply(classify_temperature)
    df["rain_class"]  = df["rainfall"].apply(classify_rainfall)
    df["flood_risk"]  = df.apply(lambda r: classify_flood_risk(r["rainfall"], r["humidity"]), axis=1)

    # Add seasonal one-hot
    for s in SEASONS:
        df[f"season_{s.replace(' ','_').replace('/','_')}"] = (df["season"] == s).astype(int)

    # Rolling features (7-day & 30-day moving average)
    for col in ["temperature_mean", "rainfall", "humidity"]:
        df[f"{col}_7d_avg"]  = df[col].rolling(7,  min_periods=1).mean().round(2)
        df[f"{col}_30d_avg"] = df[col].rolling(30, min_periods=1).mean().round(2)

    return df


def preprocess_aqi(df_aqi: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing AQI data ...")
    df = df_aqi.copy()
    outlier_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "aqi"]
    df = handle_missing(df)
    df = remove_outliers(df, outlier_cols)
    df["aqi_class"] = df["aqi"].apply(classify_aqi)
    df["aqi_7d_avg"] = df["aqi"].rolling(7, min_periods=1).mean().round(1)
    return df


def preprocess_merged(df_met: pd.DataFrame, df_aqi: pd.DataFrame) -> pd.DataFrame:
    """Merge met + AQI on date for the unified ML dataset."""
    logger.info("Merging meteorological and AQI datasets ...")
    # Ensure both dataframes have date as datetime
    df_met["date"] = pd.to_datetime(df_met["date"])
    df_aqi["date"] = pd.to_datetime(df_aqi["date"])
    # Add year/month/season to AQI if missing (new real-data format)
    for col, fn in [("year", lambda d: d.dt.year), ("month", lambda d: d.dt.month)]:
        if col not in df_aqi.columns:
            df_aqi[col] = fn(df_aqi["date"])
    if "season" not in df_aqi.columns:
        from config.settings import SEASONS
        def _s(m):
            for n, ms in SEASONS.items():
                if m in ms: return n
            return "Unknown"
        df_aqi["season"] = df_aqi["month"].apply(_s)
    merged = pd.merge(df_met, df_aqi, on=["date","year","month","season"], how="inner", suffixes=("","_aqi"))
    merged["health_risk"]  = merged.apply(
        lambda r: classify_health_risk(r["temperature_mean"], r["aqi"], r["humidity"]), axis=1)
    merged["dengue_risk"]  = merged.apply(
        lambda r: classify_dengue_risk(r["temperature_mean"], r["rainfall"], r["humidity"]), axis=1)
    return merged


def run_preprocessing():
    """Full preprocessing pipeline: load → clean → label → save."""
    os.makedirs(DATA_PROC_DIR, exist_ok=True)

    df_met = load_csv("meteorological")
    df_aqi = load_csv("aqi")

    if df_met.empty or df_aqi.empty:
        logger.error("Raw data missing. Run data_collector.py first.")
        return

    df_met_clean = preprocess_met(df_met)
    df_aqi_clean = preprocess_aqi(df_aqi)
    df_merged    = preprocess_merged(df_met_clean, df_aqi_clean)

    df_met_clean.to_csv(os.path.join(DATA_PROC_DIR, "met_processed.csv"), index=False)
    df_aqi_clean.to_csv(os.path.join(DATA_PROC_DIR, "aqi_processed.csv"), index=False)
    df_merged.to_csv(os.path.join(DATA_PROC_DIR, "merged_dataset.csv"), index=False)

    logger.info(f"Preprocessed data saved to {DATA_PROC_DIR}")
    logger.info(f"Merged dataset: {len(df_merged)} rows × {len(df_merged.columns)} columns")
    return df_merged


if __name__ == "__main__":
    run_preprocessing()
    print("\nPreprocessing complete. Files saved to data/processed/")
