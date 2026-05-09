"""
pipeline/feature_engineer.py
Climate indicator engineering for the Bengaluru Climate Intelligence Platform.

Engineered features:
  - Heat Index (Steadman's approximation)
  - Urban Heat Island (UHI) score
  - Rainfall Anomaly
  - AQI Category encoding
  - Vegetation Loss Score
  - Water-body Shrinkage Index
  - Pollution Severity Score
  - Human Discomfort Index
  - Seasonal dummy indicators
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DATA_PROC_DIR, DATA_SYN_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Individual feature calculators
# ─────────────────────────────────────────────

def heat_index(T: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """
    Steadman's heat index (°C). Applies when T > 27°C.
    HI = -8.78 + 1.61*T + 2.34*RH - 0.146*RH*T - 0.012*T^2 ...
    """
    T_f  = T * 9/5 + 32            # convert to °F for formula
    HI_f = (-42.379
             + 2.04901523 * T_f
             + 10.14333127 * RH
             - 0.22475541 * T_f * RH
             - 0.00683783 * T_f**2
             - 0.05481717 * RH**2
             + 0.00122874 * T_f**2 * RH
             + 0.00085282 * T_f * RH**2
             - 0.00000199 * T_f**2 * RH**2)
    HI_c = (HI_f - 32) * 5/9       # back to °C
    # Below threshold, use simple T
    result = np.where(T > 27, HI_c, T)
    return np.round(result, 2)


def discomfort_index(T: np.ndarray, RH: np.ndarray) -> np.ndarray:
    """Thom's Discomfort Index (°C): DI = T - 0.55*(1-RH/100)*(T-14.5)"""
    DI = T - 0.55 * (1 - RH / 100) * (T - 14.5)
    return np.round(DI, 2)


def rainfall_anomaly(rainfall: np.ndarray, window: int = 30) -> np.ndarray:
    """Rolling z-score anomaly relative to 30-day local mean."""
    s = pd.Series(rainfall)
    rolling_mean = s.rolling(window, min_periods=1).mean()
    rolling_std  = s.rolling(window, min_periods=1).std().fillna(1)
    return np.round(((s - rolling_mean) / rolling_std).values, 3)


def uhi_score(temp_mean: np.ndarray, rural_temp_base: float = 26.5) -> np.ndarray:
    """UHI intensity = urban temp - assumed rural background."""
    return np.round(np.clip(temp_mean - rural_temp_base, 0, None), 2)


def pollution_severity(pm25: np.ndarray, aqi: np.ndarray) -> np.ndarray:
    """Composite pollution severity index (0-1 normalized)."""
    pm25_norm = np.clip(pm25 / 300, 0, 1)
    aqi_norm  = np.clip(aqi  / 500, 0, 1)
    return np.round(0.6 * pm25_norm + 0.4 * aqi_norm, 3)


def vegetation_loss_score(ndvi_series: np.ndarray) -> np.ndarray:
    """
    Score = how far NDVI has dropped below a healthy baseline (0.55).
    Higher = more vegetation loss.
    """
    baseline = 0.55
    return np.round(np.clip(baseline - ndvi_series, 0, 1), 3)


def water_body_shrinkage(water_pct: np.ndarray) -> np.ndarray:
    """
    Score = how far water body % has fallen from historical max.
    Higher = more lake area lost.
    """
    baseline = float(np.max(water_pct))
    return np.round(np.clip((baseline - water_pct) / (baseline + 1e-9), 0, 1), 3)


def aqi_category_ordinal(aqi_class_series: pd.Series) -> np.ndarray:
    """Map AQI class label → ordinal integer."""
    mapping = {"Good": 0, "Moderate": 1, "Poor": 2, "Very Poor": 3}
    return aqi_class_series.map(mapping).fillna(1).astype(int).values


# ─────────────────────────────────────────────
# Main engineering runner
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to the merged daily dataset.
    Expects columns from preprocessor.preprocess_merged().
    """
    logger.info("Engineering climate features ...")

    df = df.copy()

    # Heat Index & Discomfort
    df["heat_index"]      = heat_index(df["temperature_mean"].values, df["humidity"].values)
    df["discomfort_index"] = discomfort_index(df["temperature_mean"].values, df["humidity"].values)

    # Rainfall anomaly
    df["rainfall_anomaly"] = rainfall_anomaly(df["rainfall"].values)

    # UHI score
    df["uhi_score"] = uhi_score(df["temperature_mean"].values)

    # Pollution severity
    df["pollution_severity"] = pollution_severity(df["pm25"].values, df["aqi"].values)

    # AQI ordinal
    if "aqi_class" in df.columns:
        df["aqi_ordinal"] = aqi_category_ordinal(df["aqi_class"])

    # Season-encoded
    season_map = {"Winter": 0, "Summer": 1, "Southwest Monsoon": 2, "Post-Monsoon": 3}
    df["season_code"] = df["season"].map(season_map).fillna(0).astype(int)

    # Risk ordinals
    if "health_risk" in df.columns:
        df["health_risk_ordinal"] = df["health_risk"].map(
            {"Low":0,"Moderate":1,"High":2,"Severe":3}).fillna(0).astype(int)
    if "flood_risk" in df.columns:
        df["flood_risk_ordinal"] = df["flood_risk"].map(
            {"Low":0,"Medium":1,"High":2}).fillna(0).astype(int)
    if "dengue_risk" in df.columns:
        df["dengue_risk_ordinal"] = df["dengue_risk"].map(
            {"Low":0,"Medium":1,"High":2}).fillna(0).astype(int)
    if "temp_class" in df.columns:
        df["temp_class_ordinal"] = df["temp_class"].map(
            {"Normal":0,"Hot":1,"Very Hot":2,"Heat-Wave":3}).fillna(0).astype(int)
    if "rain_class" in df.columns:
        df["rain_class_ordinal"] = df["rain_class"].map(
            {"No Rain":0,"Light":1,"Moderate":2,"Heavy":3,"Extreme":4}).fillna(0).astype(int)

    # Lag features (t-1 and t-7) for time series
    for col in ["temperature_mean", "rainfall", "aqi", "humidity"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1).bfill()
            df[f"{col}_lag7"] = df[col].shift(7).bfill()

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def run_feature_engineering():
    """Load merged dataset, engineer features, save."""
    input_path  = os.path.join(DATA_PROC_DIR, "merged_dataset.csv")
    output_path = os.path.join(DATA_PROC_DIR, "features_dataset.csv")

    if not os.path.exists(input_path):
        logger.error(f"Input not found: {input_path}. Run preprocessor.py first.")
        return None

    df = pd.read_csv(input_path, parse_dates=["date"])
    df_fe = engineer_features(df)
    df_fe.to_csv(output_path, index=False)
    logger.info(f"Features dataset saved: {output_path}")
    return df_fe


if __name__ == "__main__":
    run_feature_engineering()
    print("\nFeature engineering complete. Saved to data/processed/features_dataset.csv")
