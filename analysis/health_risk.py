"""
analysis/health_risk.py
Human health risk assessment module for the Bengaluru Climate Intelligence Platform.

Computes composite risk scores for:
  - Heat exhaustion / Heat stroke
  - Dehydration
  - Respiratory / Asthma risk
  - Flood-related health hazards
  - Dengue/chikungunya vector risk
  - Sleep disturbance due to heat
  - Human productivity reduction
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DATA_PROC_DIR, DATA_SYN_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Individual risk calculators (0–100 scale)
# ─────────────────────────────────────────────

def heat_exhaustion_risk(temp: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """Risk increases rapidly when heat index (temp+humidity effect) is high."""
    hi = temp + 0.33 * (humidity / 100 * 6.105 * np.exp(17.27 * temp / (237.7 + temp))) - 4
    score = np.clip((hi - 27) / 20 * 100, 0, 100)
    return np.round(score, 1)


def dehydration_risk(temp: np.ndarray, humidity: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """Higher temp + lower humidity + higher wind = higher dehydration risk."""
    score = np.clip(((temp - 20) / 20 * 50) + ((100 - humidity) / 100 * 30) + (wind / 40 * 20), 0, 100)
    return np.round(score, 1)


def respiratory_risk(aqi: np.ndarray, pm25: np.ndarray) -> np.ndarray:
    """Respiratory risk driven by PM2.5 and overall AQI."""
    score = np.clip(0.6 * (pm25 / 250 * 100) + 0.4 * (aqi / 500 * 100), 0, 100)
    return np.round(score, 1)


def flood_health_risk(rainfall: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """Flooding creates sanitation and injury health hazards."""
    score = np.clip(0.7 * (rainfall / 100 * 100) + 0.3 * (humidity / 100 * 100), 0, 100)
    return np.round(score, 1)


def dengue_vector_risk(temp: np.ndarray, rainfall: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """
    Mosquito breeding is optimal at 25-35°C, after rainfall, with high humidity.
    Dengue transmission peaks 2-4 weeks post-monsoon.
    """
    temp_factor     = np.exp(-0.5 * ((temp - 30) / 4)**2)       # peak at 30°C
    rain_factor     = np.clip(rainfall / 50, 0, 1)
    humidity_factor = np.clip((humidity - 50) / 50, 0, 1)
    score = (0.5 * temp_factor + 0.3 * rain_factor + 0.2 * humidity_factor) * 100
    return np.round(np.clip(score, 0, 100), 1)


def sleep_disturbance_risk(temp_min: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """Night-time temps > 24°C + humidity > 70% disrupt sleep quality."""
    score = np.clip(((temp_min - 20) / 15 * 60) + ((humidity - 50) / 50 * 40), 0, 100)
    return np.round(score, 1)


def productivity_reduction(temp: np.ndarray, aqi: np.ndarray) -> np.ndarray:
    """
    Based on research: >33°C and poor air quality reduce cognitive & physical productivity.
    """
    temp_effect = np.clip((temp - 28) / 15 * 60, 0, 60)
    aqi_effect  = np.clip(aqi / 500 * 40, 0, 40)
    score       = np.clip(temp_effect + aqi_effect, 0, 100)
    return np.round(score, 1)


def overall_health_risk_score(scores_dict: dict) -> np.ndarray:
    """Weighted composite of all individual risk scores."""
    weights = {
        "heat_exhaustion":  0.20,
        "dehydration":      0.10,
        "respiratory":      0.20,
        "flood_health":     0.10,
        "dengue_vector":    0.15,
        "sleep_disturb":    0.10,
        "productivity":     0.15,
    }
    composite = np.zeros(next(iter(scores_dict.values())).shape)
    for key, w in weights.items():
        if key in scores_dict:
            composite += w * scores_dict[key]
    return np.round(composite, 1)


def risk_label(score: float) -> str:
    if score < 25:  return "Low"
    if score < 50:  return "Moderate"
    if score < 75:  return "High"
    return "Severe"


# ─────────────────────────────────────────────
# Main health risk assessment pipeline
# ─────────────────────────────────────────────

def compute_health_risks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all health risk columns to the input dataframe.
    """
    logger.info("Computing health risk indicators ...")
    df = df.copy()

    # Safe defaults for missing columns
    temp     = df.get("temperature_mean", pd.Series(np.full(len(df), 29))).values
    temp_min = df.get("temperature_min",  pd.Series(temp - 4)).values
    humidity = df.get("humidity",         pd.Series(np.full(len(df), 65))).values
    wind     = df.get("wind_speed",       pd.Series(np.full(len(df), 12))).values
    rainfall = df.get("rainfall",         pd.Series(np.full(len(df), 3))).values
    aqi      = df.get("aqi",             pd.Series(np.full(len(df), 80))).values
    pm25     = df.get("pm25",            pd.Series(np.full(len(df), 40))).values

    scores = {
        "heat_exhaustion": heat_exhaustion_risk(temp, humidity),
        "dehydration":     dehydration_risk(temp, humidity, wind),
        "respiratory":     respiratory_risk(aqi, pm25),
        "flood_health":    flood_health_risk(rainfall, humidity),
        "dengue_vector":   dengue_vector_risk(temp, rainfall, humidity),
        "sleep_disturb":   sleep_disturbance_risk(temp_min, humidity),
        "productivity":    productivity_reduction(temp, aqi),
    }

    for key, arr in scores.items():
        df[f"risk_{key}"] = arr

    composite = overall_health_risk_score(scores)
    df["health_risk_score"] = composite
    df["health_risk_label"] = [risk_label(s) for s in composite]

    logger.info(f"Health risk scores computed for {len(df)} records.")
    return df


def load_and_assess():
    """Load merged dataset, compute risks, save."""
    for base in [DATA_PROC_DIR, DATA_SYN_DIR]:
        for fname in ["merged_dataset.csv", "meteorological.csv"]:
            p = os.path.join(base, fname)
            if os.path.exists(p):
                df = pd.read_csv(p, parse_dates=["date"])
                # Also load AQI if not merged
                if "aqi" not in df.columns:
                    aqi_p = os.path.join(DATA_SYN_DIR, "aqi.csv")
                    if os.path.exists(aqi_p):
                        df_aqi = pd.read_csv(aqi_p, parse_dates=["date"])
                        df = df.merge(df_aqi[["date","aqi","pm25"]], on="date", how="left")
                df_assessed = compute_health_risks(df)
                out = os.path.join(DATA_PROC_DIR, "health_risks.csv")
                os.makedirs(DATA_PROC_DIR, exist_ok=True)
                df_assessed.to_csv(out, index=False)
                logger.info(f"Health risks saved to {out}")
                return df_assessed
    return pd.DataFrame()


if __name__ == "__main__":
    df = load_and_assess()
    if not df.empty:
        risk_cols = [c for c in df.columns if c.startswith("risk_")]
        print("\n── Health Risk Score Summary ──")
        print(df[risk_cols + ["health_risk_score","health_risk_label"]].describe().round(2))
    print("\n✅  Health risk assessment complete.")
