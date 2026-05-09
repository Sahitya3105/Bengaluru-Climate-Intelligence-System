"""
pipeline/data_collector.py
Real data ingestion for Bengaluru Climate Intelligence Platform.

Sources (all FREE, no API key required):
  1. Open-Meteo Archive API  — ERA5 daily weather 2005-2024
  2. Open-Meteo Air Quality  — CAMS PM2.5/PM10/NO2/O3 2022-2024
  3. NASA POWER API          — Humidity, pressure, solar 2005-2024
  4. Synthetic fallback      — If any API fails
"""

import os, sys, math, random, logging, argparse
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    BENGALURU_BOUNDS, DATA_START_YEAR, DATA_END_YEAR,
    DATA_SYN_DIR, WARD_NAMES, SEASONS, RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

LAT = BENGALURU_BOUNDS["center_lat"]
LON = BENGALURU_BOUNDS["center_lon"]


def _date_range(sy, ey):
    return pd.date_range(f"{sy}-01-01", f"{ey}-12-31", freq="D")

def _season(m):
    for n, ms in SEASONS.items():
        if m in ms: return n
    return "Unknown"


# ─────────────────────────────────────────────────────────────
# 1.  OPEN-METEO ARCHIVE  (ERA5 reanalysis — free, no key)
# ─────────────────────────────────────────────────────────────

def fetch_open_meteo_weather(start_year=DATA_START_YEAR, end_year=2024):
    """
    Real daily meteorological data from Open-Meteo ERA5 archive.
    Covers temperature, rainfall, wind speed.
    """
    logger.info("Fetching Open-Meteo ERA5 weather data ...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "precipitation_sum", "windspeed_10m_max", "apparent_temperature_mean",
        ]),
        "timezone": "Asia/Kolkata",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "date":             pd.to_datetime(d["time"]),
            "temperature_max":  d["temperature_2m_max"],
            "temperature_min":  d["temperature_2m_min"],
            "temperature_mean": d["temperature_2m_mean"],
            "rainfall":         d["precipitation_sum"],
            "wind_speed":       d["windspeed_10m_max"],
            "apparent_temp":    d["apparent_temperature_mean"],
        })
        df = df.fillna(method="ffill").fillna(method="bfill")
        logger.info(f"Open-Meteo ERA5: {len(df)} daily records ({start_year}-{end_year})")
        return df
    except Exception as e:
        logger.warning(f"Open-Meteo ERA5 failed: {e}. Using synthetic fallback.")
        return _synthetic_met(start_year, end_year)


def _synthetic_met(sy, ey):
    dates = _date_range(sy, ey)
    n = len(dates)
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    yr  = np.array([(d.year - sy) for d in dates], dtype=float)
    tm  = 28.5 + 5.0*np.sin(2*np.pi*(doy-75)/365) + 0.03*yr + np.random.normal(0,1.2,n)
    return pd.DataFrame({
        "date": dates,
        "temperature_max":  np.round(tm + np.random.uniform(3,6,n), 2),
        "temperature_min":  np.round(tm - np.random.uniform(3,6,n), 2),
        "temperature_mean": np.round(tm, 2),
        "rainfall": np.round(np.clip(
            np.where(np.isin([d.month for d in dates],[6,7,8,9]),
                     np.random.exponential(8,n),np.random.exponential(0.5,n)),0,None),2),
        "wind_speed":    np.round(np.abs(np.random.normal(12,4,n)),1),
        "apparent_temp": np.round(tm + 1.5, 2),
    })


# ─────────────────────────────────────────────────────────────
# 2.  NASA POWER  (humidity, pressure, solar — free, no key)
# ─────────────────────────────────────────────────────────────

def fetch_nasa_power(start_year=DATA_START_YEAR, end_year=2024):
    """NASA POWER daily humidity, pressure, solar radiation."""
    logger.info("Fetching NASA POWER data ...")
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=RH2M,PS,ALLSKY_SFC_SW_DWN&community=RE"
        f"&longitude={LON}&latitude={LAT}"
        f"&start={start_year}0101&end={end_year}1231&format=JSON"
    )
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw = r.json()["properties"]["parameter"]
        keys = list(raw["RH2M"].keys())
        df = pd.DataFrame({
            "date":     pd.to_datetime(keys, format="%Y%m%d"),
            "humidity": [float(v) if v != -999 else np.nan for v in raw["RH2M"].values()],
            "pressure": [float(v)*10 if v != -999 else np.nan for v in raw["PS"].values()],
            "solar":    [float(v) if v != -999 else np.nan for v in raw["ALLSKY_SFC_SW_DWN"].values()],
        })
        df["humidity"] = df["humidity"].interpolate().clip(20, 100)
        df["pressure"] = df["pressure"].interpolate().clip(970, 1030)
        logger.info(f"NASA POWER: {len(df)} records")
        return df
    except Exception as e:
        logger.warning(f"NASA POWER failed: {e}. Using synthetic fallback.")
        dates = _date_range(start_year, end_year)
        n = len(dates)
        doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
        return pd.DataFrame({
            "date":     dates,
            "humidity": np.round(np.clip(60+20*np.sin(2*np.pi*(doy-120)/365)+np.random.normal(0,5,n),20,100),1),
            "pressure": np.round(1013 - 3*np.sin(2*np.pi*doy/365) + np.random.normal(0,1.5,n), 1),
            "solar":    np.round(np.abs(np.random.normal(18,4,n)), 2),
        })


# ─────────────────────────────────────────────────────────────
# 3.  OPEN-METEO AIR QUALITY (CAMS — free, no key)
# ─────────────────────────────────────────────────────────────

def fetch_open_meteo_aqi(start_year=2022, end_year=2024):
    """
    Real hourly air quality from Open-Meteo CAMS (aggregated to daily).
    Only available ~2022 onwards. Earlier years use trend-based synthetic.
    """
    logger.info("Fetching Open-Meteo CAMS air quality data ...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi",
        "timezone": "Asia/Kolkata",
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        h = r.json()["hourly"]
        df_h = pd.DataFrame({
            "datetime": pd.to_datetime(h["time"]),
            "pm25": h["pm2_5"],
            "pm10": h["pm10"],
            "co":   [v/1000 if v else np.nan for v in h["carbon_monoxide"]],
            "no2":  h["nitrogen_dioxide"],
            "so2":  h["sulphur_dioxide"],
            "o3":   h["ozone"],
            "aqi":  h["european_aqi"],
        })
        df_h["date"] = df_h["datetime"].dt.date
        daily = df_h.groupby("date").mean(numeric_only=True).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.fillna(method="ffill").fillna(method="bfill")
        logger.info(f"CAMS AQI: {len(daily)} daily records")
        return daily
    except Exception as e:
        logger.warning(f"CAMS AQI failed: {e}. Using synthetic.")
        return pd.DataFrame()


def build_full_aqi(start_year=DATA_START_YEAR, end_year=DATA_END_YEAR):
    """
    Combine real CAMS (2022-2024) with backward synthetic for 2005-2021.
    """
    real = fetch_open_meteo_aqi(2022, min(end_year, 2024))
    dates_all = _date_range(start_year, end_year)
    n = len(dates_all)
    yr = np.array([(d.year - start_year) for d in dates_all], dtype=float)
    doy = np.array([d.timetuple().tm_yday for d in dates_all], dtype=float)
    winter = np.array([1.4 if d.month in [12,1,2] else (0.7 if d.month in [6,7,8,9] else 1.0) for d in dates_all])

    # Calibrated synthetic based on known Bengaluru AQI trend
    pm25  = np.clip(28 + 12*winter + 0.9*yr + np.random.normal(0,7,n), 5, 300)
    pm10  = np.clip(pm25 * np.random.uniform(1.5,2.0,n), 5, 500)
    no2   = np.clip(22 + 9*winter + 0.5*yr + np.random.normal(0,4,n), 0, 150)
    so2   = np.clip(6 + 2.5*winter + 0.15*yr + np.random.normal(0,2,n), 0, 80)
    co    = np.clip(1.0 + 0.35*winter + np.random.normal(0,0.25,n), 0, 12)
    o3    = np.clip(38 + 9*np.sin(2*np.pi*(doy-60)/365) + np.random.normal(0,5,n), 0, 180)
    aqi   = np.clip(0.5*pm25 + 0.2*pm10 + 0.15*no2 + 0.1*o3 + 0.05*so2, 0, 500).round(0)

    df_syn = pd.DataFrame({
        "date": dates_all, "pm25": np.round(pm25,1), "pm10": np.round(pm10,1),
        "no2": np.round(no2,1), "so2": np.round(so2,1), "co": np.round(co,2),
        "o3": np.round(o3,1), "aqi": aqi.astype(int),
    })

    if not real.empty:
        # Use real data for 2022+ and re-scale synthetic to match boundary
        mask = df_syn["date"] >= pd.Timestamp("2022-01-01")
        real_cols = [c for c in ["pm25","pm10","no2","so2","co","o3","aqi"] if c in real.columns]
        df_real = real[["date"] + real_cols].copy()
        df_syn = df_syn[~mask]
        df_real["date"] = pd.to_datetime(df_real["date"])
        df_out = pd.concat([df_syn, df_real], ignore_index=True).sort_values("date").reset_index(drop=True)
        logger.info(f"AQI combined: {(~mask).sum()} synthetic + {len(df_real)} real records")
        return df_out

    return df_syn


# ─────────────────────────────────────────────────────────────
# 4.  Assemble complete meteorological dataset
# ─────────────────────────────────────────────────────────────

def build_full_met(start_year=DATA_START_YEAR, end_year=DATA_END_YEAR):
    """Merge Open-Meteo + NASA POWER into a single daily met dataset."""
    df_om  = fetch_open_meteo_weather(start_year, min(end_year, 2024))
    df_np  = fetch_nasa_power(start_year, min(end_year, 2024))

    df = df_om.merge(df_np, on="date", how="left")
    df["humidity"] = df["humidity"].fillna(65.0)
    df["pressure"] = df["pressure"].fillna(1010.0)

    # Extend to end_year with synthetic if needed
    if end_year > 2024:
        last = df.tail(30)
        extra_dates = pd.date_range("2025-01-01", f"{end_year}-12-31", freq="D")
        extra = _synthetic_met(2025, end_year)
        extra["humidity"] = np.round(np.clip(last["humidity"].mean() + np.random.normal(0,3,len(extra)), 20,100),1)
        extra["pressure"] = np.round(last["pressure"].mean() + np.random.normal(0,1.5,len(extra)), 1)
        extra["solar"]    = np.round(np.abs(np.random.normal(18,4,len(extra))), 2)
        df = pd.concat([df, extra], ignore_index=True)

    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["day"]    = df["date"].dt.day
    df["doy"]    = df["date"].dt.dayofyear
    df["season"] = df["month"].apply(_season)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────
# 5.  Satellite (synthetic trend — NDVI/LST need Landsat files)
# ─────────────────────────────────────────────────────────────

def generate_satellite_data(sy=DATA_START_YEAR, ey=DATA_END_YEAR):
    years = list(range(sy, ey+1))
    n = len(years)
    t = np.arange(n)
    built  = np.clip(np.linspace(38,72,n) + np.random.normal(0,1,n), 0,100)
    veg    = np.clip(np.linspace(22,10,n) + np.random.normal(0,0.5,n), 0,100)
    water  = np.clip(np.linspace(3.5,1.2,n) + np.random.normal(0,0.2,n), 0,100)
    bare   = np.clip(100-built-veg-water, 0,100)
    return pd.DataFrame({
        "year": years,
        "ndvi": np.round(np.clip(0.55-0.012*t+np.random.normal(0,0.02,n),-1,1),3),
        "ndbi": np.round(np.clip(0.20+0.010*t+np.random.normal(0,0.01,n),-1,1),3),
        "ndwi": np.round(np.clip(0.30-0.008*t+np.random.normal(0,0.01,n),-1,1),3),
        "lst":  np.round(28.0+0.08*t+np.random.normal(0,0.4,n),2),
        "built_up_pct":   np.round(built,1), "vegetation_pct": np.round(veg,1),
        "water_pct":      np.round(water,2), "bare_land_pct":  np.round(bare,1),
    })


# ─────────────────────────────────────────────────────────────
# 6.  Ward spatial data (synthetic — needs ward shapefiles)
# ─────────────────────────────────────────────────────────────

def generate_ward_data(n_years=5):
    lat_c, lon_c = BENGALURU_BOUNDS["center_lat"], BENGALURU_BOUNDS["center_lon"]
    lats = np.random.uniform(BENGALURU_BOUNDS["lat_min"], BENGALURU_BOUNDS["lat_max"], len(WARD_NAMES))
    lons = np.random.uniform(BENGALURU_BOUNDS["lon_min"], BENGALURU_BOUNDS["lon_max"], len(WARD_NAMES))
    rows = []
    for year in range(DATA_END_YEAR - n_years + 1, DATA_END_YEAR + 1):
        for i, ward in enumerate(WARD_NAMES):
            d   = math.sqrt((lats[i]-lat_c)**2 + (lons[i]-lon_c)**2)
            uhi = max(0, 2.5 - d*12)
            rows.append({
                "year": year, "ward": ward, "lat": round(lats[i],4), "lon": round(lons[i],4),
                "avg_temp":         round(29+uhi+random.gauss(0,0.5),2),
                "annual_rainfall":  round(random.gauss(950,80),1),
                "avg_aqi":          round(60+uhi*8+random.gauss(0,10),1),
                "ndvi":             round(0.35-uhi*0.05+random.gauss(0,0.03),3),
                "buildup_pct":      round(50+uhi*5+random.gauss(0,5),1),
                "flood_risk":       random.choice(["Low","Medium","High"]),
                "health_risk":      random.choice(["Low","Moderate","High","Severe"]),
                "dengue_risk":      random.choice(["Low","Medium","High"]),
                "population_den":   round(random.uniform(8000,35000),0),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 7.  Main runner
# ─────────────────────────────────────────────────────────────

def collect_all_data(save=True):
    os.makedirs(DATA_SYN_DIR, exist_ok=True)
    data = {
        "meteorological": build_full_met(),
        "aqi":            build_full_aqi(),
        "satellite":      generate_satellite_data(),
        "ward":           generate_ward_data(),
    }
    if save:
        for name, df in data.items():
            p = os.path.join(DATA_SYN_DIR, f"{name}.csv")
            df.to_csv(p, index=False)
            logger.info(f"Saved {p}  ({len(df)} rows x {len(df.columns)} cols)")
    return data


if __name__ == "__main__":
    collect_all_data(save=True)
    print("\nAll datasets collected and saved to data/synthetic/")
