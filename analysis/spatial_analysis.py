"""
analysis/spatial_analysis.py
Ward-wise spatial clustering, hotspot detection, and UHI mapping
for the Bengaluru Climate Intelligence Platform.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DATA_SYN_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def load_ward_data() -> pd.DataFrame:
    path = os.path.join(DATA_SYN_DIR, "ward.csv")
    if not os.path.exists(path):
        logger.error(f"Ward data not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def spatial_clustering(df_ward: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    K-Means clustering of wards based on climate & pollution features.
    Returns df_ward with 'cluster' column added.
    """
    if df_ward.empty:
        return df_ward
    feat_cols = ["avg_temp","annual_rainfall","avg_aqi","ndvi","buildup_pct"]
    feat_cols = [c for c in feat_cols if c in df_ward.columns]
    latest = df_ward.groupby("ward")[feat_cols].mean().reset_index()

    scaler = StandardScaler()
    X = scaler.fit_transform(latest[feat_cols].fillna(0))

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    latest["cluster"] = km.fit_predict(X)

    cluster_profiles = latest.groupby("cluster")[feat_cols].mean().round(2)
    logger.info(f"Spatial clustering: {n_clusters} clusters formed.")
    logger.info(f"Cluster profiles:\n{cluster_profiles}")

    return df_ward.merge(latest[["ward","cluster"]], on="ward", how="left")


def detect_hotspots(df_ward: pd.DataFrame, temp_percentile: float = 75) -> pd.DataFrame:
    """
    Identify UHI hotspot wards (avg_temp above percentile threshold).
    """
    if df_ward.empty or "avg_temp" not in df_ward.columns:
        return df_ward
    latest = df_ward.groupby("ward")[["avg_temp","avg_aqi","ndvi","lat","lon"]].mean().reset_index()
    threshold = latest["avg_temp"].quantile(temp_percentile / 100)
    latest["is_uhi_hotspot"] = (latest["avg_temp"] >= threshold).astype(int)
    n_hotspots = latest["is_uhi_hotspot"].sum()
    logger.info(f"UHI hotspots detected: {n_hotspots} wards (threshold: {threshold:.2f}°C)")
    return latest


def lake_shrinkage_analysis(df_sat: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze water body shrinkage over time from satellite data.
    """
    if df_sat.empty or "water_pct" not in df_sat.columns:
        return pd.DataFrame()
    baseline = df_sat["water_pct"].iloc[0]
    df_sat = df_sat.copy()
    df_sat["water_loss_pct"]  = ((baseline - df_sat["water_pct"]) / baseline * 100).round(2)
    df_sat["cumulative_loss"] = df_sat["water_loss_pct"].cummax()
    return df_sat[["year","water_pct","water_loss_pct","cumulative_loss"]]


def vegetation_change_analysis(df_sat: pd.DataFrame) -> pd.DataFrame:
    """
    Track NDVI decline and built-up area increase over time.
    """
    if df_sat.empty:
        return pd.DataFrame()
    df = df_sat.copy()
    df["ndvi_change"] = df["ndvi"].diff().fillna(0).round(4)
    df["buildup_change"] = df["built_up_pct"].diff().fillna(0).round(2)
    return df[["year","ndvi","ndvi_change","built_up_pct","buildup_change","vegetation_pct"]]


def flood_prone_zones(df_ward: pd.DataFrame) -> pd.DataFrame:
    """
    Return wards with 'High' flood risk and their coordinates.
    """
    if df_ward.empty or "flood_risk" not in df_ward.columns:
        return pd.DataFrame()
    return df_ward[df_ward["flood_risk"] == "High"][
        ["ward","lat","lon","avg_temp","annual_rainfall","flood_risk"]
    ].drop_duplicates("ward")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    df_ward = load_ward_data()
    if not df_ward.empty:
        df_clustered = spatial_clustering(df_ward)
        hotspots     = detect_hotspots(df_ward)
        flood_zones  = flood_prone_zones(df_ward)
        print(f"\nHotspot wards:\n{hotspots[hotspots['is_uhi_hotspot']==1][['ward','avg_temp']].to_string()}")
        print(f"\nFlood-prone zones: {len(flood_zones)} wards")
    print("\n✅  Spatial analysis complete.")
