"""
visualization/geo_maps.py
Folium/Leaflet-based geospatial maps for the Bengaluru Climate Platform.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import BENGALURU_BOUNDS


def create_ward_map(df_ward: pd.DataFrame, value_col: str = "avg_temp",
                    title: str = "Ward Climate Map") -> str:
    """
    Create a Folium choropleth-style ward map.
    Returns HTML string of the map.
    """
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        return "<p>folium not installed. Run: pip install folium</p>"

    center = [BENGALURU_BOUNDS["center_lat"], BENGALURU_BOUNDS["center_lon"]]
    m = folium.Map(location=center, zoom_start=11,
                   tiles="CartoDB dark_matter", prefer_canvas=True)

    if df_ward.empty or value_col not in df_ward.columns:
        return m._repr_html_()

    latest = df_ward.groupby("ward")[[value_col,"lat","lon"]].mean().reset_index()
    vmin   = latest[value_col].min()
    vmax   = latest[value_col].max()

    def val_to_color(v):
        ratio = (v - vmin) / (vmax - vmin + 1e-9)
        r = int(255 * ratio)
        b = int(255 * (1 - ratio))
        return f"#{r:02x}40{b:02x}"

    for _, row in latest.iterrows():
        color = val_to_color(row[value_col])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=14,
            color=color, fill=True, fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{row['ward']}</b><br>{value_col}: {row[value_col]:.2f}", max_width=200),
            tooltip=f"{row['ward']}: {row[value_col]:.2f}",
        ).add_to(m)

    return m._repr_html_()


def create_heatmap(df_ward: pd.DataFrame, value_col: str = "avg_temp") -> str:
    """Create a Folium HeatMap from ward data."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        return "<p>folium not installed.</p>"

    center = [BENGALURU_BOUNDS["center_lat"], BENGALURU_BOUNDS["center_lon"]]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB dark_matter")

    if df_ward.empty or value_col not in df_ward.columns:
        return m._repr_html_()

    latest = df_ward.groupby("ward")[[value_col,"lat","lon"]].mean().reset_index().dropna()
    heat_data = [[row["lat"], row["lon"], row[value_col]] for _, row in latest.iterrows()]

    HeatMap(heat_data, radius=25, blur=15, min_opacity=0.4,
            gradient={"0.4":"blue","0.65":"lime","1":"red"}).add_to(m)

    return m._repr_html_()


def create_flood_risk_map(df_ward: pd.DataFrame) -> str:
    """Map wards colored by flood risk level."""
    try:
        import folium
    except ImportError:
        return "<p>folium not installed.</p>"

    center = [BENGALURU_BOUNDS["center_lat"], BENGALURU_BOUNDS["center_lon"]]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB dark_matter")

    risk_colors = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f78166"}

    if not df_ward.empty and "flood_risk" in df_ward.columns:
        latest = df_ward.groupby("ward").agg(
            {"flood_risk": "last", "lat": "mean", "lon": "mean"}
        ).reset_index()
        for _, row in latest.iterrows():
            color = risk_colors.get(row["flood_risk"], "#8b949e")
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=12, color=color, fill=True,
                fill_color=color, fill_opacity=0.8,
                tooltip=f"{row['ward']}: {row['flood_risk']} flood risk",
            ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                background:#161b22;padding:12px;border-radius:8px;
                border:1px solid #30363d;font-family:Inter,sans-serif;color:#e6edf3;">
      <b>Flood Risk</b><br>
      <span style="color:#3fb950">●</span> Low<br>
      <span style="color:#d29922">●</span> Medium<br>
      <span style="color:#f78166">●</span> High
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m._repr_html_()
