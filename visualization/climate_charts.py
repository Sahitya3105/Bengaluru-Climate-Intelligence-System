"""
visualization/climate_charts.py
Plotly-based climate visualizations for the Bengaluru Climate Intelligence Platform.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PALETTE = {
    "bg":      "#0d1117",
    "card":    "#161b22",
    "border":  "#30363d",
    "accent1": "#58a6ff",
    "accent2": "#f78166",
    "accent3": "#3fb950",
    "accent4": "#d29922",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
}

LAYOUT_BASE = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor=PALETTE["card"],
    font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
)


def temp_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Annual temperature trend with warming line."""
    if df.empty or "year" not in df.columns:
        return go.Figure()
    annual = df.groupby("year")["temperature_mean"].mean().reset_index()
    z = np.polyfit(annual["year"], annual["temperature_mean"], 1)
    trend = np.poly1d(z)(annual["year"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["temperature_mean"],
        mode="lines+markers", name="Avg Temp (°C)",
        line=dict(color=PALETTE["accent2"], width=2),
        marker=dict(size=6),
        fill="tozeroy", fillcolor="rgba(247,129,102,0.12)",
    ))
    fig.add_trace(go.Scatter(
        x=annual["year"], y=trend,
        mode="lines", name="Trend",
        line=dict(color=PALETTE["accent4"], width=2, dash="dash"),
    ))
    fig.update_layout(**LAYOUT_BASE, title="📈 Annual Mean Temperature Trend — Bengaluru",
                      xaxis_title="Year", yaxis_title="Temperature (°C)", legend=dict(bgcolor=PALETTE["card"]))
    return fig


def rainfall_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Annual rainfall bar chart."""
    if df.empty or "year" not in df.columns:
        return go.Figure()
    annual = df.groupby("year")["rainfall"].sum().reset_index()
    fig = go.Figure(go.Bar(
        x=annual["year"], y=annual["rainfall"],
        marker=dict(color=annual["rainfall"], colorscale="Blues",
                    colorbar=dict(title="mm"), line=dict(width=0)),
        name="Annual Rainfall",
    ))
    fig.update_layout(**LAYOUT_BASE, title="🌧️ Annual Rainfall — Bengaluru",
                      xaxis_title="Year", yaxis_title="Total Rainfall (mm)")
    return fig


def aqi_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Monthly AQI heatmap by year."""
    if df.empty or "aqi" not in df.columns:
        return go.Figure()
    pivot = df.groupby(["year","month"])["aqi"].mean().unstack(fill_value=0)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=months[:pivot.shape[1]],
        y=pivot.index.tolist(),
        colorscale="RdYlGn_r",
        colorbar=dict(title="AQI"),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>AQI: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(**LAYOUT_BASE, title="🌫️ Monthly AQI Heatmap by Year",
                      xaxis_title="Month", yaxis_title="Year")
    return fig


def seasonal_boxplot(df: pd.DataFrame, variable: str = "temperature_mean") -> go.Figure:
    """Box-plot of a variable across four seasons."""
    if df.empty or variable not in df.columns or "season" not in df.columns:
        return go.Figure()
    season_order = ["Winter","Summer","Southwest Monsoon","Post-Monsoon"]
    colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"], PALETTE["accent4"]]
    fig = go.Figure()
    for i, season in enumerate(season_order):
        sub = df[df["season"] == season][variable].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Box(y=sub, name=season, marker_color=colors[i % 4],
                             boxmean="sd", line=dict(width=1.5)))
    label = variable.replace("_"," ").title()
    fig.update_layout(**LAYOUT_BASE, title=f"📦 {label} — Seasonal Distribution",
                      yaxis_title=label, showlegend=True)
    return fig


def satellite_lulc_chart(df_sat: pd.DataFrame) -> go.Figure:
    """Stacked area chart of land-use land-cover change."""
    if df_sat.empty or "year" not in df_sat.columns:
        return go.Figure()
    cols = ["built_up_pct","vegetation_pct","water_pct","bare_land_pct"]
    cols = [c for c in cols if c in df_sat.columns]
    labels = {"built_up_pct":"Built-Up","vegetation_pct":"Vegetation",
              "water_pct":"Water Bodies","bare_land_pct":"Bare Land"}
    colors = [PALETTE["accent2"],PALETTE["accent3"],PALETTE["accent1"],PALETTE["accent4"]]
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=df_sat["year"], y=df_sat[col], name=labels.get(col,col),
            stackgroup="one", mode="none",
            fillcolor=colors[i % len(colors)],
        ))
    fig.update_layout(**LAYOUT_BASE, title="🛰️ Land-Use Land-Cover Change (2005–2026)",
                      xaxis_title="Year", yaxis_title="Area (%)")
    return fig


def ndvi_lst_dual(df_sat: pd.DataFrame) -> go.Figure:
    """Dual-axis: NDVI decline vs LST increase."""
    if df_sat.empty:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df_sat["year"], y=df_sat.get("ndvi", []),
        name="NDVI", line=dict(color=PALETTE["accent3"], width=2),
        mode="lines+markers",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df_sat["year"], y=df_sat.get("lst", []),
        name="LST (°C)", line=dict(color=PALETTE["accent2"], width=2, dash="dot"),
        mode="lines+markers",
    ), secondary_y=True)
    fig.update_layout(**LAYOUT_BASE, title="🌿 NDVI vs Land Surface Temperature (LST)")
    fig.update_yaxes(title_text="NDVI", secondary_y=False,
                     gridcolor=PALETTE["border"], color=PALETTE["accent3"])
    fig.update_yaxes(title_text="LST (°C)", secondary_y=True, color=PALETTE["accent2"])
    return fig


def pollution_multi_line(df_aqi: pd.DataFrame) -> go.Figure:
    """Annual trend of all pollutants."""
    if df_aqi.empty:
        return go.Figure()
    annual = df_aqi.groupby("year")[["pm25","pm10","no2","so2","o3"]].mean().reset_index()
    colors = [PALETTE["accent2"],PALETTE["accent4"],PALETTE["accent1"],
              PALETTE["accent3"],"#bc8cff"]
    pollutants = ["pm25","pm10","no2","so2","o3"]
    fig = go.Figure()
    for i, p in enumerate(pollutants):
        if p in annual.columns:
            fig.add_trace(go.Scatter(
                x=annual["year"], y=annual[p], name=p.upper(),
                line=dict(color=colors[i % len(colors)], width=2),
                mode="lines+markers",
            ))
    fig.update_layout(**LAYOUT_BASE, title="🏭 Annual Pollutant Trends",
                      xaxis_title="Year", yaxis_title="Concentration (μg/m³)")
    return fig


def health_risk_radar(risk_scores: dict) -> go.Figure:
    """Radar chart of health risk dimensions."""
    categories = list(risk_scores.keys())
    values     = list(risk_scores.values())
    values.append(values[0])
    categories.append(categories[0])
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        fillcolor="rgba(88,166,255,0.2)",
        line=dict(color=PALETTE["accent1"], width=2),
        marker=dict(size=6, color=PALETTE["accent1"]),
    ))
    fig.update_layout(
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        polar=dict(
            bgcolor=PALETTE["card"],
            radialaxis=dict(visible=True, range=[0,100], gridcolor=PALETTE["border"]),
            angularaxis=dict(gridcolor=PALETTE["border"]),
        ),
        title="🩺 Health Risk Radar", margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def forecast_comparison_chart(actual: list, predicted: list, dates: list, label: str = "Temperature") -> go.Figure:
    """Actual vs predicted forecast comparison."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines",
                             name="Actual", line=dict(color=PALETTE["accent1"], width=2)))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines",
                             name="Predicted", line=dict(color=PALETTE["accent2"], width=2, dash="dash")))
    fig.update_layout(**LAYOUT_BASE, title=f"🔮 {label} — Actual vs Predicted",
                      xaxis_title="Date", yaxis_title=label)
    return fig
